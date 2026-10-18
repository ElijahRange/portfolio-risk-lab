"""Extra risk measures for the Risk Analysis tab.

Everything holds today's weights constant and uses public, textbook methods: historical, normal,
Cornish-Fisher and EWMA (RiskMetrics) value at risk, a Kupiec backtest, historical stress replays,
Euler risk budgeting, rolling diversification and up/down-market betas.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .analytics import Analysis

EWMA_LAMBDA = 0.94          # RiskMetrics daily decay
ROLL = 63                   # ~3 months: warm-up for EWMA and window for rolling diversification
TAIL_GRID = np.linspace(0.0005, 0.025, 50)  # ES 97.5% = average VaR across the worst 2.5% of outcomes

# Market sell-offs replayed with today's weights: (name, last close before, last day of the move).
SCENARIOS = [
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("Jun 2022 inflation scare", "2022-06-08", "2022-06-16"),
    ("Mar 2023 regional banks", "2023-03-08", "2023-03-13"),
    ("Aug 2024 yen carry unwind", "2024-07-31", "2024-08-05"),
    ("Apr 2025 tariff shock", "2025-04-02", "2025-04-08"),
]
BETA_SHOCKS = (-0.10, -0.20)


@dataclass
class RiskReport:
    weights: pd.Series
    port: pd.Series                 # full-history daily portfolio return with today's weights
    bench: pd.Series
    var_methods: pd.DataFrame       # index = method; var95, var99, es975 as fractions of NAV
    moments: dict
    backtest: pd.DataFrame          # ret, var95 (positive number), breach
    kupiec: dict
    stress: pd.DataFrame            # scenario, period, kind, port, bench
    budget: pd.DataFrame            # per ticker: weight, risk_share, vol, mctr, beta
    div_ratio: float
    diversification: pd.DataFrame   # rolling avg pairwise correlation and diversification ratio
    corr_regimes: dict              # avg pairwise correlation on all / S&P-down / S&P-worst-10% days
    updown: pd.DataFrame            # down_beta, up_beta per name plus "Portfolio"


_cache: dict = {}


def report(a: Analysis) -> RiskReport:
    """One report per Analysis object (the analysis itself is cached by the service)."""
    hit = _cache.get(id(a))
    if hit and hit[0] is a:
        return hit[1]
    rep = _build(a)
    _cache.clear()
    _cache[id(a)] = (a, rep)
    return rep


def _ewma_sd(r: pd.Series) -> tuple[pd.Series, float]:
    """EWMA volatility forecast for each day (using only earlier days) and for tomorrow."""
    x = r.to_numpy()
    var = np.empty(len(x) + 1)
    var[0] = np.var(x[:ROLL], ddof=1)
    for t, v in enumerate(x):
        var[t + 1] = EWMA_LAMBDA * var[t] + (1 - EWMA_LAMBDA) * v * v
    return pd.Series(np.sqrt(var[:-1]), index=r.index), float(np.sqrt(var[-1]))


def _var_methods(r: pd.Series, next_sd: float) -> tuple[pd.DataFrame, dict]:
    mu, sd = r.mean(), r.std()
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r))  # kurtosis = excess over normal

    def normal(al):
        return -(mu + stats.norm.ppf(al) * sd)

    def cornish_fisher(al):  # normal quantile bent for skew and fat tails
        z = stats.norm.ppf(al)
        zc = z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * kurt / 24 - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
        return -(mu + zc * sd)

    def ewma(al):
        return -stats.norm.ppf(al) * next_sd

    def historical(al):
        return -float(np.quantile(r, al))

    rows = {}
    for name, f in (("Historical", historical), ("Normal", normal), ("Cornish-Fisher", cornish_fisher),
                    ("EWMA (λ 0.94)", ewma)):
        rows[name] = dict(var95=f(0.05), var99=f(0.01), es975=float(np.mean([f(al) for al in TAIL_GRID])))
    rows["Historical"]["es975"] = -float(r[r <= np.quantile(r, 0.025)].mean())
    moments = dict(skew=skew, kurt=kurt, worst=float(r.min()), worst_day=r.idxmin(),
                   best=float(r.max()), best_day=r.idxmax())
    return pd.DataFrame(rows).T, moments


def _kupiec(breaches: int, n: int, p: float = 0.05) -> dict:
    """Kupiec proportion-of-failures test: are VaR breaches as rare as the model promises?"""
    phat = breaches / n if n else 0.0
    ll0 = (n - breaches) * np.log(1 - p) + breaches * np.log(p)
    ll1 = ((n - breaches) * np.log(1 - phat) if phat < 1 else 0.0) + (breaches * np.log(phat) if breaches else 0.0)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return dict(breaches=breaches, n=n, expected=n * p, p_value=float(1 - stats.chi2.cdf(lr, 1)))


def _window_return(H: pd.DataFrame, start, end) -> pd.Series:
    sub = H[(H.index > pd.Timestamp(start)) & (H.index <= pd.Timestamp(end))]
    return (1 + sub).prod() - 1


def _stress(H: pd.DataFrame, w: pd.Series, bench_name: str, down_beta: float) -> pd.DataFrame:
    tickers = list(w.index)
    rows = []

    def add(name, period, kind, growth):
        rows.append(dict(scenario=name, period=period, kind=kind,
                         port=float((w * growth[tickers]).sum()), bench=float(growth[bench_name])))

    first, last = H.index[0], H.index[-1]
    for name, start, end in SCENARIOS:
        if pd.Timestamp(start) >= first and pd.Timestamp(end) <= last:
            add(name, f"{pd.Timestamp(start):%b %d, %Y} → {pd.Timestamp(end):%b %d, %Y}", "Historical",
                _window_return(H, start, end))

    b = H[bench_name]
    day = b.idxmin()
    add("Worst S&P 500 day", f"{day:%b %d, %Y}", "Worst in data", H.loc[day])
    cum = np.log1p(b).rolling(21).sum()
    end_i = int(np.nanargmin(cum.to_numpy()))
    start = H.index[max(end_i - 21, 0)]
    add("Worst S&P 500 month", f"{start:%b %d, %Y} → {H.index[end_i]:%b %d, %Y}", "Worst in data",
        _window_return(H, start, H.index[end_i]))

    for shock in BETA_SHOCKS:
        rows.append(dict(scenario=f"S&P 500 {shock:.0%}", period=f"downside beta {down_beta:.2f}",
                         kind="Hypothetical", port=down_beta * shock, bench=shock))
    return pd.DataFrame(rows)


def _avg_corr(X: pd.DataFrame) -> float:
    c = X.corr().to_numpy()
    n = len(c)
    return float((np.nansum(c) - n) / (n * (n - 1))) if n > 1 else np.nan


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else np.nan


def _build(a: Analysis) -> RiskReport:
    w = a.positions.set_index("ticker")["weight"]
    tickers = list(w.index)
    bench_name = a.bench_returns.name
    H = a.history
    port = pd.Series(H[tickers].to_numpy() @ w.to_numpy(), index=H.index, name="Portfolio")
    bench = H[bench_name]

    sd_days, sd_next = _ewma_sd(port)
    var_methods, moments = _var_methods(a.port_returns, sd_next)

    z95 = -stats.norm.ppf(0.05)
    bt = pd.DataFrame({"ret": port, "var95": z95 * sd_days}).iloc[ROLL:]
    bt["breach"] = bt["ret"] < -bt["var95"]
    recent = bt.tail(config.TRADING_DAYS)
    kupiec = _kupiec(int(recent["breach"].sum()), len(recent))

    # Euler risk budget over the analysis window: each name's share of portfolio volatility.
    R = a.idio.returns[tickers]
    cov = R.cov().to_numpy() * config.TRADING_DAYS
    wv = w.to_numpy()
    port_vol = float(np.sqrt(wv @ cov @ wv))
    mctr = cov @ wv / port_vol
    vols = np.sqrt(np.diag(cov))
    budget = pd.DataFrame({"weight": wv, "risk_share": wv * mctr / port_vol, "vol": vols, "mctr": mctr,
                           "beta": a.positions.set_index("ticker")["beta"].reindex(tickers).to_numpy()},
                          index=tickers)
    div_ratio = float(np.abs(wv) @ vols / port_vol)

    # Rolling diversification across the full history.
    roll_vol = H[tickers].rolling(ROLL).std()
    div = pd.DataFrame({"div_ratio": (roll_vol * w.abs()).sum(axis=1) / port.rolling(ROLL).std()})
    n = len(tickers)
    rc = H[tickers].rolling(ROLL).corr().to_numpy().reshape(-1, n, n)
    div["avg_corr"] = (np.nansum(rc, axis=(1, 2)) - n) / (n * (n - 1))
    div = div.iloc[ROLL:].dropna()

    down = bench < 0
    worst = bench <= bench.quantile(0.10)
    corr_regimes = {"All days": _avg_corr(H[tickers]), "S&P 500 down days": _avg_corr(H.loc[down, tickers]),
                    "S&P 500 worst 10% days": _avg_corr(H.loc[worst, tickers])}

    cols = {**{t: H[t] for t in tickers}, "Portfolio": port}
    m = bench.to_numpy()
    updown = pd.DataFrame({
        name: {"down_beta": _slope(s.to_numpy()[m < 0], m[m < 0]), "up_beta": _slope(s.to_numpy()[m > 0], m[m > 0])}
        for name, s in cols.items()}).T

    return RiskReport(
        weights=w, port=port, bench=bench, var_methods=var_methods, moments=moments, backtest=bt,
        kupiec=kupiec, stress=_stress(H, w, bench_name, float(updown.loc["Portfolio", "down_beta"])), budget=budget, div_ratio=div_ratio,
        diversification=div, corr_regimes=corr_regimes, updown=updown,
    )
