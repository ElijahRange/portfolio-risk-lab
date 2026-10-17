"""Portfolio optimizer for the Ratio Testing page (textbook mean-variance and risk-based objectives).

Everything is estimated from daily returns of the current holdings over a chosen lookback.
Weights are fractions of the invested book and always sum to 1. Results are in-sample:
they show what *would have* been best on past data, not a forecast.
"""
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config

OBJECTIVES = {
    "max_sharpe": ("Max Sharpe ratio", "Highest expected excess return per unit of total volatility."),
    "min_vol": ("Min volatility", "Lowest total volatility (the leftmost point of the frontier)."),
    "min_idio": ("Min idiosyncratic risk", "Lowest stock-specific volatility left after removing the S&P 500."),
    "max_sortino": ("Max Sortino ratio", "Highest expected excess return per unit of downside volatility."),
    "risk_parity": ("Risk parity", "Every stock contributes the same share of total variance."),
    "max_div": ("Max diversification", "Highest ratio of average stock volatility to portfolio volatility."),
}
RETURN_MODELS = {
    "hist": "Historical mean",
    "shrunk": "Shrunk 50% to average",
    "capm": "CAPM-implied",
}


class OptimizerError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    lookback: str = "3Y"
    lo: float = 0.0               # min weight per stock
    hi: float = 0.35              # max weight per stock
    rf: float = config.RISK_FREE_RATE
    returns: str = "hist"
    target_beta: float | None = None


@dataclass
class Inputs:
    tickers: list[str]
    R: np.ndarray                 # daily returns, T x n
    m: np.ndarray                 # benchmark daily returns
    mu: np.ndarray                # expected annual returns used by the optimizer
    hist_mu: np.ndarray           # plain historical annual mean
    cov: np.ndarray               # annual covariance
    beta: np.ndarray              # beta to the benchmark
    resid_cov: np.ndarray         # annual covariance of market-model residuals
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class Result:
    settings: Settings
    inputs: Inputs
    current: np.ndarray
    weights: dict                 # objective key -> weights (np.ndarray)
    errors: dict                  # objective key -> message, for objectives that failed
    metrics: dict                 # "current" / objective key -> metrics dict
    frontier: pd.DataFrame        # vol, ret, sharpe + one column per ticker
    cloud: pd.DataFrame           # random feasible portfolios: vol, ret, sharpe
    warnings: list[str] = field(default_factory=list)


def estimate(history: pd.DataFrame, benchmark: str, s: Settings) -> Inputs:
    days = config.OPT_LOOKBACKS[s.lookback]
    H = history.tail(days) if days else history
    tickers = [c for c in H.columns if c != benchmark]
    R, m = H[tickers].to_numpy(), H[benchmark].to_numpy()
    T = config.TRADING_DAYS

    X = np.column_stack([np.ones(len(m)), m])
    coef, *_ = np.linalg.lstsq(X, R, rcond=None)
    resid = R - X @ coef
    beta = coef[1]

    hist_mu = R.mean(axis=0) * T
    if s.returns == "shrunk":
        mu = 0.5 * hist_mu + 0.5 * hist_mu.mean()
    elif s.returns == "capm":
        mu = s.rf + beta * (m.mean() * T - s.rf)
    else:
        mu = hist_mu
    return Inputs(tickers=tickers, R=R, m=m, mu=mu, hist_mu=hist_mu, cov=np.cov(R, rowvar=False) * T,
                  beta=beta, resid_cov=np.cov(resid, rowvar=False) * T, start=H.index[0], end=H.index[-1])


def _downside_vol(w: np.ndarray, inp: Inputs, rf: float) -> float:
    excess = inp.R @ w - rf / config.TRADING_DAYS
    return float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2) * config.TRADING_DAYS))


def metrics(w: np.ndarray, inp: Inputs, rf: float) -> dict:
    var = float(w @ inp.cov @ w)
    vol = np.sqrt(var)
    ret = float(w @ inp.mu)
    idio_var = float(w @ inp.resid_cov @ w)
    down = _downside_vol(w, inp, rf)
    daily = inp.R @ w
    wealth = np.cumprod(1 + daily)
    abs_w = np.abs(w) / np.abs(w).sum()
    return {
        "ret": ret,
        "hist_ret": float(daily.mean() * config.TRADING_DAYS),
        "vol": vol,
        "sharpe": (ret - rf) / vol if vol > 0 else np.nan,
        "sortino": (ret - rf) / down if down > 0 else np.nan,
        "beta": float(w @ inp.beta),
        "idio_vol": np.sqrt(max(idio_var, 0.0)),
        "idio_share": idio_var / var if var > 0 else np.nan,
        "max_dd": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "eff_n": float(1 / (abs_w ** 2).sum()),
        "div_ratio": float(w @ np.sqrt(np.diag(inp.cov)) / vol) if vol > 0 else np.nan,
    }


def _objective(key: str, inp: Inputs, rf: float):
    cov, mu, n = inp.cov, inp.mu, len(inp.mu)
    vols = np.sqrt(np.diag(cov))

    def vol(w):
        return np.sqrt(max(w @ cov @ w, 1e-16))

    if key == "max_sharpe":
        return lambda w: -(w @ mu - rf) / vol(w)
    if key == "min_vol":
        return lambda w: w @ cov @ w * 100  # scaled so the solver's tolerance bites
    if key == "min_idio":
        return lambda w: w @ inp.resid_cov @ w * 100
    if key == "max_sortino":
        return lambda w: -(w @ mu - rf) / max(_downside_vol(w, inp, rf), 1e-8)
    if key == "risk_parity":
        def f(w):
            rc = w * (cov @ w) / (w @ cov @ w)
            return float(((rc - 1 / n) ** 2).sum()) * 100
        return f
    if key == "max_div":
        return lambda w: -(w @ vols) / vol(w)
    raise KeyError(key)


def _constraints(inp: Inputs, s: Settings, target_ret: float | None = None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if s.target_beta is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ inp.beta - s.target_beta})
    if target_ret is not None:
        cons.append({"type": "eq", "fun": lambda w: (w @ inp.mu - target_ret) * 10})
    return cons


def _solve(f, inp: Inputs, s: Settings, starts: list[np.ndarray], target_ret: float | None = None) -> np.ndarray:
    bounds = [(s.lo, s.hi)] * len(inp.tickers)
    cons = _constraints(inp, s, target_ret)
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(f, np.clip(x0, s.lo, s.hi), method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        w = res.x
        feasible = all(abs(c["fun"](w)) < 1e-5 for c in cons) and w.min() > s.lo - 1e-6 and w.max() < s.hi + 1e-6
        if feasible and res.fun < best_val:
            best, best_val = w, res.fun
    if best is None:
        raise OptimizerError("No portfolio satisfies these limits. Loosen the weight bounds or the target beta.")
    best[np.abs(best) < 1e-6] = 0.0
    return best / best.sum()


def _beta_range(beta: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Lowest and highest portfolio beta reachable within the bounds (fill the extreme betas first)."""
    def greedy(order):
        w = np.full(len(beta), lo)
        left = 1 - w.sum()
        for i in order:
            add = min(hi - lo, left)
            w[i] += add
            left -= add
        return float(w @ beta)
    return greedy(np.argsort(beta)), greedy(np.argsort(beta)[::-1])


def check_settings(inp: Inputs, s: Settings) -> str | None:
    n = len(inp.tickers)
    if s.lo > s.hi:
        return "Min weight is above max weight."
    if n * s.hi < 1 - 1e-9:
        return f"Max weight {s.hi:.0%} × {n} stocks is under 100%; raise it to at least {1 / n:.1%}."
    if n * s.lo > 1 + 1e-9:
        return f"Min weight {s.lo:.0%} × {n} stocks is over 100%; lower it to at most {1 / n:.1%}."
    if s.target_beta is not None:
        b_lo, b_hi = _beta_range(inp.beta, s.lo, s.hi)
        if not b_lo - 1e-9 <= s.target_beta <= b_hi + 1e-9:
            return (f"A beta of {s.target_beta:+.2f} can't be reached with these weight limits over this lookback. "
                    f"Pick a target between {b_lo:+.2f} and {b_hi:+.2f}, or allow shorts (negative min weight).")
    return None


def frontier(inp: Inputs, s: Settings, w_min_vol: np.ndarray) -> pd.DataFrame:
    """Minimum-variance portfolio for each target return between the min-vol and max-return portfolios."""
    eq = np.full(len(inp.mu), 1 / len(inp.mu))
    w_max_ret = _solve(lambda w: -(w @ inp.mu), inp, s, [eq, w_min_vol])
    r_lo, r_hi = w_min_vol @ inp.mu, w_max_ret @ inp.mu
    targets = np.linspace(r_lo, r_hi, config.FRONTIER_POINTS) if r_hi - r_lo > 1e-6 else [r_lo]
    rows, prev = [], w_min_vol
    for t in targets:
        try:
            prev = _solve(lambda w: w @ inp.cov @ w * 100, inp, s, [prev, eq], target_ret=t)
        except OptimizerError:
            continue
        rows.append(prev)
    W = np.array(rows)
    df = pd.DataFrame(W, columns=inp.tickers)
    df.insert(0, "ret", W @ inp.mu)
    df.insert(0, "vol", np.sqrt(np.einsum("ij,jk,ik->i", W, inp.cov, W)))
    df.insert(2, "sharpe", (df["ret"] - s.rf) / df["vol"])
    return df


def random_cloud(inp: Inputs, s: Settings, k: int = 4000) -> pd.DataFrame:
    """Random long-only portfolios inside the weight bounds, to show the feasible region."""
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(len(inp.mu)), size=k)
    W = W[(W >= s.lo - 1e-9).all(axis=1) & (W <= s.hi + 1e-9).all(axis=1)]
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, inp.cov, W))
    ret = W @ inp.mu
    return pd.DataFrame({"vol": vol, "ret": ret, "sharpe": (ret - s.rf) / vol})


_cache: OrderedDict = OrderedDict()


def run(history: pd.DataFrame, benchmark: str, current: pd.Series, s: Settings) -> Result:
    """Every objective, the efficient frontier and a random cloud for one set of settings (cached)."""
    key = (history.index[-1], len(history), tuple(history.columns), tuple(current.round(8).items()), s)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    inp = estimate(history, benchmark, s)
    cur = current.reindex(inp.tickers).fillna(0.0).to_numpy()
    cur = cur / cur.sum()
    problem = check_settings(inp, s)
    if problem:
        raise OptimizerError(problem)

    eq = np.full(len(inp.tickers), 1 / len(inp.tickers))
    weights, errors = {}, {}
    for k in OBJECTIVES:
        try:
            starts = [eq, cur] + ([weights["min_vol"]] if "min_vol" in weights else [])
            weights[k] = _solve(_objective(k, inp, s.rf), inp, s, starts)
        except OptimizerError as exc:
            errors[k] = str(exc)
    if "min_vol" not in weights:
        raise OptimizerError(errors.get("min_vol", "The optimizer could not find a feasible portfolio."))

    result = Result(
        settings=s, inputs=inp, current=cur, weights=weights, errors=errors,
        metrics={"current": metrics(cur, inp, s.rf), **{k: metrics(w, inp, s.rf) for k, w in weights.items()}},
        frontier=frontier(inp, s, weights["min_vol"]),
        cloud=random_cloud(inp, s),
    )
    if len(result.frontier) == 1:
        result.warnings.append("Every allowed portfolio has the same expected return here (CAPM returns with a fixed "
                               "beta), so the frontier is a single point and Max Sharpe equals Min volatility.")
    if s.target_beta is not None:
        result.warnings.append(f"Every optimized portfolio is held to a beta of {s.target_beta:+.2f}; "
                               "the current portfolio and the random cloud are not.")
    _cache[key] = result
    while len(_cache) > 24:
        _cache.popitem(last=False)
    return result
