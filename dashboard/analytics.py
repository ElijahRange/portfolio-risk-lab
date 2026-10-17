"""Portfolio risk analytics built from public, textbook methods.

Simplification used throughout: today's holdings are held constant over the lookback
window, so portfolio return on day t = sum(weight_i * return_i,t).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class Analysis:
    as_of: pd.Timestamp
    source: str
    nav: float
    positions: pd.DataFrame          # one row per priced position
    kpis: dict
    port_returns: pd.Series
    bench_returns: pd.Series
    rolling_beta: pd.Series
    factor_table: pd.DataFrame       # index = factor name
    risk_split: dict                 # annualised variance: factor / idiosyncratic
    pca_points: pd.DataFrame         # ticker, side, weight, pc1..pc3
    pca_labels: list[str]
    corr: pd.DataFrame
    idio: "IdioRisk"
    history: pd.DataFrame            # full daily return history of the holdings + benchmark (for the optimizer)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IdioRisk:
    """Single-index (market model) split of risk against the benchmark: r = alpha + beta * r_m + e."""
    names: pd.DataFrame              # per ticker: weight, beta/alpha (+ t-stats), r2, vols, returns, idio_contrib
    kpis: dict
    returns: pd.DataFrame            # daily returns of the names and the benchmark over the window
    resid_corr: pd.DataFrame         # correlation of the residuals e (what the market doesn't explain)
    cum_split: pd.DataFrame          # summed daily return: market part, idiosyncratic part, total
    rolling: pd.DataFrame            # rolling idiosyncratic share of variance, idiosyncratic vol, beta


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an intercept. Returns (coefficients without intercept, residuals)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return coef[1:], y - Xc @ coef


def _factor_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Factor ETF returns with every non-market factor orthogonalised to the market."""
    market = rets[config.FACTORS["Market"]].to_numpy()
    out = {"Market": market}
    for name, etf in config.FACTORS.items():
        if name == "Market" or etf not in rets:
            continue
        _, resid = _ols(rets[etf].to_numpy(), market[:, None])
        out[name] = resid
    return pd.DataFrame(out, index=rets.index)


def _pca(R: pd.DataFrame, weights: pd.Series, factor_raw: pd.DataFrame):
    """Positions as points in the first three principal components of their correlation matrix."""
    std = R.std()
    R = R.loc[:, std > 0]
    if R.shape[1] < 3:
        return pd.DataFrame(), []
    Z = (R - R.mean()) / R.std()
    eigval, eigvec = np.linalg.eigh(np.corrcoef(Z.to_numpy(), rowvar=False))
    order = np.argsort(eigval)[::-1][:3]
    eigval, eigvec = eigval[order], eigvec[:, order]
    # Fix sign so the largest loading on each PC is positive (eigenvectors have arbitrary sign).
    eigvec = eigvec * np.sign(eigvec[np.abs(eigvec).argmax(axis=0), range(3)])
    loadings = eigvec * np.sqrt(eigval)
    points = pd.DataFrame(loadings, columns=["pc1", "pc2", "pc3"])
    points.insert(0, "ticker", R.columns)
    points["weight"] = points["ticker"].map(weights)

    # Name each PC after the two factor ETFs its return series tracks most closely.
    pc_series = Z.to_numpy() @ eigvec
    labels = []
    for k in range(3):
        corr = factor_raw.apply(lambda col: np.corrcoef(col, pc_series[:, k])[0, 1])
        top = corr.abs().sort_values(ascending=False).index[:2]
        labels.append(f"PC{k + 1} · {' / '.join(top)}")
    return points, labels


def _newey_west_se(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """OLS standard errors that allow for autocorrelation and uneven variance (Newey-West HAC)."""
    n = len(e)
    lags = int(4 * (n / 100) ** (2 / 9))
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        G = xe[lag:].T @ xe[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv * n / (n - X.shape[1])
    return np.sqrt(np.diag(cov))


def _idiosyncratic(R: pd.DataFrame, bench: pd.Series, w: pd.Series, full: pd.DataFrame, nav: float) -> IdioRisk:
    """Regress every name and the portfolio on the benchmark and split variance into market vs residual.

    The portfolio's residual is the weighted sum of the names' residuals (OLS is linear), so
    w' S_e w is exactly its realised residual variance. Comparing that with the diagonal
    sum(w^2 s_e^2) shows how much the residuals move together, i.e. common risk beyond the market.
    """
    n, T = len(R), config.TRADING_DAYS
    m = bench.to_numpy()
    wv = w.to_numpy()
    Y = np.column_stack([R.to_numpy(), R.to_numpy() @ wv])  # last column = portfolio
    X = np.column_stack([np.ones(n), m])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ coef
    se = np.array([_newey_west_se(X, E[:, j]) for j in range(Y.shape[1])])  # (names + 1, 2)
    alpha_d, beta = coef
    alpha_t, beta_t = alpha_d / se[:, 0], beta / se[:, 1]

    var_m = m.var(ddof=1) * T
    tot_var = Y.var(axis=0, ddof=1) * T
    idio_var = E.var(axis=0, ddof=1) * T
    sys_ret = beta * m.sum()
    idio_ret = Y.sum(axis=0) - sys_ret

    S_e = np.cov(E[:, :-1], rowvar=False) * T
    port_idio_var = idio_var[-1]
    diag_idio_var = float((wv ** 2 * idio_var[:-1]).sum())
    contrib = wv * (S_e @ wv) / port_idio_var if port_idio_var > 0 else wv * 0

    k = slice(None, -1)
    names = pd.DataFrame({
        "weight": wv,
        "beta": beta[k], "beta_t": beta_t[k],
        "alpha": alpha_d[k] * T, "alpha_t": alpha_t[k],
        "r2": 1 - idio_var[k] / tot_var[k],
        "total_vol": np.sqrt(tot_var[k]),
        "sys_vol": np.abs(beta[k]) * np.sqrt(var_m),
        "idio_vol": np.sqrt(idio_var[k]),
        "sys_ret": sys_ret[k],
        "idio_ret": idio_ret[k],
        "idio_contrib": contrib,
    }, index=R.columns)

    pb, sys_var = beta[-1], beta[-1] ** 2 * var_m
    total_var = sys_var + port_idio_var
    kpis = {
        "beta": float(pb),
        "beta_t": float(beta_t[-1]),
        "alpha": float(alpha_d[-1] * T),
        "alpha_t": float(alpha_t[-1]),
        "r2": float(1 - port_idio_var / tot_var[-1]),
        "sys_ret": float(sys_ret[-1]),
        "idio_ret": float(idio_ret[-1]),
        "sys_vol": float(np.sqrt(sys_var)),
        "idio_vol": float(np.sqrt(port_idio_var)),
        "idio_share": float(port_idio_var / total_var) if total_var > 0 else 0.0,
        "diag_idio_vol": float(np.sqrt(diag_idio_var)),
        "info_ratio": float(alpha_d[-1] * T / np.sqrt(port_idio_var)) if port_idio_var > 0 else 0.0,
        "hedge_usd": float(-pb * nav),  # benchmark notional that makes the book beta-neutral
        "start": R.index[0],
        "end": R.index[-1],
        "days": n,
    }

    cum_split = pd.DataFrame({
        "Market (β × index)": np.cumsum(pb * m),
        "Idiosyncratic (α + ε)": np.cumsum(Y[:, -1] - pb * m),
        "Total": np.cumsum(Y[:, -1]),
    }, index=R.index)

    # Rolling split over the full price history, same constant-weight holdings.
    win = config.ROLLING_BETA_DAYS
    fp = full[w.index].fillna(0.0) @ wv
    fb = full[bench.name].fillna(0.0)
    p_var, b_var = fp.rolling(win).var(), fb.rolling(win).var()
    r_beta = fp.rolling(win).cov(fb) / b_var
    r_resid_var = (p_var - r_beta ** 2 * b_var).clip(lower=0)
    rolling = pd.DataFrame({
        "idio_share": r_resid_var / p_var,
        "idio_vol": np.sqrt(r_resid_var * T),
        "beta": r_beta,
    }).dropna()

    resid_corr = pd.DataFrame(np.corrcoef(E[:, :-1], rowvar=False), index=R.columns, columns=R.columns)
    return IdioRisk(names=names, kpis=kpis, returns=R.assign(**{bench.name: m}), resid_corr=resid_corr,
                    cum_split=cum_split, rolling=rolling)


def _history(full_rets: pd.DataFrame, tickers: list[str], benchmark: str) -> pd.DataFrame:
    """Daily returns of the holdings and benchmark, minus market-holiday rows (forward-filled, all zero)."""
    h = full_rets[tickers + [benchmark]].fillna(0.0)
    return h[(h != 0).any(axis=1)]


def build(holdings: pd.DataFrame, prices: pd.DataFrame, source: str, failed: list[str],
          benchmark: str = config.BENCHMARK) -> Analysis:
    warnings = []
    if failed:
        warnings.append(f"No price data for {', '.join(failed)}; excluded.")

    is_cash = holdings["ticker"].isin(config.CASH_TICKERS)
    cash = float(holdings.loc[is_cash, "quantity"].sum())
    h = holdings[~is_cash & holdings["ticker"].isin(prices.columns)].copy()

    rets = prices.pct_change(fill_method=None).iloc[1:].tail(config.LOOKBACK_DAYS)
    coverage = rets[h["ticker"]].notna().mean()
    thin = coverage[coverage < config.MIN_HISTORY_COVERAGE].index.tolist()
    if thin:
        warnings.append(f"Less than {config.MIN_HISTORY_COVERAGE:.0%} price history for {', '.join(thin)}; excluded.")
        h = h[~h["ticker"].isin(thin)]
    if h.empty:
        raise ValueError("None of the holdings have usable price history.")

    last, prev = prices.iloc[-1], prices.iloc[-2]
    h["price"] = h["ticker"].map(last)
    h["market_value"] = h["quantity"] * h["price"]
    h["day_pnl"] = h["quantity"] * (h["price"] - h["ticker"].map(prev))
    h["side"] = np.where(h["quantity"] > 0, "Long", "Short")

    nav = float(h["market_value"].sum() + cash)
    if nav <= 0:
        nav = float(h["market_value"].abs().sum())
        warnings.append("Net value is not positive; weights are shown as a share of gross exposure.")
    h["weight"] = h["market_value"] / nav
    w = h.set_index("ticker")["weight"]

    R = rets[w.index].fillna(0.0)
    bench = rets[benchmark].fillna(0.0)
    port = pd.Series(R.to_numpy() @ w.to_numpy(), index=R.index, name="Portfolio")

    # Single-index beta of each name to the benchmark.
    b_dev = bench - bench.mean()
    h["beta"] = h["ticker"].map((R - R.mean()).mul(b_dev, axis=0).sum() / (b_dev ** 2).sum())

    # Historical-covariance risk contribution (sums to 100% of portfolio variance).
    cov = R.cov() * config.TRADING_DAYS
    port_var = float(w @ cov @ w)
    h["risk_contrib"] = h["ticker"].map(w * (cov @ w) / port_var if port_var > 0 else w * 0)

    # Multi-factor regression of every name on the (orthogonalised) factor returns.
    F = _factor_returns(rets.fillna(0.0))
    betas, resid = _ols(R.to_numpy(), F.to_numpy())
    B = pd.DataFrame(betas.T, index=R.columns, columns=F.columns)
    idio_var = pd.Series(resid.var(axis=0, ddof=1) * config.TRADING_DAYS, index=R.columns)
    h["idio_vol"] = h["ticker"].map(np.sqrt(idio_var))

    net_b = w @ B
    long_b = w.where(w > 0, 0) @ B
    short_b = w.where(w < 0, 0) @ B
    F_cov = F.cov() * config.TRADING_DAYS
    factor_var = float(net_b @ F_cov @ net_b)
    specific_var = float((w ** 2 * idio_var).sum())
    model_var = factor_var + specific_var
    factor_table = pd.DataFrame({
        "proxy": [config.FACTORS[f] for f in F.columns],
        "net": net_b,
        "long": long_b,
        "short": short_b,
        "risk_share": net_b * (F_cov @ net_b) / model_var if model_var > 0 else 0.0,
    })

    rolling_beta = (port.rolling(config.ROLLING_BETA_DAYS).cov(bench)
                    / bench.rolling(config.ROLLING_BETA_DAYS).var()).dropna()

    wealth = (1 + port).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    gross = h["market_value"].abs().sum()
    abs_w = w.abs() / w.abs().sum()

    kpis = {
        "nav": nav,
        "long_mv": float(h.loc[h["quantity"] > 0, "market_value"].sum()),
        "short_mv": float(h.loc[h["quantity"] < 0, "market_value"].sum()),
        "gross_pct": float(gross / nav),
        "net_pct": float(w.sum()),
        "beta_adj_net": float((h["weight"] * h["beta"]).sum()),
        "day_pnl": float(h["day_pnl"].sum()),
        "ann_return": float(port.mean() * config.TRADING_DAYS),
        "ann_vol": float(port.std() * np.sqrt(config.TRADING_DAYS)),
        "var95_pct": float(-np.percentile(port, 5)),
        "cvar95_pct": float(-port[port <= np.percentile(port, 5)].mean()),
        "max_drawdown": float(drawdown.min()),
        "effective_n": float(1 / (abs_w ** 2).sum()),
        "top10_pct": float(abs_w.sort_values(ascending=False).head(10).sum()),
        "factor_share": factor_var / model_var if model_var > 0 else 0.0,
        "n_positions": int(len(h)),
    }
    kpis["var95_usd"] = kpis["var95_pct"] * nav

    factor_raw = rets[[e for e in config.FACTORS.values() if e in rets]].fillna(0.0)
    factor_raw.columns = [n for n, e in config.FACTORS.items() if e in rets]
    pca_points, pca_labels = _pca(R, w, factor_raw)
    if not pca_points.empty:
        pca_points["side"] = pca_points["ticker"].map(h.set_index("ticker")["side"])

    full_rets = prices.pct_change(fill_method=None).iloc[1:]
    top = w.abs().sort_values(ascending=False).head(15).index
    return Analysis(
        as_of=prices.index[-1],
        source=source,
        nav=nav,
        positions=h.sort_values("weight", key=abs, ascending=False).reset_index(drop=True),
        kpis=kpis,
        port_returns=port,
        bench_returns=bench.rename(benchmark),
        rolling_beta=rolling_beta,
        factor_table=factor_table,
        risk_split={"factor": factor_var, "idiosyncratic": specific_var},
        pca_points=pca_points,
        pca_labels=pca_labels,
        corr=R[top].corr(),
        idio=_idiosyncratic(R, bench.rename(benchmark), w, full_rets, nav),
        history=_history(full_rets, list(w.index), benchmark),
        warnings=warnings,
    )
