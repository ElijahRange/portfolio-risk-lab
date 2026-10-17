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
    warnings: list[str] = field(default_factory=list)


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


def build(holdings: pd.DataFrame, prices: pd.DataFrame, source: str, failed: list[str]) -> Analysis:
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
    bench = rets[config.BENCHMARK].fillna(0.0)
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

    top = w.abs().sort_values(ascending=False).head(15).index
    return Analysis(
        as_of=prices.index[-1],
        source=source,
        nav=nav,
        positions=h.sort_values("weight", key=abs, ascending=False).reset_index(drop=True),
        kpis=kpis,
        port_returns=port,
        bench_returns=bench.rename(config.BENCHMARK),
        rolling_beta=rolling_beta,
        factor_table=factor_table,
        risk_split={"factor": factor_var, "idiosyncratic": specific_var},
        pca_points=pca_points,
        pca_labels=pca_labels,
        corr=R[top].corr(),
        warnings=warnings,
    )
