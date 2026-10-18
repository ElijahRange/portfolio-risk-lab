"""One cached Analysis per (portfolio file, Bloomberg file, price download)."""
import threading

import pandas as pd

from . import analytics, config, data

_lock = threading.Lock()
_memo: dict = {"key": None, "value": None}


def _key():
    path, bb = data.portfolio_path(), config.BLOOMBERG_FILE
    return str(path), path.stat().st_mtime, bb.exists() and bb.stat().st_mtime, data.cache_stamp()


def _prices(holdings: pd.DataFrame, force: bool):
    """Return (prices in USD, benchmark, failed tickers, warnings, price source label)."""
    tickers = set(holdings["ticker"]) - config.CASH_TICKERS
    factors = set(config.FACTORS.values())
    fx = data.fx_map(holdings)                       # non-USD holding -> (Yahoo FX symbol, sub-unit divisor)
    fx_symbols = {sym for t, (sym, _) in fx.items() if t in tickers}

    if not config.BLOOMBERG_FILE.exists():
        prices, failed = data.load_prices(tickers | factors | fx_symbols | {config.BENCHMARK}, force=force)
        prices, warnings = data.to_usd(prices, prices, fx)
        prices = prices.drop(columns=[c for c in fx_symbols if c in prices and c not in tickers])
        return prices, config.BENCHMARK, failed, warnings, "Yahoo prices"

    local = data.load_bloomberg_prices()
    extra = tickers - set(local.columns)  # holdings that aren't in the Bloomberg file
    web, failed = data.load_prices(factors | fx_symbols | extra, force=force)
    local, warnings = data.to_usd(local, web, fx)
    web, more = data.to_usd(web, web, {t: m for t, m in fx.items() if t in extra})
    # Put the downloaded series on the Bloomberg calendar.
    web = web.drop(columns=[c for c in fx_symbols if c in web and c not in tickers])
    web = web.reindex(local.index.union(web.index)).ffill().reindex(local.index)
    return local.join(web), config.BLOOMBERG_BENCHMARK, failed, warnings + more, config.BLOOMBERG_FILE.name


def _analyse(path, force: bool = False) -> analytics.Analysis:
    holdings = data.load_holdings(path)
    prices, benchmark, failed, warnings, price_source = _prices(holdings, force)
    if benchmark not in prices:
        raise RuntimeError(f"Could not load benchmark prices ({benchmark}).")
    holdings = data.size_positions(holdings, prices)
    result = analytics.build(holdings, prices, source=f"{path.name} · {price_source}",
                             failed=failed, benchmark=benchmark)
    result.warnings[:0] = warnings
    return result


def current_analysis(force_refresh: bool = False) -> analytics.Analysis:
    with _lock:
        if not force_refresh and _memo["key"] == _key():
            return _memo["value"]
        result = _analyse(data.portfolio_path(), force_refresh)
        _memo.update(key=_key(), value=result)
        return result


def check(path) -> analytics.Analysis:
    """Analyse a candidate portfolio file without touching the cached one (the editor runs this before saving)."""
    return _analyse(path)
