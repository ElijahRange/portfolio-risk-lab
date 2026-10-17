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
    """Return (prices, benchmark, failed tickers, warnings, price source label)."""
    tickers = set(holdings["ticker"]) - config.CASH_TICKERS
    factors = set(config.FACTORS.values())

    if not config.BLOOMBERG_FILE.exists():
        prices, failed = data.load_prices(tickers | factors | {config.BENCHMARK}, force=force)
        return prices, config.BENCHMARK, failed, [], "Yahoo prices"

    local = data.load_bloomberg_prices()
    fx_symbols = {config.LOCAL_CURRENCY_FX[t] for t in local.columns if t in config.LOCAL_CURRENCY_FX}
    extra = tickers - set(local.columns)  # holdings that aren't in the Bloomberg file
    web, failed = data.load_prices(factors | fx_symbols | extra, force=force)
    local, warnings = data.to_usd(local, web)
    # Put the downloaded series on the Bloomberg calendar.
    web = web.drop(columns=[c for c in fx_symbols if c in web])
    web = web.reindex(local.index.union(web.index)).ffill().reindex(local.index)
    return local.join(web), config.BLOOMBERG_BENCHMARK, failed, warnings, config.BLOOMBERG_FILE.name


def current_analysis(force_refresh: bool = False) -> analytics.Analysis:
    with _lock:
        if not force_refresh and _memo["key"] == _key():
            return _memo["value"]
        path = data.portfolio_path()
        holdings = data.load_holdings(path)
        prices, benchmark, failed, warnings, price_source = _prices(holdings, force_refresh)
        if benchmark not in prices:
            raise RuntimeError(f"Could not load benchmark prices ({benchmark}).")
        holdings = data.size_positions(holdings, prices)
        result = analytics.build(holdings, prices, source=f"{path.name} · {price_source}",
                                 failed=failed, benchmark=benchmark)
        result.warnings[:0] = warnings
        _memo.update(key=_key(), value=result)
        return result
