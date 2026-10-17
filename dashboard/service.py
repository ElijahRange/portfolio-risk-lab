"""One cached Analysis per (portfolio file, file version, price download)."""
import threading

from . import analytics, config, data

_lock = threading.Lock()
_memo: dict = {"key": None, "value": None}


def _key():
    path = data.portfolio_path()
    return str(path), path.stat().st_mtime, data.cache_stamp()


def current_analysis(force_refresh: bool = False) -> analytics.Analysis:
    with _lock:
        if not force_refresh and _memo["key"] == _key():
            return _memo["value"]
        holdings, path = data.load_holdings(), data.portfolio_path()
        tickers = set(holdings["ticker"]) - config.CASH_TICKERS
        tickers |= set(config.FACTORS.values()) | {config.BENCHMARK}
        prices, failed = data.load_prices(tickers, force=force_refresh)
        if config.BENCHMARK not in prices:
            raise RuntimeError(f"Could not load benchmark prices ({config.BENCHMARK}).")
        result = analytics.build(holdings, prices, source=path.name, failed=failed)
        _memo.update(key=_key(), value=result)
        return result
