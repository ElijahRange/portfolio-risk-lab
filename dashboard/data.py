"""Loading holdings from the team's spreadsheet and daily prices from Yahoo Finance."""
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

# Spreadsheet header (lower-cased) -> field. The first matching column is used.
COLUMN_ALIASES = {
    "ticker": ["ticker", "symbol", "security", "stock"],
    "quantity": ["quantity", "shares", "qty", "position", "units"],
    "sector": ["sector", "gics sector", "industry"],
}

_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (portfolio-risk-lab)"


def portfolio_path() -> Path:
    for path in config.PORTFOLIO_FILES:
        if path.exists():
            return path
    raise FileNotFoundError("No portfolio file found. Add data/private/portfolio.xlsx.")


def load_holdings(path: Path | None = None) -> pd.DataFrame:
    """Return one row per ticker with columns ticker, quantity, sector.

    Negative quantity = short. Cash rows (see config.CASH_TICKERS) keep quantity in dollars.
    """
    path = path or portfolio_path()
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    picked = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = next((a for a in aliases if a in raw.columns), None)
        if match:
            picked[field] = raw[match]
        elif field != "sector":
            raise ValueError(f"{path.name}: no '{field}' column (looked for {', '.join(aliases)})")

    df = pd.DataFrame(picked)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df[df["ticker"].ne("") & df["quantity"].notna() & df["quantity"].ne(0)]
    if "sector" not in df:
        df["sector"] = "Unclassified"
    df["sector"] = df["sector"].fillna("Unclassified").astype(str)
    return df.groupby("ticker", as_index=False).agg(quantity=("quantity", "sum"), sector=("sector", "first"))


def _cache_file(ticker: str) -> Path:
    return config.CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', ticker)}.csv"


def cache_stamp() -> float:
    """Changes whenever any price was re-downloaded; used to invalidate in-memory results."""
    stamp = config.CACHE_DIR / ".stamp"
    return stamp.stat().st_mtime if stamp.exists() else 0.0


def _fetch_yahoo(ticker: str) -> pd.Series:
    resp = _session.get(
        _YAHOO_URL.format(ticker=ticker),
        params={"range": config.PRICE_HISTORY_RANGE, "interval": "1d", "events": "div,splits"},
        timeout=20,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    indicators = result["indicators"]
    closes = (indicators.get("adjclose") or [{}])[0].get("adjclose") or indicators["quote"][0]["close"]
    index = pd.to_datetime(result["timestamp"], unit="s").normalize()
    return pd.Series(closes, index=index, name=ticker, dtype="float64").dropna()


def load_prices(tickers, force: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Adjusted daily closes, one column per ticker. Returns (prices, tickers_that_failed)."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    max_age = config.PRICE_CACHE_MAX_AGE_HOURS * 3600
    series, failed, downloaded = {}, [], False

    for ticker in sorted(set(tickers)):
        path = _cache_file(ticker)
        fresh = path.exists() and time.time() - path.stat().st_mtime < max_age
        if fresh and not force:
            series[ticker] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            continue
        try:
            s = _fetch_yahoo(ticker)
            s.to_frame().to_csv(path)
            series[ticker] = s
            downloaded = True
        except Exception:
            if path.exists():  # stale cache beats nothing
                series[ticker] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                failed.append(ticker)

    if downloaded:
        (config.CACHE_DIR / ".stamp").touch()
    prices = pd.DataFrame(series).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")].ffill()
    return prices, failed
