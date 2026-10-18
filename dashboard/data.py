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
    "weight": ["weight", "weights", "allocation"],
    "sector": ["sector", "gics sector", "industry"],
    "name": ["name", "company", "company name"],
    "currency": ["currency", "ccy"],
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
    """Return one row per ticker with columns ticker, sector and either quantity or weight.

    Negative quantity/weight = short. Cash rows (see config.CASH_TICKERS) keep quantity in
    dollars. Weights are turned into quantities by size_positions() once prices are known.
    """
    path = path or portfolio_path()
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    picked = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = next((a for a in aliases if a in raw.columns), None)
        if match:
            picked[field] = raw[match]
    if "ticker" not in picked or not {"quantity", "weight"} & picked.keys():
        raise ValueError(f"{path.name}: needs a ticker column and a quantity or weight column")

    df = pd.DataFrame(picked)
    size = "quantity" if "quantity" in df else "weight"
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df[size] = pd.to_numeric(df[size], errors="coerce")
    df = df[df["ticker"].ne("") & df[size].notna() & df[size].ne(0)]
    if "sector" not in df:
        df["sector"] = "Unclassified"
    df["sector"] = df["sector"].fillna("Unclassified").astype(str)
    extra = {c: (c, "first") for c in ("name", "currency") if c in df}
    return df.groupby("ticker", as_index=False).agg(**{size: (size, "sum"), "sector": ("sector", "first"), **extra})


def size_positions(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Convert relative weights to quantities so gross exposure = config.PORTFOLIO_AUM."""
    if "quantity" in holdings:
        return holdings
    h = holdings.copy()
    total = h["weight"].abs().sum()
    dollars = h["weight"] if total > 100 else h["weight"] / total * config.PORTFOLIO_AUM  # dollars, or relative weights
    last = prices.ffill().iloc[-1]
    is_cash = h["ticker"].isin(config.CASH_TICKERS)
    h["quantity"] = dollars.where(is_cash, dollars / h["ticker"].map(last))
    return h.drop(columns="weight")


def load_bloomberg_prices(path: Path | None = None, sheet: str | None = None) -> pd.DataFrame:
    """Daily PX_LAST per security from a Bloomberg Excel export, one column per ticker.

    Layout: a security-name row, a long-name row, then a field-code row starting with
    "Dates"; each security spans several columns and is named only in its first one.
    """
    path = path or config.BLOOMBERG_FILE
    raw = pd.read_excel(path, sheet_name=sheet or config.BLOOMBERG_PRICE_SHEET, header=None)
    header = raw.index[raw.iloc[:, 0].astype(str).eq("Dates")][0]
    securities, fields = raw.iloc[header - 2].ffill(), raw.iloc[header]
    body = raw.iloc[header + 1:]

    columns = {}
    for j in range(1, raw.shape[1]):
        if fields.iloc[j] == "PX_LAST":
            name = str(securities.iloc[j]).strip()
            ticker = config.BLOOMBERG_TICKER_MAP.get(name, name.removesuffix(" US Equity"))
            columns[ticker] = pd.to_numeric(body.iloc[:, j], errors="coerce").to_numpy()
    prices = pd.DataFrame(columns, index=pd.to_datetime(body.iloc[:, 0], errors="coerce").to_numpy())
    prices = prices[prices.index.notna()].sort_index()
    # Bloomberg repeats the previous row on days when every market was closed.
    return prices[~prices.eq(prices.shift()).all(axis=1)]


def fx_map(holdings: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Non-USD holdings -> (Yahoo FX symbol quoting local currency per USD, sub-unit divisor).

    From config.LOCAL_CURRENCY_FX plus the `currency` column the portfolio editor saves (e.g. GBp for London).
    """
    out = {t: (sym, 1.0) for t, sym in config.LOCAL_CURRENCY_FX.items()}
    if "currency" in holdings:
        for t, cur in holdings.set_index("ticker")["currency"].dropna().items():
            cur, div = config.SUBUNIT_CURRENCIES.get(str(cur), (str(cur), 1.0))
            if cur and cur.upper() != "USD" and t not in out:
                out[t] = (f"{cur.upper()}=X", div)
    return out


def to_usd(local: pd.DataFrame, fx: pd.DataFrame, mapping: dict[str, tuple[str, float]] | None = None
           ) -> tuple[pd.DataFrame, list[str]]:
    """Convert non-USD prices to USD with their FX rate (local currency per USD). Returns (prices, warnings)."""
    out, warnings = local.copy(), []
    mapping = mapping if mapping is not None else {t: (s, 1.0) for t, s in config.LOCAL_CURRENCY_FX.items()}
    for ticker, (fx_symbol, divisor) in mapping.items():
        if ticker not in out:
            continue
        if fx_symbol not in fx:
            warnings.append(f"No {fx_symbol} rate; {ticker} is shown in its local currency.")
            continue
        rate = fx[fx_symbol].reindex(out.index.union(fx.index)).ffill().reindex(out.index)
        out[ticker] = out[ticker] / divisor / rate
    return out, warnings


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


_quotes: dict = {}


def quote(ticker: str) -> dict | None:
    """Name, latest price, currency and USD price for a Yahoo ticker, or None if Yahoo doesn't know it (cached 1 h)."""
    ticker = ticker.strip().upper()
    hit = _quotes.get(ticker)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        resp = _session.get(_YAHOO_URL.format(ticker=ticker), params={"range": "5d", "interval": "1d"}, timeout=10)
        meta = resp.json()["chart"]["result"][0]["meta"] if resp.ok else None
    except Exception:
        return hit[1] if hit else None  # network trouble: keep the last answer rather than calling the ticker unknown
    out = None
    if meta and meta.get("regularMarketPrice"):
        cur, div = config.SUBUNIT_CURRENCIES.get(meta.get("currency") or "USD", (meta.get("currency") or "USD", 1.0))
        price = float(meta["regularMarketPrice"])
        usd = price / div
        if ticker.endswith("=X"):
            usd = None  # an exchange rate itself (e.g. INR=X): nothing to convert
        elif cur.upper() != "USD":
            fx = quote(f"{cur.upper()}=X")
            usd = usd / fx["price"] if fx and fx["price"] else None
        out = {"ticker": ticker, "name": meta.get("longName") or meta.get("shortName") or ticker,
               "price": price, "currency": meta.get("currency") or "USD", "usd_price": usd,
               "type": meta.get("instrumentType"), "exchange": meta.get("fullExchangeName") or meta.get("exchangeName")}
    _quotes[ticker] = (time.time(), out)
    return out
