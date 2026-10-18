"""SEC EDGAR client: free, no API key, every US-listed company.

Earnings press releases are the EX-99.1 exhibit of the 8-K a company files under Item 2.02
("Results of Operations and Financial Condition") each quarter. Reported numbers come from the
XBRL company-facts API. SEC asks for a User-Agent with a contact email and at most 10 requests/second.
"""
import json
import re
import threading
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from . import config

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"

_session = requests.Session()
_rate_lock = threading.Lock()
_last_call = [0.0]
_tickers: dict = {"loaded": 0.0, "rows": []}


class EdgarError(RuntimeError):
    pass


def _get(url: str) -> requests.Response:
    ua = config.sec_user_agent()
    if not ua:
        raise EdgarError("Set SEC_USER_AGENT (a name and contact email) in the server's .env; SEC requires it.")
    with _rate_lock:  # stay under SEC's 10 requests per second across threads
        wait = config.SEC_MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
    resp = _session.get(url, headers={"User-Agent": ua}, timeout=30)
    if resp.status_code == 404:
        raise EdgarError("Not found on EDGAR.")
    if resp.status_code == 403:
        raise EdgarError("SEC refused the request (check SEC_USER_AGENT has a contact email).")
    resp.raise_for_status()
    return resp


# ---------- companies ----------

def _ticker_rows() -> list[dict]:
    """[{cik, name, ticker, exchange}] for every listed company; cached on disk for a day."""
    if _tickers["rows"] and time.time() - _tickers["loaded"] < 86400:
        return _tickers["rows"]
    cache = config.RESEARCH_DB.parent / "company_tickers_exchange.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        raw = json.loads(cache.read_text())
    else:
        raw = _get(TICKERS_URL).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    fields = raw["fields"]
    rows = [dict(zip(fields, r)) for r in raw["data"]]
    _tickers.update(rows=rows, loaded=time.time())
    return rows


def lookup(query: str) -> dict | None:
    """Ticker (exact) or company name (contains) -> {cik, name, ticker, exchange}."""
    q = query.strip().upper().replace(" ", "")
    rows = _ticker_rows()
    for r in rows:
        if r["ticker"].upper() == q or r["ticker"].upper().replace("-", ".") == q:
            return r
    words = query.strip().lower()
    hits = [r for r in rows if words and words in r["name"].lower()]
    return min(hits, key=lambda r: len(r["name"])) if hits else None


def company_profile(cik: int) -> dict:
    sub = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    fy = sub.get("fiscalYearEnd") or ""
    return {
        "name": sub.get("name", "").title() if sub.get("name", "").isupper() else sub.get("name", ""),
        "sic": (sub.get("sicDescription") or "").title(),
        "state": sub.get("stateOfIncorporation") or "",
        "fy_end": datetime.strptime(fy, "%m%d").strftime("%B") if len(fy) == 4 else "",
        "filings": sub["filings"]["recent"],
    }


def earnings_filings(profile: dict, n: int) -> list[dict]:
    """Latest n earnings 8-Ks (Item 2.02), newest first."""
    r = profile["filings"]
    out = []
    for i, form in enumerate(r["form"]):
        if form in ("8-K", "8-K/A") and "2.02" in (r["items"][i] or ""):
            out.append({"accession": r["accessionNumber"][i], "filing_date": r["filingDate"][i],
                        "primary": r["primaryDocument"][i]})
            if len(out) == n:
                break
    return out


def press_release(cik: int, accession: str, primary: str) -> tuple[str, str]:
    """(document URL, HTML) of the press release exhibit (EX-99.1) of an earnings 8-K."""
    acc = accession.replace("-", "")
    base = ARCHIVE_URL.format(cik=cik, acc=acc)
    index = _get(f"{base}/{accession}-index.htm").text
    docs = []
    for row in BeautifulSoup(index, "html.parser").select("table.tableFile tr"):
        cells = row.find_all("td")
        link = row.find("a")
        if len(cells) >= 4 and link:
            docs.append((cells[3].get_text(strip=True).upper(), link.get_text(strip=True)))
    htm = [(t, d) for t, d in docs if d.lower().endswith((".htm", ".html", ".txt"))]
    pick = next((d for t, d in htm if t in ("EX-99.1", "EX-99", "EX-99.01")), None) \
        or next((d for t, d in htm if t.startswith("EX-99")), None) or primary
    url = f"{base}/{pick}"
    return url, _get(url).text


# ---------- reported numbers (XBRL) ----------

REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
QUARTER = re.compile(r"^CY(\d{4})Q([1-4])$")
YEAR = re.compile(r"^CY(\d{4})$")


def _series(facts: dict, concepts: list[str], unit: str, derive_q4: bool) -> dict:
    """Calendar-quarter values {(year, q): value} from the concept with the most recent data."""
    best = {}
    for c in concepts:
        entries = facts.get(c, {}).get("units", {}).get(unit, [])
        qs = {}
        years = {}
        for e in entries:
            frame = e.get("frame", "")
            if m := QUARTER.match(frame):
                qs[(int(m[1]), int(m[2]))] = e["val"]
            elif m := YEAR.match(frame):
                years[int(m[1])] = e["val"]
        if derive_q4:  # 10-Ks report the year, not Q4: Q4 = year minus the first three quarters
            for y, total in years.items():
                if (y, 4) not in qs and all((y, q) in qs for q in (1, 2, 3)):
                    qs[(y, 4)] = total - sum(qs[(y, q)] for q in (1, 2, 3))
        if qs and (not best or max(qs) > max(best)):
            best = qs
    return best


def quarterly_financials(cik: int, quarters: int = 8) -> dict:
    """Last `quarters` calendar quarters of revenue, net income and diluted EPS (US GAAP filers only)."""
    try:
        facts = _get(FACTS_URL.format(cik=cik)).json().get("facts", {}).get("us-gaap", {})
    except EdgarError:
        return {}
    rev = _series(facts, REVENUE, "USD", True)
    ni = _series(facts, NET_INCOME, "USD", True)
    eps = _series(facts, EPS, "USD/shares", False)
    keys = sorted(set(rev) | set(eps))[-quarters:]
    if not keys:
        return {}
    return {
        "quarters": [f"Q{q} {y}" for y, q in keys],
        "revenue": [rev.get(k) for k in keys],
        "net_income": [ni.get(k) for k in keys],
        "eps": [eps.get(k) for k in keys],
        "revenue_yoy": (rev[keys[-1]] / rev[(keys[-1][0] - 1, keys[-1][1])] - 1)
        if keys[-1] in rev and (keys[-1][0] - 1, keys[-1][1]) in rev and rev[(keys[-1][0] - 1, keys[-1][1])]
        else None,
    }


def is_stale(updated: str | None) -> bool:
    if not updated:
        return True
    return datetime.now() - datetime.fromisoformat(updated) > timedelta(hours=config.RESEARCH_REFRESH_HOURS)
