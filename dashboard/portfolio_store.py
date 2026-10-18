"""The on-page portfolio editor's backend: read, check, save, back up and restore the portfolio file.

Saving is guarded by a team passcode (config.edit_passcode(), PORTFOLIO_EDIT_PASSCODE in .env) because the site is
public. Before anything is written, the new portfolio is run through the full analysis (service.check), so a bad edit
(unknown ticker, no price history) is rejected instead of breaking the site. Every save keeps the previous file in
data/private/history/, and any of those can be restored from the editor.
"""
import base64
import hmac
import io
import os
import re
import shutil
import threading
import time
from datetime import datetime

import pandas as pd

from . import config, data, service

_save_lock = threading.Lock()
_failed_codes: dict = {}  # visitor -> recent wrong-passcode times


class EditError(ValueError):
    """A problem the person editing can fix; the message is shown on the page."""


# ---------- reading ----------

def current_rows(a) -> list[dict]:
    """The portfolio as editable rows: ticker, dollar amount, sector (largest first, like the positions table)."""
    return [{"ticker": r.ticker, "dollars": round(float(r.market_value)), "sector": r.sector}
            for r in a.positions.itertuples()]


def history() -> list[dict]:
    """Earlier versions, newest first: {"value": file name, "label": when + size}."""
    out = []
    for path in sorted(config.PORTFOLIO_HISTORY_DIR.glob("portfolio-*.*"), reverse=True):
        try:
            h = data.load_holdings(path)
            size = h["weight"].abs().sum() if "weight" in h else None
            when = datetime.strptime(path.stem.split("-", 1)[1], "%Y%m%d-%H%M%S")
            label = f"{when:%b %d, %H:%M} · {len(h)} holdings" + (f" · ${size:,.0f}" if size and size > 100 else "")
        except Exception:
            continue
        out.append({"value": path.name, "label": label})
    return out


# ---------- the passcode ----------

def _check_passcode(passcode: str | None, visitor: str) -> None:
    expected = config.edit_passcode()
    if not expected:
        raise EditError("Saving is turned off: set PORTFOLIO_EDIT_PASSCODE in the server's .env file.")
    now = time.time()
    recent = [t for t in _failed_codes.get(visitor, []) if now - t < 900]
    if len(recent) >= 5:
        raise EditError("Too many wrong passcodes. Try again in 15 minutes.")
    if not passcode or not hmac.compare_digest(passcode.encode(), expected.encode()):
        _failed_codes[visitor] = recent + [now]
        raise EditError("Wrong passcode, so nothing was saved.")
    _failed_codes.pop(visitor, None)


# ---------- checking and saving ----------

def clean(rows: list[dict]) -> pd.DataFrame:
    """Validate editor rows into the file's columns (ticker, weight = dollars, sector, name, currency)."""
    out, problems, seen = [], [], set()
    for i, r in enumerate(rows, 1):
        ticker = str(r.get("ticker") or "").strip().upper()
        amount = r.get("dollars")
        if not ticker and amount in (None, ""):
            continue  # an empty row
        if not ticker:
            problems.append(f"Row {i}: add a ticker.")
            continue
        if ticker in seen:
            problems.append(f"{ticker} is listed twice; combine the rows.")
            continue
        seen.add(ticker)
        try:
            dollars = float(amount)
        except (TypeError, ValueError):
            problems.append(f"{ticker}: enter a dollar amount.")
            continue
        if not dollars or dollars != dollars or abs(dollars) > 1e12:
            problems.append(f"{ticker}: the amount must be a non-zero number (negative means short).")
            continue
        sector = r.get("sector")
        if not sector:
            problems.append(f"{ticker}: pick a sector.")
            continue
        q = data.quote(ticker)
        if q is None:
            problems.append(f"{ticker}: Yahoo Finance doesn't recognise this ticker. Non-US stocks need their exchange "
                            "suffix, e.g. RELIANCE.NS or BARC.L.")
            continue
        out.append({"ticker": ticker, "weight": round(dollars, 2), "sector": sector, "name": q["name"],
                    "currency": q["currency"]})
    if not out and not problems:
        problems.append("Add at least one holding.")
    if len(out) > config.PORTFOLIO_MAX_ROWS:
        problems.append(f"At most {config.PORTFOLIO_MAX_ROWS} holdings.")
    if problems:
        raise EditError(" ".join(problems))
    return pd.DataFrame(out)


def _backup(stamp: str) -> None:
    config.PORTFOLIO_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for path in config.PORTFOLIO_FILES:
        if not path.exists() or path.parent != config.PRIVATE_DIR:
            continue  # the sample file is never touched
        target = config.PORTFOLIO_HISTORY_DIR / f"portfolio-{stamp}{path.suffix}"
        if path == config.PORTFOLIO_FILE:
            shutil.copy2(path, target)  # copy: the site keeps reading it until the new file replaces it atomically
        else:
            os.replace(path, target)  # e.g. portfolio.xlsx, which would otherwise take precedence over the saved CSV
    versions = sorted(config.PORTFOLIO_HISTORY_DIR.glob("portfolio-*.*"))
    for old in versions[:-config.PORTFOLIO_HISTORY_KEEP]:
        old.unlink()


def save(rows: list[dict], passcode: str | None, visitor: str) -> str:
    """Check the passcode and rows, test the new portfolio end to end, then back up the old file and write the new one."""
    _check_passcode(passcode, visitor)
    df = clean(rows)
    with _save_lock:
        config.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
        pending = config.PRIVATE_DIR / ".portfolio-pending.csv"
        df.to_csv(pending, index=False)
        try:
            a = service.check(pending)  # downloads prices for new tickers and runs every analysis
        except Exception as exc:
            pending.unlink(missing_ok=True)
            raise EditError(f"The new portfolio couldn't be analysed ({type(exc).__name__}: {exc}). Nothing was saved.")
        missing = sorted(set(df["ticker"]) - set(a.positions["ticker"]))
        if missing:
            pending.unlink(missing_ok=True)
            why = " ".join(w for w in a.warnings if any(t in w for t in missing))
            raise EditError(f"No usable price history for {', '.join(missing)}, so nothing was saved. {why}".strip())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _backup(stamp)
        os.replace(pending, config.PORTFOLIO_FILE)
    total = df["weight"].abs().sum()
    return (f"Saved {len(df)} holdings, ${total:,.0f} in total. Every page and the assistant now use the new portfolio; "
            "the previous version is under Restore.")


def restore(name: str | None, passcode: str | None, visitor: str) -> str:
    """Make an earlier version current again (it goes through the same checks as a normal save)."""
    if not name:
        raise EditError("Pick a version to restore.")
    path = config.PORTFOLIO_HISTORY_DIR / os.path.basename(name)
    if not path.exists() or not re.fullmatch(r"portfolio-\d{8}-\d{6}\.(csv|xlsx|xls)", path.name):
        raise EditError("That version no longer exists.")
    h = data.load_holdings(path)
    if "weight" not in h:
        raise EditError("That version lists share quantities; upload it as a spreadsheet instead.")
    if h["weight"].abs().sum() <= 100:  # relative weights from an old file: size them like the site did
        h["weight"] = h["weight"] / h["weight"].abs().sum() * config.PORTFOLIO_AUM
    rows = [{"ticker": r.ticker, "dollars": r.weight, "sector": r.sector} for r in h.itertuples()]
    return save(rows, passcode, visitor).replace("Saved", "Restored", 1)


# ---------- spreadsheet upload ----------

UPLOAD_COLUMNS = {
    "ticker": ["ticker", "symbol", "security", "stock", "bbg ticker", "bloomberg ticker"],
    "dollars": ["dollars", "amount", "value", "market value", "position value", "usd", "$", "weight", "weights",
                "allocation"],
    "shares": ["shares", "quantity", "qty", "units", "position"],
    "sector": ["sector", "gics sector", "industry"],
}


def _yahoo_ticker(raw: str) -> str:
    """'TGT US Equity' -> 'TGT', 'RELIANCE IN Equity' -> 'RELIANCE.NS'; plain tickers pass through."""
    raw = str(raw).strip()
    if raw in config.BLOOMBERG_TICKER_MAP:
        return config.BLOOMBERG_TICKER_MAP[raw]
    m = re.fullmatch(r"(\S+)\s+([A-Z]{2})\s+Equity", raw, re.I)
    if m:
        return m.group(1).upper() + config.BLOOMBERG_EXCHANGES.get(m.group(2).upper(), "")
    return raw.upper()


def parse_upload(contents: str, filename: str, nav: float, sectors: dict) -> tuple[list[dict], str]:
    """Rows for the editor from an uploaded Excel/CSV file, plus a note on what was read. Nothing is saved here."""
    try:
        raw = base64.b64decode(contents.split(",", 1)[1])
        ext = filename.lower().rsplit(".", 1)[-1]
        df = pd.read_csv(io.BytesIO(raw)) if ext == "csv" else pd.read_excel(io.BytesIO(raw))
    except Exception:
        raise EditError(f"Couldn't read {filename}. Upload an .xlsx, .xls or .csv file.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    col = {k: next((a for a in names if a in df.columns), None) for k, names in UPLOAD_COLUMNS.items()}
    if not col["ticker"] or not (col["dollars"] or col["shares"]):
        raise EditError(f"{filename} needs a ticker column and a dollar-amount (or shares) column. "
                        f"Found: {', '.join(map(str, df.columns[:12]))}.")
    df = df[df[col["ticker"]].notna()].head(config.PORTFOLIO_MAX_ROWS + 1)
    rows, notes = [], []
    if col["dollars"]:
        amounts = pd.to_numeric(df[col["dollars"]].astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
        if 0 < amounts.abs().sum() <= 100:  # weights (0.15 or 15%), not dollars: size them to today's NAV
            amounts = amounts / amounts.abs().sum() * nav
            notes.append(f"Read the {col['dollars']} column as weights and sized them to today's NAV (${nav:,.0f}).")
    else:
        notes.append("Turned share counts into dollars at the latest prices.")
    for i, r in enumerate(df.itertuples(index=False)):
        rec = dict(zip(df.columns, r))
        ticker = _yahoo_ticker(rec[col["ticker"]])
        if col["dollars"]:
            dollars = amounts.iloc[i]
        else:
            q = data.quote(ticker)
            shares = pd.to_numeric(rec[col["shares"]], errors="coerce")
            dollars = shares * q["usd_price"] if q and q["usd_price"] and shares == shares else None
        sector = rec.get(col["sector"]) if col["sector"] else None
        sector = sector if sector in config.SECTORS else sectors.get(ticker)
        rows.append({"ticker": ticker, "dollars": None if dollars is None or dollars != dollars else round(float(dollars)),
                     "sector": sector})
    note = (f"Loaded {len(rows)} rows from {filename}. Review them, then enter the passcode and save. "
            + " ".join(notes)).strip()
    return rows, note
