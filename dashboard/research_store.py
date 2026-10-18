"""Company Research: fetch earnings releases from SEC EDGAR, score them with FinBERT, store the results.

Flow for a company not seen before (about 5-10 seconds):
  1. EDGAR: profile, latest earnings 8-Ks, XBRL numbers; the newest press release is analysed at once.
  2. The older releases are analysed in a background thread; the page picks them up as they finish.
Each release is analysed in two stages:
  - heavy (once): clean the HTML, split sentences, score the useful ones with FinBERT, find management quotes;
  - light (whenever another release arrives): TF-IDF topics and "new this quarter" terms across the
    company's releases, then pick key points and guidance from the stored FinBERT scores.
Opening a page only reads SQLite, so anything already analysed shows instantly.
"""
import json
import re
import sqlite3
import threading
from datetime import datetime

import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import config, edgar, finbert

_company_locks: dict = {}
_locks_guard = threading.Lock()


def _lock_for(symbol: str) -> threading.Lock:
    with _locks_guard:
        return _company_locks.setdefault(symbol, threading.Lock())


# ---------- storage ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    symbol TEXT PRIMARY KEY, cik INTEGER, name TEXT, exchange TEXT, sic TEXT, state TEXT, fy_end TEXT,
    financials TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS releases (
    symbol TEXT, accession TEXT, filing_date TEXT, primary_doc TEXT, url TEXT, period TEXT, status TEXT,
    error TEXT, narrative TEXT, scored TEXT, analysis TEXT, PRIMARY KEY (symbol, accession));
"""


def _db() -> sqlite3.Connection:
    config.RESEARCH_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.RESEARCH_DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# ---------- text extraction ----------

STOP_SECTIONS = re.compile(
    r"^(about [A-Z0-9]|forward[- ]looking (statements|information)|cautionary (note|statement|language)|"
    r"safe harbor|non-gaap|use of non-gaap|reconciliation|conference call|webcast|earnings call|"
    r"investor (relations )?contacts?|media contacts?|contacts?:|source:)", re.I)
BOILERPLATE = re.compile(
    r"forward-looking|safe harbor|non-gaap|reconciliation|webcast|conference call|press release|"
    r"investor relations|www\.|https?://|\bdial\b|replay|undue reliance|risks and uncertainties|"
    r"incorporated by reference|form 10-[kq]|percentages may not|totals may not sum|unaudited|\bemploys\b|"
    r"headquartered|associates worldwide|for more information|is a leading|visit\b|follow us", re.I)
NUMBER = re.compile(r"\$\s?\d|\d+(?:\.\d+)?\s?(?:%|percent|basis points|bps|billion|million)|\b\d{1,3}(?:,\d{3})+\b",
                    re.I)
CATEGORIES = [
    ("Capital return", r"dividend|buyback|repurchas|share count|returned? .{0,30}shareholders"),
    ("Guidance", r"guidance|outlook|\bexpects?\b|anticipate|forecast|projected|full[- ]year|next quarter|"
                 r"for the (?:year|quarter ahead)|going forward|reaffirm|raises?\b.{0,20}(?:outlook|guidance)"),
    ("Margins & costs", r"margin|\bcosts?\b|expense|pricing|productivity|efficienc"),
    ("Results", r"revenue|\bsales\b|earnings|\beps\b|per share|profit|income|cash flow|\bgrew\b|growth|comparable|volume"),
    ("Risks", r"headwind|\brisks?\b|tariff|inflation|uncertain|pressure|declin|weak|soft|impairment|litigation"),
    ("Strategy", r"acqui|launch|invest|\bai\b|platform|partnership|customer|innovation|portfolio|pipeline|backlog"),
]
HEDGE = set("""may might could possibly perhaps uncertain uncertainty approximately roughly believe assume assumption
assumptions appear appears likely unlikely potential potentially depend depends depending somewhat unclear cautious
expect expects anticipate anticipates""".split())
STOP = list(ENGLISH_STOP_WORDS | set("""quarter quarters year years fiscal compared compare prior period periods
percent million billion basis point points company company's results financial ended ending total net share shares
also continued continue including approximately per diluted gaap adjusted inc corporation corp nyse nasdaq www com
measures measure information additional regarding respect expectations announced related reported reporting
january february march april may june july august september october november december non-gaap reconciliation
today release statements""".split()))
WORD = re.compile(r"[a-z][a-z\-']+")
QUOTE = re.compile(r"[“\"]([^”\"]{40,900})[”\"]")
ATTRIBUTION = re.compile(r"(?:said|says|commented|added|stated|noted)\s+([A-Z][A-Za-z.\-' ]{2,40}?)"
                         r"(?:,\s*([^.“”\"]{3,90}?))?(?:[.,;]|$)")
ATTRIBUTION_BEFORE = re.compile(r"([A-Z][A-Za-z.\-' ]{2,40}?),\s*([^.“”\"]{3,90}?),\s*(?:said|says|commented|added)")
PERIOD = re.compile(r"(?:quarter|three months|thirteen weeks|fourteen weeks|twelve weeks|sixteen weeks)"
                    r"[^.]{0,60}?ended\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})")


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def _category(text: str) -> str:
    for name, pattern in CATEGORIES:
        if re.search(pattern, text, re.I):
            return name
    return "Other"


def _is_numeric_table(table) -> bool:
    text = table.get_text(" ", strip=True)
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    return digits > 0.25 * (digits + letters) or len(table.find_all("tr")) > 25


def extract(html: str) -> tuple[list[str], str]:
    """(narrative paragraphs, period-ended text) from a press-release document."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    for table in soup.find_all("table"):
        if _is_numeric_table(table):
            table.decompose()  # financial statements; the XBRL numbers cover these
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "td"]):
        block.insert_before("\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in soup.get_text().split("\n")]
    paras = [ln for ln in lines if len(ln.split()) >= 6 and not ln.upper().startswith("EX-99")]
    narrative = []
    for p in paras:
        if len(narrative) >= 3 and STOP_SECTIONS.match(p):
            break
        narrative.append(p)
    m = next((PERIOD.search(p) for p in narrative[:8] if PERIOD.search(p)), None)
    return narrative, (m.group(1).replace(".", "") if m else "")


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\b(Inc|Corp|Co|Ltd|No|U\.S|vs|approx|Mr|Ms|Dr)\.", lambda m: m.group(0).replace(".", "§"), text)
    parts = re.split(r"(?<=[.!?”\"])\s+(?=[A-Z$“\"(])", text)
    return [p.replace("§", ".").strip(" •·▪●-–—\u2022\t") for p in parts if p.strip(" •·▪●-–—\u2022\t")]


def _quotes(paras: list[str]) -> list[dict]:
    out = []
    for p in paras:
        for m in QUOTE.finditer(p):
            text = m.group(1).strip()
            after, before = p[m.end():m.end() + 160], p[max(0, m.start() - 160):m.start()]
            a = ATTRIBUTION.search(after) or ATTRIBUTION_BEFORE.search(before)
            speaker, title = (a.group(1).strip(), (a.group(2) or "").strip()) if a else ("", "")
            if not speaker and out:  # a follow-on quote from the same executive
                speaker, title = out[-1]["speaker"], out[-1]["title"]
            out.append({"text": text, "speaker": speaker, "title": title})
    return out[:4]


# ---------- heavy stage: FinBERT ----------

def _heuristic(sent: str) -> float:
    category = _category(sent)
    hits = sum(bool(re.search(p, sent, re.I)) for _, p in CATEGORIES)
    return 2.0 * bool(NUMBER.search(sent)) + 0.8 * min(hits, 3) + 0.8 * (category == "Guidance") \
        - 0.6 * (category == "Other")


def score_release(narrative: list[str], background: bool = False) -> dict:
    """Pick the useful sentences and quotes, score them with FinBERT. Stored once per release."""
    sents, pos = [], 0
    for p in narrative:
        for s in _sentences(p):
            pos += 1
            n = len(s.split())
            if 8 <= n <= 70 and not BOILERPLATE.search(s):
                sents.append({"text": s, "pos": pos, "category": _category(s), "base": _heuristic(s)})
    keep = sorted(sents, key=lambda s: -s["base"])[:config.NLP_MAX_SENTENCES]
    keep.sort(key=lambda s: s["pos"])
    quotes = _quotes(narrative)
    probs = finbert.score([s["text"] for s in keep] + [q["text"][:400] for q in quotes], background=background)
    for s, p in zip(keep, probs):
        s["p"] = p
    for q, p in zip(quotes, probs[len(keep):]):
        q["p"] = p
    words = _words(" ".join(narrative))
    return {"sentences": keep, "quotes": quotes, "words": len(words), "clean": " ".join(s["text"] for s in sents),
            "hedging": 1000 * sum(w in HEDGE for w in words) / len(words) if words else None}


# ---------- light stage: selection ----------

def _label(p: dict) -> str:
    return max(p, key=p.get)


def _polarity(p: dict) -> float:
    return p.get("positive", 0.0) - p.get("negative", 0.0)


def _pick_diverse(cands: list[dict], k: int, max_sim: float = 0.5) -> list[dict]:
    if not cands:
        return []
    V = TfidfVectorizer(stop_words=STOP).fit_transform([c["text"] for c in cands])
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["score"])
    picked = []
    for i in order:
        if len(picked) == k:
            break
        if picked and (V[i] @ V[picked].T).toarray().max() > max_sim:
            continue
        picked.append(i)
    return [cands[i] for i in sorted(picked, key=lambda i: cands[i]["pos"])]


def finalise(scored: dict, novelty: dict | None) -> dict:
    sents = scored["sentences"]
    for s in sents:
        w = _words(s["text"])
        new = max((novelty or {}).get(x, 0.0) for x in w) if w else 0.0
        s["score"] = s["base"] + 1.2 * abs(_polarity(s["p"])) + 1.2 * new
    view = lambda s: {"text": s["text"], "category": s["category"], "tone": _label(s["p"]),
                      "confidence": max(s["p"].values()), "pos": s["pos"]}
    key_points = [view(s) for s in _pick_diverse(sents, config.RESEARCH_KEY_POINTS)]
    guidance = [view(s) for s in _pick_diverse(
        [s for s in sents if s["category"] == "Guidance" and NUMBER.search(s["text"])], 6)]
    pol = [_polarity(s["p"]) for s in sents]
    labels = [_label(s["p"]) for s in sents]
    by_cat = {}
    for s in sents:
        by_cat.setdefault(s["category"], []).append(_polarity(s["p"]))
    return {
        "key_points": key_points,
        "guidance": guidance,
        "quotes": [{**{k: q[k] for k in ("text", "speaker", "title")}, "tone": _label(q["p"]),
                    "polarity": _polarity(q["p"])} for q in scored["quotes"]],
        "tone": {
            "net": float(np.mean(pol)) if pol else None,
            "positive": labels.count("positive") / len(labels) if labels else None,
            "negative": labels.count("negative") / len(labels) if labels else None,
            "neutral": labels.count("neutral") / len(labels) if labels else None,
            "quotes": float(np.mean([_polarity(q["p"]) for q in scored["quotes"]])) if scored["quotes"] else None,
            "by_category": {k: {"net": float(np.mean(v)), "n": len(v)} for k, v in by_cat.items()},
        },
        "stats": {"words": scored["words"], "scored": len(sents), "hedging": scored["hedging"]},
    }


def refresh_company_analysis(con: sqlite3.Connection, symbol: str) -> None:
    """Topics and key points for every analysed release, comparing each with the one before it."""
    rows = con.execute("SELECT accession, scored FROM releases WHERE symbol=? AND status='done' "
                       "ORDER BY filing_date", (symbol,)).fetchall()
    if not rows:
        return
    name = con.execute("SELECT name FROM companies WHERE symbol=?", (symbol,)).fetchone()
    name_words = _words(name["name"] if name else "")
    own = set(_words(symbol)) | set(name_words) | {"".join(w[0] for w in name_words)}
    scored_rows = [json.loads(r["scored"]) for r in rows]
    docs = [sc.get("clean") or " ".join(x["text"] for x in sc["sentences"]) for sc in scored_rows]
    vec = TfidfVectorizer(stop_words=STOP + sorted(own), ngram_range=(1, 2), sublinear_tf=True,
                          max_df=1.0 if len(docs) < 3 else 0.9, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    for k, r in enumerate(rows):
        row = X[k].toarray().ravel()
        prev = X[k - 1].toarray().ravel() if k > 0 else None
        ranked = [j for j in row.argsort()[::-1] if row[j] > 0]
        scale = row.max() or 1.0
        novelty = {terms[j]: row[j] / scale for j in ranked if " " not in terms[j]} if len(rows) > 1 else None
        analysis = finalise(scored_rows[k], novelty)
        analysis["topics"] = {
            "top": _dedupe([terms[j] for j in ranked], 14),
            "new": _dedupe([terms[j] for j in ranked if prev is not None and prev[j] == 0], 10),
            "compared_with": rows[k - 1]["accession"] if k > 0 else None,
        }
        con.execute("UPDATE releases SET analysis=? WHERE symbol=? AND accession=?",
                    (json.dumps(analysis), symbol, r["accession"]))
    con.commit()


def _dedupe(terms: list[str], k: int) -> list[str]:
    covered = {w for p in terms[:k * 3] if " " in p for w in p.split()}
    out = [t for t in terms if " " in t or t not in covered]
    return out[:k]


# ---------- ingest ----------

def _analyse_release(con, symbol: str, cik: int, rel: sqlite3.Row, background: bool) -> None:
    try:
        url, html = edgar.press_release(cik, rel["accession"], rel["primary_doc"])
        narrative, period = extract(html)
        if not narrative:
            raise edgar.EdgarError("No readable text in the press release.")
        scored = score_release(narrative, background)
        con.execute("UPDATE releases SET url=?, period=?, narrative=?, scored=?, status='done', error=NULL "
                    "WHERE symbol=? AND accession=?",
                    (url, period, "\n\n".join(narrative), json.dumps(scored), symbol, rel["accession"]))
    except Exception as exc:  # keep the others going; the page shows which release failed
        con.execute("UPDATE releases SET status='error', error=? WHERE symbol=? AND accession=?",
                    (f"{type(exc).__name__}: {exc}"[:300], symbol, rel["accession"]))
    con.commit()
    refresh_company_analysis(con, symbol)


def _process_pending(symbol: str, limit: int | None = None, background: bool = False) -> None:
    with _lock_for(symbol):
        con = _db()
        try:
            c = con.execute("SELECT cik FROM companies WHERE symbol=?", (symbol,)).fetchone()
            pending = con.execute("SELECT * FROM releases WHERE symbol=? AND status='pending' "
                                  "ORDER BY filing_date DESC", (symbol,)).fetchall()
            for rel in pending[:limit]:
                _analyse_release(con, symbol, c["cik"], rel, background)
        finally:
            con.close()


def ensure(query: str) -> str:
    """Make sure a company is in the database (fetching and analysing its newest release if needed).

    Returns the ticker. Older releases continue in a background thread.
    """
    con = _db()
    q = query.strip()
    row = con.execute("SELECT symbol, updated FROM companies WHERE symbol=? OR lower(name) LIKE ?",
                      (q.upper(), f"%{q.lower()}%")).fetchone() if q else None
    if row and not edgar.is_stale(row["updated"]):
        con.close()
        if pending_count(row["symbol"]):
            _start_background(row["symbol"])
        return row["symbol"]
    hit = edgar.lookup(row["symbol"] if row else q)
    if not hit:
        con.close()
        raise edgar.EdgarError(f"No US-listed company matches '{query}' on SEC EDGAR.")
    symbol, cik = hit["ticker"].upper(), int(hit["cik"])
    with _lock_for(symbol):
        profile = edgar.company_profile(cik)
        filings = edgar.earnings_filings(profile, config.RESEARCH_RELEASES)
        if not filings:
            con.close()
            raise edgar.EdgarError(f"{hit['name']} has no earnings releases (8-K Item 2.02) on EDGAR. "
                                   "Foreign companies that report on Form 6-K aren't covered yet.")
        financials = edgar.quarterly_financials(cik)
        con.execute("INSERT OR REPLACE INTO companies VALUES (?,?,?,?,?,?,?,?,?)", (
            symbol, cik, profile["name"] or hit["name"], hit.get("exchange") or "", profile["sic"], profile["state"],
            profile["fy_end"], json.dumps(financials), datetime.now().isoformat(timespec="seconds")))
        for f in filings:
            con.execute("INSERT OR IGNORE INTO releases (symbol, accession, filing_date, primary_doc, status) "
                        "VALUES (?,?,?,?, 'pending')", (symbol, f["accession"], f["filing_date"], f["primary"]))
        keep = {f["accession"] for f in filings}
        old = [r["accession"] for r in con.execute("SELECT accession FROM releases WHERE symbol=?", (symbol,))
               if r["accession"] not in keep]
        con.executemany("DELETE FROM releases WHERE symbol=? AND accession=?", [(symbol, a) for a in old])
        con.commit()
    con.close()
    _process_pending(symbol, limit=1)  # newest release now, so the page has something to show
    _start_background(symbol)
    return symbol


def _start_background(symbol: str) -> None:
    threading.Thread(target=_process_pending, args=(symbol,), kwargs={"background": True}, daemon=True).start()


# ---------- reads for the page ----------

def library() -> list[dict]:
    con = _db()
    out = []
    for c in con.execute("SELECT * FROM companies ORDER BY symbol").fetchall():
        rels = con.execute("SELECT filing_date, period, analysis FROM releases WHERE symbol=? AND status='done' "
                           "ORDER BY filing_date DESC", (c["symbol"],)).fetchall()
        tones = [json.loads(r["analysis"])["tone"]["net"] for r in rels if r["analysis"]]
        fin = json.loads(c["financials"] or "{}")
        out.append({"symbol": c["symbol"], "name": c["name"], "sic": c["sic"], "exchange": c["exchange"],
                    "releases": len(rels), "latest": rels[0]["filing_date"] if rels else None,
                    "tone": tones[0] if tones else None,
                    "tone_change": tones[0] - tones[1] if len(tones) > 1 else None,
                    "revenue_yoy": fin.get("revenue_yoy")})
    con.close()
    return out


def company(symbol: str) -> dict | None:
    con = _db()
    c = con.execute("SELECT * FROM companies WHERE symbol=?", (symbol,)).fetchone()
    if not c:
        con.close()
        return None
    rels = []
    for r in con.execute("SELECT accession, filing_date, period, status, error, url, analysis FROM releases "
                         "WHERE symbol=? ORDER BY filing_date DESC", (symbol,)).fetchall():
        a = json.loads(r["analysis"]) if r["analysis"] else None
        rels.append({**{k: r[k] for k in r.keys() if k != "analysis"},
                     "tone": a["tone"] if a else None, "stats": a["stats"] if a else None})
    con.close()
    return {**{k: c[k] for k in c.keys() if k != "financials"},
            "financials": json.loads(c["financials"] or "{}"), "releases": rels}


def release(symbol: str, accession: str) -> dict | None:
    con = _db()
    r = con.execute("SELECT * FROM releases WHERE symbol=? AND accession=?", (symbol, accession)).fetchone()
    con.close()
    if not r or not r["analysis"]:
        return None
    return {**{k: r[k] for k in r.keys() if k not in ("scored", "analysis")},
            "analysis": json.loads(r["analysis"])}


def pending_count(symbol: str) -> int:
    con = _db()
    n = con.execute("SELECT COUNT(*) FROM releases WHERE symbol=? AND status='pending'", (symbol,)).fetchone()[0]
    con.close()
    return n
