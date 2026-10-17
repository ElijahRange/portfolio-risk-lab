"""Company Research data: earnings-call transcripts from Alpha Vantage, analysed once and stored.

Fetching a company (a CLI script or the page's "Add company" button) does all the work:
  Alpha Vantage OVERVIEW + EARNINGS + EARNINGS_CALL_TRANSCRIPT  ->  SQLite (data/research/research.db)
  -> one NLP pass per call (sections, tone, key points, guidance, analyst questions, topics) -> stored JSON.
The research page only reads the database, so it opens instantly and makes no API or model calls.

The NLP is classic and runs in about a second per call on LinuxONE: sentence scoring on finance keywords,
numbers and TF-IDF novelty (scikit-learn), small tone and hedging word lists, and Alpha Vantage's own
turn-by-turn sentiment scores.
"""
import json
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import config

_ingest_lock = threading.Lock()


class AlphaVantageError(RuntimeError):
    pass


# ---------- Alpha Vantage ----------

def _av(function: str, **params) -> dict:
    key = config.alphavantage_key()
    if not key:
        raise AlphaVantageError("No Alpha Vantage key yet. Add ALPHAVANTAGE_API_KEY=<your key> to the server's "
                                ".env file (free key: alphavantage.co/support/#api-key).")
    for attempt in range(2):  # an empty {} reply usually means brief throttling: wait and retry once
        resp = requests.get(config.ALPHAVANTAGE_URL, params={"function": function, **params, "apikey": key},
                            timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data or attempt:
            break
        time.sleep(5)
    message = data.get("Information") or data.get("Note") or data.get("Error Message")
    if message and len(data) <= 2:
        raise AlphaVantageError(message)
    time.sleep(config.ALPHAVANTAGE_PAUSE)  # stay under the free plan's per-second limit
    return data


def fiscal_quarter(period_end: str, fy_end_month: int) -> str:
    """'2026-08-02' with a January fiscal year end -> '2027Q2' (fiscal years are named by the year they end).

    The date is moved back 10 days so 52/53-week quarters that end a few days into a month count
    for the month before.
    """
    d = datetime.strptime(period_end, "%Y-%m-%d").date() - timedelta(days=10)
    k = (d.month - fy_end_month) % 12
    q = round(k / 3) or 4
    fy = d.year if d.month <= fy_end_month else d.year + 1
    return f"{fy}Q{q}"


def calendar_quarter(period_end: str) -> str:
    d = datetime.strptime(period_end, "%Y-%m-%d").date() - timedelta(days=10)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------- storage ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    symbol TEXT PRIMARY KEY, name TEXT, exchange TEXT, sector TEXT, industry TEXT, description TEXT,
    fy_end TEXT, market_cap REAL, updated TEXT);
CREATE TABLE IF NOT EXISTS calls (
    symbol TEXT, quarter TEXT, period_end TEXT, call_date TEXT, eps_reported REAL, eps_estimate REAL,
    eps_surprise_pct REAL, segments TEXT, analysis TEXT, fetched TEXT, PRIMARY KEY (symbol, quarter));
"""


def _db() -> sqlite3.Connection:
    config.RESEARCH_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.RESEARCH_DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def db_stamp() -> float:
    return config.RESEARCH_DB.stat().st_mtime if config.RESEARCH_DB.exists() else 0.0


# ---------- ingest ----------

def ingest(symbol: str, quarters: int = config.RESEARCH_QUARTERS, only: list[str] | None = None) -> dict:
    """Fetch a company's latest calls (or the `only` fiscal quarters) and analyse them. Returns a short report.

    Calls already stored are not fetched again. If Alpha Vantage stops us part-way (e.g. the free plan's
    daily limit), whatever was fetched is still analysed and saved, and the error is reported.
    """
    symbol = symbol.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", symbol):
        raise AlphaVantageError(f"'{symbol}' doesn't look like a ticker.")
    with _ingest_lock:
        report = {"symbol": symbol, "fetched": [], "missing": [], "skipped": [], "error": None}
        con = _db()
        try:
            _ingest(con, symbol, quarters, only, report)
        except AlphaVantageError as exc:
            report["error"] = str(exc)
        finally:
            if con.execute("SELECT 1 FROM calls WHERE symbol=?", (symbol,)).fetchone():
                analyse_company(con, symbol)
            con.commit()
            con.close()
        return report


def _ingest(con, symbol, quarters, only, report):
    ov = _av("OVERVIEW", symbol=symbol)
    if not ov.get("Symbol"):
        raise AlphaVantageError(f"Alpha Vantage returned no company data for {symbol}. Check the ticker (US "
                                "listings work best), or wait a minute in case the API is throttling.")
    fy_month = MONTHS.get(ov.get("FiscalYearEnd", "December"), 12)
    con.execute("INSERT OR REPLACE INTO companies VALUES (?,?,?,?,?,?,?,?,?)", (
        symbol, ov.get("Name", symbol), ov.get("Exchange", ""), ov.get("Sector", "").title(),
        ov.get("Industry", "").title(), ov.get("Description", ""), ov.get("FiscalYearEnd", ""),
        _num(ov.get("MarketCapitalization")), datetime.now().isoformat(timespec="seconds")))
    con.commit()

    earnings = _av("EARNINGS", symbol=symbol).get("quarterlyEarnings", [])
    today = date.today().isoformat()
    rows = [e for e in earnings if e.get("reportedDate") and e["reportedDate"] <= today]
    for e in rows:
        e["label"] = fiscal_quarter(e["fiscalDateEnding"], fy_month)
    if only:
        wanted = {q.upper() for q in only}
        rows = [e for e in rows if e["label"] in wanted] + \
               [{"label": q, "fiscalDateEnding": None, "reportedDate": None} for q in
                wanted - {e["label"] for e in rows}]
    else:
        rows = rows[:quarters]

    have = {r["quarter"] for r in con.execute(
        "SELECT quarter FROM calls WHERE symbol=? AND segments != '[]'", (symbol,))}
    for e in rows:
        label = e["label"]
        if label in have:
            report["skipped"].append(label)
            continue
        segments = _av("EARNINGS_CALL_TRANSCRIPT", symbol=symbol, quarter=label).get("transcript") or []
        if not segments and e.get("fiscalDateEnding"):
            alt = calendar_quarter(e["fiscalDateEnding"])
            if alt != label:
                segments = _av("EARNINGS_CALL_TRANSCRIPT", symbol=symbol, quarter=alt).get("transcript") or []
        segments = [{"speaker": s.get("speaker", ""), "title": s.get("title", ""),
                     "content": s.get("content", ""), "sentiment": _num(s.get("sentiment"))} for s in segments]
        con.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)", (
            symbol, label, e.get("fiscalDateEnding"), e.get("reportedDate"), _num(e.get("reportedEPS")),
            _num(e.get("estimatedEPS")), _num(e.get("surprisePercentage")), json.dumps(segments), None,
            datetime.now().isoformat(timespec="seconds")))
        con.commit()
        (report["fetched"] if segments else report["missing"]).append(label)


# ---------- NLP ----------

POS = set("""strong stronger strongest strength growth grew grow growing record increase increased increasing improve
improved improvement improving exceeded exceed beat raise raised raising momentum accelerate accelerated accelerating
robust expand expanded expansion outperform outperformed solid healthy gain gains upside confident favorable tailwind
tailwinds double-digit resilient resilience pleased progress opportunity opportunities""".split())
NEG = set("""decline declined declining decrease decreased decreasing headwind headwinds weak weaker weakness soft
softer softness pressure pressures pressured challenging challenge challenges uncertain uncertainty delay delayed
delays slowdown slower slowing loss losses miss missed cut cuts reduced impairment tariff tariffs inflation volatile
volatility difficult unfavorable deteriorate deteriorated deterioration disappointing shortfall lower""".split())
HEDGE = set("""may might could possibly perhaps uncertain uncertainty approximately roughly believe assume assumption
assumptions appear appears likely unlikely potential potentially depend depends depending somewhat unclear cautious
hope hopefully expect anticipate""".split())
BOILERPLATE = re.compile(
    r"forward-looking|safe harbor|non-gaap|reconciliation|webcast|replay|press release|investor relations|"
    r"supplemental|turn (the call|it) over|hand (the call|it) over|good (morning|afternoon|evening)|"
    r"joining us|thank you|thanks,|question-and-answer|q&a session|operator", re.I)
NUMBER = re.compile(r"\$\s?\d|\d+(?:\.\d+)?\s?(?:%|percent|basis points|bps|billion|million)|\b\d{1,3}(?:,\d{3})+\b",
                    re.I)
CATEGORIES = [
    ("Guidance", r"guidance|outlook|\bexpect|anticipate|forecast|full[- ]year|next quarter|fiscal (?:year|20\d\d)|"
                 r"for the (?:year|quarter ahead)|going forward"),
    ("Capital return", r"dividend|buyback|repurchas|share count|returned? .{0,30}shareholders"),
    ("Margins & costs", r"margin|\bcosts?\b|expense|pricing|productivity|efficienc"),
    ("Results", r"revenue|\bsales\b|earnings|\beps\b|profit|income|cash flow|\bgrew\b|growth|comparable|volume"),
    ("Risks", r"headwind|\brisks?\b|tariff|inflation|uncertain|pressure|declin|weak|soft"),
    ("Strategy", r"acqui|launch|invest|\bai\b|platform|partnership|customer|innovation|portfolio|pipeline|backlog"),
]
STOP = list(ENGLISH_STOP_WORDS | set("""quarter quarters year years think going just really know lot little bit
yes yeah okay ok sure great good thank thanks question questions mean like said say look looking things thing kind
want way maybe right today able continue continued continues also us we our year-over-year percent million billion
point points basis term terms time number numbers make made let talk talked talking""".split()))
WORD = re.compile(r"[a-z][a-z\-']+")


def _role(title: str, speaker: str) -> str:
    t = f"{title} {speaker}".lower()
    if "operator" in t:
        return "Operator"
    if "analyst" in t:
        return "Analyst"
    if "investor relations" in t or re.search(r"\bir\b", t):
        return "IR"
    return "Executive"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z$\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _category(text: str) -> str:
    for name, pattern in CATEGORIES:
        if re.search(pattern, text, re.I):
            return name
    return "Other"


def _words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def _tone_label(text: str) -> str:
    w = _words(text)
    net = sum(x in POS for x in w) - sum(x in NEG for x in w)
    return "good" if net > 0 else ("bad" if net < 0 else "neutral")


def _weighted(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w]
    return float(np.average([v for v, _ in pairs], weights=[w for _, w in pairs])) if pairs else None


def _per_k(words: list[str], lexicon: set) -> float | None:
    return 1000 * sum(w in lexicon for w in words) / len(words) if words else None


def _pick_diverse(cands: list[dict], vectors, k: int, max_sim: float = 0.5) -> list[dict]:
    """Highest-scoring sentences, skipping any too similar to one already picked (maximal marginal relevance)."""
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["score"])
    picked = []
    for i in order:
        if len(picked) == k:
            break
        if picked and (vectors[i] @ vectors[picked].T).toarray().max() > max_sim:
            continue
        picked.append(i)
    return [cands[i] for i in sorted(picked, key=lambda i: cands[i]["pos"])]


def analyse_call(segments: list[dict], novelty: dict[str, float] | None = None) -> dict:
    """One-off NLP pass over a call. `novelty` = term -> how unusual it is versus the company's other calls."""
    for i, s in enumerate(segments):
        s["role"] = _role(s["title"], s["speaker"])
        s["words"] = len(s["content"].split())
    first_analyst = next((i for i, s in enumerate(segments) if s["role"] == "Analyst"), len(segments))
    qa_start = first_analyst - 1 if first_analyst > 0 and segments[first_analyst - 1]["role"] == "Operator" \
        else first_analyst
    for i, s in enumerate(segments):
        s["section"] = "Q&A" if i >= qa_start else "Prepared remarks"

    mgmt = [s for s in segments if s["role"] in ("Executive", "IR")]
    prepared = [s for s in mgmt if s["section"] == "Prepared remarks"]
    answers = [s for s in mgmt if s["section"] == "Q&A"]
    analysts = [s for s in segments if s["role"] == "Analyst"]

    def tone(group):
        return _weighted([s["sentiment"] for s in group], [s["words"] for s in group])

    w_prep = _words(" ".join(s["content"] for s in prepared))
    w_ans = _words(" ".join(s["content"] for s in answers))
    w_mgmt = w_prep + w_ans

    # Candidate key-point sentences: management, not boilerplate, a sensible length.
    cands, pos = [], 0
    for seg_i, s in enumerate(segments):
        for sent in _sentences(s["content"]):
            pos += 1
            n = len(sent.split())
            if s["role"] != "Executive" or not 10 <= n <= 60 or BOILERPLATE.search(sent):
                continue
            words = _words(sent)
            category = _category(sent)
            hits = sum(bool(re.search(p, sent, re.I)) for _, p in CATEGORIES)
            new = max((novelty or {}).get(w, 0.0) for w in words) if words else 0.0
            score = (2.0 * bool(NUMBER.search(sent)) + 0.8 * min(hits, 3) + 0.8 * (category == "Guidance")
                     + 1.5 * new + 0.4 * (s["section"] == "Prepared remarks") - 0.6 * (category == "Other"))
            cands.append({"text": sent, "score": score, "pos": pos, "category": category, "tone": _tone_label(sent),
                          "speaker": s["speaker"], "title": s["title"], "section": s["section"], "segment": seg_i})

    key_points, guidance = [], []
    if cands:
        vec = TfidfVectorizer(stop_words=STOP).fit([c["text"] for c in cands])
        V = vec.transform([c["text"] for c in cands])
        key_points = _pick_diverse(cands, V, config.RESEARCH_KEY_POINTS)
        g_idx = [i for i, c in enumerate(cands) if c["category"] == "Guidance" and NUMBER.search(c["text"])]
        guidance = _pick_diverse([cands[i] for i in g_idx], V[g_idx], 6) if g_idx else []
    for c in key_points + guidance:
        c.pop("score", None)

    # Analyst questions: who asked, from which firm, about what, who answered and how.
    questions = []
    for i, s in enumerate(segments):
        if s["role"] != "Analyst" or (i > 0 and segments[i - 1]["speaker"] == s["speaker"]):
            continue
        firm = ""
        prev = segments[i - 1] if i > 0 and segments[i - 1]["role"] == "Operator" else None
        if prev and s["speaker"] and s["speaker"] in prev["content"]:
            after = prev["content"].split(s["speaker"], 1)[1]
            m = re.match(r"\s*(?:with|from|at|of)\s+(?:the\s+)?(.+?)(?:\.|,|;|\s+Your\b|\s+Please\b|$)", after)
            firm = m.group(1).strip() if m and len(m.group(1)) < 60 else ""
        sents = _sentences(s["content"])
        asks = [x for x in sents if x.endswith("?")] or sents[:2]
        gist = " ".join(asks[:2])
        gist = gist if len(gist) <= 320 else gist[:317].rsplit(" ", 1)[0] + "…"
        replies = []
        for t in segments[i + 1:]:
            if t["role"] in ("Analyst", "Operator"):
                break
            replies.append(t)
        questions.append({
            "analyst": s["speaker"], "firm": firm, "question": gist, "topic": _category(s["content"]),
            "answered_by": list(dict.fromkeys(r["speaker"] for r in replies))[:2],
            "answer_tone": tone(replies), "segment": i,
        })

    return {
        "stats": {
            "words": sum(s["words"] for s in segments),
            "prepared_words": sum(s["words"] for s in segments if s["section"] == "Prepared remarks"),
            "qa_words": sum(s["words"] for s in segments if s["section"] == "Q&A"),
            "analysts": len({s["speaker"] for s in analysts}),
            "questions": len(questions),
        },
        "tone": {
            "management": tone(mgmt), "prepared": tone(prepared), "answers": tone(answers),
            "analysts": tone(analysts),
            "net_words": (_per_k(w_mgmt, POS) or 0) - (_per_k(w_mgmt, NEG) or 0) if w_mgmt else None,
            "hedge_prepared": _per_k(w_prep, HEDGE), "hedge_answers": _per_k(w_ans, HEDGE),
        },
        "speakers": [
            {"speaker": sp, "title": next(s["title"] for s in segments if s["speaker"] == sp),
             "role": next(s["role"] for s in segments if s["speaker"] == sp),
             "words": sum(s["words"] for s in segments if s["speaker"] == sp),
             "tone": tone([s for s in segments if s["speaker"] == sp])}
            for sp in dict.fromkeys(s["speaker"] for s in segments if s["role"] != "Operator")
        ],
        "timeline": [{"i": i, "speaker": s["speaker"], "role": s["role"], "section": s["section"],
                      "sentiment": s["sentiment"], "words": s["words"]} for i, s in enumerate(segments)],
        "key_points": key_points,
        "guidance": guidance,
        "questions": questions,
    }


def _dedupe_terms(terms: list[str], k: int) -> list[str]:
    """Keep the best-ranked terms, dropping single words already covered by a kept two-word phrase."""
    phrases = [t for t in terms[:k * 3] if " " in t]
    covered = {w for p in phrases for w in p.split()}
    out = []
    for t in terms:
        if " " not in t and t in covered:
            continue
        out.append(t)
        if len(out) == k:
            break
    return out


def analyse_company(con: sqlite3.Connection, symbol: str) -> None:
    """(Re)analyse every stored call of a company. Topic novelty compares each call with the others."""
    rows = con.execute("SELECT quarter, segments FROM calls WHERE symbol=? AND segments != '[]' ORDER BY quarter",
                       (symbol,)).fetchall()
    if not rows:
        return
    calls = [(r["quarter"], json.loads(r["segments"])) for r in rows]
    docs = [" ".join(s["content"] for s in segs if _role(s["title"], s["speaker"]) in ("Executive", "IR"))
            for _, segs in calls]
    name = con.execute("SELECT name FROM companies WHERE symbol=?", (symbol,)).fetchone()
    own = set(_words(f"{symbol} {name['name'] if name else ''}")) - {"inc", "corp", "co", "the"}
    vec = TfidfVectorizer(stop_words=STOP + sorted(own), ngram_range=(1, 2), min_df=1,
                          max_df=1.0 if len(docs) < 3 else 0.9, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    for k, (quarter, segs) in enumerate(calls):
        row = X[k].toarray().ravel()
        prev = X[k - 1].toarray().ravel() if k > 0 else None
        ranked = [j for j in row.argsort()[::-1] if row[j] > 0]
        top = _dedupe_terms([terms[j] for j in ranked], 14)
        new = _dedupe_terms([terms[j] for j in ranked if prev is not None and prev[j] == 0], 10)
        scale = row.max() or 1.0
        novelty = {t: row[j] / scale for j, t in enumerate(terms) if row[j] > 0 and " " not in t} \
            if len(calls) > 1 else None
        analysis = analyse_call(segs, novelty)
        analysis["topics"] = {"top": top, "new": new, "compared_with": calls[k - 1][0] if k > 0 else None}
        con.execute("UPDATE calls SET analysis=? WHERE symbol=? AND quarter=?",
                    (json.dumps(analysis), symbol, quarter))
    con.commit()


def reanalyse_all() -> None:
    con = _db()
    for (symbol,) in con.execute("SELECT DISTINCT symbol FROM calls").fetchall():
        analyse_company(con, symbol)
    con.close()


# ---------- reads for the page (no network) ----------

def library() -> list[dict]:
    con = _db()
    out = []
    for c in con.execute("SELECT * FROM companies ORDER BY symbol").fetchall():
        calls = con.execute("SELECT quarter, call_date, eps_surprise_pct, analysis FROM calls "
                            "WHERE symbol=? AND analysis IS NOT NULL ORDER BY quarter DESC", (c["symbol"],)).fetchall()
        tones = [json.loads(r["analysis"])["tone"]["management"] for r in calls]
        out.append({**dict(c), "description": None, "calls": len(calls),
                    "latest": calls[0]["quarter"] if calls else None,
                    "latest_date": calls[0]["call_date"] if calls else None,
                    "tone": tones[0] if tones else None,
                    "tone_change": tones[0] - tones[1] if len(tones) > 1 and None not in tones[:2] else None,
                    "eps_surprise": calls[0]["eps_surprise_pct"] if calls else None})
    con.close()
    return out


def company(symbol: str) -> dict | None:
    con = _db()
    c = con.execute("SELECT * FROM companies WHERE symbol=?", (symbol,)).fetchone()
    if not c:
        con.close()
        return None
    calls = []
    for r in con.execute("SELECT quarter, period_end, call_date, eps_reported, eps_estimate, eps_surprise_pct, "
                         "analysis FROM calls WHERE symbol=? ORDER BY quarter DESC", (symbol,)).fetchall():
        a = json.loads(r["analysis"]) if r["analysis"] else None
        calls.append({**{k: r[k] for k in r.keys() if k != "analysis"}, "available": a is not None,
                      "tone": a["tone"] if a else None, "stats": a["stats"] if a else None})
    con.close()
    return {**dict(c), "calls": calls}


def call(symbol: str, quarter: str) -> dict | None:
    con = _db()
    r = con.execute("SELECT * FROM calls WHERE symbol=? AND quarter=?", (symbol, quarter)).fetchone()
    con.close()
    if not r or not r["analysis"]:
        return None
    return {**{k: r[k] for k in r.keys() if k not in ("segments", "analysis")},
            "segments": json.loads(r["segments"]), "analysis": json.loads(r["analysis"])}
