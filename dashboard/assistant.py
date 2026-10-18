"""Portfolio assistant behind the chat bubble: answers simple questions about the portfolio.

Deterministic and fast on purpose (same question -> same answer, and the numbers always match the charts):
1. Understand: a small Hugging Face sentence-embedding model (config.ASSISTANT_MODEL, loaded with `transformers`
   and run locally on LinuxONE like FinBERT) embeds the question. The intent whose example phrasings are closest
   (cosine similarity) wins. Below config.ASSISTANT_THRESHOLD it lists what it can answer instead of guessing.
2. Extract: plain code finds tickers / company names, confidence levels, stress scenarios and optimizer goals.
3. Answer: a template filled from the same objects the pages use (service, risk_metrics, optimizer,
   research_store). No model writes text, so no number is ever invented.
"""
import re
import threading
import logging
import time
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from . import components as ui
from . import config, optimizer, research_store, risk_metrics, service

config.setup_hugging_face()

KEYWORD_BONUS = 0.1  # added to an intent's similarity when the question contains one of its telltale words

# Company names and nicknames of the holdings (the tickers themselves always match).
COMPANIES = {
    "VZ": ("Verizon", ["verizon"]),
    "MMM": ("3M", ["3M", "three m"]),
    "RELIANCE.NS": ("Reliance Industries", ["reliance", "reliance industries", "RIL"]),
    "WMT": ("Walmart", ["walmart", "wal-mart"]),
    "TPB": ("Turning Point Brands", ["turning point", "turning point brands"]),
    "TGT": ("Target", ["target"]),
    "CB": ("Chubb", ["chubb"]),
    "BA": ("Boeing", ["boeing"]),
    "PM": ("Philip Morris", ["philip morris", "phillip morris", "PMI"]),
}
CASE_SENSITIVE = {"PM", "3M", "RIL", "PMI"}  # only match these when typed in capitals ("3m" may mean 3 months)
NOT_A_STOCK = r"(?!\s+(?:beta|weights?|vol|volatility|returns?|risk|allocation|portfolio))"  # "target beta"
# Capitalised words that look like tickers but aren't.
NOT_TICKERS = {"I", "A", "VAR", "CVAR", "ES", "NAV", "PNL", "SPX", "SPY", "ETF", "ETFS", "PCA", "YTD", "EWMA", "USD",
               "INR", "AI", "ML", "OK", "CEO", "CFO", "EPS", "US", "USA", "IBM", "Z", "SEC", "EDGAR", "MTD", "QTD",
               "AUM", "GDP", "CPI", "FED", "VS", "OR", "AND", "THE", "IS", "IT", "MY", "WE", "OUR", "ME", "DO", "HOW",
               "WHAT", "WHO", "WHY", "P", "L", "S", "MPT", "PE", "ROE", "ROI", "FAQ", "LLM", "FX", "TL", "DR", "SP"}

# Shorthand the embedding model doesn't know, spelled out before embedding (applied to lowercase text).
EXPAND = [
    (r"\bcvar\b", "expected shortfall"),
    (r"\bvar\b", "value at risk"),
    (r"\bes\b", "expected shortfall"),
    (r"\bp ?& ?l\b|\bpnl\b|\bp/l\b", "profit and loss"),
    (r"\bnav\b", "net asset value"),
    (r"\bvols?\b|\bvola\b", "volatility"),
    (r"\bmax ?dd\b|\bmdd\b", "maximum drawdown"),
    (r"\bidio\b", "idiosyncratic"),
    (r"\bcorr\b", "correlation"),
    (r"\bytd\b", "year to date"),
    (r"\b(the )?(s ?& ?p( ?500)?|spx|spy|sp ?500)\b", "the s&p 500"),
]

SCENARIOS = [  # (pattern, scenario name in risk_metrics.SCENARIOS / the stress table)
    (r"june? 2022|inflation scare|\binflation\b", "Jun 2022 inflation scare"),
    (r"bear market|\b2022\b", "2022 bear market"),
    (r"\bbanks?\b|banking|\bsvb\b|silicon valley|mar(ch)? 2023|\b2023\b", "Mar 2023 regional banks"),
    (r"\byen\b|carry|aug(ust)? 2024|\b2024\b", "Aug 2024 yen carry unwind"),
    (r"tariff|liberation day|apr(il)? 2025|\b2025\b", "Apr 2025 tariff shock"),
    (r"worst (single |(the )?s&p 500 |market |trading )?day", "Worst S&P 500 day"),
    (r"worst (single |(the )?s&p 500 |market )?month", "Worst S&P 500 month"),
]
SHOCK = re.compile(r"(?:s&p 500|market|index|stocks?|equities)\D{0,25}?(?:falls?|drops?|declines?|crash(?:es)?|"
                   r"tanks?|loses?|down|sinks?|plunges?|fell|dropped)\D{0,6}?(\d{1,2}(?:\.\d+)?)\s*%|"
                   r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:fall|drop|decline|crash|sell-?off|correction|move down|lower)")
GOALS = [
    (r"sortino", "max_sortino"),
    (r"sharpe|risk.adjusted", "max_sharpe"),
    (r"idiosyncratic|stock.specific", "min_idio"),
    (r"risk parity|equal risk", "risk_parity"),
    (r"diversif", "max_div"),
    (r"min(imum|imise|imize)? ?vol|(least|lowest|lower|less) (volatil|risk)|safest|(reduce|cut) (our )?(risk|volatil)|de-?risk",
     "min_vol"),
]
WORST = re.compile(r"\b(worst|least|lowest|smallest|bottom|weakest|lagg|loser|negative(ly)?|uncorrelated|"
                   r"less correlated|least correlated)")

TABS = {"overview": "Portfolio View", "factors": "Factor Model", "idio": "Idiosyncratic Risk",
        "risk": "Risk Analysis"}  # labels as in app.TABS
RATIO_LINK = {"label": "Open Ratio Testing", "href": "/ratio-testing"}
LOOKBACK_WORDS = {"1Y": "1-year", "3Y": "3-year", "Full": "full-history"}
DEFAULT_CHIPS = ["What is our 1-day VaR?", "What's the portfolio beta?", "Which stock adds the most risk?",
                 "How did we do in the 2022 bear market?", "How should we rebalance for a better Sharpe?"]


# ---------- parsing ----------

@dataclass
class Query:
    text: str                                       # as typed
    norm: str = ""                                  # what the model sees
    tickers: list = field(default_factory=list)     # holdings mentioned, in order
    others: list = field(default_factory=list)      # stored Company Research symbols that aren't holdings
    unknown: list = field(default_factory=list)     # capitalised words that look like tickers we don't know
    level: float | None = None
    scenario: str | None = None
    shock: float | None = None
    goal: str | None = None
    worst: bool = False


_library = {"t": 0.0, "symbols": set()}


def _research_symbols() -> set:
    """Companies already stored by Company Research (re-read at most once a minute; never fetches EDGAR)."""
    if time.time() - _library["t"] > 60:
        try:
            _library["symbols"] = {c["symbol"] for c in research_store.library()}
        except Exception:
            _library["symbols"] = set()
        _library["t"] = time.time()
    return _library["symbols"]


def _patterns(holdings: list[str]) -> list[tuple[re.Pattern, str]]:
    out = []
    for t in holdings:
        names = [t, t.split(".")[0]] + COMPANIES.get(t, ("", []))[1]
        for n in dict.fromkeys(names):
            sensitive = n in CASE_SENSITIVE
            tail = NOT_A_STOCK if n == "target" else ""
            out.append((re.compile(rf"(?<![\w.]){re.escape(n)}(?![\w]){tail}", 0 if sensitive else re.I), t))
    return out


def parse(text: str, holdings: list[str]) -> Query:
    q = Query(text)
    spans = []
    for pat, t in _patterns(holdings):
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), t))
    others = _research_symbols() - set(holdings)
    shouting = text.upper() == text
    for m in re.finditer(r"(?<![\w.$])[A-Z]{1,5}(?![\w])", text):
        word = m.group()
        if any(s <= m.start() < e for s, e, _ in spans) or word in NOT_TICKERS or shouting:
            continue
        if word in others:
            q.others.append(word)
            spans.append((m.start(), m.end(), word))
        elif len(word) >= 2:
            q.unknown.append(word)
    # Keep the longest match where names overlap ("turning point brands" over "turning point").
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept, end = [], -1
    for s in spans:
        if s[0] >= end:
            kept.append(s)
            end = s[1]
    q.tickers = list(dict.fromkeys(t for *_, t in kept if t in holdings))
    q.others = list(dict.fromkeys(q.others))

    # Names become "the stock" so the model focuses on what is being asked, not on which company.
    masked = text
    for s, e, _ in reversed(kept):
        masked = masked[:s] + "the stock" + masked[e:]
    low = masked.lower()
    for pat, rep in EXPAND:
        low = re.sub(pat, rep, low)
    q.norm = re.sub(r"\s+", " ", re.sub(r"[^\w%&.+\-' ]", " ", low)).strip()

    if re.search(r"\b99(\.0)?\s*%|\b99th|\b1\s*%\s*(var|value at risk)", low):
        q.level = 0.99
    elif re.search(r"\b97\.5", low):
        q.level = 0.975
    elif re.search(r"\b95\s*%|\b95th|\b5\s*%\s*(var|value at risk)|\b95\b", low):
        q.level = 0.95
    m = SHOCK.search(low)
    if m:
        q.shock = -float(m.group(1) or m.group(2)) / 100
    for pat, name in SCENARIOS:
        if re.search(pat, low):
            q.scenario = name
            break
    for pat, goal in GOALS:
        if re.search(pat, low):
            q.goal = goal
            break
    q.worst = bool(WORST.search(low))
    return q


# ---------- the embedding model ----------

_lock = threading.Lock()
_model: dict = {}


def _embed(texts: list[str]) -> np.ndarray:
    import torch

    batch = _model["tok"](texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = _model["model"](**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    vec = (out * mask).sum(1) / mask.sum(1)  # mean pooling, as the model was trained
    return torch.nn.functional.normalize(vec, dim=1).numpy()


def _load() -> dict:
    if "examples" not in _model:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(config.NLP_THREADS)
        _model["tok"] = AutoTokenizer.from_pretrained(config.ASSISTANT_MODEL)
        _model["model"] = AutoModel.from_pretrained(config.ASSISTANT_MODEL).eval()
        holdings = list(COMPANIES)
        texts, owners = [], []
        for intent in INTENTS:
            for ex in intent.examples:
                texts.append(parse(ex, holdings).norm)
                owners.append(intent.key)
        _model["examples"] = np.vstack([_embed(texts[i:i + 64]) for i in range(0, len(texts), 64)])
        _model["owners"] = np.array(owners)
    return _model


def warm_up() -> None:
    """Load the model, embed the example questions and fill the data caches ahead of the first chat.

    Runs in a background thread at start-up, so even the first question answers in milliseconds.
    """
    try:
        with _lock:
            _load()
        d = Data()
        d.rep, d.opt  # noqa: B018  (computing them fills the risk-report and optimizer caches)
    except Exception:
        logging.getLogger(__name__).exception("Assistant warm-up failed")  # the first question shows it too


def classify(q: Query) -> list[tuple[str, float]]:
    """Intents ranked by similarity: best example match, plus a bonus when a telltale word appears."""
    with _lock:
        m = _load()
        v = _embed([q.norm])[0]
    sims = m["examples"] @ v
    scores = {}
    for intent in INTENTS:
        s = float(sims[m["owners"] == intent.key].max())
        if intent.keywords and re.search(intent.keywords, q.norm):
            s += KEYWORD_BONUS
        scores[intent.key] = s
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ---------- data shared by the answers ----------

class Data:
    """The same cached objects the pages render from."""

    def __init__(self):
        self.a = service.current_analysis()
        self.p = self.a.positions.set_index("ticker")
        self.nav = self.a.nav

    @cached_property
    def rep(self):
        return risk_metrics.report(self.a)

    @cached_property
    def opt(self):
        return optimizer.run(self.a.history, self.a.bench_returns.name, self.p["weight"], optimizer.Settings())


def _ans(value, label, text, tone="", rows=None, note=None, link=None, chips=None) -> dict:
    return {"value": value, "label": label, "text": text, "tone": tone, "rows": rows, "note": note, "link": link,
            "chips": chips or []}


def _tab(key: str) -> dict:
    return {"label": f"Open {TABS[key]}", "href": f"/?tab={key}", "tab": TABS[key]}


def _research_link(symbol: str) -> dict:
    return {"label": f"Open {symbol} on Company Research", "href": f"/research?symbol={symbol}"}


def _name(t: str) -> str:
    return COMPANIES[t][0] if t in COMPANIES else t


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _date(ts) -> str:
    return f"{pd.Timestamp(ts):%b %d, %Y}"


def _window(d: Data) -> str:
    r = d.a.port_returns
    return f"the last {len(r)} trading days ({_date(r.index[0])} to {_date(r.index[-1])})"


def _beta_meaning(b: float, dollars: float, who: str) -> str:
    move = -0.10 * b
    if b < 0:
        desc = "moves against the market on average"
    elif b < 0.5:
        desc = "is defensive"
    elif b < 0.9:
        desc = "is less sensitive than the market"
    elif b <= 1.1:
        desc = "moves roughly in line with the market"
    else:
        desc = "is aggressive, amplifying market moves"
    return (f"{who} {desc}: a 10% S&P 500 fall implies about {ui.pct(move, 1, signed=True)} "
            f"({ui.usd(move * dollars, signed=True, compact=True)}).")


def _corr_meaning(c: float) -> str:
    if c >= 0.7:
        return "strongly tied: they tend to move together"
    if c >= 0.4:
        return "moderately linked"
    if c >= 0.15:
        return "only weakly linked, so each still diversifies the other"
    if c > -0.15:
        return "almost unrelated, which is good for diversification"
    return "inclined to move in opposite directions: a natural hedge"


def _risk_vs_weight(rs: float, w: float) -> str:
    if rs < 0:
        return "Its risk share is negative: it lowers the portfolio's total volatility."
    ratio = rs / w if w else np.nan
    if ratio > 1.25:
        return "That is more than its weight, so it is a risk hot spot."
    if ratio < 0.75:
        return "That is less than its weight, so it diversifies the rest of the book."
    return "That is roughly in line with its weight."


# ---------- answers ----------

def h_help(q: Query, d: Data) -> dict:
    return _ans(None, None, "I answer questions about this portfolio using the dashboard's own numbers. Ask about "
                "any holding by name or ticker, or try one of these:",
                rows=[["Size", "NAV, holdings, weights, sectors"], ["Returns", "return, Sharpe, alpha, today's P&L"],
                      ["Risk", "volatility, VaR, expected shortfall, drawdown"],
                      ["Stress", "2022, banks, yen, tariffs, S&P −10%"],
                      ["Market", "beta, factor tilts, correlations"],
                      ["Ideas", "optimizer suggestions, earnings tone"]],
                chips=DEFAULT_CHIPS)


def h_smalltalk(q: Query, d: Data) -> dict:
    if re.search(r"thank|cheers|great|nice|awesome|perfect", q.norm):
        return _ans(None, None, "Happy to help. Ask me anything else about the portfolio.", chips=DEFAULT_CHIPS[:3])
    return _ans(None, None, "Hi! I'm the portfolio assistant. I answer questions about this portfolio's numbers: "
                "risk, returns, beta, stress tests and each holding.", chips=DEFAULT_CHIPS)


def h_nav(q: Query, d: Data) -> dict:
    a, k = d.a, d.a.kpis
    cash = a.nav - float(a.positions["market_value"].sum())
    cash_text = f" plus {ui.usd(cash)} cash" if abs(cash) >= 1 else ""
    return _ans(ui.usd(a.nav), "Net asset value (NAV)",
                f"Market value of the {k['n_positions']} positions{cash_text} at the {_date(a.as_of)} close. "
                f"Today's P&L is {ui.usd(k['day_pnl'], signed=True)} ({ui.pct(k['day_pnl'] / a.nav, signed=True)}).",
                link=_tab("overview"), chips=["What do we hold?", "What's today's P&L?", "Sector breakdown"])


def h_holdings(q: Query, d: Data) -> dict:
    if q.tickers:
        return h_weight(q, d)
    p = d.a.positions
    shorts = int((p["weight"] < 0).sum())
    rows = [[f"{r.ticker} · {_name(r.ticker)}", f"{ui.pct(r.weight, 1)} · {ui.usd(r.market_value, compact=True)}"]
            for r in p.itertuples()]
    side = "All long" if not shorts else f"{len(p) - shorts} long, {shorts} short"
    return _ans(f"{len(p)} stocks", "Holdings", f"{side}, {ui.usd(d.nav)} in total, sorted by weight.", rows=rows,
                link=_tab("overview"),
                chips=["What's our biggest position?", "Sector breakdown", "Which stock adds the most risk?"])


def h_weight(q: Query, d: Data) -> dict:
    if not q.tickers:
        return h_largest(q, d)
    p, n = d.p, len(d.p)
    if len(q.tickers) > 1:
        rows = [[f"{t} · {_name(t)}", f"{ui.pct(p.loc[t, 'weight'], 1)} · {ui.usd(p.loc[t, 'market_value'], compact=True)}"]
                for t in q.tickers]
        total = float(p.loc[q.tickers, "weight"].sum())
        return _ans(ui.pct(total, 1), f"{' + '.join(q.tickers)} combined",
                    f"{_join(q.tickers)} together are {ui.usd(total * d.nav)} of the {ui.usd(d.nav)} portfolio.",
                    rows=rows, link=_tab("overview"))
    t = q.tickers[0]
    w, mv = float(p.loc[t, "weight"]), float(p.loc[t, "market_value"])
    rank = 1 + int((p["weight"].abs() > abs(w) + 1e-9).sum())
    tied = int(((p["weight"].abs() - abs(w)).abs() < 1e-9).sum())
    place = ("tied for " if tied > 1 else "the ") + ("largest" if rank == 1 else f"{_ordinal(rank)}-largest")
    rs = float(d.rep.budget.loc[t, "risk_share"])
    return _ans(ui.pct(w, 1), f"{t} weight",
                f"{_name(t)} is {ui.usd(mv)} of the {ui.usd(d.nav)} portfolio, {place} of {n} positions. It drives "
                f"{ui.pct(rs, 1)} of portfolio risk. {_risk_vs_weight(rs, w)}",
                link=_tab("overview"), chips=[f"What is {t}'s beta?", f"How correlated is {t} with the rest?",
                                              f"Tell me about {t}"])


def h_largest(q: Query, d: Data) -> dict:
    if q.tickers:
        return h_weight(q, d)
    w = d.p["weight"]
    pick = w.abs().min() if q.worst else w.abs().max()
    names = list(w.index[(w.abs() - pick).abs() < 1e-9])
    other = w.abs().max() if q.worst else w.abs().min()
    others = list(w.index[(w.abs() - other).abs() < 1e-9])
    word, opp = ("Smallest", "largest") if q.worst else ("Largest", "smallest")
    if len(names) > 1:
        label = f"{word} positions (tied): {', '.join(names)}"
        text = (f"{_join(names)} are each {ui.usd(pick * d.nav)}; together they hold "
                f"{ui.pct(pick * len(names), 1)} of the portfolio.")
    else:
        label = f"{word} position: {names[0]}"
        text = f"{_name(names[0])} is {ui.usd(pick * d.nav)} of the {ui.usd(d.nav)} portfolio."
    text += f" The {opp} {'are' if len(others) > 1 else 'is'} {_join(others)} at {ui.pct(other, 1)} each."
    if len(others) == 1:
        text = text.replace(" each.", ".")
    return _ans(ui.pct(pick, 1), label, text, link=_tab("overview"),
                chips=["What do we hold?", "Which stock adds the most risk?", "How concentrated is the portfolio?"])


def h_sectors(q: Query, d: Data) -> dict:
    p = d.a.positions
    s = p.groupby("sector")["weight"].sum().sort_values(ascending=False)
    members = p.groupby("sector")["ticker"].apply(list)
    rows = [[sec, f"{ui.pct(w, 1)} · {', '.join(members[sec])}"] for sec, w in s.items()]
    top = s.index[0]
    return _ans(ui.pct(s.iloc[0], 1), f"{top}: the largest sector",
                f"{len(s)} sectors. The biggest exposure is {top} ({_join(members[top])}).", rows=rows,
                link=_tab("overview"), chips=["Which factor tilts do we have?", "What's our biggest position?"])


def h_day_pnl(q: Query, d: Data) -> dict:
    a, p = d.a, d.p
    day = f"{a.as_of:%b %d}"
    if q.tickers:
        t = q.tickers[0]
        pnl, mv = float(p.loc[t, "day_pnl"]), float(p.loc[t, "market_value"])
        return _ans(ui.usd(pnl, signed=True), f"{t} day P&L ({day})",
                    f"{_name(t)} moved {ui.pct(pnl / (mv - pnl), 2, signed=True)} on the day, "
                    f"{ui.pct(pnl / d.nav, 2, signed=True)} of NAV.", tone=ui.tone_class(pnl), link=_tab("overview"))
    pnl = a.kpis["day_pnl"]
    s = p["day_pnl"].sort_values(ascending=False)
    rows = [[t, ui.usd(v, signed=True)] for t, v in s.items()]
    return _ans(ui.usd(pnl, signed=True), f"Day P&L ({day})",
                f"{ui.pct(pnl / d.nav, 2, signed=True)} of NAV at the {_date(a.as_of)} close. Biggest gain: "
                f"{s.index[0]} {ui.usd(s.iloc[0], signed=True)}; biggest loss: {s.index[-1]} "
                f"{ui.usd(s.iloc[-1], signed=True)}.", tone=ui.tone_class(pnl), rows=rows, link=_tab("overview"),
                chips=["How have we done over the last year?", "What's our NAV?"])


def _cum(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def h_performance(q: Query, d: Data) -> dict:
    a = d.a
    pc, bc = _cum(a.port_returns), _cum(a.bench_returns)
    n = len(a.port_returns)
    if q.tickers:
        t = q.tickers[0]
        c = _cum(a.idio.returns[t])
        usd_note = " in US dollars" if t in config.LOCAL_CURRENCY_FX else ""
        return _ans(ui.pct(c, 1, signed=True), f"{t} return, last {n} trading days",
                    f"{_name(t)}'s price return{usd_note} over {_window(d)}, against {ui.pct(pc, 1, signed=True)} for "
                    f"the portfolio and {ui.pct(bc, 1, signed=True)} for the S&P 500.", tone=ui.tone_class(c),
                    link=_tab("idio"), chips=["Which stock performed best?", f"What is {t}'s alpha?"])
    diff = pc - bc
    verdict = (f"beat it by {abs(diff) * 100:.1f} percentage points" if diff > 0
               else f"trailed it by {abs(diff) * 100:.1f} percentage points")
    return _ans(ui.pct(pc, 1, signed=True), f"Portfolio return, last {n} trading days",
                f"Over {_window(d)}, holding today's weights the whole time. The S&P 500 returned "
                f"{ui.pct(bc, 1, signed=True)}, so the portfolio {verdict}. Average daily return × 252 = "
                f"{ui.pct(a.kpis['ann_return'], 1, signed=True)} a year.", tone=ui.tone_class(pc),
                link=_tab("overview"),
                chips=["Which stock performed best?", "What is our alpha?", "What's our Sharpe ratio?"])


def h_performers(q: Query, d: Data) -> dict:
    if q.tickers:
        return h_performance(q, d)
    R = d.a.idio.returns[list(d.p.index)]
    cum = ((1 + R).prod() - 1).sort_values(ascending=False)
    t = cum.index[-1] if q.worst else cum.index[0]
    o = cum.index[0] if q.worst else cum.index[-1]
    word, opp = ("Worst", "best") if q.worst else ("Best", "worst")
    return _ans(ui.pct(cum[t], 1, signed=True), f"{word} performer: {t}",
                f"{_name(t)} over {_window(d)}. The {opp} was {o} at {ui.pct(cum[o], 1, signed=True)}.",
                tone=ui.tone_class(cum[t]), rows=[[k, ui.pct(v, 1, signed=True)] for k, v in cum.items()],
                link=_tab("idio"), chips=["How have we done over the last year?", "Which stock adds the most risk?"])


def h_volatility(q: Query, d: Data) -> dict:
    k, rep = d.a.kpis, d.rep
    vol = k["ann_vol"]
    bvol = float(rep.bench.tail(config.TRADING_DAYS).std() * np.sqrt(config.TRADING_DAYS))
    if q.tickers:
        t = q.tickers[0]
        v = float(rep.budget.loc[t, "vol"])
        rows = [[x, ui.pct(float(rep.budget.loc[x, "vol"]), 1)] for x in q.tickers] if len(q.tickers) > 1 else None
        compare = "more" if v > vol else "less"
        return _ans(ui.pct(v, 1), f"{t} annualised volatility",
                    f"{_name(t)} swings about ±{ui.pct(v, 1)} in a typical year, {compare} than the whole "
                    f"portfolio ({ui.pct(vol, 1)}) and against {ui.pct(bvol, 1)} for the S&P 500. Mixing stocks that "
                    "don't move together is why the portfolio is calmer than most of its parts.", rows=rows,
                    link=_tab("risk"))
    ratio = vol / bvol if bvol else np.nan
    vs = ("calmer than the market" if ratio < 0.9 else "more volatile than the market" if ratio > 1.1
          else "about as volatile as the market")
    return _ans(ui.pct(vol, 1), "Annualised volatility (last year)",
                f"A typical year moves the portfolio about ±{ui.pct(vol, 1)} (±{ui.usd(vol * d.nav, compact=True)}); a "
                f"typical day about ±{ui.pct(vol / np.sqrt(config.TRADING_DAYS), 2)}. The S&P 500's volatility is "
                f"{ui.pct(bvol, 1)}, so the portfolio is {vs}.", link=_tab("risk"),
                chips=["What is our 1-day VaR?", "What's our Sharpe ratio?", "How diversified are we?"])


def h_sharpe(q: Query, d: Data) -> dict:
    r = d.opt
    mc, s, inp = r.metrics["current"], r.settings, r.inputs
    sh = mc["sharpe"]
    rating = ("below the risk-free rate, so the risk hasn't been paid" if sh < 0 else "weak" if sh < 0.5
              else "decent" if sh < 1 else "good" if sh < 2 else "very strong")
    tan = r.metrics.get("max_sharpe")
    best = f" The best mix of the same stocks reached {ui.num(tan['sharpe'], 2)} on the same data." if tan else ""
    return _ans(ui.num(sh, 2), "Sharpe ratio (current weights)",
                f"({ui.pct(mc['ret'], 1)} return − {ui.pct(s.rf, 1)} risk-free) ÷ {ui.pct(mc['vol'], 1)} volatility, "
                f"from {len(inp.R)} trading days ({LOOKBACK_WORDS.get(s.lookback, s.lookback)} lookback, "
                f"{inp.start:%b %Y} to {inp.end:%b %Y}): {rating}. Sortino ratio {ui.num(mc['sortino'], 2)}.{best}",
                link=RATIO_LINK, chips=["How should we rebalance for a better Sharpe?", "What is our alpha?"])


def h_beta(q: Query, d: Data) -> dict:
    a = d.a
    if q.tickers:
        t = q.tickers[0]
        b, mv = float(d.p.loc[t, "beta"]), float(d.p.loc[t, "market_value"])
        r2 = float(a.idio.names.loc[t, "r2"])
        expl = (f" But the market explains only {r2:.0%} of its daily moves, so company news matters more."
                if r2 < 0.3 else f" The market explains {r2:.0%} of its daily moves (R²).")
        rows = [[x, ui.num(float(d.p.loc[x, "beta"]), 2)] for x in q.tickers] if len(q.tickers) > 1 else None
        return _ans(ui.num(b, 2), f"{t} beta to the S&P 500",
                    _beta_meaning(b, mv, _name(t)) + expl, rows=rows, link=_tab("idio"),
                    chips=[f"What is {t}'s weight?", "What's the portfolio beta?", f"Tell me about {t}"])
    k = a.idio.kpis
    b = k["beta"]
    down = float(d.rep.updown.loc["Portfolio", "down_beta"])
    return _ans(ui.num(b, 3, signed=True), "Portfolio beta to the S&P 500",
                _beta_meaning(b, d.nav, "The portfolio") + f" The S&P 500 explains {k['r2']:.0%} of its daily moves. "
                f"On S&P 500 down days (full history) the beta is {down:.2f}; the stress tests use that. "
                f"{'Shorting' if b > 0 else 'Buying'} {ui.usd(abs(k['hedge_usd']), compact=True)} of the S&P 500 "
                "would make it market-neutral.",
                link=_tab("idio"),
                chips=["What happens if the S&P 500 falls 20%?", "Which factor tilts do we have?",
                       "What is Boeing's beta?"])


def h_var(q: Query, d: Data) -> dict:
    level = 0.99 if q.level == 0.99 else 0.95
    col = "var99" if level == 0.99 else "var95"
    vm = d.rep.var_methods[col]
    v = float(vm["Historical"])
    others = vm.drop("Historical")
    odds = "1 trading day in 100" if level == 0.99 else "1 trading day in 20"
    rows = [[m, f"{ui.usd(x * d.nav)} · {ui.pct(x)}"] for m, x in vm.items()]
    note = "VaR at 97.5% isn't on the dashboard; this is the 95% figure." if q.level == 0.975 else None
    other = "What is the 95% VaR?" if level == 0.99 else "What is the 99% VaR?"
    return _ans(ui.usd(v * d.nav), f"1-day VaR ({level:.0%})",
                f"On about {odds}, expect to lose more than {ui.usd(v * d.nav)} ({ui.pct(v)} of NAV). That's the "
                f"historical method on {_window(d)}; the other methods give {ui.usd(others.min() * d.nav)} to "
                f"{ui.usd(others.max() * d.nav)}.", tone="neg", rows=rows, note=note, link=_tab("risk"),
                chips=[other, "What is the expected shortfall?", "Does the VaR model pass its backtest?"])


def h_es(q: Query, d: Data) -> dict:
    hist = d.rep.var_methods.loc["Historical"]
    if q.level == 0.95:
        v, label, tail = d.a.kpis["cvar95_pct"], "Expected shortfall (95%)", "worst 5% of days"
    else:
        v, label, tail = float(hist["es975"]), "Expected shortfall (97.5%)", "worst 2.5% of days (about 6 a year)"
    return _ans(ui.usd(v * d.nav), label,
                f"On the {tail}, the average loss is {ui.usd(v * d.nav)} ({ui.pct(v)} of NAV). That's "
                f"{v / hist['var95']:.1f}× the 95% VaR: VaR says where the bad days start, expected shortfall how bad "
                "they get.", tone="neg", link=_tab("risk"),
                chips=["What is our 1-day VaR?", "What's our worst drawdown?", "What was the worst sell-off?"])


def h_drawdown(q: Query, d: Data) -> dict:
    r = d.a.port_returns
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth[:trough].idxmax()
    cur = float(dd.iloc[-1])
    now = "It is at a new high today." if cur > -1e-9 else f"Today it is {ui.pct(-cur, 1)} below its peak."
    m = float(d.a.kpis["max_drawdown"])
    return _ans(ui.pct(m, 1), "Maximum drawdown (last year)",
                f"The worst fall from a peak: {_date(peak)} to {_date(trough)}, about "
                f"{ui.usd(m * d.nav, signed=True, compact=True)} on today's NAV. {now}", tone="neg",
                link=_tab("overview"), chips=["What was the worst sell-off?", "What is our 1-day VaR?"])


def h_diversification(q: Query, d: Data) -> dict:
    rep = d.rep
    dr, cr = rep.div_ratio, rep.corr_regimes
    return _ans(f"{dr:.2f}×", "Diversification ratio",
                f"The stocks' stand-alone volatilities add up to {dr:.2f}× the portfolio's actual volatility: mixing them "
                f"cuts risk by {1 - 1 / dr:.0%}. Average correlation between holdings is {cr['All days']:.2f}, rising to "
                f"{cr['S&P 500 worst 10% days']:.2f} on the S&P 500's worst 10% of days.", link=_tab("risk"),
                chips=["Which stock is the best diversifier?", "Which two stocks move together most?",
                       "How concentrated is the portfolio?"])


def h_effective_n(q: Query, d: Data) -> dict:
    k = d.a.kpis
    n, eff = k["n_positions"], k["effective_n"]
    top3 = float(d.p["weight"].abs().sort_values(ascending=False).head(3).sum() / d.p["weight"].abs().sum())
    return _ans(f"{eff:.1f}", "Effective number of positions",
                f"The {n} weights are as concentrated as {eff:.1f} equal-sized positions (1 ÷ sum of squared "
                f"weights; {n} would mean equal weights). The top three hold {ui.pct(top3, 0)}.", link=_tab("overview"),
                chips=["How diversified are we?", "What's our biggest position?"])


def h_idio(q: Query, d: Data) -> dict:
    a = d.a
    if q.tickers:
        t = q.tickers[0]
        nm = a.idio.names.loc[t]
        return _ans(ui.pct(1 - nm["r2"], 1), f"{t}: stock-specific share of variance",
                    f"The S&P 500 explains {nm['r2']:.0%} of {_name(t)}'s daily moves; the other {1 - nm['r2']:.0%} "
                    f"({ui.pct(nm['idio_vol'], 1)} a year of volatility) is company-specific.", link=_tab("idio"),
                    chips=[f"What is {t}'s alpha?", f"What is {t}'s beta?"])
    k = a.idio.kpis
    share = k["idio_share"]
    meaning = ("so returns depend more on the individual stocks than on the market" if share > 0.5
               else "so the market drives most of the swings")
    return _ans(ui.pct(share, 1), "Idiosyncratic (stock-specific) share of variance",
                f"Only {ui.pct(1 - share, 0)} of the portfolio's variance comes from the S&P 500 (systematic volatility "
                f"{ui.pct(k['sys_vol'], 1)}); the rest ({ui.pct(k['idio_vol'], 1)} volatility) is stock-specific news, "
                f"{meaning}.", link=_tab("idio"), chips=["What is our alpha?", "What's the portfolio beta?"])


def h_alpha(q: Query, d: Data) -> dict:
    a = d.a
    k = a.idio.kpis
    if q.tickers:
        t = q.tickers[0]
        nm = a.idio.names.loc[t]
        al, tt, who, ir = float(nm["alpha"]), float(nm["alpha_t"]), _name(t), None
        label = f"{t} alpha vs the S&P 500"
    else:
        al, tt, who, ir = k["alpha"], k["alpha_t"], "the portfolio", k["info_ratio"]
        label = "Alpha vs the S&P 500"
    sig = abs(tt) > 1.96
    ir_text = f" Information ratio {ui.num(ir, 2, signed=True)} (alpha ÷ stock-specific volatility)." if ir is not None else ""
    return _ans(f"{ui.pct(al, 1, signed=True)}/yr", label,
                f"Return of {who} not explained by the S&P 500, annualised, from {_date(k['start'])} to "
                f"{_date(k['end'])}. t-stat {tt:+.2f}: "
                f"{'statistically significant at 5%' if sig else 'not statistically significant at 5%, so it could be luck'}."
                f"{ir_text}", tone=ui.tone_class(al), link=_tab("idio"),
                chips=["How much of our risk is stock-specific?", "How have we done over the last year?"])


def h_risk_contrib(q: Query, d: Data) -> dict:
    b = d.rep.budget
    if q.tickers:
        t = q.tickers[0]
        rs, w = float(b.loc[t, "risk_share"]), float(b.loc[t, "weight"])
        return _ans(ui.pct(rs, 1), f"{t}'s share of portfolio risk",
                    f"{_name(t)} is {ui.pct(w, 1)} of the money but {ui.pct(rs, 1)} of the volatility (weight × how much "
                    f"it moves with the rest of the book). {_risk_vs_weight(rs, w)}", link=_tab("risk"),
                    chips=[f"What is {t}'s beta?", "Which stock adds the most risk?"])
    s = b["risk_share"].sort_values(ascending=q.worst)
    t = s.index[0]
    word = "Smallest" if q.worst else "Biggest"
    top3 = list(b["risk_share"].sort_values(ascending=False).index[:3])
    rows = [[x, f"{ui.pct(b.loc[x, 'risk_share'], 1)} of risk · {ui.pct(b.loc[x, 'weight'], 1)} weight"]
            for x in b["risk_share"].sort_values(ascending=False).index]
    return _ans(ui.pct(s.iloc[0], 1), f"{word} risk contributor: {t}",
                f"{_name(t)} is {ui.pct(b.loc[t, 'weight'], 1)} of the portfolio but drives {ui.pct(s.iloc[0], 1)} of "
                f"its volatility. The top three ({_join(top3)}) drive "
                f"{ui.pct(b.loc[top3, 'risk_share'].sum(), 0)}.", rows=rows, link=_tab("risk"),
                chips=["Which stock is the best diversifier?", "How should we rebalance to lower volatility?"])


def _pairs(C: pd.DataFrame) -> pd.Series:
    i, j = np.triu_indices(len(C), 1)
    return pd.Series(C.to_numpy()[i, j], index=[(C.index[a], C.columns[b]) for a, b in zip(i, j)])


def h_correlation(q: Query, d: Data) -> dict:
    C = d.a.pca_corr
    window = f"Daily returns over {_window(d)}."
    if len(q.tickers) >= 2:
        x, y = q.tickers[:2]
        c = float(C.loc[x, y])
        return _ans(ui.num(c, 2), f"Correlation {x} / {y}", f"{x} and {y} are {_corr_meaning(c)}. {window}",
                    link=_tab("risk"), chips=["Which two stocks move together most?", "Which stock is the best diversifier?"])
    if q.tickers:
        x = q.tickers[0]
        s = C[x].drop(x).sort_values(ascending=False)
        return _ans(ui.num(s.iloc[0], 2), f"{x} moves most with {s.index[0]}",
                    f"Its least correlated holding is {s.index[-1]} ({ui.num(s.iloc[-1], 2)}); average correlation with "
                    f"the other {len(s)} is {ui.num(s.mean(), 2)}. {window}",
                    rows=[[k, ui.num(v, 2)] for k, v in s.items()], link=_tab("risk"))
    pairs = _pairs(C).sort_values(ascending=q.worst)
    (x, y), c = pairs.index[0], float(pairs.iloc[0])
    (ox, oy), oc = pairs.index[-1], float(pairs.iloc[-1])
    word, opp = ("Least", "most") if q.worst else ("Most", "least")
    return _ans(ui.num(c, 2), f"{word} correlated pair: {x} / {y}",
                f"{x} and {y} are {_corr_meaning(c)}. The {opp} correlated pair is {ox} / {oy} ({ui.num(oc, 2)}); "
                f"the average across all pairs is {ui.num(pairs.mean(), 2)}. {window}", link=_tab("risk"),
                chips=["Which stock is the best diversifier?", "How diversified are we?"])


def h_diversifier(q: Query, d: Data) -> dict:
    if q.tickers:
        return h_correlation(q, d)
    C = d.a.pca_corr
    avg = ((C.sum() - 1) / (len(C) - 1)).sort_values()
    t, worst = avg.index[0], avg.index[-1]
    b = d.rep.budget
    return _ans(t, f"Best diversifier · average correlation {ui.num(avg[t], 2)}",
                f"{_name(t)} has the lowest average correlation with the other holdings, so it does the most to smooth "
                f"the portfolio: {ui.pct(b.loc[t, 'risk_share'], 1)} of risk for a {ui.pct(b.loc[t, 'weight'], 1)} "
                f"weight. The least diversifying is {worst} ({ui.num(avg[worst], 2)}).",
                rows=[[k, ui.num(v, 2)] for k, v in avg.items()], link=_tab("risk"),
                chips=["Which two stocks move together most?", "How diversified are we?"])


def h_stress(q: Query, d: Data) -> dict:
    rep, nav = d.rep, d.nav
    st = rep.stress
    db = float(rep.updown.loc["Portfolio", "down_beta"])
    chips = ["How did we do in the 2022 bear market?", "What about the April 2025 tariff shock?",
             "What happens if the S&P 500 falls 20%?"]
    if q.shock is not None:
        port = db * q.shock
        return _ans(ui.usd(port * nav, signed=True, compact=True), f"If the S&P 500 falls {abs(q.shock):.0%}",
                    f"About {ui.pct(port, 1, signed=True)} of NAV: the fall scaled by the portfolio's downside beta of "
                    f"{db:.2f} (its slope on S&P 500 down days since {rep.port.index[0]:%Y}). An estimate, not a replay.",
                    tone="neg" if port < 0 else "pos", link=_tab("risk"), chips=chips)
    row = st[st["scenario"] == q.scenario]
    if q.scenario and len(row):
        r = row.iloc[0]
        held = ("held up better than the market" if r["port"] > r["bench"] + 0.005
                else "fell more than the market" if r["port"] < r["bench"] - 0.005 else "moved with the market")
        return _ans(ui.usd(r["port"] * nav, signed=True, compact=True), r["scenario"],
                    f"{r['period']}: replaying today's weights, the portfolio returned {ui.pct(r['port'], 1, signed=True)} "
                    f"against {ui.pct(r['bench'], 1, signed=True)} for the S&P 500, so it {held}.",
                    tone=ui.tone_class(r["port"]), link=_tab("risk"), chips=chips)
    hist = st[st["kind"] != "Hypothetical"].sort_values("port")
    w = hist.iloc[0]
    rows = [[r.scenario, f"{ui.pct(r.port, 1, signed=True)} · S&P {ui.pct(r.bench, 1, signed=True)}"]
            for r in st.itertuples()]
    return _ans(ui.usd(w["port"] * nav, signed=True, compact=True), f"Worst replayed sell-off: {w['scenario']}",
                f"{w['period']}: {ui.pct(w['port'], 1, signed=True)} for the portfolio against "
                f"{ui.pct(w['bench'], 1, signed=True)} for the S&P 500, replaying today's weights. Every scenario:",
                tone="neg", rows=rows, link=_tab("risk"), chips=chips)


def h_backtest(q: Query, d: Data) -> dict:
    kp = d.rep.kupiec
    ok = kp["p_value"] >= 0.05
    verdict = ("passes: breaches are about as rare as promised" if ok
               else "fails: breaches are more (or less) frequent than a 95% VaR promises")
    return _ans(f"{kp['breaches']} / {kp['n']}", "VaR backtest: days losses beat the 95% EWMA VaR",
                f"Expected about {kp['expected']:.1f} breaches in {kp['n']} days. Kupiec p-value {kp['p_value']:.2f}, so the "
                f"model {verdict}. Each day's VaR was forecast from earlier days only.", tone="" if ok else "neg",
                link=_tab("risk"), chips=["What is our 1-day VaR?", "What is the expected shortfall?"])


def h_factors(q: Query, d: Data) -> dict:
    a = d.a
    ft = a.factor_table
    tilt = ft.drop("Market")["net"]
    order = tilt.abs().sort_values(ascending=False).index
    top = order[0]
    lean = "leans toward" if tilt[top] > 0 else "leans away from"
    rows = [[f"{f} ({ft.loc[f, 'proxy']})", f"{ui.num(ft.loc[f, 'net'], 2, signed=True)} · "
             f"{ui.pct(ft.loc[f, 'risk_share'], 0)} of risk"] for f in ["Market", *order[:5]]]
    return _ans(ui.num(tilt[top], 2, signed=True), f"Largest factor tilt: {top} ({ft.loc[top, 'proxy']})",
                f"The portfolio {lean} {top} stocks beyond what the market explains (factors are measured net of the "
                f"market). Factors explain {ui.pct(a.kpis['factor_share'], 0)} of risk; the rest is stock-specific.",
                rows=rows, link=_tab("factors"),
                chips=["What's the portfolio beta?", "How much of our risk is stock-specific?"])


GOAL_CHIPS = {"max_sharpe": "How should we rebalance for a better Sharpe?",
              "min_vol": "How should we rebalance to lower volatility?", "risk_parity": "What does risk parity suggest?"}
OPT_METRIC = {"max_sharpe": ("sharpe", "Sharpe ratio", "num"), "min_vol": ("vol", "volatility", "pct"),
              "min_idio": ("idio_vol", "stock-specific volatility", "pct"), "max_sortino": ("sortino", "Sortino ratio", "num"),
              "risk_parity": ("vol", "volatility", "pct"), "max_div": ("div_ratio", "diversification ratio", "num")}


def h_optimizer(q: Query, d: Data) -> dict:
    goal = q.goal or "max_sharpe"
    r = d.opt
    title = optimizer.OBJECTIVES[goal][0]
    if goal not in r.weights:
        return _ans(None, None, f"The optimizer couldn't solve “{title}”: {r.errors.get(goal, 'no feasible portfolio')}.",
                    link=RATIO_LINK)
    key, metric, kind = OPT_METRIC[goal]
    fmt = (lambda x: ui.pct(x, 1)) if kind == "pct" else (lambda x: ui.num(x, 2))
    w = pd.Series(r.weights[goal], index=r.inputs.tickers)
    cur = pd.Series(r.current, index=r.inputs.tickers)
    diff = (w - cur).sort_values()
    adds = [f"{t} (+{x * 100:.0f} pts)" for t, x in diff[::-1].items() if x > 0.005][:3]
    cuts = [f"{t} (−{abs(x) * 100:.0f} pts)" for t, x in diff.items() if x < -0.005][:3]
    moves = "; ".join(filter(None, [f"add {_join(adds)}" if adds else "", f"cut {_join(cuts)}" if cuts else ""]))
    s = r.settings
    order = diff.abs().sort_values(ascending=False).index
    rows = [[t, f"{ui.pct(cur[t], 1)} → {ui.pct(w[t], 1)}"] for t in order if abs(diff[t]) > 0.005][:6]
    label = f"{title} portfolio" if metric.lower() in title.lower() else f"{title} portfolio · {metric}"
    return _ans(fmt(r.metrics[goal][key]), label,
                f"Against {fmt(r.metrics['current'][key])} for today's weights. Biggest moves: {moves or 'none'}. Fit "
                f"in-sample on {LOOKBACK_WORDS.get(s.lookback, s.lookback)} data with {s.lo:.0%}–{s.hi:.0%} per stock, "
                "so read it as a direction, not a forecast.", rows=rows, link=RATIO_LINK,
                chips=[c for g, c in GOAL_CHIPS.items() if g != goal][:2])


def _tone_word(net: float) -> str:
    from .views.research import net_class  # same thresholds as the Company Research page

    return {"good": "positive", "bad": "negative"}.get(net_class(net), "neutral")


def h_research(q: Query, d: Data) -> dict:
    syms = q.tickers + q.others
    if not syms:
        rows = []
        for t in d.p.index:
            c = research_store.company(t)
            rel = next((r for r in (c or {}).get("releases", []) if r["tone"]), None)
            if rel:
                rows.append([t, f"{rel['tone']['net']:+.2f} · {_tone_word(rel['tone']['net'])}"])
        return _ans(f"{len(rows)} of {len(d.p)}", "Holdings with earnings research",
                    "Net FinBERT tone of each holding's latest earnings release (−1 to +1). Ask about one for details.",
                    rows=rows, link={"label": "Open Company Research", "href": "/research"},
                    chips=[f"What's the earnings tone for {t}?" for t, _ in rows[:2]])
    sym = syms[0]
    if sym in config.LOCAL_CURRENCY_FX:
        return _ans(None, None, f"{_name(sym)} files with the SEC as a foreign issuer (Form 6-K), so Company Research "
                    "doesn't cover its earnings releases.", link={"label": "Open Company Research", "href": "/research"})
    c = research_store.company(sym)
    rels = [r for r in (c or {}).get("releases", []) if r["tone"]]
    if not rels:
        return _ans(None, None, f"No earnings research is stored for {sym} yet. Open it on Company Research to analyse "
                    "its latest releases (about 8 seconds the first time).", link=_research_link(sym))
    latest, tone = rels[0], rels[0]["tone"]
    net, word = tone["net"], _tone_word(tone["net"])
    change = ""
    if len(rels) > 1:
        delta = net - rels[1]["tone"]["net"]
        change = f" That's {'up' if delta >= 0 else 'down'} {abs(delta):.2f} from the previous release."
    yoy = c["financials"].get("revenue_yoy")
    rev = f" Revenue {ui.pct(yoy, 1, signed=True)} year on year." if yoy is not None else ""
    note = None
    rel = research_store.release(sym, latest["accession"])
    points = (rel or {}).get("analysis", {}).get("key_points", [])
    point = next((p for p in points if p["tone"] == word), points[0] if points else None)
    if point:
        text = point["text"] if len(point["text"]) <= 220 else point["text"][:217].rsplit(" ", 1)[0] + "…"
        note = f"Key point: “{text}”"
    held = "" if sym in d.p.index else f" ({sym} isn't in the portfolio.)"
    return _ans(f"{net:+.2f}", f"{sym} earnings tone (FinBERT)",
                f"Latest release {_date(latest['filing_date'])}: {ui.pct(tone['positive'], 0)} of the sentences FinBERT "
                f"scored are positive, {ui.pct(tone['negative'], 0)} negative, so the tone is {word}.{change}{rev}{held}",
                tone="pos" if word == "positive" else "neg" if word == "negative" else "", note=note,
                link=_research_link(sym))


def h_summary(q: Query, d: Data) -> dict:
    if not q.tickers:
        return h_research(q, d) if q.others else h_help(q, d)
    t = q.tickers[0]
    p, b, a = d.p.loc[t], d.rep.budget.loc[t], d.a
    rows = [["Weight", f"{ui.pct(p['weight'], 1)} · {ui.usd(p['market_value'])}"],
            ["Beta", ui.num(p["beta"], 2)],
            ["Share of risk", ui.pct(b["risk_share"], 1)],
            ["Volatility", ui.pct(b["vol"], 1)],
            [f"Return, last {len(a.port_returns)} days", ui.pct(_cum(a.idio.returns[t]), 1, signed=True)],
            ["Day P&L", ui.usd(p["day_pnl"], signed=True)]]
    c = research_store.company(t) if t not in config.LOCAL_CURRENCY_FX else None
    rel = next((r for r in (c or {}).get("releases", []) if r["tone"]), None)
    if rel:
        rows.append(["Earnings tone", f"{rel['tone']['net']:+.2f} · {_tone_word(rel['tone']['net'])}"])
    return _ans(ui.pct(p["weight"], 1), f"{t} · {_name(t)}",
                f"{p['sector']}. " + _beta_meaning(float(p["beta"]), float(p["market_value"]), _name(t)), rows=rows,
                link=_tab("idio"), chips=[f"How correlated is {t} with the rest?", f"What is {t}'s alpha?",
                                          f"What's the earnings tone for {t}?"])


def not_held(q: Query) -> dict:
    sym = q.unknown[0]
    return _ans(None, None, f"{sym} isn't one of our holdings, so I can't describe it. Company Research can analyse "
                "any US-listed company's earnings releases.", link=_research_link(sym), chips=["What do we hold?"])


def fallback(q: Query | None = None) -> dict:
    return _ans(None, None, "I'm not sure what you're asking. I answer questions about this portfolio's numbers, "
                "for example:", chips=DEFAULT_CHIPS)


# ---------- intents ----------

@dataclass
class Intent:
    key: str
    title: str
    handler: object
    examples: list
    keywords: str = ""


INTENTS = [
    Intent("help", "Help", h_help, [
        "what can you do", "help", "what can I ask you", "what questions can you answer", "how do I use this",
        "what do you know", "show me some example questions", "how does this chat work", "what are you for"],
        r"\bhelp\b|what can (you|i)|how do i use|how does this"),
    Intent("smalltalk", "Small talk", h_smalltalk, [
        "hi", "hello", "hey there", "thanks", "thank you", "good morning", "great, thanks", "who are you"],
        r"^(hi|hello|hey|thanks?|thank you|cheers|good (morning|afternoon|evening))\b"),
    Intent("nav", "NAV", h_nav, [
        "what is our NAV", "how much is the portfolio worth", "total portfolio value", "what's the fund worth today",
        "how much money do we have", "what is the net asset value", "portfolio size in dollars"],
        r"net asset value|worth|aum|total value|how much money"),
    Intent("holdings", "Holdings", h_holdings, [
        "what do we hold", "list our holdings", "which stocks are in the portfolio", "show me all positions",
        "what stocks do we own", "how many stocks do we have", "what's in the portfolio",
        "which companies are in the portfolio", "who do we own"],
        r"\bholdings?\b|what do we (own|hold)|(stocks|companies|names) do we|in the portfolio|positions do we|list"),
    Intent("weight", "Weight of a stock", h_weight, [
        "what is the weight of Boeing", "how much Walmart do we own", "what percent of the portfolio is TGT",
        "how big is our Chubb position", "allocation to Verizon", "how much do we have in 3M",
        "what's our exposure to Philip Morris", "position size of TPB"],
        r"\bweights?\b|allocation|position size|how much .* (own|hold|have)|exposure to"),
    Intent("largest", "Largest position", h_largest, [
        "what is our biggest position", "largest holding", "which stock has the most weight",
        "what's our smallest position", "top holding", "which position is the largest"],
        r"largest|biggest (position|holding)|smallest|top (position|holding)"),
    Intent("sectors", "Sectors", h_sectors, [
        "sector breakdown", "what sectors are we in", "how much is in consumer staples", "sector exposure",
        "which industries do we own", "sector allocation"], r"sector|industr"),
    Intent("day_pnl", "Day P&L", h_day_pnl, [
        "what's today's P&L", "how much did we make today", "did we make or lose money today",
        "daily profit and loss", "how did the portfolio do today", "what did Boeing do today",
        "how much did we lose yesterday", "how did we do on the day", "how did the market treat us today"],
        r"profit and loss|today|yesterday|daily (gain|loss|move)|on the day"),
    Intent("performance", "Return", h_performance, [
        "how have we performed", "what is our return over the last year", "did we beat the S&P 500",
        "portfolio performance", "what's the annual return", "how did Target perform this year",
        "what return did we make", "how much did we make over the last 12 months", "did we outperform the market",
        "how has the portfolio done over the year"],
        r"\breturns?\b|perform|beat(en)? the|outperform|underperform|year to date|this year|past year|last year|12 months"),
    Intent("performers", "Best / worst performer", h_performers, [
        "which stock performed best", "best performing stock", "who was the worst performer",
        "top gainer this year", "which holding lost the most", "biggest winner and loser"],
        r"(best|worst|top|bottom)\b.{0,25}(perform|stock|gain|return)|winners?|losers?"),
    Intent("volatility", "Volatility", h_volatility, [
        "what is our volatility", "how volatile is the portfolio", "annualised volatility", "how risky are we",
        "how much does the portfolio swing", "how volatile is Boeing", "standard deviation of returns",
        "how much does our value fluctuate"], r"volatil|swing|how risky|standard deviation|fluctuat"),
    Intent("sharpe", "Sharpe ratio", h_sharpe, [
        "what is our Sharpe ratio", "risk adjusted return", "Sharpe", "what's the Sortino ratio",
        "are we being paid for the risk", "return per unit of risk", "are we compensated for the risk we take"],
        r"sharpe|sortino|risk.adjusted|for the risk"),
    Intent("beta", "Beta", h_beta, [
        "what's the portfolio beta", "beta to the S&P 500", "how sensitive are we to the market",
        "what is Boeing's beta", "beta of Verizon", "how much do we move with the market", "market exposure",
        "how defensive is the portfolio", "if the market goes up 1% how much do we move"],
        r"\bbeta\b|sensitiv|move with the market|when the (s&p 500|market) moves|market exposure|defensive"),
    Intent("var", "Value at risk", h_var, [
        "what is our VaR", "1-day value at risk", "99% VaR", "how much could we lose in a day",
        "what is the 95% VaR in dollars", "daily value at risk", "worst expected daily loss"],
        r"value at risk|lose in a day|daily loss"),
    Intent("es", "Expected shortfall", h_es, [
        "what is the expected shortfall", "CVaR", "how bad are the worst days", "average loss on the worst days",
        "tail loss", "conditional value at risk", "how much do we lose when value at risk is exceeded"],
        r"expected shortfall|tail loss|conditional|worst days|beyond"),
    Intent("drawdown", "Max drawdown", h_drawdown, [
        "what's our max drawdown", "worst drawdown", "biggest drop from the peak", "how far did we fall from the high",
        "peak to trough loss", "are we below our peak"], r"drawdown|peak|from the high|trough"),
    Intent("diversification", "Diversification", h_diversification, [
        "how diversified are we", "diversification ratio", "is the portfolio diversified", "average correlation",
        "how much does diversification help", "how much risk does diversification remove",
        "how correlated are the stocks on average"],
        r"diversified|diversification|average correlation|on average|mixing"),
    Intent("effective_n", "Effective N", h_effective_n, [
        "effective number of positions", "how concentrated is the portfolio", "concentration", "effective N",
        "are we too concentrated"], r"effective|concentrat"),
    Intent("idio", "Idiosyncratic risk", h_idio, [
        "how much of our risk is stock-specific", "idiosyncratic risk", "systematic vs idiosyncratic",
        "what share of variance is idiosyncratic", "how much does the market explain", "R squared",
        "how much of Boeing's risk is company-specific", "what fraction of our variance is idiosyncratic",
        "how much of the risk is explained by the market"],
        r"idiosyncratic|stock.specific|company.specific|systematic|r squared|(from|explained by) the market"),
    Intent("alpha", "Alpha", h_alpha, [
        "what is our alpha", "are we generating alpha", "is the alpha significant", "information ratio",
        "excess return over the market", "what is Walmart's alpha"], r"\balpha\b|information ratio|excess return"),
    Intent("risk_contrib", "Risk contribution", h_risk_contrib, [
        "which stock adds the most risk", "biggest risk contributor", "what drives our risk", "riskiest position",
        "risk contribution of Target", "how much risk does Boeing add", "risk budget", "where does our risk come from"],
        r"risk contribut|share of risk|drives? (our |the )?risk|riskiest|(most|least) risk|risk budget|risk does|risk come"),
    Intent("correlation", "Correlation", h_correlation, [
        "correlation between Boeing and Walmart", "how correlated are Target and Walmart", "which two stocks move together most",
        "most correlated pair", "least correlated pair", "how correlated is Chubb with the rest", "do PM and TPB move together"],
        r"correlat|move together|co-?move"),
    Intent("diversifier", "Best diversifier", h_diversifier, [
        "which stock is the best diversifier", "what diversifies the portfolio most", "which holding hedges the others",
        "least correlated stock", "which stock reduces risk the most", "which stock has the lowest average correlation",
        "what is our best hedge"],
        r"diversifier|hedges?\b|lowest (average )?correlation|(lowers?|reduces?) (our |the )?risk the most|(with|to) the (others|rest)"),
    Intent("stress", "Stress test", h_stress, [
        "what was the worst sell-off", "stress test", "how did we do in the 2022 bear market",
        "what happens if the S&P 500 falls 20%", "tariff shock loss", "yen carry unwind", "a market crash",
        "how much would we lose in a crash", "regional bank crisis", "which historical crisis hurt us most",
        "historical scenarios"],
        r"stress|crash|sell.?off|scenario|crisis|bear market|tariff|\byen\b|banks?\b|falls? \d|drops? \d|recession|shock"),
    Intent("backtest", "VaR backtest", h_backtest, [
        "does the VaR model pass its backtest", "VaR backtest", "Kupiec test", "how many VaR breaches",
        "is the VaR model accurate", "how often do losses exceed the VaR"], r"backtest|kupiec|breach|exception|exceed"),
    Intent("factors", "Factor exposures", h_factors, [
        "what are our factor exposures", "which factor tilts do we have", "momentum exposure", "factor model",
        "are we tilted to value or growth", "style factors", "sector factor betas", "what's our value factor beta",
        "momentum factor beta"],
        r"factor|tilt|momentum|value or growth|style|low vol"),
    Intent("optimizer", "Optimizer", h_optimizer, [
        "how should we rebalance", "what weights would maximise the Sharpe ratio", "optimal portfolio",
        "how can we reduce risk", "what does risk parity suggest", "what should we buy or sell",
        "suggest better weights", "minimum volatility portfolio"],
        r"optimi|rebalanc|suggest|should we (buy|sell|add|cut|trim)|best weights|better weights|risk parity|reduce risk"),
    Intent("research", "Earnings research", h_research, [
        "what's the earnings tone for Walmart", "FinBERT sentiment", "how was Boeing's last earnings release",
        "earnings sentiment", "what did management say", "news sentiment for Target"],
        r"\btone\b|sentiment|earnings|finbert|press release|research|management"),
    Intent("summary", "Stock summary", h_summary, [
        "tell me about Boeing", "Boeing", "how is Target doing", "give me a summary of Verizon", "what about Chubb",
        "overview of 3M"], r"tell me about|summary|overview of|how is"),
]
_BY_KEY = {i.key: i for i in INTENTS}
PER_STOCK = {"weight", "day_pnl", "performance", "volatility", "beta", "idio", "alpha", "risk_contrib", "correlation",
             "research", "summary"}
REDIRECT_WITH_TICKER = {"holdings": "weight", "largest": "weight", "nav": "weight", "performers": "performance",
                        "diversifier": "correlation", "help": "summary", "smalltalk": "summary", "sectors": "summary"}
PORTFOLIO_ONLY = {"sharpe", "var", "es", "drawdown", "diversification", "effective_n", "backtest", "factors"}


def answer(question: str) -> dict:
    """The full reply for one question: the answer fields plus the intent, its similarity and the time taken."""
    started = time.perf_counter()
    question = (question or "").strip()[:300]
    if not question:
        return {**fallback(), "intent": None, "title": None, "score": None, "ms": 0}
    try:
        d = Data()
        q = parse(question, list(d.p.index))
        ranked = classify(q)
        key, score = ranked[0]
        if q.tickers and key in REDIRECT_WITH_TICKER:
            key = REDIRECT_WITH_TICKER[key]
        if score < config.ASSISTANT_THRESHOLD:
            key = "summary" if q.tickers else ("research" if q.others else None)
        if key is None:
            out = not_held(q) if q.unknown else fallback(q)
        elif q.unknown and not q.tickers and not q.others and key in PER_STOCK:
            out = not_held(q)
        else:
            if q.others and not q.tickers and key != "research":
                key = "research"
            out = _BY_KEY[key].handler(q, d)
            if q.tickers and key in PORTFOLIO_ONLY:
                out["note"] = "This is for the whole portfolio; the dashboard doesn't break it down by stock."
        out.update(intent=key, title=_BY_KEY[key].title if key else None, score=round(score, 3))
    except Exception as exc:  # never break the page over one question
        out = _ans(None, None, f"Sorry, I couldn't work that out ({type(exc).__name__}: {exc}).", chips=DEFAULT_CHIPS)
        out.update(intent="error", title=None, score=None)
    out["ms"] = round((time.perf_counter() - started) * 1000)
    return out
