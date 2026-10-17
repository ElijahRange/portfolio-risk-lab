"""Company Research page: a library of companies whose earnings calls were fetched and analysed ahead of time.

Everything shown here is read from the local research database (dashboard/transcripts.py), so the page
opens instantly. Only the "Add company" button talks to Alpha Vantage.
"""
from datetime import datetime
from urllib.parse import parse_qs

import plotly.graph_objects as go
from dash import dcc, html

from .. import components as ui
from .. import config
from .. import transcripts as tr
from ..theme import C

ROLE_COLORS = {"Executive": C["primary"], "IR": C["violet"], "Analyst": C["warn"], "Operator": C["muted"]}


def qlabel(quarter: str | None) -> str:
    return f"{quarter[4:]} {quarter[:4]}" if quarter else "–"


def tone_class(x: float | None) -> str:
    if x is None:
        return "neutral"
    return "good" if x >= config.TONE_GOOD else ("bad" if x <= config.TONE_BAD else "neutral")


def tone_text(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def _score(label: str, value: str, tone: str = "neutral"):
    return html.Div([html.Div(label, className="score-label"), html.Div(value, className=f"score-box {tone}")],
                    className="score")


def _date(d: str | None) -> str:
    if not d:
        return ""
    return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y")


# ---------- page shell ----------

def parse_search(search: str | None) -> dict:
    q = parse_qs((search or "").lstrip("?"))
    return {"symbol": (q.get("symbol") or [None])[0], "quarter": (q.get("quarter") or [None])[0]}


def layout(search: str | None = None):
    lib = tr.library()
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Company research", className="page-title"),
                html.Div("Earnings calls from Alpha Vantage, analysed once on LinuxONE and stored, so every page "
                         "opens instantly.", className="muted small"),
            ]),
        ], className="page-head"),
        dcc.Store(id="rs-selected", data=parse_search(search)),
        dcc.Store(id="rs-fetched"),
        html.Div([
            html.Div([
                dcc.Input(id="rs-search", type="text", list="rs-names", debounce=True, className="search-input",
                          placeholder="Search the library: ticker or company name"),
                html.Datalist([html.Option(value=c["symbol"], label=c["name"]) for c in lib], id="rs-names"),
                html.Button("Open", id="rs-open", className="btn"),
            ], className="search-row"),
            html.Div([
                dcc.Input(id="rs-add", type="text", debounce=True, className="num-input add-input",
                          placeholder="Ticker"),
                html.Button("Fetch calls", id="rs-add-btn", className="btn btn-primary"),
            ], className="search-row add-row", title="Download a company's latest earnings calls from Alpha "
                                                     "Vantage and analyse them"),
        ], className="research-bar"),
        dcc.Loading(html.Div(id="rs-add-status"), type="dot", color=C["primary"]),
        html.Div(id="rs-body"),
    ])


def resolve(text: str | None) -> str | None:
    """Ticker or (part of a) company name -> ticker in the library."""
    if not text:
        return None
    t = text.strip().lower()
    lib = tr.library()
    for c in lib:
        if c["symbol"].lower() == t:
            return c["symbol"]
    for c in lib:
        if t in (c["name"] or "").lower():
            return c["symbol"]
    return None


def body(selected: dict | None):
    symbol = (selected or {}).get("symbol")
    if symbol:
        return company_view(symbol, (selected or {}).get("quarter"))
    return library_view()


# ---------- library ----------

def library_view():
    lib = tr.library()
    if not lib:
        return ui.card("Library", html.Div([
            html.P("No companies yet. Type a ticker in the box on the right and press Fetch calls.", className="muted"),
            html.P(f"Each new company uses {2 + config.RESEARCH_QUARTERS} Alpha Vantage requests (company profile, "
                   f"earnings dates and the last {config.RESEARCH_QUARTERS} calls). The free plan allows 25 a day.",
                   className="muted small"),
        ]))
    rows = []
    for c in lib:
        surprise = c["eps_surprise"]
        rows.append(html.Div([
            html.Div([
                html.Div(c["name"], className="lib-name"),
                html.Div(" · ".join(x for x in (c["symbol"], c["sector"]) if x), className="muted small mono"),
            ]),
            html.Div([
                _score("CALLS", str(c["calls"])),
                _score("LATEST", qlabel(c["latest"])),
                _score("TONE", tone_text(c["tone"]), tone_class(c["tone"])),
                _score("Δ TONE", "–" if c["tone_change"] is None else f"{c['tone_change']:+.2f}",
                       "neutral" if c["tone_change"] is None else ("good" if c["tone_change"] > 0.02 else
                                                                   "bad" if c["tone_change"] < -0.02 else "neutral")),
                _score("EPS SURP.", "–" if surprise is None else f"{surprise:+.1f}%",
                       "neutral" if surprise is None else ("good" if surprise > 0 else "bad")),
            ], className="score-row"),
        ], id={"type": "rs-company", "index": c["symbol"]}, n_clicks=0, className="lib-row"))
    return ui.card("Library", html.Div(rows, className="lib-list"),
                   f"{len(lib)} companies. Click one to open its research page. Tone = Alpha Vantage's sentiment "
                   "score for management's remarks on the latest call (0 = negative, 1 = positive).")


# ---------- company ----------

def tone_trend_figure(calls: list[dict]) -> go.Figure:
    cs = [c for c in reversed(calls) if c["available"]]
    x = [qlabel(c["quarter"]) for c in cs]
    fig = go.Figure()
    for key, name, color in (("prepared", "Management, prepared", C["primary"]),
                             ("answers", "Management, Q&A answers", C["violet"]),
                             ("analysts", "Analysts' questions", C["warn"])):
        fig.add_scatter(x=x, y=[c["tone"][key] for c in cs], name=name, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=7),
                        hovertemplate="%{x}<br>%{y:.2f}<extra>" + name + "</extra>")
    fig.update_yaxes(range=[0, 1], title=dict(text="Sentiment (0–1)"))
    fig.update_layout(hovermode="x unified", margin=dict(l=56, r=16, t=36, b=36))
    return fig


def call_timeline_figure(analysis: dict) -> go.Figure:
    tl = analysis["timeline"]
    fig = go.Figure()
    qa = [t["i"] for t in tl if t["section"] == "Q&A"]
    if qa:
        fig.add_vrect(x0=min(qa) + 0.5, x1=max(qa) + 1.5, fillcolor=C["surface_2"], line_width=0, layer="below",
                      annotation_text="Q&A", annotation_position="top left",
                      annotation_font=dict(color=C["muted"], size=10))
    for role, color in ROLE_COLORS.items():
        pts = [t for t in tl if t["role"] == role and t["sentiment"] is not None]
        if not pts:
            continue
        fig.add_bar(x=[t["i"] + 1 for t in pts], y=[t["sentiment"] for t in pts], name=role, marker_color=color,
                    width=0.8,
                    customdata=[[t["speaker"], t["words"]] for t in pts],
                    hovertemplate="turn %{x} · %{customdata[0]}<br>sentiment %{y:.1f} · %{customdata[1]} words"
                                  "<extra></extra>")
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title=dict(text="Speaker turn"))
    fig.update_layout(barmode="overlay", bargap=0.15, margin=dict(l=40, r=16, t=36, b=40))
    return fig


def _chip(text: str, cls: str = ""):
    return html.Span(text, className=f"chip {cls}".strip())


def key_points_card(d: dict):
    a = d["analysis"]
    items = [html.Div([
        html.Div([_chip(k["category"]), html.Span(f"{k['speaker']} · {k['title']}", className="muted small")],
                 className="kp-meta"),
        html.Div(k["text"], className="kp-text"),
    ], className=f"kp {k['tone']}") for k in a["key_points"]]
    legend = html.Div([html.Span([html.Span(className=f"dot-s {t}"), lbl], className="legend-item")
                       for t, lbl in (("bad", "Negative"), ("neutral", "Neutral"), ("good", "Positive"))],
                      className="legend")
    return ui.card(f"Key points · {qlabel(d['quarter'])} call",
                   html.Div(items or html.P("No key sentences found.", className="muted")),
                   "The most informative management sentences (numbers, guidance, results), picked by NLP when the "
                   "call was fetched.", actions=legend, className="col-card")


def calls_card(c: dict, quarter: str):
    rows = []
    for k in c["calls"]:
        surprise = k["eps_surprise_pct"]
        sub = [html.Span(_date(k["call_date"]), className="muted small")]
        if surprise is not None:
            sub.append(_chip(f"EPS {surprise:+.1f}%", "good" if surprise > 0 else "bad"))
        if k["available"]:
            sub.append(_chip(f"tone {tone_text(k['tone']['management'])}", tone_class(k["tone"]["management"])))
        else:
            sub.append(_chip("no transcript", "muted-chip"))
        rows.append(html.Div([
            html.Div(f"{qlabel(k['quarter'])} earnings call", className="call-title"),
            html.Div(sub, className="call-sub"),
        ], id={"type": "rs-call", "index": k["quarter"]}, n_clicks=0,
            className="call-row" + (" selected" if k["quarter"] == quarter else "")
                      + ("" if k["available"] else " unavailable")))
    return ui.card("Calls", html.Div(rows, className="call-list"),
                   "Fiscal quarters. EPS surprise = reported vs analysts' estimate.", className="col-card")


def guidance_card(d: dict):
    items = [html.Div([
        html.Div(className=f"flag-dot {g['tone']}"),
        html.Div([html.Div(g["speaker"], className="muted small"), html.Div(g["text"], className="kp-text")]),
    ], className="flag") for g in d["analysis"]["guidance"]]
    return ui.card("Guidance & outlook", html.Div(items or html.P("No forward-looking numbers found.",
                                                                   className="muted")),
                   "Forward-looking statements that include a number.", className="col-card")


def questions_card(d: dict):
    rows = [[
        html.Div([html.Div(q["analyst"], className="strong"), html.Div(q["firm"], className="muted small")]),
        _chip(q["topic"]),
        html.Div(q["question"], className="wrap"),
        html.Div(", ".join(q["answered_by"]) or "–", className="small"),
        html.Span(tone_text(q["answer_tone"]), className=f"tone-num {tone_class(q['answer_tone'])}"),
    ] for q in d["analysis"]["questions"]]
    return ui.card("Analyst questions", ui.table(["Analyst", "Topic", "Question", "Answered by", "Answer tone"],
                                                 rows, numeric_cols={4}) if rows
                   else html.P("No analyst Q&A in this transcript.", className="muted"),
                   "Who asked what in the Q&A, and how upbeat management's answer was (Alpha Vantage sentiment).")


def speakers_topics(d: dict):
    a = d["analysis"]
    sp_rows = [[html.Div([html.Div(s["speaker"], className="strong"), html.Div(s["title"], className="muted small")]),
                s["role"], f"{s['words']:,}",
                html.Span(tone_text(s["tone"]), className=f"tone-num {tone_class(s['tone'])}")]
               for s in sorted(a["speakers"], key=lambda s: -s["words"])]
    t, topics = a["tone"], a.get("topics", {})
    hedge = html.Div([
        _score("HEDGING · PREPARED", "–" if t["hedge_prepared"] is None else f"{t['hedge_prepared']:.1f}"),
        _score("HEDGING · Q&A", "–" if t["hedge_answers"] is None else f"{t['hedge_answers']:.1f}"),
        _score("NET TONE WORDS", "–" if t["net_words"] is None else f"{t['net_words']:+.1f}",
               "neutral" if t["net_words"] is None else ("good" if t["net_words"] > 0 else "bad")),
    ], className="score-row left")
    topic_body = html.Div([
        html.Div("Most distinctive terms", className="detail-sub"),
        html.Div([_chip(x) for x in topics.get("top", [])], className="chips"),
        html.Div(f"New vs {qlabel(topics['compared_with'])}" if topics.get("compared_with")
                 else "New vs previous call", className="detail-sub"),
        html.Div([_chip(x, "good") for x in topics.get("new", [])] or
                 [html.Span("Needs an earlier call in the library to compare with.", className="muted small")],
                 className="chips"),
        html.Div("Language", className="detail-sub"),
        hedge,
        html.P("Hedging = words like may, could, uncertain per 1,000 management words. Net tone words = positive "
               "minus negative finance words per 1,000.", className="muted small"),
    ])
    return html.Div([
        ui.card("Speakers", ui.table(["Speaker", "Role", "Words", "Tone"], sp_rows, numeric_cols={2, 3})),
        ui.card("Topics & language", topic_body),
    ], className="grid-2")


def transcript_card(d: dict):
    tl = d["analysis"]["timeline"]
    rows, section = [], None
    for s, t in zip(d["segments"], tl):
        if t["section"] != section:
            section = t["section"]
            rows.append(html.Div(section, className="script-section"))
        rows.append(html.Div([
            html.Div([html.Div(s["speaker"], className="strong"), html.Div(s["title"], className="muted small")],
                     className="script-who"),
            html.Div(s["content"], className="script-text"),
            html.Span(tone_text(s["sentiment"]), className=f"tone-num {tone_class(s['sentiment'])}"),
        ], className="script-row"))
    words = d["analysis"]["stats"]["words"]
    return ui.card("Transcript", html.Details([
        html.Summary(f"Show the full call ({words:,} words, {len(tl)} speaker turns)"),
        html.Div(rows, className="script"),
    ]), "Source: Alpha Vantage earnings call transcript, with its sentiment score for each speaker turn.")


def company_view(symbol: str, quarter: str | None):
    c = tr.company(symbol)
    back = html.Button("← Library", id={"type": "rs-back", "index": 0}, n_clicks=0, className="link-btn")
    if not c:
        return html.Div([back, ui.alert(f"{symbol} isn't in the library yet",
                                        "Use Fetch calls to download its earnings calls.", "info")])
    available = [k for k in c["calls"] if k["available"]]
    if quarter not in {k["quarter"] for k in available}:
        quarter = available[0]["quarter"] if available else None
    d = tr.call(symbol, quarter) if quarter else None
    sel = next((k for k in c["calls"] if k["quarter"] == quarter), None)

    t = d["analysis"]["tone"] if d else {}
    surprise = sel["eps_surprise_pct"] if sel else None
    header = html.Div(ui._keep(
        back,
        html.Div([
            html.Div([
                html.H2(c["name"], className="company-name"),
                html.Div(" · ".join(x for x in (symbol, c["exchange"], c["sector"], c["industry"]) if x),
                         className="muted small mono"),
            ]),
            html.Div([
                _score("MGMT TONE", tone_text(t.get("management")), tone_class(t.get("management"))),
                _score("Q&A TONE", tone_text(t.get("answers")), tone_class(t.get("answers"))),
                _score("ANALYSTS", tone_text(t.get("analysts")), tone_class(t.get("analysts"))),
                _score("EPS SURP.", "–" if surprise is None else f"{surprise:+.1f}%",
                       "neutral" if surprise is None else ("good" if surprise > 0 else "bad")),
            ], className="score-row"),
        ], className="company-head"),
        html.Details([html.Summary("About the company"), html.P(c["description"] or "–", className="small")],
                     className="about") if c["description"] else None,
    ), className="card company-card")

    if not d:
        return html.Div([header, ui.alert("No transcripts stored for this company",
                                          "Alpha Vantage had no call transcripts for the quarters we tried.", "info")])

    calls_tab = html.Div([
        html.Div([key_points_card(d), calls_card(c, quarter), guidance_card(d)], className="grid-3"),
        html.Div([
            ui.card("Tone across quarters", ui.graph(tone_trend_figure(c["calls"]), 300),
                    "Alpha Vantage sentiment, weighted by words, for each part of every stored call."),
            ui.card(f"Tone through the {qlabel(quarter)} call", ui.graph(call_timeline_figure(d["analysis"]), 300),
                    "Each bar is one speaker turn, in order. The shaded part is the Q&A."),
        ], className="grid-2"),
        questions_card(d),
        speakers_topics(d),
        transcript_card(d),
    ])
    return html.Div([
        header,
        dcc.Tabs(value="calls", className="tabs", parent_className="tabs-parent", children=[
            dcc.Tab(label="Earnings Calls", value="calls", className="tab", selected_className="tab--active",
                    children=html.Div(calls_tab, className="tab-content")),
        ]),
    ])


def fetch_status(report: dict):
    sym = report["symbol"]
    parts = []
    if report["fetched"]:
        parts.append(f"fetched {', '.join(qlabel(q) for q in report['fetched'])}")
    if report["skipped"]:
        parts.append(f"already had {', '.join(qlabel(q) for q in report['skipped'])}")
    if report["missing"]:
        parts.append(f"no transcript for {', '.join(qlabel(q) for q in report['missing'])}")
    summary = f"{sym}: " + ("; ".join(parts) if parts else "nothing new") + "."
    if report["error"]:
        return ui.alert(f"Alpha Vantage stopped the fetch for {sym}", f"{report['error']} {summary}", "warn")
    return ui.alert(f"{sym} is ready", summary, "info")
