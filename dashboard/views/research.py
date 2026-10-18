"""Company Research page: any US-listed company's earnings releases from SEC EDGAR, scored by FinBERT.

Companies already looked up open instantly from the local database (dashboard/research_store.py).
A new company takes a few seconds: its newest release is analysed at once, older ones in the background.
"""
from datetime import datetime
from urllib.parse import parse_qs

import plotly.graph_objects as go
from dash import dcc, html

from .. import components as ui
from .. import research_store as rs
from ..theme import C

TONE_CLASS = {"positive": "good", "negative": "bad", "neutral": "neutral"}


def net_class(x: float | None) -> str:
    if x is None:
        return "neutral"
    return "good" if x >= 0.15 else ("bad" if x <= -0.05 else "neutral")


def net_text(x: float | None) -> str:
    return "–" if x is None else f"{x:+.2f}"


def _date(d: str | None) -> str:
    return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y") if d else ""


def _score(label: str, value: str, tone: str = "neutral"):
    return html.Div([html.Div(label, className="score-label"), html.Div(value, className=f"score-box {tone}")],
                    className="score")


def _chip(text: str, cls: str = ""):
    return html.Span(text, className=f"chip {cls}".strip())


# ---------- page shell ----------

def parse_search(search: str | None) -> dict:
    q = parse_qs((search or "").lstrip("?"))
    return {"symbol": (q.get("symbol") or [None])[0], "release": None}


def layout(search: str | None = None):
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Company research", className="page-title"),
                html.Div("Earnings press releases from SEC EDGAR for any US-listed company, scored by FinBERT "
                         "(Hugging Face) running on LinuxONE. Companies you've opened before load instantly.",
                         className="muted small"),
            ]),
        ], className="page-head"),
        dcc.Store(id="rs-selected", data=parse_search(search)),
        dcc.Store(id="rs-fetched"),
        dcc.Interval(id="rs-poll", interval=3000, disabled=True),
        html.Div([
            dcc.Input(id="rs-search", type="text", debounce=True, className="search-input",
                      placeholder="Any US stock: ticker or company name (e.g. TPB, Walmart)"),
            html.Button("Research", id="rs-go", className="btn btn-primary"),
        ], className="search-row research-bar"),
        dcc.Loading(html.Div(id="rs-status"), type="dot", color=C["primary"]),
        html.Div(id="rs-body"),
    ])


def body(selected: dict | None):
    symbol = (selected or {}).get("symbol")
    return company_view(symbol, (selected or {}).get("release")) if symbol else library_view()


# ---------- library ----------

def library_view():
    lib = rs.library()
    if not lib:
        return ui.card("Library", html.P("Nothing looked up yet. Search any US stock above.", className="muted"))
    rows = [html.Div([
        html.Div([
            html.Div(c["name"], className="lib-name"),
            html.Div(" · ".join(x for x in (c["symbol"], c["exchange"], c["sic"]) if x), className="muted small mono"),
        ]),
        html.Div([
            _score("RELEASES", str(c["releases"])),
            _score("LATEST", _date(c["latest"]) or "–"),
            _score("TONE", net_text(c["tone"]), net_class(c["tone"])),
            _score("Δ TONE", net_text(c["tone_change"]),
                   "neutral" if c["tone_change"] is None else ("good" if c["tone_change"] > 0.05 else
                                                               "bad" if c["tone_change"] < -0.05 else "neutral")),
            _score("REV YOY", "–" if c["revenue_yoy"] is None else ui.pct(c["revenue_yoy"], 1, signed=True),
                   "neutral" if c["revenue_yoy"] is None else ("good" if c["revenue_yoy"] > 0 else "bad")),
        ], className="score-row"),
    ], id={"type": "rs-company", "index": c["symbol"]}, n_clicks=0, className="lib-row") for c in lib]
    return ui.card("Library", html.Div(rows, className="lib-list"),
                   f"{len(lib)} companies looked up so far. Tone = FinBERT's average (positive minus negative) "
                   "over the latest release's key sentences, from −1 to +1.")


# ---------- charts ----------

def tone_trend_figure(releases: list[dict]) -> go.Figure:
    done = [r for r in reversed(releases) if r["tone"]]
    x = [_date(r["filing_date"]) for r in done]
    fig = go.Figure()
    fig.add_bar(x=x, y=[r["tone"]["positive"] for r in done], name="Positive sentences", marker_color=C["pos"],
                opacity=0.55, hovertemplate="%{x}<br>%{y:.0%} positive<extra></extra>")
    fig.add_bar(x=x, y=[-r["tone"]["negative"] for r in done], name="Negative sentences", marker_color=C["neg"],
                opacity=0.55, customdata=[r["tone"]["negative"] for r in done],
                hovertemplate="%{x}<br>%{customdata:.0%} negative<extra></extra>")
    fig.add_scatter(x=x, y=[r["tone"]["net"] for r in done], name="Net tone", mode="lines+markers",
                    line=dict(color=C["text"], width=2), marker=dict(size=8),
                    hovertemplate="%{x}<br>net %{y:+.2f}<extra></extra>")
    fig.update_yaxes(range=[-1, 1], tickformat="+.1f", zeroline=True, zerolinecolor=C["muted"])
    fig.update_layout(barmode="relative", margin=dict(l=48, r=16, t=36, b=36), hovermode="x unified")
    return fig


def financials_figure(fin: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=fin["quarters"], y=fin["revenue"], name="Revenue", marker_color=C["primary"],
                hovertemplate="%{x}<br>revenue $%{y:,.0f}<extra></extra>")
    if any(v is not None for v in fin.get("eps", [])):
        fig.add_scatter(x=fin["quarters"], y=fin["eps"], name="Diluted EPS", yaxis="y2", mode="lines+markers",
                        line=dict(color=C["warn"], width=2),
                        hovertemplate="%{x}<br>EPS $%{y:.2f}<extra></extra>")
    fig.update_layout(margin=dict(l=56, r=48, t=36, b=36), yaxis=dict(tickprefix="$", tickformat=".2s"),
                      yaxis2=dict(overlaying="y", side="right", tickprefix="$", showgrid=False,
                                  tickfont=dict(color=C["muted"], size=11)))
    return fig


def tone_by_topic_figure(analysis: dict) -> go.Figure:
    cats = sorted(analysis["tone"]["by_category"].items(), key=lambda kv: kv[1]["net"])
    fig = go.Figure(go.Bar(
        x=[v["net"] for _, v in cats], y=[k for k, _ in cats], orientation="h",
        marker_color=[C["pos"] if v["net"] >= 0 else C["neg"] for _, v in cats],
        customdata=[v["n"] for _, v in cats],
        hovertemplate="%{y}<br>net tone %{x:+.2f} · %{customdata} sentences<extra></extra>"))
    fig.update_xaxes(range=[-1, 1], tickformat="+.1f", zeroline=True, zerolinecolor=C["muted"])
    fig.update_layout(margin=dict(l=110, r=16, t=8, b=32))
    return fig


# ---------- cards ----------

def key_points_card(r: dict):
    items = [html.Div([
        html.Div([_chip(k["category"]), html.Span(f"FinBERT: {k['tone']} {k['confidence']:.0%}",
                                                  className="muted small")], className="kp-meta"),
        html.Div(k["text"], className="kp-text"),
    ], className=f"kp {TONE_CLASS[k['tone']]}") for k in r["analysis"]["key_points"]]
    legend = html.Div([html.Span([html.Span(className=f"dot-s {t}"), lbl], className="legend-item")
                       for t, lbl in (("bad", "Negative"), ("neutral", "Neutral"), ("good", "Positive"))],
                      className="legend")
    return ui.card(f"Key points · {_date(r['filing_date'])} release",
                   html.Div(items or html.P("No key sentences found.", className="muted")),
                   "The most informative sentences (numbers, guidance, results, what's new), with FinBERT's tone.",
                   actions=legend, className="col-card")


def releases_card(c: dict, selected: str):
    rows = []
    for k in c["releases"]:
        sub = [html.Span(f"Quarter ended {k['period']}" if k["period"] else "Earnings release",
                         className="muted small")]
        if k["status"] == "done" and k["tone"]:
            sub.append(_chip(f"tone {net_text(k['tone']['net'])}", net_class(k["tone"]["net"])))
        elif k["status"] == "pending":
            sub.append(_chip("analysing…", "muted-chip"))
        else:
            sub.append(_chip("couldn't read", "bad"))
        rows.append(html.Div([
            html.Div(f"{_date(k['filing_date'])} release", className="call-title"),
            html.Div(sub, className="call-sub"),
        ], id={"type": "rs-release", "index": k["accession"]}, n_clicks=0,
            className="call-row" + (" selected" if k["accession"] == selected else "")
                      + ("" if k["status"] == "done" else " unavailable")))
    return ui.card("Earnings releases", html.Div(rows, className="call-list"),
                   "8-K filings (Item 2.02) with the press release, newest first.", className="col-card")


def guidance_card(r: dict):
    items = [html.Div([
        html.Div(className=f"flag-dot {TONE_CLASS[g['tone']]}"),
        html.Div(g["text"], className="kp-text"),
    ], className="flag") for g in r["analysis"]["guidance"]]
    return ui.card("Guidance & outlook", html.Div(items or html.P(
        "No forward-looking numbers in this release's text (some companies put guidance in tables or on the call).",
        className="muted")), "Forward-looking statements that include a number.", className="col-card")


def quotes_card(r: dict):
    items = [html.Div([
        html.Div(className=f"flag-dot {TONE_CLASS[q['tone']]}"),
        html.Div([
            html.Div(f"“{q['text']}”", className="kp-text"),
            html.Div(" · ".join(x for x in (q["speaker"], q["title"]) if x) or "Management", className="muted small"),
        ]),
    ], className="flag") for q in r["analysis"]["quotes"]]
    return ui.card("Management commentary", html.Div(items or html.P("No quotes in this release.", className="muted")),
                   "Executive quotes from the release, coloured by FinBERT's tone.")


def topics_card(r: dict, c: dict):
    a = r["analysis"]
    topics = a.get("topics", {})
    prev = next((x for x in c["releases"] if x["accession"] == topics.get("compared_with")), None)
    hedge = a["stats"]["hedging"]
    return ui.card("Topics & language", html.Div([
        html.Div("Most distinctive terms", className="detail-sub"),
        html.Div([_chip(x) for x in topics.get("top", [])], className="chips"),
        html.Div(f"New vs the {_date(prev['filing_date'])} release" if prev else "New vs previous release",
                 className="detail-sub"),
        html.Div([_chip(x, "good") for x in topics.get("new", [])] or
                 [html.Span("Needs an earlier release to compare with.", className="muted small")], className="chips"),
        html.Div("Language", className="detail-sub"),
        html.Div([
            _score("HEDGING /1K WORDS", "–" if hedge is None else f"{hedge:.1f}"),
            _score("SENTENCES SCORED", str(a["stats"]["scored"])),
            _score("WORDS", f"{a['stats']['words']:,}"),
        ], className="score-row left"),
    ]))


def company_view(symbol: str, accession: str | None):
    c = rs.company(symbol)
    back = html.Button("← Library", id={"type": "rs-back", "index": 0}, n_clicks=0, className="link-btn")
    if not c:
        return html.Div([back, ui.alert(f"{symbol} hasn't been looked up yet", "Search for it above.", "info")])
    done = [k for k in c["releases"] if k["status"] == "done"]
    if accession not in {k["accession"] for k in done}:
        accession = done[0]["accession"] if done else None
    r = rs.release(symbol, accession) if accession else None
    fin = c["financials"]
    tone = r["analysis"]["tone"] if r else {}
    eps = next((v for v in reversed(fin.get("eps", [])) if v is not None), None)
    edgar_link = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={c['cik']}&type=8-K"
    header = html.Div([
        back,
        html.Div([
            html.Div([
                html.H2(c["name"], className="company-name"),
                html.Div(" · ".join(x for x in (symbol, c["exchange"], c["sic"],
                                                f"FY ends {c['fy_end']}" if c["fy_end"] else "") if x),
                         className="muted small mono"),
                html.A("EDGAR filings ↗", href=edgar_link, target="_blank", className="small ext-link"),
            ]),
            html.Div([
                _score("TONE", net_text(tone.get("net")), net_class(tone.get("net"))),
                _score("POSITIVE", "–" if tone.get("positive") is None else ui.pct(tone["positive"], 0)),
                _score("NEGATIVE", "–" if tone.get("negative") is None else ui.pct(tone["negative"], 0)),
                _score("REV YOY", "–" if fin.get("revenue_yoy") is None else ui.pct(fin["revenue_yoy"], 1, signed=True),
                       "neutral" if fin.get("revenue_yoy") is None else ("good" if fin["revenue_yoy"] > 0 else "bad")),
                _score("LATEST EPS", "–" if eps is None else f"${eps:.2f}"),
            ], className="score-row"),
        ], className="company-head"),
    ], className="card company-card")

    pending = sum(k["status"] == "pending" for k in c["releases"])
    notes = [ui.alert(f"Analysing {pending} earlier release{'s' if pending > 1 else ''} in the background",
                      "They appear here as they finish.", "info")] if pending else []
    if not r:
        errors = [k["error"] for k in c["releases"] if k["error"]]
        return html.Div([header, *notes, ui.alert("No release could be analysed yet",
                                                  "; ".join(errors[:2]) or None, "warn")])

    releases_tab = html.Div([
        html.Div([key_points_card(r), releases_card(c, accession), guidance_card(r)], className="grid-3"),
        html.Div([
            ui.card("Tone across releases", ui.graph(tone_trend_figure(c["releases"]), 300),
                    "Share of positive and negative key sentences per release (FinBERT), and the net tone."),
            ui.card("Revenue & EPS", ui.graph(financials_figure(fin), 300) if fin.get("quarters") else
                    html.P("No US-GAAP numbers in SEC's XBRL data for this company.", className="muted"),
                    "Calendar quarters from SEC XBRL filings (Q4 = full year minus Q1–Q3)."),
        ], className="grid-2"),
        html.Div([
            quotes_card(r),
            ui.card("Tone by topic", ui.graph(tone_by_topic_figure(r["analysis"]), 300),
                    "FinBERT net tone of the scored sentences in each topic."),
        ], className="grid-2"),
        topics_card(r, c),
        ui.card("Press release text", html.Details([
            html.Summary(f"Show the release text ({r['analysis']['stats']['words']:,} words)"),
            html.Div([html.P(p, className="small") for p in (r["narrative"] or "").split("\n\n")], className="script"),
        ]), "Narrative part of the press release (financial tables removed).",
            actions=html.A("Open on sec.gov ↗", href=r["url"], target="_blank", className="small ext-link")
            if r["url"] else None),
    ])
    return html.Div([
        header, *notes,
        dcc.Tabs(value="releases", className="tabs", parent_className="tabs-parent", children=[
            dcc.Tab(label="Earnings Releases", value="releases", className="tab", selected_className="tab--active",
                    children=html.Div(releases_tab, className="tab-content")),
        ]),
    ])
