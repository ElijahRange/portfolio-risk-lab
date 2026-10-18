"""Portfolio Risk Lab: Dash entry point. Gunicorn serves `app:server`."""
import os
import platform
import threading
import time
from urllib.parse import parse_qs

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from dashboard import components as ui
from dashboard import assistant, config, finbert, research_store, service, theme  # noqa: F401  (theme registers the Plotly template)
from dashboard.views import chat, factors, idio, notes, overview, ratio_testing, research, risk

app = Dash(__name__, title=config.APP_TITLE, update_title=None, suppress_callback_exceptions=True)
server = app.server

TABS = {
    "overview": ("Portfolio View", overview),
    "factors": ("Factor Model", factors),
    "idio": ("Idiosyncratic Risk", idio),
    "risk": ("Risk Analysis", risk),
}
PAGES = {"/": "Portfolio", "/ratio-testing": "Ratio Testing", "/research": "Company Research",
         "/notes": "Model notes"}
NAV_IDS = {path: f"nav-{label.lower().replace(' ', '-')}" for path, label in PAGES.items()}


def nav_link(label, href):
    return dcc.Link(html.Span(label), href=href, className="nav-link", id=f"nav-{label.lower().replace(' ', '-')}")


sidebar = html.Aside([
    html.Div([html.Div(config.APP_TITLE, className="brand-name"), html.Div("IBM Z Datathon", className="brand-sub")],
             className="brand"),
    html.Nav([
        html.Div("Workspace", className="nav-group"),
        nav_link("Portfolio", "/"),
        nav_link("Ratio Testing", "/ratio-testing"),
        html.Div("Research", className="nav-group"),
        nav_link("Company Research", "/research"),
        html.Div("Reference", className="nav-group"),
        nav_link("Model notes", "/notes"),
    ]),
    html.Div([html.Span(className="dot"), f"LinuxONE · {platform.machine()}"], className="sidebar-foot"),
], className="sidebar")


def portfolio_page(tab: str = "overview"):
    return html.Div([
        html.Div([
            html.Div([html.H2("Portfolio", className="page-title"), html.Div(id="source-line", className="muted small")]),
            html.Button("Refresh prices", id="refresh", className="btn"),
        ], className="page-head"),
        html.Div(id="alerts"),
        dcc.Tabs(
            id="tabs", value=tab if tab in TABS else "overview", className="tabs", parent_className="tabs-parent",
            children=[dcc.Tab(label=label, value=key, className="tab", selected_className="tab--active")
                      for key, (label, _) in TABS.items()],
        ),
        dcc.Loading(html.Div(id="tab-content", className="tab-content"), type="dot", color=theme.C["primary"]),
    ])


app.layout = html.Div([
    html.Div([
        dcc.Location(id="url"),
        dcc.Store(id="refreshed-at"),
        sidebar,
        html.Div([
            html.Header([html.H1(id="header-title", className="header-title")], className="topbar"),
            html.Main(id="page", className="page"),
        ], className="main"),
    ], className="shell"),
    chat.layout(),  # outside the routed page, so the assistant stays open across pages and tabs
])


@app.callback(Output("page", "children"), Output("header-title", "children"),
              *[Output(nav_id, "className") for nav_id in NAV_IDS.values()],
              Input("url", "pathname"), Input("url", "search"))
def route(pathname, search):
    pathname = pathname if pathname in PAGES else "/"
    active = ["nav-link active" if p == pathname else "nav-link" for p in PAGES]
    if pathname == "/notes":
        page = notes.layout()
    elif pathname == "/research":
        page = research.layout(search)
    elif pathname == "/ratio-testing":
        try:
            page = ratio_testing.layout(service.current_analysis())
        except Exception as exc:
            page = ui.alert("Could not load the portfolio", f"{type(exc).__name__}: {exc}", "error")
    else:  # /?tab=risk opens a portfolio tab directly (the assistant's links use this)
        page = portfolio_page(parse_qs((search or "").lstrip("?")).get("tab", ["overview"])[0])
    return page, PAGES[pathname], *active


@app.callback(Output("refreshed-at", "data"), Input("refresh", "n_clicks"), prevent_initial_call=True)
def refresh(_):
    try:
        service.current_analysis(force_refresh=True)
    except Exception:
        pass  # the tab callback shows the error
    return time.time()


@app.callback(Output("tab-content", "children"), Output("source-line", "children"), Output("alerts", "children"),
              Input("tabs", "value"), Input("refreshed-at", "data"))
def render_tab(tab, _):
    try:
        a = service.current_analysis()
    except Exception as exc:  # show the problem instead of a blank page
        return None, "", ui.alert("Could not load the portfolio", f"{type(exc).__name__}: {exc}", "error")
    source = f"{a.source} · prices as of {a.as_of:%b %d, %Y}"
    alerts = [ui.alert(w) for w in a.warnings]
    if a.source.startswith("sample_"):
        alerts.insert(0, ui.alert("Showing the sample portfolio",
                                  "Add the team's spreadsheet as data/private/portfolio.xlsx to see real holdings.",
                                  "info"))
    return TABS[tab][1].layout(a), source, alerts


@app.callback(Output("positions-table", "children"), Input("positions-side", "value"), prevent_initial_call=True)
def filter_positions(side):
    try:
        return overview.positions_table(service.current_analysis(), side)
    except Exception:
        return no_update


@app.callback(Output("geo-graph", "figure"), Input("geo-color", "value"), Input("geo-layers", "value"),
              prevent_initial_call=True)
def update_geometry(color_by, layers):
    try:
        return risk.geometry_figure(service.current_analysis(), color_by, layers or [])
    except Exception:
        return no_update


@app.callback(Output("idio-detail", "children"), Output({"type": "idio-row", "index": ALL}, "className"),
              Input({"type": "idio-row", "index": ALL}, "n_clicks"), State({"type": "idio-row", "index": ALL}, "id"),
              prevent_initial_call=True)
def select_idio_name(clicks, ids):
    if not ctx.triggered_id or not any(clicks):
        return no_update, no_update
    ticker = ctx.triggered_id["index"]
    try:
        detail = idio.detail(service.current_analysis(), ticker)
    except Exception:
        return no_update, no_update
    return detail, [idio._row_class(i["index"], ticker) for i in ids]


# ---------- Ratio Testing page ----------

RT_CONTROLS = [State("rt-lookback", "value"), State("rt-returns", "value"), State("rt-lo", "value"),
               State("rt-hi", "value"), State("rt-rf", "value"), State("rt-beta-on", "value"), State("rt-beta", "value")]


@app.callback(Output("rt-content", "children"),
              Input("rt-tabs", "value"), Input("rt-objective", "value"),
              *[Input(c.component_id, c.component_property) for c in RT_CONTROLS],
              State("rt-custom-store", "data"))
def render_ratio_testing(tab, objective, *args):
    *controls, custom = args
    try:
        return ratio_testing.content(service.current_analysis(), tab, objective,
                                     ratio_testing.settings_from(*controls), custom)
    except Exception as exc:
        return ui.alert("Could not run the optimizer", f"{type(exc).__name__}: {exc}", "error")


@app.callback(Output("rt-frontier-graph", "figure"), Output("rt-custom-summary", "children"),
              Output("rt-point-detail", "children"), Output("rt-custom-store", "data"),
              Input({"type": "rt-w", "index": ALL}, "value"), Input("rt-frontier-graph", "clickData"),
              State("rt-objective", "value"), *RT_CONTROLS, prevent_initial_call=True)
def frontier_interact(values, click, objective, *controls):
    try:
        a = service.current_analysis()
        r = ratio_testing.run(a, ratio_testing.settings_from(*controls))
    except Exception:
        return no_update, no_update, no_update, no_update
    custom = ratio_testing._custom_weights(r, values)
    if ctx.triggered_id == "rt-frontier-graph":
        if not click or not click.get("points") or "customdata" not in click["points"][0]:
            return no_update, no_update, no_update, no_update
        return no_update, no_update, ratio_testing.point_detail(a, r, click["points"][0]["customdata"], custom), no_update
    return (ratio_testing.frontier_figure(r, objective, custom), ratio_testing.custom_summary(r, values),
            no_update, values)


@app.callback(Output({"type": "rt-w", "index": ALL}, "value"), Input("rt-reset", "n_clicks"),
              State({"type": "rt-w", "index": ALL}, "id"), prevent_initial_call=True)
def reset_custom_weights(_, ids):
    try:
        w = service.current_analysis().positions.set_index("ticker")["weight"]
    except Exception:
        return [no_update] * len(ids)
    w = w / w.sum()
    return [round(float(w.get(i["index"], 0.0)) * 100, 1) for i in ids]


# ---------- Company Research page ----------

@app.callback(Output("rs-status", "children"), Output("rs-fetched", "data"),
              Input("rs-go", "n_clicks"), Input("rs-search", "n_submit"), State("rs-search", "value"),
              prevent_initial_call=True)
def research_search(_clicks, _submit, query):
    if not query or not query.strip():
        return ui.alert("Type a ticker or company name", tone="info"), no_update
    started = time.time()
    try:
        symbol = research_store.ensure(query)
    except research_store.edgar.EdgarError as exc:
        return ui.alert("Couldn't research that company", str(exc), "warn"), no_update
    except Exception as exc:
        return ui.alert("Research failed", f"{type(exc).__name__}: {exc}", "error"), no_update
    took = time.time() - started
    note = None if took < 1 else ui.alert(f"{symbol} analysed in {took:.0f} s", "Saved: it opens instantly next time.",
                                          "info")
    return note, {"symbol": symbol, "t": time.time()}


@app.callback(Output("rs-selected", "data"),
              Input("rs-fetched", "data"),
              Input({"type": "rs-company", "index": ALL}, "n_clicks"),
              Input({"type": "rs-release", "index": ALL}, "n_clicks"),
              Input({"type": "rs-back", "index": ALL}, "n_clicks"),
              Input("rs-poll", "n_intervals"),
              State("rs-selected", "data"), prevent_initial_call=True)
def research_select(fetched, _companies, _releases, _back, _poll, selected):
    trig = ctx.triggered_id
    if not ctx.triggered or not ctx.triggered[0]["value"]:
        return no_update  # a list re-render, not a click
    selected = selected or {}
    if trig == "rs-fetched":
        return {"symbol": fetched["symbol"], "release": None}
    if trig == "rs-poll":  # background analysis finished another release: re-render
        symbol = selected.get("symbol")
        pending = research_store.pending_count(symbol) if symbol else 0
        return {**selected, "pending": pending} if pending != selected.get("pending") else no_update
    if isinstance(trig, dict) and trig["type"] == "rs-company":
        return {"symbol": trig["index"], "release": None}
    if isinstance(trig, dict) and trig["type"] == "rs-release":
        return {**selected, "release": trig["index"]}
    if isinstance(trig, dict) and trig["type"] == "rs-back":
        return {"symbol": None, "release": None}
    return no_update


@app.callback(Output("rs-body", "children"), Output("rs-poll", "disabled"), Input("rs-selected", "data"))
def research_body(selected):
    symbol = (selected or {}).get("symbol")
    try:
        if symbol and not research_store.company(symbol):
            symbol = research_store.ensure(symbol)  # deep link (/research?symbol=X) to a new company
            selected = {"symbol": symbol, "release": None}
        return research.body(selected), not (symbol and research_store.pending_count(symbol))
    except Exception as exc:
        return ui.alert("Could not open the research page", f"{type(exc).__name__}: {exc}", "error"), True


# ---------- Portfolio assistant (chat bubble on every page) ----------



def _warm_up_models():
    assistant.warm_up()  # sentence-embedding model + example questions + data caches
    finbert.warm_up()


# Load both Hugging Face models while the worker starts, so neither the first chat question nor the first research
# search waits. One thread on purpose: importing transformers from two threads at once fails.
threading.Thread(target=_warm_up_models, daemon=True).start()


@app.callback(Output("chat-log", "data"), Output("chat-input", "value"),
              Input("chat-send", "n_clicks"), Input("chat-input", "n_submit"),
              Input({"type": "chat-chip", "index": ALL}, "n_clicks"), Input("chat-clear", "n_clicks"),
              State({"type": "chat-chip", "index": ALL}, "children"), State("chat-input", "value"),
              State("chat-log", "data"), prevent_initial_call=True)
def chat_ask(_send, _submit, _chips, _clear, chip_texts, text, log):
    trig = ctx.triggered_id
    if trig == "chat-clear":
        return [], ""
    if isinstance(trig, dict):
        if not ctx.triggered[0]["value"]:
            return no_update, no_update  # the chips were re-rendered, not clicked
        text, new_input = chip_texts[trig["index"]], no_update  # a chip: leave anything typed in the box
    elif text and text.strip():
        new_input = ""
    else:
        return no_update, no_update
    reply = assistant.answer(text)  # never raises: errors come back as a polite answer
    log = (log or []) + [{"role": "user", "text": text.strip()[:300]}, {"role": "bot", **reply}]
    return log[-config.ASSISTANT_HISTORY:], new_input


@app.callback(Output("chat-messages", "children"), Output("chat-chips", "children"), Input("chat-log", "data"))
def chat_render(log):
    return chat.render(log)


app.clientside_callback(
    "function(_open, _close, isOpen) { return !isOpen; }",
    Output("chat-open", "data"), Input("chat-launcher", "n_clicks"), Input("chat-close", "n_clicks"),
    State("chat-open", "data"), prevent_initial_call=True)

app.clientside_callback(
    """function(isOpen) {
        if (isOpen && window.innerWidth > 760) {
            setTimeout(function () { var el = document.getElementById('chat-input'); if (el) { el.focus(); } }, 80);
        }
        var open = isOpen ? ' open' : '';
        return ['chat-panel' + open, 'chat-launcher' + open];
    }""",
    Output("chat-panel", "className"), Output("chat-launcher", "className"), Input("chat-open", "data"))

# Scroll so the newest question sits at the top of the list, with its answer right below it.
app.clientside_callback(
    """function() {
        setTimeout(function () {
            var box = document.getElementById('chat-messages');
            if (!box) { return; }
            var asked = box.querySelectorAll('.chat-msg.user');
            var last = asked[asked.length - 1];
            box.scrollTop = last ? last.offsetTop - 8 : 0;
        }, 30);
        return Date.now();
    }""",
    Output("chat-scroll", "data"), Input("chat-messages", "children"), Input("chat-open", "data"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8050)), debug=True)
