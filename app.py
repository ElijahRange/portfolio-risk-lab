"""Portfolio Risk Lab: Dash entry point. Gunicorn serves `app:server`."""
import os
import platform
import time

from dash import Dash, Input, Output, State, dcc, html, no_update

from dashboard import components as ui
from dashboard import config, service, theme  # noqa: F401  (theme registers the Plotly template)
from dashboard.views import factors, notes, overview, risk

app = Dash(__name__, title=config.APP_TITLE, update_title=None, suppress_callback_exceptions=True)
server = app.server

# Fail closed: the site is on a public IP, so a team login is required unless explicitly disabled.
if os.environ.get("DASH_AUTH_DISABLED") == "1":
    pass
elif os.environ.get("DASH_USERS"):
    import dash_auth

    users = dict(pair.split(":", 1) for pair in os.environ["DASH_USERS"].split(",") if ":" in pair)
    dash_auth.BasicAuth(app, users)
else:
    raise RuntimeError("Set DASH_USERS=user:password[,user2:password2] or DASH_AUTH_DISABLED=1 (local only).")

TABS = {"overview": ("Portfolio View", overview), "factors": ("Factor Model", factors), "risk": ("Risk Analysis", risk)}
PAGES = {"/": "Portfolio", "/notes": "Model notes"}


def nav_link(label, href):
    return dcc.Link(html.Span(label), href=href, className="nav-link", id=f"nav-{label.lower().replace(' ', '-')}")


sidebar = html.Aside([
    html.Div([html.Div(config.APP_TITLE, className="brand-name"), html.Div("IBM Z Datathon", className="brand-sub")],
             className="brand"),
    html.Nav([
        html.Div("Workspace", className="nav-group"),
        nav_link("Portfolio", "/"),
        html.Div("Reference", className="nav-group"),
        nav_link("Model notes", "/notes"),
    ]),
    html.Div([html.Span(className="dot"), f"LinuxONE · {platform.machine()}"], className="sidebar-foot"),
], className="sidebar")


def portfolio_page():
    return html.Div([
        html.Div([
            html.Div([html.H2("Portfolio", className="page-title"), html.Div(id="source-line", className="muted small")]),
            html.Button("Refresh prices", id="refresh", className="btn"),
        ], className="page-head"),
        html.Div(id="alerts"),
        dcc.Tabs(
            id="tabs", value="overview", className="tabs", parent_className="tabs-parent",
            children=[dcc.Tab(label=label, value=key, className="tab", selected_className="tab--active")
                      for key, (label, _) in TABS.items()],
        ),
        dcc.Loading(html.Div(id="tab-content", className="tab-content"), type="dot", color=theme.C["primary"]),
    ])


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="refreshed-at"),
    sidebar,
    html.Div([
        html.Header([html.H1(id="header-title", className="header-title")], className="topbar"),
        html.Main(id="page", className="page"),
    ], className="main"),
], className="shell")


@app.callback(Output("page", "children"), Output("header-title", "children"),
              Output("nav-portfolio", "className"), Output("nav-model-notes", "className"),
              Input("url", "pathname"))
def route(pathname):
    pathname = pathname if pathname in PAGES else "/"
    active = ["nav-link active" if p == pathname else "nav-link" for p in PAGES]
    page = notes.layout() if pathname == "/notes" else portfolio_page()
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8050)), debug=True)
