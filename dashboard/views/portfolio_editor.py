"""Edit portfolio dialog on the Portfolio page: change holdings and dollar amounts, upload a spreadsheet, restore.

Two stages: a passcode prompt (#pe-lock), then the editor (#pe-editor) with Save changes at the bottom.

The backend (checks, passcode, backups) is dashboard/portfolio_store.py; callbacks are in app.py (pe_*).
Rows are plain inputs with pattern-matching ids; their values are read back when rows are added, removed or saved.
"""
from dash import dcc, html

from .. import components as ui
from .. import config, data


def open_button():
    return html.Button("Edit portfolio", id="pe-open", className="btn")


def row(i: int, r: dict):
    return html.Div([
        dcc.Input(id={"type": "pe-ticker", "index": i}, value=r.get("ticker"), placeholder="Ticker", debounce=True,
                  maxLength=20, name="ticker", className="pe-field pe-ticker"),
        dcc.Input(id={"type": "pe-amount", "index": i}, type="number", value=r.get("dollars"), placeholder="Dollars",
                  step="any", debounce=True, name="dollars", className="pe-field pe-amount"),
        dcc.Dropdown(config.SECTORS, r.get("sector"), id={"type": "pe-sector", "index": i}, placeholder="Sector",
                     clearable=False, searchable=False, className="pe-sector"),
        html.Div(id={"type": "pe-info", "index": i}, className="pe-info"),
        html.Button("×", id={"type": "pe-remove", "index": i}, className="pe-remove", title="Remove this holding",
                    **{"aria-label": "Remove this holding"}),
    ], className="pe-row")


def rows(items: list[dict]):
    return [row(i, r) for i, r in enumerate(items)]


def ticker_info(ticker: str | None):
    """Company name and latest price next to a ticker, or why it isn't usable."""
    if not ticker or not ticker.strip():
        return None
    q = data.quote(ticker)
    if q is None:
        return html.Span("Not found on Yahoo Finance", className="pe-bad")
    price = f"{q['price']:,.2f} {q['currency']}" if q["currency"] != "USD" else f"${q['price']:,.2f}"
    return [html.Span(q["name"], className="pe-name", title=q["name"]), html.Span(price, className="pe-price")]


def total_line(amounts: list) -> str:
    vals = [float(a) for a in amounts if a not in (None, "")]
    return f"{len(vals)} holdings · {ui.usd(sum(vals))} total (this becomes NAV)"


def dialog():
    return html.Div([
        dcc.Store(id="pe-pass-token", storage_type="memory"),  # signed edit pass from unlock; gone on reload
        dcc.Store(id="pe-focus"),
        html.Section([
            html.Button("×", id="pe-close", className="chat-icon-btn chat-x pe-close", title="Close",
                        **{"aria-label": "Close the editor"}),
            html.Div([
                html.H3("Edit portfolio", id="pe-title", className="card-title"),
                html.P("Enter the team passcode to change the portfolio.", className="card-desc"),
                html.Div([
                    dcc.Input(id="pe-pass", type="password", placeholder="Team passcode", debounce=False,
                              name="passcode", autoComplete="current-password", className="pe-field pe-pass"),
                    html.Button("Unlock", id="pe-unlock", className="btn btn-primary"),
                ], className="pe-unlock-row"),
                html.Div(id="pe-lock-status"),
            ], id="pe-lock", className="pe-stage"),
            html.Div([
                html.Div([
                    html.H3("Edit portfolio", className="card-title"),
                    html.P("Change holdings and their dollar amounts; the total becomes NAV. Save changes updates "
                           "every page and the assistant, and keeps the previous version so you can restore it.",
                           className="card-desc"),
                ], className="pe-head"),
                html.Div([
                    dcc.Upload(html.Button("Upload spreadsheet", className="btn"), id="pe-upload",
                               accept=".xlsx,.xls,.csv", max_size=2_000_000),
                    html.Span("Excel or CSV with a ticker column and a dollar (or shares) column. It fills the table "
                              "below for you to review; nothing changes until you press Save changes.",
                              className="muted small"),
                ], className="pe-toolbar"),
                html.Div([html.Span("Ticker"), html.Span("Dollars"), html.Span("Sector"),
                          html.Span("Company · price"), html.Span("")], className="pe-row pe-row-head"),
                html.Div(id="pe-rows", className="pe-rows"),
                html.Div([
                    html.Button("+ Add holding", id="pe-add", className="btn"),
                    html.Span(id="pe-total", className="pe-total mono"),
                ], className="pe-add-row"),
                html.Div([
                    dcc.Dropdown(id="pe-history", placeholder="Earlier versions", clearable=True, searchable=False,
                                 className="pe-history"),
                    html.Button("Restore", id="pe-restore", className="btn"),
                ], className="pe-restore-group"),
                html.Div(id="pe-status"),
                html.Div([
                    html.Button("Cancel", id="pe-cancel", className="btn"),
                    html.Button("Save changes", id="pe-save", className="btn btn-primary"),
                ], className="pe-foot"),
            ], id="pe-editor", className="pe-stage hidden"),
        ], className="pe-dialog", role="dialog", **{"aria-modal": "true", "aria-labelledby": "pe-title"}),
    ], id="pe-overlay", className="pe-overlay")
