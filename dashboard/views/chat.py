"""Portfolio assistant chat bubble: fixed bottom-right on every page (mounted outside the routed page).

The conversation lives in the browser session (dcc.Store). Open/close and auto-scroll run in the browser
(clientside callbacks in app.py); one server callback turns a question into an answer (dashboard/assistant.py).
"""
from dash import dcc, html

from .. import assistant
from .. import components as ui

WELCOME = {"role": "bot", "value": None, "label": None, "tone": "", "rows": None, "note": None, "link": None,
           "text": "Hi! I answer questions about this portfolio with the dashboard's own numbers: risk, returns, "
                   "beta, stress tests and each holding. Ask in your own words, or pick one:",
           "chips": assistant.DEFAULT_CHIPS}


def layout():
    return html.Div([
        dcc.Store(id="chat-log", storage_type="session", data=[]),
        dcc.Store(id="chat-open", storage_type="session", data=False),
        dcc.Store(id="chat-scroll"),
        html.Section([
            html.Div([
                html.Div([
                    html.Div("Portfolio assistant", className="chat-title"),
                    html.Div([html.Span(className="dot"), "MiniLM on LinuxONE · answers from live dashboard data"],
                             className="chat-sub"),
                ]),
                html.Div([
                    html.Button("Clear", id="chat-clear", className="chat-icon-btn", title="Clear the conversation"),
                    html.Button("×", id="chat-close", className="chat-icon-btn chat-x", title="Close",
                                **{"aria-label": "Close the assistant"}),
                ], className="chat-head-actions"),
            ], className="chat-head"),
            html.Div(id="chat-messages", className="chat-messages", role="log", **{"aria-live": "polite"}),
            html.Div(id="chat-chips", className="chat-chips"),
            html.Div([
                dcc.Input(id="chat-input", type="text", placeholder="Ask about risk, returns or a stock…",
                          autoComplete="off", maxLength=300, debounce=False, className="chat-input", n_submit=0),
                html.Button("Send", id="chat-send", className="btn btn-primary chat-send"),
            ], className="chat-form"),
        ], id="chat-panel", className="chat-panel", **{"aria-label": "Portfolio assistant"}),
        html.Button(id="chat-launcher", className="chat-launcher", title="Ask the portfolio assistant",
                    **{"aria-label": "Open the portfolio assistant"}),
    ], className="chat")


def _bot(m: dict):
    head = None
    if m.get("value"):
        head = html.Div([html.Div(m["value"], className=f"chat-value {m.get('tone') or ''}".strip()),
                         html.Div(m.get("label"), className="chat-label")], className="chat-stat")
    rows = html.Div([html.Div([html.Span(k, className="chat-row-k"), html.Span(v, className="chat-row-v")],
                              className="chat-row") for k, v in m["rows"]],
                    className="chat-rows") if m.get("rows") else None
    link = m.get("link")
    meta = None
    if m.get("title"):
        meta = html.Div(f"{m['title']} · match {m['score']:.2f} · {m.get('ms', 0)} ms", className="chat-meta",
                        title="Intent picked by the sentence-embedding model, its cosine similarity, and the time taken")
    return html.Div(ui._keep(
        head,
        html.P(m["text"], className="chat-text"),
        rows,
        html.P(m["note"], className="chat-note") if m.get("note") else None,
        html.Div(dcc.Link(f"{link['label']} →", href=link["href"], className="chat-link"),
                 **({"data-tab": link["tab"]} if link.get("tab") else {})) if link else None,
        meta,
    ), className="chat-msg bot")


def render(log: list | None):
    """Message list and suggestion chips for the stored conversation."""
    log = log or []
    msgs = [_bot(WELCOME)] + [html.Div(m["text"], className="chat-msg user") if m["role"] == "user" else _bot(m)
                              for m in log]
    last = next((m for m in reversed(log) if m["role"] == "bot"), WELCOME)
    asked = {m["text"] for m in log if m["role"] == "user"}
    chips = [c for c in last.get("chips") or [] if c not in asked][:3]
    return msgs, [html.Button(c, id={"type": "chat-chip", "index": i}, className="chip chat-chip")
                  for i, c in enumerate(chips)]
