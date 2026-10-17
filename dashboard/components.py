"""Reusable layout pieces (KPI tiles, cards, alerts, tables) and number formatting."""
import math

from dash import dcc, html

from .theme import GRAPH_CONFIG


def _keep(*children):
    return [c for c in children if c is not None]


# ---------- formatting ----------

def usd(x: float, signed: bool = False, compact: bool = False) -> str:
    sign = "+" if signed and x > 0 else ("-" if x < 0 else "")
    x = abs(x)
    if compact:
        for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if x >= div:
                return f"{sign}${x / div:,.2f}{suffix}"
    return f"{sign}${x:,.0f}"


def pct(x: float, digits: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def tone_class(x: float) -> str:
    return "pos" if x > 0 else ("neg" if x < 0 else "")


# ---------- building blocks ----------

def kpi(label: str, value: str, tone: str = "", sub: str | None = None):
    return html.Div(_keep(
        html.Div(label, className="kpi-label"),
        html.Div(value, className=f"kpi-value {tone}"),
        html.Div(sub, className="kpi-sub") if sub else None,
    ), className="kpi")


def kpi_row(*tiles):
    return html.Div(list(tiles), className="kpi-row")


def card(title: str, body, description: str | None = None, actions=None, className: str = ""):
    head = html.Div(_keep(
        html.Div(_keep(
            html.H3(title, className="card-title"),
            html.P(description, className="card-desc") if description else None,
        )),
        html.Div(actions, className="card-actions") if actions is not None else None,
    ), className="card-head")
    return html.Section([head, html.Div(body, className="card-body")], className=f"card {className}".strip())


def alert(title: str, detail: str | None = None, tone: str = "warn"):
    return html.Div(_keep(
        html.Div(title, className="alert-title"),
        html.Div(detail, className="alert-detail") if detail else None,
    ), className=f"alert alert-{tone}")


def graph(fig, height: int = 320, graph_id: str | None = None):
    kwargs = {"id": graph_id} if graph_id else {}
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": f"{height}px"}, **kwargs)


def bar_cell(value: float, scale: float):
    """Small centred bar like a sparkline: right/green for positive, left/red for negative."""
    width = 0 if not scale else min(abs(value) / scale, 1) * 50
    side = "left" if value >= 0 else "right"
    return html.Div(
        html.Div(className=f"minibar-fill {'pos' if value >= 0 else 'neg'}",
                 style={side: "50%", "width": f"{width:.1f}%"}),
        className="minibar",
    )


def table(header: list[str], rows: list[list], numeric_cols: set[int] = frozenset()):
    def cell(tag, content, i):
        return tag(content, className="num" if i in numeric_cols else None)

    return html.Div(html.Table([
        html.Thead(html.Tr([cell(html.Th, h, i) for i, h in enumerate(header)])),
        html.Tbody([html.Tr([cell(html.Td, c, i) for i, c in enumerate(r)]) for r in rows]),
    ], className="data-table"), className="table-wrap")
