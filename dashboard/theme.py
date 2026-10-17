"""Colours and the Plotly template. Keep in sync with the CSS variables in assets/style.css."""
import plotly.graph_objects as go
import plotly.io as pio

from . import s390x_fix  # noqa: F401  (must load before any figure is serialised)

C = {
    "bg": "#0c0e13",
    "surface": "#12151c",
    "surface_2": "#171b24",
    "border": "#1f2633",
    "grid": "#1a202b",
    "text": "#cad0d8",
    "muted": "#6b7686",
    "primary": "#3a8ee6",
    "pos": "#22a35a",
    "neg": "#e5484d",
    "warn": "#e0a339",
    "bench": "#8a94a6",
    "violet": "#9b7ae6",
}
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, Consolas, monospace"

_axis = dict(
    gridcolor=C["grid"],
    zerolinecolor=C["border"],
    linecolor=C["border"],
    tickfont=dict(color=C["muted"], size=11),
    title=dict(font=dict(color=C["muted"], size=11)),
)

pio.templates["risklab"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=C["text"], size=12),
        colorway=[C["primary"], C["bench"], C["warn"], C["pos"], C["neg"], C["violet"]],
        xaxis=_axis,
        yaxis=_axis,
        margin=dict(l=52, r=16, t=12, b=36),
        hoverlabel=dict(bgcolor=C["surface_2"], bordercolor=C["border"], font=dict(family=MONO, color=C["text"], size=12)),
        legend=dict(orientation="h", x=0, y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"], size=11)),
    )
)
pio.templates.default = "risklab"

# scrollZoom off so the mouse wheel scrolls the page instead of zooming charts (esp. 3D).
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

# Diverging scale for correlations: red (negative) -> card surface (0) -> blue (positive).
DIVERGING = [[0.0, C["neg"]], [0.5, C["surface_2"]], [1.0, C["primary"]]]


def tone(x: float) -> str:
    return C["pos"] if x >= 0 else C["neg"]
