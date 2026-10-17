"""Factor Model tab: net factor betas, long/short split and rolling market beta."""
import plotly.graph_objects as go
from dash import html

from .. import components as ui
from .. import config
from ..analytics import Analysis
from ..theme import C


def factor_beta_figure(a: Analysis) -> go.Figure:
    ft = a.factor_table.iloc[::-1]  # first factor at the top
    fig = go.Figure(go.Bar(
        x=ft["net"], y=ft.index, orientation="h",
        marker_color=[C["pos"] if v >= 0 else C["neg"] for v in ft["net"]],
        customdata=ft["proxy"],
        hovertemplate="%{y} (%{customdata})<br>net beta %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=96, r=16, t=8, b=32), bargap=0.25)
    fig.update_xaxes(zeroline=True, zerolinecolor=C["muted"], zerolinewidth=1)
    return fig


def rolling_beta_figure(a: Analysis) -> go.Figure:
    rb = a.rolling_beta
    fig = go.Figure(go.Scatter(x=rb.index, y=rb, mode="lines", line=dict(color=C["primary"], width=2),
                               hovertemplate="%{x|%b %d, %Y}<br>beta %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C["muted"], dash="dot", width=1))
    return fig


def layout(a: Analysis):
    k, ft = a.kpis, a.factor_table
    scale = ft["net"].abs().max()
    rows = [[
        html.Span(name, className="strong"),
        html.Span(r.proxy, className="muted mono"),
        ui.bar_cell(r.net, scale),
        html.Span(ui.num(r.net, 2, signed=True), className=ui.tone_class(r.net)),
        ui.num(r.long, 2, signed=True),
        ui.num(r.short, 2, signed=True),
        ui.pct(r.risk_share, 1),
    ] for name, r in zip(ft.index, ft.itertuples())]

    return html.Div([
        ui.kpi_row(
            ui.kpi("Gross exposure", ui.usd(k["long_mv"] - k["short_mv"], compact=True)),
            ui.kpi("Net exposure", ui.usd(k["long_mv"] + k["short_mv"], compact=True)),
            ui.kpi("Long / Short", f"{ui.usd(k['long_mv'], compact=True)} / {ui.usd(-k['short_mv'], compact=True)}"),
            ui.kpi("Risk explained by factors", ui.pct(k["factor_share"], 1),
                   sub="Rest is stock-specific"),
        ),
        ui.card(
            "Factor model analysis",
            ui.table(["Factor", "Proxy", "", "Net β", "Long β", "Short β", "% of risk"], rows,
                     numeric_cols={3, 4, 5, 6}),
            f"Market + style + sector model fit on the last {config.LOOKBACK_DAYS} trading days of daily "
            "returns. Every factor except Market is measured beyond what the market already explains.",
        ),
        html.Div([
            ui.card("Net factor betas", ui.graph(factor_beta_figure(a), 380),
                    "How much the portfolio moves when each factor moves 1%."),
            ui.card(f"Rolling {config.ROLLING_BETA_DAYS}-day market beta", ui.graph(rolling_beta_figure(a), 380),
                    "Above 1 = swings more than the market; below 0 = moves against it."),
        ], className="grid-2"),
    ])
