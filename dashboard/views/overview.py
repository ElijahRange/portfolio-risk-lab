"""Portfolio View tab: headline numbers, performance, drawdown and positions."""
import plotly.graph_objects as go
from dash import dcc, html

from .. import components as ui
from ..analytics import Analysis
from ..theme import C


def performance_figure(a: Analysis) -> go.Figure:
    fig = go.Figure()
    for series, color, name in ((a.port_returns, C["primary"], "Portfolio"),
                                (a.bench_returns, C["bench"], f"Benchmark ({a.bench_returns.name})")):
        cum = (1 + series).cumprod() - 1
        fig.add_scatter(x=cum.index, y=cum, name=name, mode="lines", line=dict(color=color, width=2),
                        hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2%}<extra>" + name + "</extra>")
    fig.update_yaxes(tickformat="+.0%")
    fig.update_layout(hovermode="x unified")
    return fig


def drawdown_figure(a: Analysis) -> go.Figure:
    wealth = (1 + a.port_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd, mode="lines", fill="tozeroy", line=dict(color=C["neg"], width=1.5),
        fillcolor="rgba(229,72,77,0.18)", hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%")
    return fig


def positions_table(a: Analysis, side: str = "All"):
    df = a.positions if side == "All" else a.positions[a.positions["side"] == side]
    scale = a.positions["weight"].abs().max()
    rows = [[
        html.Span(r.ticker, className="ticker"),
        r.sector,
        html.Span(r.side, className=f"pill {'pos' if r.side == 'Long' else 'neg'}"),
        ui.bar_cell(r.weight, scale),
        ui.pct(r.weight),
        ui.usd(r.market_value),
        html.Span(ui.usd(r.day_pnl, signed=True), className=ui.tone_class(r.day_pnl)),
        ui.num(r.beta),
        ui.pct(r.risk_contrib, 1),
    ] for r in df.itertuples()]
    return ui.table(
        ["Ticker", "Sector", "Side", "", "Weight", "Market value", "Day P&L", "Beta", "Risk contrib."],
        rows, numeric_cols={4, 5, 6, 7, 8},
    )


def layout(a: Analysis):
    k = a.kpis
    return html.Div([
        ui.kpi_row(
            ui.kpi("NAV", ui.usd(k["nav"])),
            ui.kpi("Net exposure", ui.pct(k["net_pct"])),
            ui.kpi("Gross exposure", ui.pct(k["gross_pct"])),
            ui.kpi("Beta-adj. net", ui.pct(k["beta_adj_net"], signed=True), ui.tone_class(k["beta_adj_net"])),
            ui.kpi("Day P&L", ui.usd(k["day_pnl"], signed=True), ui.tone_class(k["day_pnl"]),
                   sub=ui.pct(k["day_pnl"] / k["nav"], signed=True)),
        ),
        html.Div([
            ui.card("Cumulative return", ui.graph(performance_figure(a)),
                    "Today's holdings held constant over the last year, against the S&P 500."),
            ui.card("Drawdown", ui.graph(drawdown_figure(a)),
                    "Distance below the previous peak of the same constant-weight portfolio."),
        ], className="grid-2"),
        ui.card(
            "Positions",
            html.Div(positions_table(a), id="positions-table"),
            f"{k['n_positions']} priced positions, sorted by absolute weight.",
            actions=dcc.RadioItems(["All", "Long", "Short"], "All", id="positions-side",
                                   inline=True, className="segmented"),
        ),
    ])
