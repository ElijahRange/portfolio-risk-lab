"""Idiosyncratic Risk tab: market-model split of the book against the benchmark, with a per-name drill-down."""
import numpy as np
import plotly.graph_objects as go
from dash import html

from .. import components as ui
from .. import config
from ..analytics import Analysis
from ..theme import C, DIVERGING, MONO


def _window(k: dict) -> str:
    return f"{k['start']:%Y-%m-%d} to {k['end']:%Y-%m-%d} · {k['days']} trading days"


def return_split_figure(a: Analysis) -> go.Figure:
    cs = a.idio.cum_split
    fig = go.Figure()
    for col, color, dash in ((cs.columns[0], C["bench"], "solid"), (cs.columns[1], C["violet"], "solid"),
                             (cs.columns[2], C["primary"], "dot")):
        fig.add_scatter(x=cs.index, y=cs[col], name=col, mode="lines", line=dict(color=color, width=2, dash=dash),
                        hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2%}<extra>" + col + "</extra>")
    fig.update_yaxes(tickformat="+.0%")
    fig.update_layout(hovermode="x unified")
    return fig


def rolling_share_figure(a: Analysis) -> go.Figure:
    r = a.idio.rolling
    fig = go.Figure(go.Scatter(
        x=r.index, y=r["idio_share"], mode="lines", fill="tozeroy", line=dict(color=C["violet"], width=1.5),
        fillcolor="rgba(155,122,230,0.15)", customdata=np.column_stack([r["idio_vol"], r["beta"]]),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.0%} idiosyncratic<br>idio vol %{customdata[0]:.1%}"
                      "<br>beta %{customdata[1]:.2f}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def variance_split_figure(a: Analysis) -> go.Figure:
    nm = a.idio.names.sort_values("r2")  # most market-driven name at the top
    fig = go.Figure()
    fig.add_bar(x=nm["r2"], y=nm.index, orientation="h", name="Market (R²)", marker_color=C["bench"],
                hovertemplate="%{y}<br>%{x:.0%} explained by the index<extra></extra>")
    fig.add_bar(x=1 - nm["r2"], y=nm.index, orientation="h", name="Stock-specific (1 − R²)",
                marker_color=C["violet"], hovertemplate="%{y}<br>%{x:.0%} stock-specific<extra></extra>")
    fig.update_layout(barmode="stack", margin=dict(l=96, r=16, t=36, b=32), bargap=0.3, legend=dict(y=1.14))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    return fig


def residual_corr_figure(a: Analysis) -> go.Figure:
    corr = a.idio.resid_corr
    z = corr.to_numpy().copy()
    np.fill_diagonal(z, np.nan)  # the 1.0 diagonal would wash out the colour scale
    # Residual correlations are small, so scale colours to the largest one (labels give exact values).
    zmax = max(0.3, np.ceil(np.nanmax(np.abs(z)) * 10) / 10)
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index, zmin=-zmax, zmax=zmax, colorscale=DIVERGING,
        texttemplate="%{z:.2f}", textfont=dict(family=MONO, size=10, color=C["text"]),
        colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(color=C["muted"], size=10)),
        hovertemplate="%{y} × %{x}<br>residual ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=96, r=8, t=8, b=72))
    return fig


def market_model_figure(a: Analysis, ticker: str) -> go.Figure:
    rets, r = a.idio.returns, a.idio.names.loc[ticker]
    bench = a.bench_returns.name
    x, y = rets[bench], rets[ticker]
    line_x = np.array([x.min(), x.max()])
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", marker=dict(size=5, color=C["primary"], opacity=0.55),
                    customdata=rets.index, name="Days",
                    hovertemplate="%{customdata|%b %d, %Y}<br>" + bench + " %{x:+.2%}<br>"
                                  + ticker + " %{y:+.2%}<extra></extra>")
    fig.add_scatter(x=line_x, y=r["alpha"] / config.TRADING_DAYS + r["beta"] * line_x, mode="lines",
                    line=dict(color=C["warn"], width=2), name=f"Fit: β = {r['beta']:.2f}", hoverinfo="skip")
    fig.update_xaxes(tickformat=".0%", title=dict(text=f"{bench} daily return"))
    fig.update_yaxes(tickformat=".0%", title=dict(text=f"{ticker} daily return"))
    fig.update_layout(showlegend=False, margin=dict(l=56, r=12, t=8, b=44))
    return fig


def _tile(label: str, value: str, sub: str | None = None, tone: str = ""):
    return html.Div(ui._keep(
        html.Div(label, className="tile-label"),
        html.Div(value, className=f"tile-value {tone}"),
        html.Div(sub, className="tile-sub") if sub else None,
    ), className="tile")


def detail(a: Analysis, ticker: str):
    r, k = a.idio.names.loc[ticker], a.idio.kpis
    pos = a.positions.set_index("ticker").loc[ticker]
    sys_share = r["r2"]
    stat_rows = [
        ["Alpha (annualized)", html.Span(ui.pct(r["alpha"], 1, signed=True), className=ui.tone_class(r["alpha"])),
         ui.num(r["alpha_t"], 2, signed=True)],
        [f"Beta to {a.bench_returns.name}", ui.num(r["beta"], 2), ui.num(r["beta_t"], 2, signed=True)],
    ]
    return html.Div([
        html.Div([
            html.Span(ticker, className="ticker detail-ticker"),
            html.Span(pos["side"].lower(), className=f"pill {'pos' if pos['side'] == 'Long' else 'neg'}"),
            html.Span(pos["sector"] if isinstance(pos.get("sector"), str) else "", className="muted small"),
        ], className="detail-head"),
        html.Div([
            _tile("Exposure", ui.usd(pos["market_value"], compact=True), f"{ui.pct(r['weight'], 1)} of NAV"),
            _tile("R²", ui.num(r["r2"], 2), f"{ui.pct(sys_share, 0)} explained by the index"),
            _tile("% of book idio risk", ui.pct(r["idio_contrib"], 1), "share of residual variance"),
            _tile("Systematic return", ui.pct(r["sys_ret"], 1, signed=True), "β × index, lookback",
                  ui.tone_class(r["sys_ret"])),
            _tile("Idiosyncratic return", ui.pct(r["idio_ret"], 1, signed=True), "α + residual, lookback",
                  ui.tone_class(r["idio_ret"])),
            _tile("Total vol", ui.pct(r["total_vol"], 1), "annualized"),
            _tile("Systematic vol", ui.pct(r["sys_vol"], 1), "|β| × index vol"),
            _tile("Specific vol", ui.pct(r["idio_vol"], 1), f"{ui.pct(1 - sys_share, 0)} of variance"),
        ], className="tile-grid"),
        html.Div("Market model fit", className="detail-sub"),
        ui.graph(market_model_figure(a, ticker), 240),
        ui.table(["Term", "Estimate", "t-stat"], stat_rows, numeric_cols={1, 2}),
        html.P(f"Newey–West (HAC) t-stats over {_window(k)}; |t| > 1.96 is significant at 5%.",
               className="muted small"),
    ], className="detail")


def _row_class(ticker: str, selected: str) -> str:
    return "row-click selected" if ticker == selected else "row-click"


def names_table(a: Analysis, selected: str):
    nm = a.idio.names.sort_values("idio_contrib", key=abs, ascending=False)
    side = a.positions.set_index("ticker")["side"]
    scale = nm["idio_contrib"].abs().max()
    head = ["Ticker", "Side", "Wgt%", "β", "R²", "Spec. vol", "", "% idio risk"]
    num_cols = {2, 3, 4, 5, 7}
    rows = [html.Tr([
        html.Td(html.Span(t, className="ticker")),
        html.Td(html.Span(side[t].lower(), className="pos" if side[t] == "Long" else "neg")),
        html.Td(ui.pct(r.weight, 1), className="num"),
        html.Td(ui.num(r.beta, 2), className="num"),
        html.Td(ui.num(r.r2, 2), className="num"),
        html.Td(ui.pct(r.idio_vol, 1), className="num"),
        html.Td(ui.bar_cell(r.idio_contrib, scale)),
        html.Td(html.Span(ui.pct(r.idio_contrib, 1), className=ui.tone_class(r.idio_contrib)), className="num"),
    ], id={"type": "idio-row", "index": t}, n_clicks=0, className=_row_class(t, selected))
        for t, r in zip(nm.index, nm.itertuples())]
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(h, className="num" if i in num_cols else None) for i, h in enumerate(head)])),
        html.Tbody(rows),
    ], className="data-table"), className="table-wrap")


def default_ticker(a: Analysis) -> str:
    return a.idio.names["idio_contrib"].abs().idxmax()


def layout(a: Analysis):
    k, nav = a.idio.kpis, a.nav
    bench = a.bench_returns.name
    selected = default_ticker(a)
    div_ratio = k["idio_vol"] / k["diag_idio_vol"] if k["diag_idio_vol"] else float("nan")
    return html.Div([
        html.Div([
            html.H3("Idiosyncratic risk analysis", className="section-title"),
            html.P(f"Single-index market model against the {bench} (S&P 500) as the market portfolio: each "
                   f"day's return = alpha + beta × {bench} + a stock-specific residual. Fit on {_window(k)}, "
                   "holding today's weights constant.", className="card-desc"),
        ], className="section-head"),
        ui.kpi_row(
            ui.kpi(f"Market beta ({bench})", ui.num(k["beta"], 3, signed=True),
                   sub=f"t = {k['beta_t']:+.1f} · R² {k['r2']:.2f}"),
            ui.kpi("Systematic return (lookback)", ui.pct(k["sys_ret"], 2, signed=True), ui.tone_class(k["sys_ret"]),
                   sub=f"{ui.usd(k['sys_ret'] * nav, signed=True, compact=True)} · β × {bench}"),
            ui.kpi("Idiosyncratic return (lookback)", ui.pct(k["idio_ret"], 2, signed=True),
                   ui.tone_class(k["idio_ret"]),
                   sub=f"{ui.usd(k['idio_ret'] * nav, signed=True, compact=True)} · alpha / selection"),
            ui.kpi("Systematic risk", ui.pct(k["sys_vol"], 2), sub=f"{ui.pct(1 - k['idio_share'], 1)} of variance · ann. vol"),
            ui.kpi("Idiosyncratic risk", ui.pct(k["idio_vol"], 2),
                   sub=f"{ui.pct(k['idio_share'], 1)} of variance · ann. vol"),
        ),
        ui.kpi_row(
            ui.kpi("Alpha t-stat", ui.num(k["alpha_t"], 2, signed=True),
                   "pos" if abs(k["alpha_t"]) > 1.96 else "",
                   sub=f"α {ui.pct(k['alpha'], 1, signed=True)}/yr · " +
                       ("significant at 5%" if abs(k["alpha_t"]) > 1.96 else "not significant at 5%")),
            ui.kpi("Information ratio", ui.num(k["info_ratio"], 2, signed=True), ui.tone_class(k["info_ratio"]),
                   sub="alpha ÷ idiosyncratic vol"),
            ui.kpi("Beta-neutral hedge", ui.usd(k["hedge_usd"], signed=True, compact=True),
                   sub=f"{bench} notional · leaves {ui.pct(k['idio_vol'], 1)} vol"),
            ui.kpi("Residual co-movement", f"{div_ratio:.2f}×",
                   sub=f"{ui.pct(k['idio_vol'], 1)} actual vs {ui.pct(k['diag_idio_vol'], 1)} if independent"),
        ),
        html.Div([
            ui.card("Where the return came from", ui.graph(return_split_figure(a), 320),
                    f"Summed daily return split into the part explained by {bench} (β × index) and the "
                    "stock-specific remainder (alpha + residual)."),
            ui.card(f"Idiosyncratic share of variance, rolling {config.ROLLING_BETA_DAYS} days",
                    ui.graph(rolling_share_figure(a), 320),
                    "1 − R² of the book against the index over the full price history. Higher = the book "
                    "moves on its own stock bets rather than with the market."),
        ], className="grid-2"),
        ui.card(
            "Per-name market model",
            html.Div([
                html.Div(names_table(a, selected), className="split-left"),
                html.Div(detail(a, selected), id="idio-detail", className="split-right"),
            ], className="split"),
            "Sorted by share of the book's idiosyncratic variance (using the residual covariance, so it sums "
            "to 100%). Click a row to inspect that name's fit.",
        ),
        html.Div([
            ui.card("Market vs stock-specific variance by name", ui.graph(variance_split_figure(a), 360),
                    f"How much of each name's daily variance {bench} explains (R²) and how much is its own."),
            ui.card("Residual correlation", ui.graph(residual_corr_figure(a), 360),
                    "Correlation left over after removing the market. Clusters here are shared bets "
                    "(sector, style, country) that the single-index model counts as idiosyncratic."),
        ], className="grid-2"),
    ])
