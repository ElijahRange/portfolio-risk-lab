"""Ratio Testing page: optimize the weights for a chosen goal and explore the efficient frontier."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from .. import components as ui
from .. import config
from .. import optimizer as opt
from ..analytics import Analysis
from ..theme import C

# One colour per stock for the stacked weights chart.
PALETTE = ["#3a8ee6", "#22a35a", "#e0a339", "#9b7ae6", "#e5484d", "#2bb3b1", "#d9719f", "#8a94a6", "#b5c45a",
           "#f08c4a"]
SHORT_LABELS = {"max_sharpe": "Max Sharpe", "min_vol": "Min vol", "min_idio": "Min idio risk",
                "max_sortino": "Max Sortino", "risk_parity": "Risk parity", "max_div": "Max diversification"}
METRIC_ROWS = [
    ("Expected return", "ret", "pct"), ("Volatility", "vol", "pct"), ("Sharpe ratio", "sharpe", "num"),
    ("Sortino ratio", "sortino", "num"), ("Beta to index", "beta", "num"), ("Idiosyncratic vol", "idio_vol", "pct"),
    ("Max drawdown", "max_dd", "pct"), ("Effective # of stocks", "eff_n", "num1"),
]


def _fmt(x: float, kind: str, signed: bool = False) -> str:
    if kind == "pct":
        return ui.pct(x, 1, signed=signed)
    return ui.num(x, 1 if kind == "num1" else 2, signed=signed)


# ---------- controls ----------

def settings_from(lookback, returns, lo, hi, rf, beta_on, beta) -> opt.Settings:
    def frac(x, default):
        return default if x is None or x == "" else float(x) / 100

    return opt.Settings(
        lookback=lookback or "3Y", returns=returns or "hist",
        lo=frac(lo, 0.0), hi=frac(hi, 0.35), rf=frac(rf, config.RISK_FREE_RATE),
        target_beta=float(beta) if beta_on and beta not in (None, "") else None,
    )


def _control(label: str, child, hint: str | None = None):
    return html.Div(ui._keep(html.Div(label, className="control-label"), child,
                             html.Div(hint, className="control-hint") if hint else None), className="control")


def _number(id_, value, min_=None, max_=None):
    return dcc.Input(id=id_, type="number", value=value, step="any", min=min_, max=max_, debounce=True,
                     className="num-input")


def controls():
    return ui.card("Optimizer settings", html.Div([
        _control("Optimize for", dcc.RadioItems(
            [{"label": SHORT_LABELS[k], "value": k} for k in opt.OBJECTIVES], "max_sharpe",
            id="rt-objective", inline=True, className="segmented wrap")),
        html.Div([
            _control("Lookback", dcc.RadioItems(list(config.OPT_LOOKBACKS), "3Y", id="rt-lookback", inline=True,
                                                className="segmented")),
            _control("Expected returns", dcc.RadioItems(
                [{"label": v, "value": k} for k, v in opt.RETURN_MODELS.items()], "hist",
                id="rt-returns", inline=True, className="segmented")),
            _control("Min weight %", _number("rt-lo", 0, -100, 100), "negative = shorts allowed"),
            _control("Max weight %", _number("rt-hi", 35, 1, 100)),
            _control("Risk-free rate %", _number("rt-rf", config.RISK_FREE_RATE * 100, 0, 20)),
            _control("Target beta", html.Div([
                dcc.Checklist([{"label": "Hold at", "value": "on"}], [], id="rt-beta-on", className="check"),
                _number("rt-beta", 0.3, -2, 3),
            ], className="inline")),
        ], className="control-row"),
    ]), "Pick a goal and limits. Every portfolio is fully invested (weights sum to 100%) in the current stocks.")


def layout(a: Analysis):
    return html.Div([
        html.Div([
            html.Div([html.H2("Ratio testing", className="page-title"),
                      html.Div(f"{a.source} · prices as of {a.as_of:%b %d, %Y}", className="muted small")]),
        ], className="page-head"),
        controls(),
        dcc.Store(id="rt-custom-store", storage_type="session"),
        dcc.Tabs(
            id="rt-tabs", value="optimizer", className="tabs", parent_className="tabs-parent",
            children=[dcc.Tab(label=label, value=key, className="tab", selected_className="tab--active")
                      for key, label in (("optimizer", "Optimizer"), ("frontier", "Efficient Frontier"))],
        ),
        dcc.Loading(html.Div(id="rt-content", className="tab-content"), type="dot", color=C["primary"]),
    ])


# ---------- shared ----------

def run(a: Analysis, s: opt.Settings) -> opt.Result:
    return opt.run(a.history, a.bench_returns.name, a.positions.set_index("ticker")["weight"], s)


def _estimation_note(r: opt.Result) -> str:
    i, s = r.inputs, r.settings
    return (f"Estimated from {len(i.R)} trading days of daily returns ({i.start:%Y-%m-%d} to {i.end:%Y-%m-%d}), "
            f"{opt.RETURN_MODELS[s.returns].lower()} expected returns, risk-free rate {s.rf:.1%}, "
            f"weights between {s.lo:.0%} and {s.hi:.0%}"
            + (f", beta held at {s.target_beta:+.2f}." if s.target_beta is not None else "."))


def _sectors(a: Analysis) -> pd.Series:
    if "sector" not in a.positions:
        return pd.Series(dtype=str)
    return a.positions.set_index("ticker")["sector"].fillna("")


def _grid_table(header: list, rows: list, numeric_from: int = 1, highlight: int | None = None, row_classes=None):
    def cls(i, base=None):
        names = [base] if base else []
        if i >= numeric_from:
            names.append("num")
        if highlight is not None and i == highlight:
            names.append("col-selected")
        return " ".join(names) or None

    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(h, className=cls(i)) for i, h in enumerate(header)])),
        html.Tbody([html.Tr([html.Td(c, className=cls(i)) for i, c in enumerate(r)],
                            className=(row_classes or {}).get(n)) for n, r in enumerate(rows)]),
    ], className="data-table"), className="table-wrap")


def content(a: Analysis, tab: str, objective: str, s: opt.Settings, custom: dict | None):
    try:
        r = run(a, s)
    except opt.OptimizerError as exc:
        return ui.alert("Can't optimize with these settings", str(exc), "warn")
    alerts = [ui.alert(w, tone="info") for w in r.warnings]
    alerts += [ui.alert(f"{SHORT_LABELS[k]} failed", msg) for k, msg in r.errors.items()]
    body = frontier_view(a, r, objective, custom) if tab == "frontier" else optimizer_view(a, r, objective)
    return html.Div(alerts + [body])


# ---------- Optimizer tab ----------

def weights_figure(r: opt.Result, w: np.ndarray, label: str) -> go.Figure:
    t = r.inputs.tickers
    fig = go.Figure()
    fig.add_bar(x=t, y=r.current, name="Current", marker_color=C["bench"],
                hovertemplate="%{x}<br>current %{y:.1%}<extra></extra>")
    fig.add_bar(x=t, y=w, name=label, marker_color=C["primary"],
                hovertemplate="%{x}<br>suggested %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group", bargap=0.25, margin=dict(l=52, r=16, t=36, b=48))
    return fig


def optimizer_view(a: Analysis, r: opt.Result, objective: str):
    if objective not in r.weights:
        return ui.alert(f"{SHORT_LABELS[objective]} has no solution with these settings",
                        r.errors.get(objective), "warn")
    label, desc = opt.OBJECTIVES[objective]
    w, mc, mo = r.weights[objective], r.metrics["current"], r.metrics[objective]
    invested = float(a.positions["market_value"].sum())
    sectors = _sectors(a)
    delta = w - r.current
    scale = np.abs(delta).max() or 1

    tiles = [ui.kpi(name, _fmt(mo[key], kind),
                    sub=f"current {_fmt(mc[key], kind)} · Δ {_fmt(mo[key] - mc[key], kind, signed=True)}")
             for name, key, kind in METRIC_ROWS[:6]]

    rows = [[
        html.Div([html.Div(t, className="ticker"), html.Div(sectors.get(t, ""), className="muted small")]),
        ui.pct(r.current[i], 1),
        html.Span(ui.pct(w[i], 1), className="strong"),
        ui.bar_cell(delta[i], scale),
        html.Span(ui.pct(delta[i], 1, signed=True), className=ui.tone_class(round(delta[i], 4))),
        html.Span(ui.usd(delta[i] * invested, signed=True) if abs(delta[i]) > 5e-4 else "–",
                  className=ui.tone_class(round(delta[i], 4))),
    ] for i, t in enumerate(r.inputs.tickers)]

    keys = [k for k in opt.OBJECTIVES if k in r.weights]
    cols = ["current"] + keys
    highlight = cols.index(objective) + 1
    comp_rows = [[html.Span(t, className="ticker")] + [ui.pct(r.current[i], 1)] +
                 [ui.pct(r.weights[k][i], 1) for k in keys] for i, t in enumerate(r.inputs.tickers)]
    comp_rows += [[html.Span(name, className="strong")] + [_fmt(r.metrics[c][key], kind) for c in cols]
                  for name, key, kind in METRIC_ROWS]
    n_t = len(r.inputs.tickers)

    return html.Div([
        html.Div([
            html.H3(f"{label}: suggested weights", className="section-title"),
            html.P(f"{desc} {_estimation_note(r)}", className="card-desc"),
        ], className="section-head"),
        html.Div(tiles, className="kpi-row kpi-row-6"),
        html.Div([
            ui.card("Suggested weights", ui.table(
                ["Ticker", "Current", "Suggested", "", "Change", "Trade"], rows, numeric_cols={1, 2, 4, 5}),
                f"Trade = change in weight × today's invested value ({ui.usd(invested, compact=True)})."),
            ui.card("Current vs suggested", ui.graph(weights_figure(r, w, SHORT_LABELS[objective]), 360)),
        ], className="grid-2 wide-left"),
        ui.card("Compare every objective",
                _grid_table([""] + ["Current"] + [SHORT_LABELS[k] for k in keys], comp_rows,
                            highlight=highlight, row_classes={n_t: "row-divider"}),
                "Weights each goal would pick, and what they would have delivered over the lookback."),
        ui.card("Read with care", html.Ul([
            html.Li("These are in-sample answers: the weights that would have been best over the lookback, "
                    "not a forecast. Re-run with a different lookback to see how stable they are."),
            html.Li("Return-seeking goals (Sharpe, Sortino) chase past winners and lean on a few names. The weight "
                    "cap, the shrunk or CAPM return estimates, and the risk-only goals (Min vol, Min idio risk, "
                    "Risk parity) are the usual ways to keep that in check."),
            html.Li("Min idio risk uses each stock's leftover moves after removing the index (single-index market "
                    "model), so it targets the stock-specific part of risk only."),
        ], className="notes")),
    ])


# ---------- Efficient Frontier tab ----------

def _custom_weights(r: opt.Result, values: list | None) -> np.ndarray | None:
    if values is None:
        return None
    w = np.array([float(v) if v not in (None, "") else 0.0 for v in values]) / 100
    return w / w.sum() if abs(w.sum()) > 1e-9 else None


def frontier_figure(r: opt.Result, objective: str, custom: np.ndarray | None) -> go.Figure:
    i, fr, cl = r.inputs, r.frontier, r.cloud
    fig = go.Figure()
    if not cl.empty:
        fig.add_scatter(x=cl["vol"], y=cl["ret"], mode="markers", name="Random portfolios",
                        marker=dict(size=4, color=cl["sharpe"], opacity=0.55,
                                    colorscale=[[0, "#2a3140"], [1, C["primary"]]],
                                    colorbar=dict(title=dict(text="Sharpe", font=dict(size=10, color=C["muted"])),
                                                  thickness=8, outlinewidth=0, len=0.6,
                                                  tickfont=dict(color=C["muted"], size=10))),
                        hovertemplate="random portfolio<br>vol %{x:.1%} · return %{y:.1%}<extra></extra>")
    vols = np.sqrt(np.diag(i.cov))
    fig.add_scatter(x=vols, y=i.mu, mode="markers+text", name="Single stocks", text=i.tickers,
                    textposition="top center", textfont=dict(size=10, color=C["muted"]),
                    marker=dict(size=7, color=C["bench"], symbol="circle-open", line=dict(width=1.5)),
                    hovertemplate="%{text}<br>vol %{x:.1%} · return %{y:.1%}<extra></extra>")
    if "max_sharpe" in r.weights and len(fr) > 1:
        tan = r.metrics["max_sharpe"]
        x_end = max(fr["vol"].max(), tan["vol"]) * 1.15
        fig.add_scatter(x=[0, x_end], y=[r.settings.rf, r.settings.rf + tan["sharpe"] * x_end], mode="lines",
                        name="Capital market line", line=dict(color=C["muted"], dash="dot", width=1),
                        hoverinfo="skip")
    fig.add_scatter(x=fr["vol"], y=fr["ret"], mode="lines+markers", name="Efficient frontier",
                    line=dict(color=C["text"], width=2.5), marker=dict(size=5, color=C["text"]),
                    customdata=[f"f:{k}" for k in range(len(fr))],
                    hovertemplate="frontier<br>vol %{x:.1%} · return %{y:.1%}<extra>click for weights</extra>")
    keys = list(r.weights)
    fig.add_scatter(
        x=[r.metrics[k]["vol"] for k in keys], y=[r.metrics[k]["ret"] for k in keys], mode="markers",
        name="Optimized", text=[SHORT_LABELS[k] for k in keys], customdata=[f"o:{k}" for k in keys],
        marker=dict(symbol="diamond", color=C["violet"], size=[15 if k == objective else 10 for k in keys],
                    line=dict(color=[C["text"] if k == objective else C["violet"] for k in keys], width=1.5)),
        hovertemplate="%{text}<br>vol %{x:.1%} · return %{y:.1%}<extra>click for weights</extra>")
    mc = r.metrics["current"]
    fig.add_scatter(x=[mc["vol"]], y=[mc["ret"]], mode="markers", name="Current", customdata=["c:current"],
                    marker=dict(size=13, color=C["warn"], line=dict(color=C["bg"], width=1.5)),
                    hovertemplate="current portfolio<br>vol %{x:.1%} · return %{y:.1%}<extra></extra>")
    if custom is not None:
        m = opt.metrics(custom, i, r.settings.rf)
        fig.add_scatter(x=[m["vol"]], y=[m["ret"]], mode="markers", name="Your weights", customdata=["u:custom"],
                        marker=dict(size=16, symbol="star", color=C["pos"], line=dict(color=C["bg"], width=1)),
                        hovertemplate="your weights<br>vol %{x:.1%} · return %{y:.1%}<extra></extra>")
    fig.update_xaxes(tickformat=".0%", title=dict(text="Volatility (annualized)"), rangemode="tozero")
    fig.update_yaxes(tickformat=".0%", title=dict(text="Expected return (annualized)"))
    fig.update_layout(margin=dict(l=64, r=16, t=40, b=48), legend=dict(y=1.08), clickmode="event")
    return fig


def frontier_weights_figure(r: opt.Result) -> go.Figure:
    fr = r.frontier
    fig = go.Figure()
    for k, t in enumerate(r.inputs.tickers):
        fig.add_scatter(x=fr["vol"], y=fr[t], name=t, mode="lines", stackgroup="w",
                        line=dict(width=0.5, color=PALETTE[k % len(PALETTE)]),
                        hovertemplate=t + " %{y:.1%}<extra></extra>")
    fig.update_xaxes(tickformat=".1%", title=dict(text="Volatility along the frontier"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(hovermode="x unified", margin=dict(l=52, r=8, t=12, b=44),
                      legend=dict(orientation="v", x=1.02, y=1, traceorder="reversed"))
    return fig


def point_detail(a: Analysis, r: opt.Result, code: str, custom: np.ndarray | None):
    kind, _, key = code.partition(":")
    if kind == "f" and key.isdigit() and int(key) < len(r.frontier):
        w, title = r.frontier[r.inputs.tickers].iloc[int(key)].to_numpy(), \
            f"Frontier point {int(key) + 1} of {len(r.frontier)}"
    elif kind == "o" and key in r.weights:
        w, title = r.weights[key], opt.OBJECTIVES[key][0]
    elif kind == "u" and custom is not None:
        w, title = custom, "Your weights"
    else:
        w, title = r.current, "Current portfolio"
    m, mc = opt.metrics(w, r.inputs, r.settings.rf), r.metrics["current"]
    tiles = [html.Div([
        html.Div(name, className="tile-label"),
        html.Div(_fmt(m[k], kind_), className="tile-value"),
        html.Div(f"current {_fmt(mc[k], kind_)}", className="tile-sub"),
    ], className="tile") for name, k, kind_ in METRIC_ROWS[:6]]
    rows = [[html.Span(t, className="ticker"), ui.pct(w[n], 1),
             html.Span(ui.pct(w[n] - r.current[n], 1, signed=True), className=ui.tone_class(round(w[n] - r.current[n], 4)))]
            for n, t in enumerate(r.inputs.tickers)]
    return html.Div([
        html.Div(title, className="strong detail-title"),
        html.Div(tiles, className="tile-grid"),
        ui.table(["Ticker", "Weight", "vs current"], rows, numeric_cols={1, 2}),
    ])


def custom_summary(r: opt.Result, values: list | None):
    w = _custom_weights(r, values)
    if w is None:
        return ui.alert("Weights add up to zero", "Enter at least one non-zero weight.", "warn")
    raw = sum(float(v) for v in values if v not in (None, ""))
    m, mc = opt.metrics(w, r.inputs, r.settings.rf), r.metrics["current"]
    fr = r.frontier.sort_values("vol")
    gap = (np.interp(m["vol"], fr["vol"], fr["ret"]) - m["ret"]
           if len(fr) > 1 and fr["vol"].min() <= m["vol"] <= fr["vol"].max() else None)
    note = (f"Entered weights add up to {raw:.1f}%, so they are scaled to 100%. " if abs(raw - 100) > 0.05 else "")
    if gap is not None:
        note += (f"At this volatility the frontier offers {ui.pct(gap, 1)} more expected return per year."
                 if gap > 5e-4 else "This mix sits on the efficient frontier.")
    return html.Div(ui._keep(
        html.Div([ui.kpi(name, _fmt(m[k], kind), sub=f"current {_fmt(mc[k], kind)}")
                  for name, k, kind in METRIC_ROWS[:6]], className="kpi-row kpi-row-6"),
        html.P(note, className="muted small") if note else None,
    ))


def frontier_view(a: Analysis, r: opt.Result, objective: str, custom_values: list | None):
    t = r.inputs.tickers
    if custom_values is None or len(custom_values) != len(t):
        custom_values = [round(x * 100, 1) for x in r.current]
    custom = _custom_weights(r, custom_values)
    mc = r.metrics["current"]
    tan = r.metrics.get("max_sharpe")
    mv = r.metrics["min_vol"]
    fr = r.frontier.sort_values("vol")
    gap = (np.interp(mc["vol"], fr["vol"], fr["ret"]) - mc["ret"]
           if len(fr) > 1 and fr["vol"].min() <= mc["vol"] <= fr["vol"].max() else None)

    inputs = html.Div([
        html.Div([html.Label(tk, className="ticker"),
                  dcc.Input(id={"type": "rt-w", "index": tk}, type="number", value=v, step="any", debounce=True,
                            className="num-input")], className="weight-input")
        for tk, v in zip(t, custom_values)
    ], className="weights-grid")

    return html.Div([
        ui.kpi_row(
            ui.kpi("Max Sharpe portfolio", ui.num(tan["sharpe"], 2) if tan else "–",
                   sub=f"{ui.pct(tan['ret'], 1)} return · {ui.pct(tan['vol'], 1)} vol" if tan else None),
            ui.kpi("Min volatility portfolio", ui.pct(mv["vol"], 1),
                   sub=f"{ui.pct(mv['ret'], 1)} return · Sharpe {ui.num(mv['sharpe'], 2)}"),
            ui.kpi("Current portfolio", ui.num(mc["sharpe"], 2),
                   sub=f"Sharpe · {ui.pct(mc['ret'], 1)} return · {ui.pct(mc['vol'], 1)} vol"),
            ui.kpi("Return left on the table", ui.pct(gap, 1) if gap is not None else "–",
                   "neg" if gap and gap > 5e-4 else "",
                   sub="frontier return at the current volatility, minus current"),
        ),
        ui.card("Efficient frontier",
                dcc.Graph(id="rt-frontier-graph", figure=frontier_figure(r, objective, custom),
                          config=ui.GRAPH_CONFIG, style={"height": "520px"}),
                "Each grey-blue dot is a random mix of the current stocks (brighter = higher Sharpe). The white "
                "curve is the lowest-risk mix for each level of expected return. Click any frontier point or "
                f"diamond to see its weights. {_estimation_note(r)}"),
        html.Div([
            ui.card("Weights along the frontier", ui.graph(frontier_weights_figure(r), 380),
                    "How the mix changes as you move from the lowest-risk portfolio (left) to the highest-return "
                    "one (right)."),
            ui.card("Selected portfolio",
                    html.Div(point_detail(a, r, f"o:{objective}", custom), id="rt-point-detail"),
                    "Click a point on the frontier chart to inspect it."),
        ], className="grid-2"),
        ui.card("Try your own weights", html.Div([
            inputs,
            html.Div(custom_summary(r, custom_values), id="rt-custom-summary"),
        ]), "Type a weight (%) for each stock; the green star on the chart moves to your mix. Weights are scaled "
            "to add up to 100%.",
                actions=html.Button("Reset to current", id="rt-reset", className="btn")),
    ])
