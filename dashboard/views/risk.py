"""Risk Analysis tab: VaR, concentration, 3D risk geometry, contributors and correlations."""
import numpy as np
import plotly.graph_objects as go
from dash import html

from .. import components as ui
from ..analytics import Analysis
from ..theme import C, DIVERGING


def geometry_figure(a: Analysis) -> go.Figure:
    pts = a.pca_points
    fig = go.Figure()
    scale = pts["weight"].abs().max()
    for side, color in (("Long", C["pos"]), ("Short", C["neg"])):
        p = pts[pts["side"] == side]
        if p.empty:
            continue
        fig.add_scatter3d(
            x=p["pc1"], y=p["pc2"], z=p["pc3"], mode="markers", name=f"{side} ({len(p)})",
            marker=dict(size=4 + 14 * p["weight"].abs() / scale, color=color, opacity=0.85,
                        line=dict(width=0)),
            text=p["ticker"], customdata=p["weight"],
            hovertemplate="<b>%{text}</b><br>weight %{customdata:.2%}<extra></extra>",
        )
        w = p["weight"].abs()
        fig.add_scatter3d(  # weight-averaged centre of each book
            x=[np.average(p["pc1"], weights=w)], y=[np.average(p["pc2"], weights=w)],
            z=[np.average(p["pc3"], weights=w)], mode="markers", name=f"{side} centroid",
            marker=dict(size=9, color=color, symbol="diamond", line=dict(color=C["text"], width=1)),
            hovertemplate=f"{side} book centroid<extra></extra>",
        )
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=C["grid"], zerolinecolor=C["border"],
                showspikes=False, tickfont=dict(color=C["muted"], size=9), title=dict(font=dict(size=10, color=C["muted"])))
    labels = a.pca_labels
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=dict(axis, title=dict(text=labels[0])), yaxis=dict(axis, title=dict(text=labels[1])),
                   zaxis=dict(axis, title=dict(text=labels[2])), bgcolor="rgba(0,0,0,0)"),
        legend=dict(y=1, x=0),
    )
    return fig


def contributors_figure(a: Analysis) -> go.Figure:
    top = a.positions.reindex(a.positions["risk_contrib"].abs().sort_values(ascending=False).index).head(12)
    top = top.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["risk_contrib"], y=top["ticker"], orientation="h",
        marker_color=[C["pos"] if s == "Long" else C["neg"] for s in top["side"]],
        customdata=top["weight"],
        hovertemplate="%{y}<br>%{x:.1%} of variance<br>weight %{customdata:.2%}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=64, r=16, t=8, b=32))
    return fig


def factor_risk_figure(a: Analysis) -> go.Figure:
    ft = a.factor_table
    total = a.risk_split["factor"] + a.risk_split["idiosyncratic"]
    names = list(ft.index) + ["Stock-specific"]
    shares = list(ft["risk_share"]) + [a.risk_split["idiosyncratic"] / total if total else 0]
    order = np.argsort(np.abs(shares))
    names, shares = [names[i] for i in order], [shares[i] for i in order]
    fig = go.Figure(go.Bar(
        x=shares, y=names, orientation="h",
        marker_color=[C["bench"] if n == "Stock-specific" else (C["primary"] if s >= 0 else C["warn"])
                      for n, s in zip(names, shares)],
        hovertemplate="%{y}<br>%{x:.1%} of model variance<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(margin=dict(l=96, r=16, t=8, b=32))
    return fig


def correlation_figure(a: Analysis) -> go.Figure:
    corr = a.corr
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING,
        colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(color=C["muted"], size=10)),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=64, r=8, t=8, b=56))
    return fig


def layout(a: Analysis):
    k = a.kpis
    geometry = (ui.graph(geometry_figure(a), 520) if not a.pca_points.empty
                else html.P("Needs at least three positions with price history.", className="muted"))
    return html.Div([
        ui.kpi_row(
            ui.kpi("Annualized volatility", ui.pct(k["ann_vol"], 1)),
            ui.kpi("1-day VaR (95%)", ui.usd(k["var95_usd"]), "neg",
                   sub=f"{ui.pct(k['var95_pct'])} of NAV · CVaR {ui.pct(k['cvar95_pct'])}"),
            ui.kpi("Max drawdown", ui.pct(k["max_drawdown"], 1), "neg"),
            ui.kpi("Effective # of positions", ui.num(k["effective_n"], 1),
                   sub=f"Top 10 = {ui.pct(k['top10_pct'], 0)} of gross"),
        ),
        ui.card("Risk geometry (PCA → 3D)", geometry,
                "Each dot is a position placed by how its returns load on the portfolio's three main risk "
                "directions. Names close together carry similar risk; dot size = weight. Drag to rotate."),
        html.Div([
            ui.card("Top risk contributors", ui.graph(contributors_figure(a), 380),
                    "Share of total portfolio variance from each position (negative = hedging)."),
            ui.card("Factor vs stock-specific risk", ui.graph(factor_risk_figure(a), 380),
                    "Share of modelled variance from each factor and from stock-specific moves."),
        ], className="grid-2"),
        ui.card("Correlation of largest positions", ui.graph(correlation_figure(a), 460),
                "Daily return correlation over the lookback window for the 15 largest positions."),
    ])
