"""Risk Analysis tab: VaR, concentration, 3D risk geometry, contributors and correlations."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from .. import components as ui
from ..analytics import Analysis
from ..theme import C, DIVERGING, MONO


# Risk geometry: the 3D PCA map and the panels that explain it.
GROUP_COLORS = [C["primary"], C["warn"], C["violet"], C["pos"], "#3fb8c4", C["neg"]]
COLOR_MODES = {"group": "Groups", "risk": "Risk share", "beta": "Beta", "sector": "Sector"}
LAYERS = {"links": "Links", "drops": "Drop lines", "hulls": "Shading"}


def _geo_points(a: Analysis) -> pd.DataFrame:
    """PCA coordinates plus what the hover and colours need, and each name's closest partner."""
    p = a.pca_points.merge(a.positions[["ticker", "sector", "beta", "risk_contrib"]], on="ticker", how="left")
    off = a.pca_corr.where(~np.eye(len(a.pca_corr), dtype=bool))
    p["avg_corr"] = p["ticker"].map(off.mean())
    p["twin"] = p["ticker"].map(off.idxmax())
    p["twin_corr"] = p["ticker"].map(off.max())
    p["label"] = p["ticker"].str.replace(r"\.[A-Z]+$", "", regex=True)
    return p


def _group_colors(p: pd.DataFrame) -> dict:
    return {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(sorted(p["group"].unique()))}


def _pairs(a: Analysis) -> list[tuple[str, str, float]]:
    c = a.pca_corr
    return [(c.index[i], c.columns[j], float(c.iat[i, j])) for i in range(len(c)) for j in range(i + 1, len(c))]


def _link_cut(pairs) -> float:
    """Draw roughly the top 30% of pairs, but never ones that barely move together."""
    return max(0.15, float(np.quantile([r for *_, r in pairs], 0.7))) if pairs else 1.0


def _segments(xyz: dict, pairs) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for t1, t2, _ in pairs:
        for k, out in enumerate((xs, ys, zs)):
            out += [xyz[t1][k], xyz[t2][k], None]
    return xs, ys, zs


def geometry_figure(a: Analysis, color_by: str = "group", layers=tuple(LAYERS)) -> go.Figure:
    p = _geo_points(a)
    xyz = p.set_index("ticker")[["pc1", "pc2", "pc3"]].apply(tuple, axis=1).to_dict()
    pts = p[["pc1", "pc2", "pc3"]].to_numpy()
    # Fit each axis to the data (always including the origin) so the points fill the cube.
    lo, hi = np.minimum(pts.min(axis=0), 0), np.maximum(pts.max(axis=0), 0)
    pad = 0.12 * (hi - lo).max()
    ranges = [(float(l - pad), float(h + pad)) for l, h in zip(lo, hi)]
    floor = ranges[2][0]
    fig = go.Figure()

    for k in range(3):  # axes through the origin: the "average stock" sits at 0 on every direction
        seg = np.zeros((2, 3))
        seg[:, k] = ranges[k]
        fig.add_scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", showlegend=False,
                          line=dict(color="#2a3342", width=2), hoverinfo="skip")

    colors = _group_colors(p)
    if "hulls" in layers and color_by == "group":
        for g, grp in p.groupby("group"):
            q = grp[["pc1", "pc2", "pc3"]].to_numpy()
            if len(q) < 3:
                continue
            mesh = dict(x=q[:, 0], y=q[:, 1], z=q[:, 2], color=colors[g], opacity=0.13, hoverinfo="skip",
                        showlegend=False, flatshading=True, lighting=dict(ambient=1, diffuse=0, specular=0))
            fig.add_mesh3d(**mesh, **(dict(alphahull=0) if len(q) >= 4 else dict(i=[0], j=[1], k=[2])))

    if "drops" in layers:  # stems to the floor make depth readable without rotating
        xs, ys, zs = [], [], []
        for x, y, z in pts:
            xs += [x, x, None]
            ys += [y, y, None]
            zs += [z, floor, None]
        fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", showlegend=False, hoverinfo="skip",
                          line=dict(color="#3a4454", width=1.5, dash="dot"))
        fig.add_scatter3d(x=pts[:, 0], y=pts[:, 1], z=[floor] * len(pts), mode="markers", showlegend=False,
                          hoverinfo="skip", marker=dict(size=3, color="#3a4454"))

    if "links" in layers:
        pairs = _pairs(a)
        cut = _link_cut(pairs)
        top = max([r for *_, r in pairs] + [cut])
        strong = [q for q in pairs if q[2] >= (cut + top) / 2]
        weak = [q for q in pairs if cut <= q[2] < (cut + top) / 2]
        hedges = [q for q in pairs if q[2] <= -0.1]
        for subset, name, line in (
            (strong, f"ρ ≥ {(cut + top) / 2:.2f}", dict(color="rgba(202,208,216,0.42)", width=4)),
            (weak, f"ρ {cut:.2f}–{(cut + top) / 2:.2f}", dict(color="rgba(202,208,216,0.2)", width=2.5)),
            (hedges, "Offsetting (ρ < −0.1)", dict(color="rgba(229,72,77,0.6)", width=3, dash="dash")),
        ):
            if not subset:
                continue
            xs, ys, zs = _segments(xyz, subset)
            fig.add_scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, line=line, hoverinfo="skip",
                              legendgroup="links")
            mid = np.array([[(xyz[t1][k] + xyz[t2][k]) / 2 for k in range(3)] for t1, t2, _ in subset])
            fig.add_scatter3d(  # invisible midpoints so the links themselves have a hover label
                x=mid[:, 0], y=mid[:, 1], z=mid[:, 2], mode="markers", showlegend=False, legendgroup="links",
                marker=dict(size=4, color="rgba(0,0,0,0)"),
                text=[f"{t1} ↔ {t2}".replace(".NS", "") for t1, t2, _ in subset], customdata=[r for *_, r in subset],
                hovertemplate="<b>%{text}</b><br>correlation %{customdata:.2f}<extra></extra>",
            )

    size = 8 + 18 * np.sqrt(p["weight"].abs() / p["weight"].abs().max())
    hover = ("<b>%{text}</b> · %{customdata[3]}<br>"
             "weight %{customdata[0]:.1%} · risk share %{customdata[1]:.1%}<br>"
             "beta vs SPX %{customdata[2]:.2f} · group %{customdata[4]}<br>"
             "PC1 %{x:.2f} · PC2 %{y:.2f} · PC3 %{z:.2f}<br>"
             "avg ρ with the rest %{customdata[5]:.2f}<br>"
             "moves most with %{customdata[6]} (ρ %{customdata[7]:.2f})<extra></extra>")
    custom = p[["weight", "risk_contrib", "beta", "sector", "group", "avg_corr", "twin", "twin_corr"]]
    common = dict(mode="markers+text", textposition="top center",
                  textfont=dict(family=MONO, size=11, color=C["text"]), hovertemplate=hover)

    def add_points(rows, name, marker):
        fig.add_scatter3d(x=rows["pc1"], y=rows["pc2"], z=rows["pc3"], name=name, text=rows["label"],
                          customdata=custom.loc[rows.index].to_numpy().tolist(), marker=marker, **common)

    outline = dict(width=1, color=C["bg"])
    if color_by in ("group", "sector"):
        key = "group" if color_by == "group" else "sector"
        palette = colors if key == "group" else {s: GROUP_COLORS[i % len(GROUP_COLORS)]
                                                 for i, s in enumerate(sorted(p["sector"].unique()))}
        for g, rows in p.groupby(key, sort=True):
            name = f"Group {g} · {', '.join(rows['label'])}" if key == "group" else g
            add_points(rows, name, dict(size=size[rows.index], color=palette[g], opacity=0.95, line=outline))
    else:
        col, title, scale, fmt = (
            ("risk_contrib", "Risk share", [[0, "#5b6a80"], [0.5, C["warn"]], [1, C["neg"]]], ".0%")
            if color_by == "risk" else
            ("beta", "Beta", [[0, C["primary"]], [0.5, "#8a94a6"], [1, C["warn"]]], ".2f"))
        add_points(p, COLOR_MODES[color_by], dict(
            size=size, color=p[col], colorscale=scale, opacity=0.95, line=outline, showscale=True,
            colorbar=dict(title=dict(text=title, font=dict(size=11, color=C["muted"])), thickness=8,
                          len=0.55, x=1.0, outlinewidth=0, tickformat=fmt,
                          tickfont=dict(family=MONO, size=10, color=C["muted"]))))

    for side in p["side"].unique():  # weighted centre of each book: where the portfolio as a whole sits
        rows = p[p["side"] == side]
        w = rows["weight"].abs()
        c = [float(np.average(rows[f"pc{k}"], weights=w)) for k in (1, 2, 3)]
        name = "Portfolio centre" if p["side"].nunique() == 1 else f"{side} book centre"
        fig.add_scatter3d(x=[0, c[0]], y=[0, c[1]], z=[0, c[2]], mode="lines", showlegend=False,
                          line=dict(color=C["text"], width=3), hoverinfo="skip")
        fig.add_scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", name=name,
                          marker=dict(size=7, color=C["text"], symbol="diamond", line=dict(width=0)),
                          hovertemplate=f"<b>{name}</b><br>PC1 %{{x:.2f}} · PC2 %{{y:.2f}} · PC3 %{{z:.2f}}"
                                        "<extra></extra>")

    def axis(k):
        pc, _, factors = a.pca_labels[k].partition(" · ")
        return dict(
            title=dict(text=f"{pc} {a.pca_explained[k]:.0%} · {factors}", font=dict(size=11, color=C["text"])),
            range=list(ranges[k]), backgroundcolor="rgba(23,27,36,0.55)", showbackground=True,
            gridcolor="#222a36", zeroline=False, showspikes=False, nticks=5,
            tickfont=dict(family=MONO, color=C["muted"], size=9))

    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        uirevision="geometry",  # keep the user's rotation when the colour mode or layers change
        scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode="cube", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=1.3, y=-1.25, z=0.72), center=dict(x=0, y=0, z=-0.06))),
        legend=dict(orientation="v", x=0, y=1, itemsizing="constant", font=dict(size=11, color=C["muted"])),
        hoverlabel=dict(align="left"),
    )
    return fig


def geometry_stats(a: Analysis):
    p = _geo_points(a)
    short = dict(zip(p["ticker"], p["label"]))
    pairs = sorted(((short[t1], short[t2], r) for t1, t2, r in _pairs(a)), key=lambda q: q[2])
    hi, lo = pairs[-1], pairs[0]
    loner = p.loc[p["avg_corr"].idxmin()]
    items = [
        ("3D view captures", f"{sum(a.pca_explained[:3]):.0%}", "of how the names move together"),
        ("Risk groups", str(p["group"].nunique()), "clusters of names that move together"),
        ("Tightest pair", f"{hi[0]} ↔ {hi[1]}", f"ρ {hi[2]:.2f}"),
        ("Best diversifier", loner["label"], f"avg ρ {loner['avg_corr']:.2f} with the rest"),
        ("Most offsetting pair", f"{lo[0]} ↔ {lo[1]}", f"ρ {lo[2]:.2f}"),
    ]
    return html.Div([html.Div([html.Div(label, className="stat-label"), html.Div(value, className="stat-value"),
                               html.Div(sub, className="stat-sub")], className="stat")
                     for label, value, sub in items], className="stat-strip")


def scree_figure(a: Analysis) -> go.Figure:
    ev = np.array(a.pca_explained)
    names = [f"PC{i + 1}" for i in range(len(ev))]
    fig = go.Figure()
    fig.add_bar(x=names, y=ev, name="This component",
                marker_color=[C["primary"] if i < 3 else "#2c3646" for i in range(len(ev))],
                hovertemplate="%{x}: %{y:.1%} of variance<extra></extra>")
    fig.add_scatter(x=names, y=np.cumsum(ev), name="Cumulative", mode="lines+markers",
                    line=dict(color=C["warn"], width=2), marker=dict(size=5),
                    hovertemplate="PC1–%{x}: %{y:.1%}<extra></extra>")
    fig.add_hline(y=1 / len(ev), line=dict(color=C["muted"], width=1, dash="dot"),
                  annotation=dict(text="one stock's worth", font=dict(size=10, color=C["muted"])),
                  annotation_position="top right")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    fig.update_layout(margin=dict(l=44, r=8, t=8, b=28), legend=dict(y=1.12), bargap=0.3)
    return fig


def loadings_figure(a: Analysis) -> go.Figure:
    p = _geo_points(a).sort_values(["group", "pc1"], ascending=[True, False])
    z = p[["pc1", "pc2", "pc3"]].to_numpy()
    rows = [f"{g} · {t}" for g, t in zip(p["group"], p["label"])]
    fig = go.Figure(go.Heatmap(
        z=z, x=["PC1", "PC2", "PC3"], y=rows, zmin=-1, zmax=1, colorscale=DIVERGING, showscale=False,
        text=np.round(z, 2), texttemplate="%{text:.2f}", textfont=dict(family=MONO, size=10, color=C["text"]),
        hovertemplate="%{y} on %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(family=MONO, size=10))
    fig.update_xaxes(side="top")
    fig.update_layout(margin=dict(l=110, r=8, t=24, b=8))
    return fig


def groups_table(a: Analysis):
    p = _geo_points(a)
    colors = _group_colors(p)
    corr = a.pca_corr
    rows = []
    for g, grp in p.groupby("group"):
        names = list(grp["ticker"])
        inner = corr.loc[names, names].to_numpy()
        avg_in = inner[~np.eye(len(names), dtype=bool)].mean() if len(names) > 1 else np.nan
        rows.append([
            html.Span([html.Span(className="swatch", style={"background": colors[g]}), g]),
            ", ".join(grp["label"]),
            ui.pct(grp["weight"].sum(), 0),
            ui.pct(grp["risk_contrib"].sum(), 0),
            "–" if np.isnan(avg_in) else ui.num(avg_in, 2),
        ])
    return ui.table(["Group", "Names", "Weight", "Risk", "Avg ρ"], rows, numeric_cols={2, 3, 4})


def geometry_controls():
    return html.Div([
        dcc.RadioItems([{"label": v, "value": k} for k, v in COLOR_MODES.items()], "group",
                       id="geo-color", inline=True, className="segmented"),
        dcc.Checklist([{"label": v, "value": k} for k, v in LAYERS.items()], list(LAYERS),
                      id="geo-layers", inline=True, className="segmented"),
    ], className="geo-controls")


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
    has_geo = not a.pca_points.empty
    geometry = (html.Div([geometry_stats(a), ui.graph(geometry_figure(a), 600, graph_id="geo-graph")])
                if has_geo else html.P("Needs at least three positions with price history.", className="muted"))
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
                "Each position sits where its returns load on the portfolio's three main risk directions. "
                "Names close together move together; lines join the most correlated pairs (red dashed = they "
                "tend to offset); colours are groups found by clustering. Dot size = weight; the white diamond "
                "is the portfolio as a whole. Drag to rotate, hover for detail.",
                actions=geometry_controls() if has_geo else None),
        html.Div([
            ui.card("Variance explained", ui.graph(scree_figure(a), 300),
                    "How much of the names' joint movement each direction explains. The blue bars are the "
                    "three axes above; anything above the dotted line beats a single stock."),
            ui.card("Loadings", ui.graph(loadings_figure(a), 300),
                    "Each name's correlation with each direction (blue = moves with it, red = against)."),
            ui.card("Risk groups", groups_table(a),
                    "Clusters of names whose returns move together, with their share of weight and of "
                    "portfolio variance."),
        ], className="grid-3 geo-panels") if has_geo else None,
        html.Div([
            ui.card("Top risk contributors", ui.graph(contributors_figure(a), 380),
                    "Share of total portfolio variance from each position (negative = hedging)."),
            ui.card("Factor vs stock-specific risk", ui.graph(factor_risk_figure(a), 380),
                    "Share of modelled variance from each factor and from stock-specific moves."),
        ], className="grid-2"),
        ui.card("Correlation of largest positions", ui.graph(correlation_figure(a), 460),
                "Daily return correlation over the lookback window for the 15 largest positions."),
    ])
