"""PRISM's night sky: pixel stars, little glinting prisms and the odd shooting star behind the sidebar and page.

Purely decorative (pointer-events off, hidden from screen readers). Positions come from a fixed seed so the sky is the
same on every page load and every server worker. The pixel sprites and animations live in assets/style.css (.sky);
everything stops for visitors who ask their system for reduced motion.
"""
import math
import random

from dash import html

SPECTRUM = ["#e5484d", "#e0a339", "#22a35a", "#3a8ee6", "#9b7ae6"]  # the logo's rays, P R I S M


def _star(rng: random.Random, kind: str, x: float, y: float, dim: bool):
    hi = rng.uniform(0.25, 0.45) if dim else rng.uniform(0.7, 1.0)
    style = {"left": f"{x:.1f}%", "top": f"{y:.1f}%", "--hi": f"{hi:.2f}", "--lo": f"{hi * 0.12:.2f}",
             "--dur": f"{rng.uniform(2.8, 7.5):.1f}s", "--delay": f"{-rng.uniform(0, 8):.1f}s"}
    if kind == "prism":
        style["--glint"] = rng.choice(SPECTRUM)
    elif rng.random() < 0.25:
        style["color"] = rng.choice(SPECTRUM[2:])  # a few cool-tinted stars
    return html.Span(className=f"star {kind}", style=style)


def _field(seed: int, bands: list[tuple[float, float, int, bool]], mix=(0.55, 0.3, 0.15)) -> list:
    """Stars scattered over horizontal bands: (top %, bottom %, how many, dim?). mix = share of dots/sparkles/prisms."""
    rng = random.Random(seed)
    stars = []
    for top, bottom, count, dim in bands:
        for _ in range(count):
            kind = rng.choices(["dot", "sparkle", "prism"], weights=mix)[0]
            stars.append(_star(rng, kind, rng.uniform(4, 94), rng.uniform(top, bottom), dim))
    return stars


def _shooting(left: str, top: str, cycle: float, delay: float, dx: int, dy: int):
    return html.Span(className="shooting", style={
        "left": left, "top": top, "--cycle": f"{cycle}s", "--delay": f"{delay}s", "--dx": f"{dx}px", "--dy": f"{dy}px",
        "--angle": f"{math.degrees(math.atan2(dy, dx)):.1f}deg"})  # point the head along the flight path


def sidebar():
    """Faint over the navigation, a full sky in the empty space below it."""
    return html.Div(_field(7, [(0, 40, 9, True), (40, 96, 24, False)]) + [
        _shooting("88%", "38%", 13, 3, -230, 150),
        _shooting("70%", "58%", 21, 11, -200, 130),
    ], className="sky", **{"aria-hidden": "true"})


def page():
    """A sparse, dim sky behind the page: it shows in the gaps between cards and on emptier pages."""
    return html.Div(_field(11, [(2, 98, 34, True)], mix=(0.6, 0.25, 0.15)) + [
        _shooting("82%", "6%", 24, 8, -480, 260),
    ], className="sky sky-page", **{"aria-hidden": "true"})


def topbar():
    """A few stars in the empty right half of the top bar."""
    rng = random.Random(5)
    return html.Div([_star(rng, kind, rng.uniform(55, 97), rng.uniform(18, 78), True)
                     for kind in ["dot", "sparkle", "dot", "prism", "dot", "sparkle", "dot"]],
                    className="sky sky-top", **{"aria-hidden": "true"})
