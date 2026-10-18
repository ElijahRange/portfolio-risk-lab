"""AI replies for the portfolio assistant, written by Claude (Anthropic API) from the dashboard's own numbers.

Every question goes to Claude with a text snapshot of everything the dashboard shows (positions, sectors, risk,
stress tests, factors, correlations, optimizer results, FinBERT earnings tone), so it can answer free-form questions
("which sectors do we perform best in?") while every figure still comes from the same objects as the charts.

The on-device model in assistant.py (MiniLM on LinuxONE) still runs first: it picks the related chart for the reply's
link and follow-up chips, and its template answer is the fallback whenever Claude is unavailable (no API key, network
or API error, refusal, rate limit). So the chat never breaks during a demo.
"""
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
from flask import has_request_context, request

from . import assistant, config, optimizer, research_store
from . import components as ui

SYSTEM = """You are the portfolio assistant inside Portfolio Risk Lab, a dashboard a student team built for the IBM Z \
Datathon 2026. It analyses the team's real, $1,000,000, nine-stock portfolio. Answer the user's question from the \
DASHBOARD DATA below.

How to answer:
- Ground every portfolio number in the dashboard data and quote it as it appears there. When you combine numbers \
(sums, averages, rankings), say briefly how. Never invent a figure: if the data doesn't contain it, say so and give the \
closest thing it does contain.
- Lead with the direct answer and its key number in **bold**, then one to three sentences on what it means in plain \
English (the judges aren't all finance people). Use a short bullet list only when comparing several items. Stay under \
about 120 words unless the user asks for more detail.
- When it helps, say which page or tab shows the chart (see WHERE THINGS ARE).
- General finance questions (what beta means, how VaR works) are welcome: explain briefly and tie the answer back to \
this portfolio.
- You may discuss what the optimizer suggests and its trade-offs, framed as analysis of past data, not a forecast or \
personal financial advice.
- If a request has nothing to do with the portfolio, investing or this dashboard, say in one sentence that you only \
help with this portfolio and suggest a question you can answer.
- Format with plain markdown only: bold and bullet lists. No headings, tables, links, images or code blocks."""

WHERE = """WHERE THINGS ARE
- Portfolio page > Portfolio View tab: NAV, exposure, day P&L, cumulative return vs the S&P 500, drawdown, positions table.
- Portfolio page > Factor Model tab: factor betas (style and sector ETF proxies), rolling 63-day market beta.
- Portfolio page > Idiosyncratic Risk tab: market beta, alpha, R², systematic vs stock-specific return and risk per stock.
- Portfolio page > Risk Analysis tab: VaR by method, expected shortfall, VaR backtest, stress tests, risk budget, 3D risk \
geometry (correlations, risk groups), diversification over time, upside/downside beta.
- Ratio Testing page: optimizer (max Sharpe, min volatility, min idiosyncratic risk, max Sortino, risk parity, max \
diversification) and the efficient frontier.
- Company Research page: earnings press releases from SEC EDGAR scored by FinBERT (tone from -1 to +1).
- Model notes page: methodology."""


# ---------- snapshot of the dashboard's numbers ----------

def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join(" | ".join(str(c) for c in r) for r in [header, *rows])


def snapshot(d: "assistant.Data") -> str:
    """Everything the dashboard shows, as compact text. Same data -> same bytes, so the prompt cache keeps hitting."""
    a, p, rep, opt, nav = d.a, d.p, d.rep, d.opt, d.nav
    k, ik = a.kpis, a.idio.kpis
    R = a.idio.returns[list(p.index)]
    n_days = len(a.port_returns)
    window = f"{a.port_returns.index[0]:%Y-%m-%d} to {a.port_returns.index[-1]:%Y-%m-%d}"
    cum = (1 + R).prod() - 1
    contrib = p["weight"] * R.mean() * config.TRADING_DAYS
    b, ud, names = rep.budget, rep.updown, a.idio.names
    hist = rep.var_methods.loc["Historical"]
    bench_vol = float(rep.bench.tail(config.TRADING_DAYS).std() * np.sqrt(config.TRADING_DAYS))
    wealth = (1 + a.port_returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    tones = {}
    for t in p.index:
        c = research_store.company(t) if t not in config.LOCAL_CURRENCY_FX else None
        rel = next((r for r in (c or {}).get("releases", []) if r["tone"]), None)
        if rel:
            yoy = c["financials"].get("revenue_yoy")
            tones[t] = (f"{rel['tone']['net']:+.2f} ({assistant._tone_word(rel['tone']['net'])}, release "
                        f"{rel['filing_date']}, revenue YoY {ui.pct(yoy, 1, signed=True) if yoy is not None else 'n/a'})")

    out = [f"DASHBOARD DATA (prices as of {a.as_of:%Y-%m-%d}; risk window = last {n_days} trading days, {window}; "
           f"benchmark = S&P 500; today's weights held constant over the window)",
           "",
           "PORTFOLIO",
           f"NAV {ui.usd(nav)} in {k['n_positions']} stocks, all long. Day P&L {ui.usd(k['day_pnl'], signed=True)} "
           f"({ui.pct(k['day_pnl'] / nav, 2, signed=True)}).",
           f"Return over the window {ui.pct(_cum(a.port_returns), 1, signed=True)} vs S&P 500 "
           f"{ui.pct(_cum(a.bench_returns), 1, signed=True)}; annualised mean return {ui.pct(k['ann_return'], 1, signed=True)}.",
           f"Annualised volatility {ui.pct(k['ann_vol'], 1)} (S&P 500 {ui.pct(bench_vol, 1)}).",
           f"Max drawdown {ui.pct(k['max_drawdown'], 1)} ({wealth[:trough].idxmax():%Y-%m-%d} to {trough:%Y-%m-%d}); "
           f"now {ui.pct(float(dd.iloc[-1]), 1)} from peak.",
           f"Market beta {ui.num(ik['beta'], 3, signed=True)} (R² {ik['r2']:.2f}); downside beta {ud.loc['Portfolio', 'down_beta']:.2f}, "
           f"upside beta {ud.loc['Portfolio', 'up_beta']:.2f} (full history since {rep.port.index[0]:%Y}). "
           f"Beta-neutral hedge: {ui.usd(ik['hedge_usd'], signed=True, compact=True)} of S&P 500.",
           f"Alpha {ui.pct(ik['alpha'], 1, signed=True)}/yr (t-stat {ik['alpha_t']:+.2f}), information ratio "
           f"{ik['info_ratio']:+.2f}. Idiosyncratic (stock-specific) share of variance {ui.pct(ik['idio_share'], 1)}; "
           f"systematic vol {ui.pct(ik['sys_vol'], 1)}, idiosyncratic vol {ui.pct(ik['idio_vol'], 1)}.",
           f"1-day VaR (historical): 95% {ui.usd(hist['var95'] * nav)} ({ui.pct(hist['var95'])}), 99% "
           f"{ui.usd(hist['var99'] * nav)} ({ui.pct(hist['var99'])}). Expected shortfall 97.5% {ui.usd(hist['es975'] * nav)}; "
           f"CVaR 95% {ui.usd(k['cvar95_pct'] * nav)}.",
           "VaR 95% by method: " + "; ".join(f"{m} {ui.usd(v * nav)}" for m, v in rep.var_methods["var95"].items()) + ".",
           f"VaR backtest (EWMA 95%): {rep.kupiec['breaches']} breaches in {rep.kupiec['n']} days vs "
           f"{rep.kupiec['expected']:.1f} expected, Kupiec p-value {rep.kupiec['p_value']:.2f} "
           f"({'passes' if rep.kupiec['p_value'] >= 0.05 else 'fails'}).",
           f"Diversification ratio {rep.div_ratio:.2f}x; effective number of positions {k['effective_n']:.1f}; "
           "average pairwise correlation " + ", ".join(f"{name} {v:.2f}" for name, v in rep.corr_regimes.items()) + ".",
           f"Multi-factor model explains {ui.pct(k['factor_share'], 0)} of risk.",
           "",
           "POSITIONS (window return = price return over the risk window; contribution = weight x mean daily return x 252, "
           "sums to the portfolio's annualised mean return; risk share = share of portfolio volatility; betas vs S&P 500)",
           _table(["ticker", "company", "sector", "weight", "value", "day P&L", "window return", "contribution", "beta",
                   "down beta", "up beta", "volatility", "risk share", "stock-specific share", "alpha/yr (t)", "earnings tone"],
                  [[t, assistant._name(t), r.sector, ui.pct(r.weight, 1), ui.usd(r.market_value), ui.usd(r.day_pnl, signed=True),
                    ui.pct(cum[t], 1, signed=True), ui.pct(contrib[t], 2, signed=True), ui.num(r.beta, 2),
                    f"{ud.loc[t, 'down_beta']:.2f}", f"{ud.loc[t, 'up_beta']:.2f}", ui.pct(b.loc[t, 'vol'], 1),
                    ui.pct(b.loc[t, 'risk_share'], 1), ui.pct(1 - names.loc[t, 'r2'], 0),
                    f"{ui.pct(names.loc[t, 'alpha'], 1, signed=True)} ({names.loc[t, 'alpha_t']:+.1f})",
                    tones.get(t, "none stored")] for t, r in p.iterrows()]),
           "",
           "SECTORS (weighted return = weight-averaged window return of the sector's stocks)"]
    rows = []
    for sec, g in p.groupby("sector"):
        w = g["weight"]
        rows.append([sec, ", ".join(g.index), ui.pct(w.sum(), 1), ui.pct(float((w * cum[g.index]).sum() / w.sum()), 1, signed=True),
                     ui.pct(float(contrib[g.index].sum()), 2, signed=True), ui.pct(float(b.loc[g.index, "risk_share"].sum()), 1),
                     ui.usd(float(g["day_pnl"].sum()), signed=True)])
    rows.sort(key=lambda r: -float(r[2].rstrip("%")))
    out += [_table(["sector", "stocks", "weight", "weighted return", "contribution", "risk share", "day P&L"], rows), ""]

    C = a.pca_corr
    out += ["RETURN CORRELATIONS (daily, risk window)",
            _table(["", *C.columns], [[i, *[f"{v:.2f}" for v in C.loc[i]]] for i in C.index]), ""]

    ft = a.factor_table
    out += ["FACTOR BETAS (non-market factors measured beyond the market; Factor Model tab)",
            _table(["factor", "ETF proxy", "net beta", "share of risk"],
                   [[f, ft.loc[f, "proxy"], ui.num(ft.loc[f, "net"], 2, signed=True), ui.pct(ft.loc[f, "risk_share"], 0)]
                    for f in ft.index]), ""]

    out += ["STRESS TESTS (today's weights replayed; hypothetical rows scale the S&P move by downside beta)",
            _table(["scenario", "period", "portfolio", "portfolio $", "S&P 500"],
                   [[r.scenario, r.period, ui.pct(r.port, 1, signed=True), ui.usd(r.port * nav, signed=True),
                     ui.pct(r.bench, 1, signed=True)] for r in rep.stress.itertuples()]), ""]

    s, inp = opt.settings, opt.inputs
    cols = ["current", *opt.weights]
    labels = {"current": "Current", **{key: v[0] for key, v in optimizer.OBJECTIVES.items()}}
    out += [f"OPTIMIZER (Ratio Testing page; {len(inp.R)} trading days, {inp.start:%Y-%m-%d} to {inp.end:%Y-%m-%d}, "
            f"historical mean returns, risk-free {s.rf:.1%}, weights {s.lo:.0%}-{s.hi:.0%} per stock; in-sample, not a forecast)",
            _table(["portfolio", "exp. return", "volatility", "Sharpe", "Sortino", "beta", "max drawdown",
                    "weights (" + ", ".join(inp.tickers) + ")"],
                   [[labels[c], ui.pct(opt.metrics[c]["ret"], 1), ui.pct(opt.metrics[c]["vol"], 1),
                     ui.num(opt.metrics[c]["sharpe"], 2), ui.num(opt.metrics[c]["sortino"], 2), ui.num(opt.metrics[c]["beta"], 2),
                     ui.pct(opt.metrics[c]["max_dd"], 1),
                     " ".join(f"{x * 100:.0f}" for x in (opt.current if c == "current" else opt.weights[c]))]
                    for c in cols]), "", WHERE]
    return "\n".join(out)


def _cum(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


# ---------- rate limits (the site is public, so cap what a visitor can spend) ----------

_limit_lock = threading.Lock()
_by_visitor: dict = {}
_today: dict = {"day": None, "count": 0}


def _allowed(visitor: str) -> str | None:
    """None if this visitor may ask Claude now, else why not."""
    per_min, per_hour = config.AI_LIMIT_PER_VISITOR
    now = time.time()
    with _limit_lock:
        day = time.strftime("%Y-%m-%d")
        if _today["day"] != day:
            _today.update(day=day, count=0)
        if _today["count"] >= config.AI_LIMIT_PER_DAY:
            return "the site's daily AI limit is used up"
        q = _by_visitor.setdefault(visitor, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= per_hour or sum(1 for t in q if now - t < 60) >= per_min:
            return "you're asking faster than the AI limit allows"
        q.append(now)
        _today["count"] += 1
    return None


# ---------- calling Claude ----------

_client: dict = {}


def _anthropic():
    key = config.anthropic_api_key()
    if not key:
        return None
    if _client.get("key") != key:
        import anthropic

        _client.update(key=key, client=anthropic.Anthropic(api_key=key, timeout=config.AI_TIMEOUT, max_retries=1))
    return _client["client"]


def _history(log: list) -> list[dict]:
    """The last few question/answer pairs as API messages (alternating user / assistant, oldest first)."""
    pairs = []
    for i in range(len(log) - 1):
        if log[i]["role"] == "user" and log[i + 1]["role"] == "bot":
            m = log[i + 1]
            answer = m.get("md") or " ".join(x for x in (m.get("value"), m.get("label"), m.get("text")) if x)
            pairs.append((log[i]["text"], answer))
    msgs = []
    for q, ans in pairs[-config.AI_HISTORY_TURNS:]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
    return msgs


def _fallback(base: dict, why: str) -> dict:
    return {**base, "note": f"AI reply unavailable ({why}), so this is the built-in answer."
            + (f" {base['note']}" if base.get("note") else "")}


def reply(question: str, log: list | None = None) -> dict:
    """Claude's answer, with the on-device route's link and chips; the built-in answer if Claude can't reply."""
    started = time.perf_counter()
    base = assistant.answer(question)  # MiniLM route on LinuxONE + template answer (also the fallback)
    if not (question or "").strip() or base.get("intent") == "error":
        return base
    try:
        client = _anthropic()
    except Exception as exc:  # e.g. the SDK isn't installed yet
        return _fallback(base, type(exc).__name__)
    if client is None:
        return base  # no API key configured: the built-in answers are the assistant
    visitor = (request.remote_addr or "?") if has_request_context() else "local"  # no proxy in front of the site
    blocked = _allowed(visitor)
    if blocked:
        return _fallback(base, blocked)

    import anthropic

    try:
        d = assistant.Data()
        resp = client.messages.create(
            model=config.AI_MODEL,
            max_tokens=config.AI_MAX_TOKENS,
            output_config={"effort": config.AI_EFFORT},
            system=[{"type": "text", "text": SYSTEM},
                    {"type": "text", "text": snapshot(d), "cache_control": {"type": "ephemeral"}}],
            messages=_history(log or []) + [{"role": "user", "content": question.strip()[:500]}],
        )
    except anthropic.AuthenticationError:
        return _fallback(base, "the API key was rejected")
    except anthropic.RateLimitError:
        return _fallback(base, "Anthropic rate limit")
    except anthropic.APIStatusError as exc:
        return _fallback(base, f"API error {exc.status_code}")
    except anthropic.APIConnectionError:
        return _fallback(base, "couldn't reach the API")
    except Exception as exc:
        return _fallback(base, type(exc).__name__)

    if resp.stop_reason == "refusal":
        return _fallback(base, "the model declined")
    text = "".join(blk.text for blk in resp.content if blk.type == "text").strip()
    if not text:
        return _fallback(base, "empty reply")
    if resp.stop_reason == "max_tokens":
        text += " …"
    routed = base.get("score") is not None and base["score"] >= config.ASSISTANT_THRESHOLD and base.get("link")
    return {"md": text, "value": None, "label": None, "text": text, "tone": "", "rows": None, "note": None,
            "link": base.get("link") if routed else None, "chips": base.get("chips") or assistant.DEFAULT_CHIPS,
            "intent": base.get("intent"), "title": base.get("title"), "score": base.get("score"),
            "model": config.AI_MODEL_LABEL, "ms": round((time.perf_counter() - started) * 1000),
            "usage": {"in": resp.usage.input_tokens, "cached": resp.usage.cache_read_input_tokens or 0,
                      "out": resp.usage.output_tokens}}
