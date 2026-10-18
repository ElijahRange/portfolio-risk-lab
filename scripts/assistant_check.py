"""Check the portfolio assistant: intent accuracy, numbers vs the dashboard's sources, consistency and speed.

Run on the server (inside the container, from the repo root):
    podman exec dashboard .venv/bin/python scripts/assistant_check.py [-v]

The paraphrases below are deliberately different from the examples in dashboard/assistant.py, so the
accuracy figure measures how well the model generalises, not whether it memorised its own examples.
Exits non-zero if accuracy < 90%, any number disagrees with its source, or answers are slow.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import assistant, optimizer, risk_metrics, service  # noqa: E402
from dashboard import components as ui  # noqa: E402

PARAPHRASES = {
    "help": ["what sort of things can I ask", "what are you able to answer", "give me ideas for questions",
             "how does this assistant work", "help me"],
    "smalltalk": ["hey", "thanks a lot", "hello there", "good afternoon", "thank you so much"],
    "nav": ["what's the total value of our portfolio", "how much are we managing", "current portfolio value",
            "NAV please", "what is the fund's market value"],
    "holdings": ["which companies do we own", "show the holdings", "what positions are in the book",
                 "name all the stocks we hold", "how many names are in the portfolio"],
    "weight": ["how much of the portfolio is in Boeing", "what's the WMT weight", "how large is the Verizon position",
               "what share of the fund is Reliance", "how much Turning Point do we hold"],
    "largest": ["which is our largest stock", "what's the smallest holding", "biggest single position",
                "what stock has the highest weight", "which position is the smallest"],
    "sectors": ["what's our sector mix", "how are we split across sectors", "which sector do we have the most of",
                "industry breakdown", "sector weights"],
    "day_pnl": ["how did we do today", "today's gain or loss", "what was the P&L on the last day",
                "how much money did we make on the day", "how did Chubb do today"],
    "performance": ["what was our one-year return", "have we beaten the market", "how much did we make this year",
                    "what was Walmart's return", "how has the portfolio performed over the past year"],
    "performers": ["which stock did the best", "who were the winners", "which name did the worst",
                   "what was our top performer", "which holding has done worst this year"],
    "volatility": ["how much do our returns fluctuate", "what's the annual vol", "is the portfolio volatile",
                   "what is TGT's volatility", "volatility of the book"],
    "sharpe": ["what's the sharpe", "is our return good for the risk we take", "what Sharpe ratio do we have",
               "sortino ratio", "how good is our risk-adjusted performance"],
    "beta": ["what is our market beta", "how much do we move when the S&P moves", "Chubb beta",
             "what's VZ's beta to the market", "are we sensitive to the market"],
    "var": ["what's the value at risk", "VaR at 99%", "how much could we lose tomorrow", "one day 95% VaR",
            "what's our daily VaR in dollars"],
    "es": ["what's the CVaR", "average loss beyond VaR", "expected shortfall at 97.5%",
           "how much do we lose on a really bad day on average", "what is the ES"],
    "drawdown": ["what's the largest drawdown", "how much did we fall from our peak", "maximum drawdown this year",
                 "worst peak-to-trough decline", "how deep was our worst drawdown"],
    "diversification": ["what's the diversification ratio", "are we well diversified",
                        "how much does mixing stocks reduce our risk", "is our portfolio diversified enough",
                        "how correlated are our holdings on average"],
    "effective_n": ["how concentrated are we", "effective number of stocks", "is the portfolio concentrated",
                    "how many equal-weighted positions is this like", "concentration of the portfolio"],
    "idio": ["how much risk is idiosyncratic", "what percent of variance is stock specific",
             "how much of our risk comes from the market", "systematic vs idiosyncratic risk split",
             "how much of Target's risk is idiosyncratic"],
    "alpha": ["do we have alpha", "what's the portfolio alpha", "is our alpha statistically significant",
              "what's our information ratio", "what alpha does Boeing have"],
    "risk_contrib": ["which position contributes the most risk", "what is our biggest source of risk",
                     "how much risk does TPB contribute", "which stock is riskiest for the portfolio",
                     "risk contribution by stock"],
    "correlation": ["how correlated are BA and WMT", "which stocks are most correlated",
                    "correlation of Verizon and Chubb", "which pair of stocks is least correlated",
                    "what does Target move with"],
    "diversifier": ["what's our best diversifier", "which stock diversifies us the most",
                    "which holding has the lowest correlation to the others", "which stock is the best hedge",
                    "what lowers our risk the most"],
    "stress": ["how did we do during the yen unwind", "what if the market drops 15%", "stress test results",
               "how did we fare in the 2025 tariff selloff", "what is our worst historical scenario"],
    "backtest": ["does VaR pass the backtest", "how many times was VaR breached", "Kupiec result",
                 "is the VaR model reliable", "VaR exceptions"],
    "factors": ["what factors are we exposed to", "what's our momentum beta", "factor tilts",
                "do we lean value or growth", "which factor exposure is largest"],
    "optimizer": ["what's the optimal rebalance", "which weights maximise Sharpe", "how can we cut risk",
                  "what does the optimizer recommend", "what should we buy more of"],
    "research": ["what's the FinBERT tone for Target", "how did Walmart's earnings sound", "earnings tone of Chubb",
                 "sentiment of Boeing's latest results", "what's the earnings sentiment of our holdings"],
    "summary": ["tell me about Verizon", "how is Walmart doing", "TPB", "give me an overview of Chubb",
                "what about Boeing"],
}


OFF_TOPIC = ["what's the weather in Dallas", "who won the super bowl", "write me a poem", "what is the capital of France",
             "tell me a joke", "how do I cook pasta"]


def check_numbers() -> list[str]:
    """The headline number of key answers must equal the dashboard's own value, formatted the same way."""
    a = service.current_analysis()
    rep = risk_metrics.report(a)
    opt = optimizer.run(a.history, a.bench_returns.name, a.positions.set_index("ticker")["weight"], optimizer.Settings())
    hist = rep.var_methods.loc["Historical"]
    p = a.positions.set_index("ticker")
    worst = rep.stress[rep.stress["kind"] != "Hypothetical"].sort_values("port").iloc[0]
    top_risk = rep.budget["risk_share"].idxmax()
    bear = rep.stress[rep.stress["scenario"] == "2022 bear market"].iloc[0]
    shock = rep.stress[rep.stress["scenario"] == "S&P 500 -20%"].iloc[0]
    expected = {
        "what is our NAV": ui.usd(a.nav),
        "what is our 1-day VaR": ui.usd(hist["var95"] * a.nav),
        "what is the 99% VaR": ui.usd(hist["var99"] * a.nav),
        "what is the expected shortfall": ui.usd(hist["es975"] * a.nav),
        "what is our volatility": ui.pct(a.kpis["ann_vol"], 1),
        "what's the portfolio beta": ui.num(a.idio.kpis["beta"], 3, signed=True),
        "what is Boeing's beta": ui.num(p.loc["BA", "beta"], 2),
        "what is the weight of Walmart": ui.pct(p.loc["WMT", "weight"], 1),
        "what's today's P&L": ui.usd(a.kpis["day_pnl"], signed=True),
        "what's our max drawdown": ui.pct(a.kpis["max_drawdown"], 1),
        "what's the diversification ratio": f"{rep.div_ratio:.2f}×",
        "effective number of positions": f"{a.kpis['effective_n']:.1f}",
        "what is our Sharpe ratio": ui.num(opt.metrics["current"]["sharpe"], 2),
        "which stock adds the most risk": ui.pct(rep.budget.loc[top_risk, "risk_share"], 1),
        "what was the worst sell-off": ui.usd(worst["port"] * a.nav, signed=True, compact=True),
        "how did we do in the 2022 bear market": ui.usd(bear["port"] * a.nav, signed=True, compact=True),
        "what happens if the S&P 500 falls 20%": ui.usd(shock["port"] * a.nav, signed=True, compact=True),
        "correlation between Boeing and Walmart": ui.num(a.pca_corr.loc["BA", "WMT"], 2),
        "how much of our risk is stock-specific": ui.pct(a.idio.kpis["idio_share"], 1),
        "VaR backtest": f"{rep.kupiec['breaches']} / {rep.kupiec['n']}",
    }
    bad = []
    for question, want in expected.items():
        got = assistant.answer(question)["value"]
        if got != want:
            bad.append(f"  {question!r}: got {got!r}, source says {want!r}")
    return bad


def main(verbose: bool) -> int:
    assistant.warm_up()
    assistant.answer("warm-up")  # loads the analysis, risk report and optimizer caches
    ok, total, misses, times = 0, 0, [], []
    for want, questions in PARAPHRASES.items():
        for q in questions:
            t = time.perf_counter()
            out = assistant.answer(q)
            times.append((time.perf_counter() - t) * 1000)
            total += 1
            if out["intent"] == want:
                ok += 1
            else:
                misses.append(f"  {q!r}: expected {want}, got {out['intent']} ({out['score']})")
            if verbose:
                print(f"{out['intent'] or '-':>15} {out['score'] or 0:.2f}  {q}  ->  {out['value']} | {out['label']}")
    acc = ok / total
    print(f"Intent accuracy: {ok}/{total} = {acc:.1%}")
    print("\n".join(misses) or "  (no misses)")

    stray = [f"  {q!r}: answered as {out['intent']} ({out['score']})" for q in OFF_TOPIC
             if (out := assistant.answer(q))["intent"] is not None]
    print(f"Off-topic questions declined: {len(OFF_TOPIC) - len(stray)}/{len(OFF_TOPIC)}")
    print("\n".join(stray))

    bad = check_numbers()
    print(f"Numbers vs source: {'all match' if not bad else f'{len(bad)} mismatches'}")
    print("\n".join(bad))

    same = all(assistant.answer(q)["text"] == assistant.answer(q)["text"] for q in ("what is our VaR", "tell me about BA"))
    print(f"Same question, same answer: {same}")

    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"Latency: median {statistics.median(times):.0f} ms, 95th percentile {p95:.0f} ms, max {max(times):.0f} ms")
    return 0 if acc >= 0.9 and not bad and not stray and same and p95 < 300 else 1


if __name__ == "__main__":
    sys.exit(main("-v" in sys.argv))
