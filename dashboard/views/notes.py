"""Model notes page: plain-English methodology for teammates and judges."""
import platform

from dash import html

from .. import components as ui
from .. import config


def layout():
    factors = ", ".join(f"{name} ({etf})" for name, etf in config.FACTORS.items())
    return html.Div([
        ui.card("How the numbers are made", html.Ul([
            html.Li("Holdings come from the portfolio spreadsheet; prices are daily adjusted closes."),
            html.Li(f"Every risk number uses the last {config.LOOKBACK_DAYS} trading days and assumes "
                    "today's holdings were held the whole time."),
            html.Li("Beta: how much a stock moved per 1% move in the S&P 500 (SPY)."),
            html.Li(f"Factor model: each stock's returns are regressed on ETF proxies — {factors}. "
                    "Non-market factors are measured beyond the market."),
            html.Li("Idiosyncratic risk: each stock and the whole book are regressed on the S&P 500 (single-index "
                    "market model). Beta × index is the systematic part; alpha + the leftover residual is the "
                    "stock-specific (idiosyncratic) part. R² is the share of variance the index explains. "
                    "t-stats use Newey–West standard errors."),
            html.Li("Ratio testing: SciPy (SLSQP) searches for the weights of the current stocks that best meet a goal "
                    "(max Sharpe, min volatility, min idiosyncratic risk, max Sortino, risk parity, max "
                    "diversification) within the weight limits. The efficient frontier is the lowest-volatility "
                    "mix for each level of expected return. All of it is fit on past returns (in-sample)."),
            html.Li("Company research: for any US-listed company, the last earnings press releases (8-K, Item "
                    "2.02, exhibit 99.1) come from SEC EDGAR and reported revenue/EPS from SEC XBRL data. FinBERT, a "
                    "financial-sentiment model from Hugging Face, runs on LinuxONE and scores the most informative "
                    "sentences and executive quotes as positive, negative or neutral. Key points, guidance and "
                    "'new this quarter' topics (TF-IDF) are picked from those, then stored so pages open instantly."),
            html.Li("Portfolio assistant (the chat bubble on every page): Claude (Anthropic API, Haiku) writes each "
                    "reply from a text snapshot of this dashboard's own numbers (positions, sectors, risk, stress tests, "
                    "factors, correlations, optimizer results, FinBERT tone) and is told to quote only those figures. "
                    "A small sentence-embedding model from Hugging Face (all-MiniLM-L6-v2) runs on LinuxONE alongside "
                    "it: it matches the question to the closest of about 30 question types to pick the related chart "
                    "and follow-up questions, and its template answers take over if the AI is unavailable."),
            html.Li("Editing the portfolio: Portfolio > Edit portfolio changes holdings and dollar amounts (NAV is their "
                    "total) or loads them from a spreadsheet. Every ticker is checked against Yahoo Finance and the "
                    "whole analysis is run on the new portfolio before it is saved; earlier versions can be restored. "
                    "Stocks outside the team's Bloomberg export are priced from Yahoo (5 years of history) and "
                    "converted to US dollars at daily exchange rates."),
            html.Li("VaR (95%): the 1-day loss exceeded on only 5% of days in the window. "
                    "CVaR is the average loss on those worst days."),
            html.Li("Risk contribution: each position's share of portfolio variance, using the historical "
                    "covariance of returns."),
            html.Li("Value at risk: historical (empirical 5% / 1% quantiles of the last 252 days), normal, "
                    "Cornish-Fisher (normal quantile adjusted for skew and excess kurtosis) and EWMA "
                    "(RiskMetrics, λ = 0.94). Expected shortfall 97.5% = average loss beyond the 97.5% quantile."),
            html.Li("VaR backtest: each day's EWMA 95% VaR is forecast from earlier days only; breaches over the "
                    "last 252 days are scored with the Kupiec proportion-of-failures test."),
            html.Li("Stress tests: historical sell-off windows replayed buy-and-hold with today's weights; "
                    "hypothetical S&P 500 shocks are scaled by the portfolio's downside beta (full history)."),
            html.Li("Risk budget: Euler decomposition of portfolio volatility (weight × marginal contribution). "
                    "Diversification ratio = weighted sum of stand-alone vols ÷ portfolio vol."),
            html.Li("Upside / downside beta: OLS slope on S&P 500 up days and down days separately."),
            html.Li("Risk geometry: principal component analysis (PCA) of the positions' return correlations; "
                    "each axis is named after the factor ETFs it tracks most closely and shows its share of "
                    "variance. Risk groups come from Ward hierarchical clustering on the correlation distance "
                    "sqrt((1 - ρ) / 2), about one group per three names; links join the "
                    "most correlated ~30% of pairs (ρ ≥ 0.15)."),
        ], className="notes")),
        ui.card("Where this runs", html.Ul([
            html.Li(f"Machine architecture: {platform.machine()} (IBM LinuxONE / IBM Z)"),
            html.Li(f"Python {platform.python_version()} in a Podman container"),
        ], className="notes")),
    ])
