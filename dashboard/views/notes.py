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
            html.Li("VaR (95%): the 1-day loss exceeded on only 5% of days in the window. "
                    "CVaR is the average loss on those worst days."),
            html.Li("Risk contribution: each position's share of portfolio variance, using the historical "
                    "covariance of returns."),
            html.Li("Risk geometry: principal component analysis (PCA) of the positions' return correlations; "
                    "each axis is named after the factor ETFs it tracks most closely."),
        ], className="notes")),
        ui.card("Where this runs", html.Ul([
            html.Li(f"Machine architecture: {platform.machine()} (IBM LinuxONE / IBM Z)"),
            html.Li(f"Python {platform.python_version()} in a Podman container"),
        ], className="notes")),
    ])
