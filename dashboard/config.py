"""Project-wide settings. Change factor proxies, lookbacks and file locations here."""
import os
from pathlib import Path

APP_TITLE = "Portfolio Risk Lab"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRIVATE_DIR = DATA_DIR / "private"  # git-ignored: put the team's real portfolio here
CACHE_DIR = DATA_DIR / "cache"      # git-ignored: downloaded price history

# First file that exists wins. The sample is only a placeholder until the real file is added.
PORTFOLIO_FILES = [
    PRIVATE_DIR / "portfolio.xlsx",
    PRIVATE_DIR / "portfolio.csv",
    DATA_DIR / "sample_portfolio.csv",
]

# Rows with these tickers are treated as cash (quantity = dollars).
CASH_TICKERS = {"CASH", "$CASH", "USD"}

# When the portfolio file gives a `weight` column instead of `quantity`, weights are
# relative (any scale) and positions are sized so gross exposure equals this amount.
PORTFOLIO_AUM = 1_000_000

BENCHMARK = "SPY"  # used when there is no Bloomberg file

# The team's Bloomberg export (git-ignored). When present it is the price source for the
# holdings and the benchmark; Yahoo is still used for factor ETFs and FX rates.
BLOOMBERG_FILE = PRIVATE_DIR / "IMBZportfolio_hist_data.xlsx"
BLOOMBERG_PRICE_SHEET = "Sheet3"
BLOOMBERG_BENCHMARK = "SPX"
# Bloomberg security -> ticker used in the app. Anything else just drops " US Equity".
BLOOMBERG_TICKER_MAP = {"SPX Index": "SPX", "RELIANCE IN Equity": "RELIANCE.NS"}
# Non-USD tickers -> Yahoo FX symbol quoting local currency per 1 USD.
LOCAL_CURRENCY_FX = {"RELIANCE.NS": "INR=X"}

# Factor name -> ETF proxy. Every factor except the market is orthogonalised to the
# market before the regression, so its beta reads as exposure *beyond* market beta.
STYLE_FACTORS = {
    "Market": "SPY",
    "Size": "IWM",
    "Value": "VLUE",
    "Quality": "QUAL",
    "Momentum": "MTUM",
    "Low Vol": "USMV",
}
SECTOR_FACTORS = {
    "Tech": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Cons. Disc.": "XLY",
    "Industrials": "XLI",
}
FACTORS = {**STYLE_FACTORS, **SECTOR_FACTORS}

TRADING_DAYS = 252
LOOKBACK_DAYS = 252        # window for betas, volatility, VaR and correlations
ROLLING_BETA_DAYS = 63     # about one quarter
PRICE_HISTORY_RANGE = "2y"
PRICE_CACHE_MAX_AGE_HOURS = 12
MIN_HISTORY_COVERAGE = 0.6  # drop tickers with less than 60% of the lookback window

# Ratio Testing page (portfolio optimizer + efficient frontier).
RISK_FREE_RATE = 0.04            # annual; default for Sharpe/Sortino, editable on the page
OPT_LOOKBACKS = {"1Y": 252, "3Y": 756, "Full": None}  # None = all history in the price file
FRONTIER_POINTS = 30

# Company Research page: earnings calls from Alpha Vantage, analysed once and stored locally.
RESEARCH_DB = DATA_DIR / "research" / "research.db"  # git-ignored: licensed transcript text
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_PAUSE = 1.2      # seconds between requests (free plan rate limit)
RESEARCH_QUARTERS = 4         # latest calls fetched per company
RESEARCH_KEY_POINTS = 8       # key sentences kept per call
TONE_GOOD, TONE_BAD = 0.5, 0.2  # Alpha Vantage turn sentiment (0 to 1) thresholds for colours


def alphavantage_key() -> str | None:
    """ALPHAVANTAGE_API_KEY from the environment, or from .env (read live, so no restart is needed)."""
    if os.environ.get("ALPHAVANTAGE_API_KEY"):
        return os.environ["ALPHAVANTAGE_API_KEY"].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ALPHAVANTAGE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None
