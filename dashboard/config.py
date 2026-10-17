"""Project-wide settings. Change factor proxies, lookbacks and file locations here."""
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

BENCHMARK = "SPY"

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
