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

# Company Research page: earnings press releases from SEC EDGAR (any US-listed company), scored by the
# FinBERT model from Hugging Face running locally on LinuxONE. Results are stored so pages open instantly.
RESEARCH_DB = DATA_DIR / "research" / "edgar.db"     # git-ignored
MODELS_DIR = DATA_DIR / "models"                     # git-ignored Hugging Face cache (HF_HOME)
FINBERT_MODEL = "ProsusAI/finbert"
NLP_THREADS = 3               # CPU threads for FinBERT (the VM has 3 cores)
NLP_MAX_SENTENCES = 28        # sentences FinBERT scores per release (keeps a new company to a few seconds)
RESEARCH_RELEASES = 4         # latest earnings releases kept per company
RESEARCH_KEY_POINTS = 8
RESEARCH_REFRESH_HOURS = 24   # re-check EDGAR for new releases after this long
SEC_MIN_INTERVAL = 0.12       # seconds between SEC requests (SEC allows 10 per second)


def _env_file_value(name: str) -> str | None:
    """A variable from the environment, or from .env (read live, so no restart is needed)."""
    if os.environ.get(name):
        return os.environ[name].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def sec_user_agent() -> str | None:
    """SEC requires a User-Agent with a contact email on every automated request (SEC_USER_AGENT in .env)."""
    return _env_file_value("SEC_USER_AGENT")
