# PRISM (Portfolio Risk Insights, Stress-testing, and Monitoring)

A chart-heavy portfolio risk dashboard for the IBM Z Datathon, running on an
IBM LinuxONE (s390x) virtual server.

- **Live site:** http://148.100.98.18:8050 (public, no login)
- **Stack:** Python 3.12 · Dash 4 · Plotly 7 · pandas · NumPy · Gunicorn · Podman

**New here (or using an AI coding agent)? Start with [AGENTS.md](AGENTS.md).**

## How the team works on it

GitHub `main` is the source of truth. The server pulls `main` every minute and the site
reloads by itself, so **push to `main` and the live site updates within about a minute.**

```bash
git clone https://github.com/ElijahRange/portfolio-risk-lab.git
cd portfolio-risk-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-local.txt
export SEC_USER_AGENT="PRISM your.email@example.com"   # for Company Research
python app.py                           # http://127.0.0.1:8050 on your laptop
```

Then commit and push (`git pull --rebase` first if someone else pushed).

## Project layout

| Path | What it is |
| --- | --- |
| `app.py` | Page shell (sidebar, header, tabs), routing and callbacks |
| `dashboard/analytics.py` | All risk maths: betas, factor model, VaR, risk contributions, PCA |
| `dashboard/data.py` | Reads the portfolio spreadsheet; downloads and caches daily prices |
| `dashboard/views/` | One file per page/tab: `overview.py`, `factors.py`, `idio.py`, `risk.py`, `ratio_testing.py`, `research.py`, `notes.py` |
| `dashboard/optimizer.py` | Ratio Testing: SciPy portfolio optimizer and efficient frontier |
| `dashboard/edgar.py`, `finbert.py`, `research_store.py` | Company Research: SEC EDGAR client, Hugging Face FinBERT, analysis + SQLite store |
| `dashboard/components.py` | KPI tiles, cards, alerts, tables, number formatting |
| `dashboard/theme.py`, `assets/style.css` | Colours, fonts and the Plotly chart theme |
| `dashboard/config.py` | Factor ETF proxies, lookback windows, file locations |
| `dashboard/s390x_fix.py` | Fixes Plotly charts on big-endian IBM Z (see the file) |
| `data/sample_portfolio.csv` | Made-up placeholder holdings |
| `data/private/` | The real portfolio goes here (git-ignored) |
| `scripts/deploy.sh` | (Re)starts the dashboard container on the server |
| `scripts/sync.sh` | Pulls GitHub `main` onto the server (runs every minute) |

## Adding the real portfolio

Copy the spreadsheet to the server as `data/private/portfolio.xlsx` (see
`data/private/README.md` for the expected columns). It is git-ignored on purpose, so
holdings never land on GitHub:

```bash
scp portfolio.xlsx datathon:portfolio-risk-lab/data/private/portfolio.xlsx
```

## Server operations (on the LinuxONE server)

```bash
cd ~/portfolio-risk-lab
podman logs -f dashboard          # live logs
scripts/deploy.sh                 # restart the dashboard
scripts/deploy.sh --install       # after changing requirements.txt
```
