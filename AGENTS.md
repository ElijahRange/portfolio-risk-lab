# Portfolio Risk Lab: guide for the team and AI coding agents

This file is the starting point for anyone (or any coding agent: Codex, Claude Code, Copilot…)
working on this repo. Read it fully before changing code. `CLAUDE.md` has the same technical
notes in more detail; keep both in sync when you change something important.

## 1. What this is

- **IBM Z Datathon 2026** team project (24 hours, started 2026-10-17), team of four. Elijah Range owns the
  server and the GitHub repo.
- **Rules that shape the work:** everything must run on **IBM LinuxONE** (our VM is an IBM z17,
  `s390x`, big-endian) and must use **AI/ML**. Judges score community value, novelty, technical use
  of LinuxONE + AI/ML, and execution (each out of 10).
- **Deliverable:** a video of at most 3 minutes covering the problem, the data, the solution, and how
  it uses IBM Z (tech stack, LinuxONE instance, model).
- **The product:** "Portfolio Risk Lab", a dark-themed Dash dashboard that analyses the team's real
  9-stock, $1,000,000 portfolio and researches any US company.

**Live site:** http://148.100.98.18:8050 (public, no login).

## 2. What exists now

| Page / tab | Path | What it does | Main code |
| --- | --- | --- | --- |
| Portfolio › Portfolio View | `/` | NAV, exposure, performance vs S&P 500, drawdown, positions | `views/overview.py` |
| Portfolio › Factor Model | `/` | Multi-factor betas (ETF proxies), rolling beta | `views/factors.py` |
| Portfolio › Idiosyncratic Risk | `/` | Single-index model vs SPX: systematic vs stock-specific return/risk, per-name drill-down | `views/idio.py`, `analytics._idiosyncratic` |
| Portfolio › Risk Analysis | `/` | VaR/CVaR, drawdown, 3D PCA "risk geometry", risk contributors, correlations | `views/risk.py` |
| Ratio Testing | `/ratio-testing` | Optimizer (max Sharpe, min vol, min idio risk, max Sortino, risk parity, max diversification) + efficient frontier with custom weights | `optimizer.py`, `views/ratio_testing.py` |
| Company Research | `/research` | Any US stock: earnings press releases from SEC EDGAR, scored by **Hugging Face FinBERT running on LinuxONE**; key points, guidance, tone trend, revenue/EPS, quotes, topics | `edgar.py`, `finbert.py`, `research_store.py`, `views/research.py` |
| Model notes | `/notes` | Plain-English methodology for judges | `views/notes.py` |

**Data:**
- Prices come from the team's Bloomberg export (VZ, MMM, RELIANCE (converted INR→USD), WMT, TPB,
  TGT, CB, BA, PM, plus SPX).
- Real position sizes (dollar amounts, $1,000,000 total) live only in the git-ignored
  `data/private/portfolio.csv` on the server. Ask Elijah for them; never write them into the repo.
- Without those private files (e.g. on your laptop), the app falls back to `data/sample_portfolio.csv`
  and public Yahoo prices.

## 3. How to work on it

GitHub `main` is the source of truth. **The server pulls `main` every minute and the live site reloads
by itself, so every push to `main` goes live within about a minute.** A broken push breaks the demo
site, so always run it locally first.

```bash
git clone https://github.com/ElijahRange/portfolio-risk-lab.git
cd portfolio-risk-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-local.txt
export SEC_USER_AGENT="Portfolio Risk Lab your.email@example.com"   # needed by Company Research
python app.py                                     # http://127.0.0.1:8050
```

- `requirements.txt` is what the server installs. The server image already has numpy, pandas, scipy,
  scikit-learn, torch and transformers built for s390x. `requirements-local.txt` adds those for
  laptops. **Don't add or upgrade torch/transformers in `requirements.txt`**; that would replace the
  IBM Z builds on the server.
- The first Company Research search downloads FinBERT (~440 MB) into `data/models/` (git-ignored).
- Before pushing: open every page you touched, check the browser console and terminal for errors, then
  `git pull --rebase` and push. After pushing, check the live site a minute later.
- Small, descriptive commits. Don't commit anything under `data/` except `sample_portfolio.csv`.
- New dependency? Add it to `requirements.txt` only if it is pure Python or has s390x wheels, and tell
  Elijah: the server must reinstall (`scripts/deploy.sh --install`, run on the server).

**Server access:** only Elijah has SSH to the LinuxONE VM (alias `ssh datathon`). Ask him to run server
commands (logs, restarts, preloading research companies). You don't need SSH for normal work.

## 4. Architecture

- `app.py`: shell (sidebar, header), routing (`PAGES`), the portfolio tabs (`TABS`), and every callback.
  Gunicorn serves `app:server`.
- `dashboard/service.py`: `current_analysis()` caches one `Analysis` per (portfolio file, price data).
  Portfolio views get their data from here.
- `dashboard/analytics.py`: `build()` returns the `Analysis` dataclass.
  - Today's holdings are held constant over the lookback (252 trading days).
  - Contents: betas, factor model (non-market factors orthogonalised to the market), VaR/CVaR, risk
    contributions, PCA, the idiosyncratic-risk block, and the full return history used by the optimizer.
- `dashboard/optimizer.py`: SciPy SLSQP objectives, efficient frontier and random-portfolio cloud,
  cached per settings.
- `dashboard/edgar.py` / `finbert.py` / `research_store.py`: Company Research.
  - EDGAR is free with no key. SEC requires a `User-Agent` with a contact email and ≤10 requests/second.
  - `research_store.ensure(query)` analyses the newest release at once (~8 s for a new company) and
    older ones in a background thread. Results go to SQLite `data/research/edgar.db`, so companies
    already looked up open in under a second.
- `dashboard/views/*.py`: each page/tab exposes `layout(...)`.
  - To add a portfolio tab: write a view, then register it in `TABS` in `app.py`.
  - To add a page: add it to `PAGES` and the sidebar `nav_link`s.
- `dashboard/config.py`: every tunable number (lookbacks, factor proxies, file paths, research limits).

## 5. Rules

- **Private data stays private.** Holdings and the Bloomberg export (licensed data) live in
  `data/private/` (git-ignored). Never commit them, print them in logs, or paste them anywhere public.
- **No secrets in the repo.** `.env` (`SEC_USER_AGENT`) is git-ignored. Use `.env.example`
  to document new variables with fake values.
- **The site is public with no login** (a deliberate choice for this hackathon). Don't add secrets to it.
- **Keep `dashboard/s390x_fix.py` imported** (theme.py does it). Without it every Plotly chart is
  garbage on IBM Z because of byte order.
- **The AI/ML has to run on LinuxONE** for judging. Prefer local models (Hugging Face models via the
  image's torch/transformers, the provided IBM Granite GGUF model via llama.cpp) over cloud AI APIs,
  and never call a model while a page is rendering: precompute and store, like Company Research does.

## 6. Design system (read this: you don't have the original reference)

The look was modelled on a private institutional research portal that only Elijah can access (the
Sabrepoint research portal at his employer). **You can't open it**, so your visual reference is weaker.
To make up for that:
- **Treat the existing pages as the reference.** Copy their structure and reuse the helpers in
  `dashboard/components.py`; don't invent new styles.
- Never copy that portal's name, branding, data, numbers or proprietary methods (it's fine that our
  layout follows the same general dashboard conventions).
- Company Research also borrowed layout ideas from Canary (app.canary-data.com, a commercial research
  product; needs a login): company header with score boxes, document list, colour-coded key points.
- If you need to see the original, ask Elijah for screenshots of the specific screen.

Design language:
- **Dark, dense, quiet.**
  - Background `#0c0e13`, cards `#12151c` with a 1px `#1f2633` border and 2px radius.
  - Muted labels `#6b7686`, body text `#cad0d8`, headings `#e6e9ee`.
  - Accent blue `#3a8ee6` (links, active nav/tab, main series).
  - Green `#22a35a` = positive/long/good, red `#e5484d` = negative/short/bad, amber `#e0a339` =
    warning/secondary series, violet `#9b7ae6` = stock-specific/alternative series, grey `#8a94a6` =
    benchmark.
  - Tokens are in `assets/style.css` (`:root`) and mirrored in `dashboard/theme.py` (`C`). Keep both in sync.
- **Type:** Inter for text, JetBrains Mono for every number (KPI values, table numbers, scores).
- **Page anatomy:**
  - left sidebar (grouped nav: Workspace / Research / Reference)
  - sticky top bar with the page title
  - a page head (title + muted source line)
  - then a row of **KPI tiles** (`ui.kpi_row(ui.kpi(label, value, tone, sub))`)
  - then **cards** (`ui.card(title, body, description, actions)`) arranged in `grid-2` / `grid-3`
- **Tables:** `ui.table(header, rows, numeric_cols)`. Right-aligned mono numbers, `ui.bar_cell` mini
  bars (green right / red left of centre), tickers in mono bold, a hover highlight. Use pills/chips for
  categories.
- **Charts:** Plotly with the `risklab` template (transparent background, faint grid). Height 300–380px
  in a card, 520px for a hero chart. Diverging bars for betas/exposures. Always set hover templates with
  formatted numbers. No mode bar, no scroll-zoom (`ui.graph` handles this).
- **Words:** short card titles, one-sentence muted descriptions that say what the chart means in plain
  English (judges aren't all finance people), and `ui.alert` for warnings/info.
- **Numbers:** `ui.pct`, `ui.usd`, `ui.num`. Signed where direction matters; colour with `ui.tone_class`.

## 7. IBM Z / s390x gotchas (already solved; don't undo)

- Plotly byte order: `s390x_fix.py` (see above).
- `podman build` fails on the event image (podman 5.8 bug), so the server layers `.venv` on the image
  via `scripts/deploy.sh` instead of a Containerfile.
- Hugging Face models load and run correctly on s390x with the image's torch/transformers (tested with
  FinBERT). `transformers` is v5: the `summarization` pipeline was removed. Load seq2seq models
  directly if you need them, but small summarisers took about 60 s per paragraph on this 3-core VM.
  Too slow for live use.
- The provided IBM Granite 4.0 350M (base) model and llama.cpp with the z17's NNPA accelerator are in the
  event's Jupyter image (`/home/ibm-user/*.gguf`). It's good at short extraction prompts (~6 s), bad at
  summaries, and slow on long text.

## 8. Ideas and open items

- **Video and demo:** walk through Portfolio → Idiosyncratic Risk → Ratio Testing → Company Research.
  Search a stock live, e.g. one already preloaded so it's instant.
- **Research:** more tabs per company (10-Q/10-K MD&A and risk factors from EDGAR, insider trades via
  Form 4, news). Guidance in tables isn't extracted yet. Foreign filers on Form 6-K (e.g. RELIANCE)
  aren't covered.
- **ML on LinuxONE:** a volatility or drawdown forecaster using the Bloomberg options and 200-day MA
  fields (Sheet1 of the private export). Or try running FinBERT on the z17 NNPA accelerator via
  `torch_nnpa` (in the image) for the "uses IBM Z AI hardware" story.
- **Portfolio:** upload a portfolio from the UI, live prices, per-position detail, stress tests.
