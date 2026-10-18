# Portfolio Risk Lab: notes for Claude

Team-wide guide (also read by Codex and other agents): `AGENTS.md`. Keep both in sync.

IBM Z Datathon project (team of four). A Dash dashboard that analyses the team's own
stock portfolio. It runs on an IBM LinuxONE VM (RHEL 9.8, **s390x, big-endian**).

## Rules

- **Design reference only.** The look (dark theme, left sidebar, tabbed portfolio page,
  KPI tiles, cards, mini-bar tables, diverging factor-beta bars, 3D PCA "risk geometry")
  is modelled on a private research portal from a teammate's employer. Don't copy that
  portal's code, data, figures, proprietary scoring, name or branding. Rebuild everything
  from public, textbook methods.
- **Real holdings stay private.** They live in `data/private/` (git-ignored). Never
  commit them, print them in logs or paste them into issues.
- `.env` holds `SEC_USER_AGENT`. It is git-ignored; never commit or echo it.
- The site is public with no login (the team chose this: low-stakes hackathon project).

## Architecture

- `app.py`: shell, routing (`/`, `/notes`) and callbacks. Gunicorn serves `app:server`.
- `dashboard/service.py`: `current_analysis()` caches one `Analysis` per
  (portfolio file + mtime, price-cache stamp). Every view gets data from here.
- `dashboard/analytics.py`: `build()` returns the `Analysis` dataclass. Today's holdings
  are held constant over `LOOKBACK_DAYS`. Contents:
  - single-index betas to SPY
  - multi-factor OLS on ETF proxies, with non-market factors orthogonalised to the market
  - factor vs idiosyncratic variance split
  - historical-covariance risk contributions
  - historical VaR/CVaR, drawdown, effective N
  - PCA of position return correlations
  - idiosyncratic risk (`Analysis.idio`): single-index market model of every name and the
    book on the benchmark (SPX); systematic vs idiosyncratic return/variance, Newey-West
    t-stats, residual covariance contributions and correlations, rolling 63-day split over
    the full price history. Shown in the "Idiosyncratic Risk" tab (`views/idio.py`), whose
    row-click detail panel is wired up in `app.py` (`select_idio_name`).
- `dashboard/optimizer.py`: maths for the Ratio Testing page. `run()` estimates returns, covariance and
  betas from `Analysis.history` (full daily return history, holiday rows dropped) over a chosen lookback,
  then solves six objectives with SciPy SLSQP (max Sharpe, min vol, min idiosyncratic risk, max Sortino,
  risk parity, max diversification), the efficient frontier and a random-portfolio cloud. Results are
  cached per settings. Page: `views/ratio_testing.py` at `/ratio-testing`; callbacks in `app.py`.
- Company Research (`/research`, `views/research.py`) works for **any US-listed company**:
  - `dashboard/edgar.py`: SEC EDGAR client (free, no key; needs `SEC_USER_AGENT` with a contact email in
    `.env`, max 10 req/s). Ticker/name lookup, latest earnings 8-Ks (Item 2.02), the EX-99.1 press release,
    and quarterly revenue / net income / EPS from XBRL company facts.
  - `dashboard/finbert.py`: Hugging Face `ProsusAI/finbert` run locally on LinuxONE (CPU, 3 threads),
    cached in `data/models` (git-ignored). Page requests get priority over background work.
  - `dashboard/research_store.py`: SQLite at `data/research/edgar.db` (git-ignored). `ensure(query)`
    analyses the newest release at once (about 8 s for a new company) and older ones in a background
    thread; the page polls until they finish. Stored companies open in well under a second.
  - Preload demo companies: `podman exec dashboard .venv/bin/python scripts/research_preload.py [TICKERS]`.
  - Limits: press releases, not call transcripts (no analyst Q&A); foreign filers on Form 6-K
    (e.g. RELIANCE) aren't covered.
- Portfolio assistant: a chat bubble fixed bottom-right on **every page**. **Claude writes the replies**
  (`dashboard/ai_chat.py`); the on-device MiniLM router below picks the related chart link and follow-up chips and is
  the fallback.
  - `ai_chat.reply(question, log)`: runs `assistant.answer()` first (route + fallback), then calls the Anthropic API
    (`config.AI_MODEL` = `claude-haiku-5-5`, effort `low`, ~1–3 s) with a system prompt plus `snapshot(d)`, a ~2.5K-token
    text dump of every number the dashboard shows (positions, sector aggregates, KPIs, VaR, stress, factors,
    correlations, optimizer table, FinBERT tones, where each chart lives). The snapshot is deterministic so prompt
    caching hits (~4.9K cached tokens per question). The last 3 Q&A pairs go along for follow-ups.
  - Key: `ANTHROPIC_API_KEY` in `.env` (read live via `config.anthropic_api_key()`; never print it). No key, API error,
    refusal or rate limit → the built-in template answer with a short note. Limits (public site): 6/min and 60/hour
    per IP, 1,500/day site-wide (`config.AI_LIMIT_*`, in-memory per worker).
  - The `anthropic` SDK is in `requirements.txt` (pure Python; httpx2/jiter have s390x wheels).
  - Replies are markdown (`dcc.Markdown`, no raw HTML). A clientside callback shows the question + typing dots
    while the server works; it must not also listen to `chat-log` (Dash defers callbacks whose inputs a running callback
    will change), so a second callback hides it.
  - Widget: `views/chat.py`, mounted in `app.layout` outside `html.Main(id="page")`, so it survives
    navigation. Callbacks are in `app.py` (`chat_ask`, `chat_render`, plus clientside open/close, typing and
    scroll); `assets/chat.js` adds Escape-to-close and tab switching for its links.
- The built-in assistant (router for the reply's chart link and chips, and the fallback answer):
  - `dashboard/assistant.py`: deterministic on purpose. Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
    (`config.ASSISTANT_MODEL`, loaded with `transformers` `AutoModel` + mean pooling, cached in `data/models`)
    embeds the question; the closest example phrasing (cosine) picks one of ~30 intents in `INTENTS`, plus a
    small bonus for telltale words. Below `config.ASSISTANT_THRESHOLD` it lists what it can answer instead.
  - Regex code pulls out tickers/company names (`COMPANIES` aliases), VaR level, stress scenario, S&P shock
    size and optimizer goal; names are masked to "the stock" and shorthand (VaR, P&L, NAV…) spelled out
    before embedding.
  - Every answer is a template filled from the same objects the pages use (`service`, `risk_metrics.report`,
    `optimizer.run` with default `Settings()`, `research_store` reads only, never an EDGAR fetch), so the chat
    always matches the charts. Headline values use the same `ui.pct`/`ui.usd`/`ui.num` formatting as the tiles.
  - ~10–20 ms per answer once warm. `assistant.warm_up()` loads the model and fills the analysis, risk-report
    and optimizer caches at start-up, in the **same thread as FinBERT's warm-up** (importing transformers from
    two threads at once fails with `cannot import name 'AutoModel'`).
  - `scripts/assistant_check.py` checks intent accuracy on unseen paraphrases (≥ 90%), that headline numbers
    equal their source, off-topic questions are declined, answers repeat exactly, and latency. Run it after
    changing intents: `podman exec dashboard .venv/bin/python scripts/assistant_check.py -v`.
  - To add a question type: write a handler `h_x(q, d)` returning `_ans(...)`, add an `Intent` with 5–8
    example phrasings, then add paraphrases to the check script.
  - Portfolio tabs can be deep-linked: `/?tab=overview|factors|idio|risk` (`route()` reads `url.search`).
- `dashboard/data.py`: reads the portfolio file (flexible column aliases; `CASH` row =
  dollars; negative quantity = short). Prices come from Yahoo's public chart endpoint and
  are cached per ticker as CSV in `data/cache/` for 12 hours.
- `dashboard/views/*.py`: each exposes `layout(analysis)`. Adding a tab means adding a
  view and registering it in `TABS` in `app.py`.
- Styling: CSS tokens in `assets/style.css` and the Plotly template in
  `dashboard/theme.py`. Keep the colours in sync. Numbers use the mono font.
- `dashboard/s390x_fix.py` **must stay imported** (theme.py imports it). Plotly encodes
  arrays as base64 in native byte order and plotly.js reads them as little-endian, so
  without the fix every chart on s390x shows garbage.

## Runtime on the server

- SSH alias on Elijah's Mac: `ssh datathon` (user `linux1`, repo at `~/portfolio-risk-lab`).
- Container `dashboard` (podman, rootless, `--network host`, `--restart always`).
- The container uses the event's Jupyter image
  `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x`. That image already has
  numpy, pandas, scipy, sklearn, TF and torch built for s390x.
- Extra packages from `requirements.txt` go in `./.venv`, which sees the image's
  `/opt/venv` through a `.pth` file.
- The repo is mounted at `/app`. `gunicorn --reload` means code edits go live with no
  restart.
- `podman build` / `podman commit` on top of that image fail (podman 5.8 bug: "unsupported
  MIME type for compression"). That's why there is no Containerfile. Use
  `scripts/deploy.sh [--install]`.
- Port 8050 is opened in iptables. Rules persist via `/etc/sysconfig/iptables.save`,
  which `/etc/rc.d/rc.local` restores at boot.
- Jupyter Lab (the event's lab environment) runs separately as container `notebook` on
  port 38888.
- `loginctl enable-linger linux1` plus `podman-restart.service` keep both containers up
  after logout and reboot.
- The systemd user timer `dashboard-sync.timer` runs `scripts/sync.sh` every minute. It
  fast-forwards to `origin/main`, and reinstalls the venv if `requirements.txt` changed.

## Workflow

- Verify changes before pushing:
  ```bash
  podman exec dashboard .venv/bin/python -c \
    "from dashboard import service; from dashboard.views import overview, factors, risk; \
     a = service.current_analysis(); [v.layout(a) for v in (overview, factors, risk)]"
  ```
- Commit small, descriptive changes to `main`. Teammates push from laptops; the server
  pulls.

## Team data file: `data/private/IMBZportfolio_hist_data.xlsx`

This is Bloomberg Terminal export data. It is git-ignored; never commit it or publish
the raw file. The file has **prices only, no position sizes**.

- **Sheet3** (main price sheet):
  - Daily `PX_LAST`, `PX_OPEN`, `PX_VOLUME`, `BETA_RAW_OVERRIDABLE` for VZ, MMM, SPX Index,
    RELIANCE IN, WMT, TPB, TGT, CB, BA and PM.
  - 1,448 rows from 2021-03-31 to 2026-10-16.
- **Sheet1**: the same names, with `VOLUME_TOTAL_CALL`, `VOLUME_TOTAL_PUT`,
  `PUT_CALL_OPEN_INTEREST_TOTAL`, `PUT_CALL_VOLUME_RATIO_CUR_DAY` and `MOV_AVG_200D`.
- **Sheet2**: `SPXL1 Index`, with price, volume, beta and put/call fields. It has 1,394 rows
  and no holiday rows. Its prices match SPX.
- **Layout:** each sheet has a 2-row preamble (`Start Date`/`End Date`). Then comes a
  security-name row, a long-name row, and a field-code row whose first cell is `Dates`.
  Each security spans N columns; the name appears only in its first column.
- **Quirks:**
  - US/India market holidays are forward-filled (e.g. 2021-04-02 repeats 2021-04-01), so
    drop repeated rows before computing returns.
  - Sheet1 `PUT_CALL_OPEN_INTEREST_TOTAL` is all `#N/A N/A` for SPX and RELIANCE IN.
  - Sheet2 lists `VOLUME_TOTAL_PUT` twice.
  - RELIANCE IN is priced in **INR** on India's NSE (Yahoo: `RELIANCE.NS`). Convert it to
    USD (e.g. USDINR) before mixing it with USD names.
  - Bloomberg tickers map to Yahoo tickers by dropping " US Equity". SPX Index = `^GSPC`.

## Status (2026-10-17)

- Live: dashboard on the team's data, GitHub repo + auto-sync, team invites sent.
- **Portfolio:** `data/private/portfolio.csv` on the server (git-ignored) holds the team's real position
  sizes since 2026-10-17: the `weight` column is each position's dollar amount, and they sum to
  `PORTFOLIO_AUM = 1_000_000`, so sizing is exact. The old equal-weight file is kept as
  `portfolio_equal_weight_backup.csv.bak`.
  - The team may later switch to share quantities and live API prices; don't build that until asked.
- **Prices:**
  - `data.load_bloomberg_prices()` reads Sheet3 `PX_LAST` and is the price source for the
    holdings and the SPX benchmark (`service._prices`).
  - Yahoo is still used for the factor ETFs, for `INR=X` (RELIANCE.NS is converted to
    USD) and for any holding missing from the file.
- **Verified:**
  - The file's prices match Yahoo to the cent.
  - Negative 1-year betas for VZ, CB and PM are real; Yahoo gives the same.
- **Portfolio assistant** (chat bubble, every page) added 2026-10-17: 97% intent accuracy on the check script's
  145 unseen paraphrases, all headline numbers equal to the dashboard's, ~12 ms median answer.
- 2026-10-18: chat replies now written by Claude Haiku 5.5 (Elijah's choice: real AI answers to free-form
  questions); the MiniLM assistant stays as router + fallback. Key in the server `.env`.
- Options and 200-day MA fields (Sheet1) are unused so far. They're a natural input for
  the ML feature.

## Ideas not built yet

- Upload a new portfolio snapshot from the UI (`dcc.Upload`) instead of `scp`.
- More tabs: per-position detail, sector exposure, stress tests / scenario shocks,
  forecast risk (e.g. EWMA or GARCH volatility).
- An ML component that runs on LinuxONE (judging criterion), e.g. a volatility or
  drawdown forecaster trained in the Jupyter container and exported to ONNX.
- Swap the unofficial Yahoo endpoint for a documented free data API if it rate-limits.
