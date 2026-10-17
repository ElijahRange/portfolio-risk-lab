# Portfolio Risk Lab: notes for Claude

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
- `.env` holds the site login (`DASH_USERS`). It is git-ignored; never commit or echo it.
- The site is on a public IP, so `app.py` refuses to start without `DASH_USERS` unless
  `DASH_AUTH_DISABLED=1`. Only use that on 127.0.0.1.

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
  podman exec -e DASH_AUTH_DISABLED=1 dashboard .venv/bin/python -c \
    "from dashboard import service; from dashboard.views import overview, factors, risk; \
     a = service.current_analysis(); [v.layout(a) for v in (overview, factors, risk)]"
  ```
- For a visual check without the password, run a second gunicorn on `127.0.0.1:8051` with
  `DASH_AUTH_DISABLED=1` and SSH-tunnel to it.
- Commit small, descriptive changes to `main`. Teammates push from laptops; the server
  pulls.

## Ideas not built yet

- Upload a new portfolio snapshot from the UI (`dcc.Upload`) instead of `scp`.
- More tabs: per-position detail, sector exposure, stress tests / scenario shocks,
  forecast risk (e.g. EWMA or GARCH volatility).
- An ML component that runs on LinuxONE (judging criterion), e.g. a volatility or
  drawdown forecaster trained in the Jupyter container and exported to ONNX.
- Swap the unofficial Yahoo endpoint for a documented free data API if it rate-limits.
