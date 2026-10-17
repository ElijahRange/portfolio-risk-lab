Private data. Everything in this folder except this note is git-ignored, so it never
reaches GitHub.

`portfolio.xlsx` or `portfolio.csv` holds the holdings (first one found wins). Header
names are case-insensitive:
- ticker: `ticker`, `symbol`, `security` or `stock`
- size, either:
  - `quantity` / `shares` / `qty` / `position` / `units`: shares held (negative = short)
  - `weight` / `weights` / `allocation`: relative weights of any scale (negative = short).
    Positions are sized so gross exposure = `PORTFOLIO_AUM` in dashboard/config.py.
    Equal weight = put 1 on every row.
- sector (optional): `sector`, `gics sector` or `industry`

A row with ticker `CASH` holds cash (dollars if `quantity`, a weight if `weight`).

`IMBZportfolio_hist_data.xlsx` is the team's Bloomberg export. When present, it supplies
the prices for the holdings and the SPX benchmark (see `config.BLOOMBERG_*`).
