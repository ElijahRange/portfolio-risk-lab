Put the team's real portfolio here as `portfolio.xlsx` (or `portfolio.csv`).
Everything in this folder except this note is git-ignored, so holdings never reach GitHub.

Required columns (header names are case-insensitive; first match wins):
- ticker: `ticker`, `symbol`, `security` or `stock` (Yahoo Finance symbols, e.g. BRK-B)
- quantity: `quantity`, `shares`, `qty`, `position` or `units` (negative = short)
- sector (optional): `sector`, `gics sector` or `industry`

A row with ticker `CASH` holds cash; its quantity is the dollar amount.
