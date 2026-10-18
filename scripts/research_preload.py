"""Look companies up ahead of time so the Company Research page opens them instantly (e.g. before a demo).

Run inside the dashboard container, from the repo root:
    podman exec dashboard .venv/bin/python scripts/research_preload.py              # every portfolio stock
    podman exec dashboard .venv/bin/python scripts/research_preload.py AAPL NVDA    # specific tickers
Waits until every release is analysed. Non-US companies (e.g. RELIANCE.NS) are skipped with a message.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import config, data, research_store  # noqa: E402


def main():
    tickers = sys.argv[1:] or [t for t in data.load_holdings()["ticker"] if t not in config.CASH_TICKERS]
    for t in tickers:
        started = time.time()
        try:
            symbol = research_store.ensure(t)
            research_store._process_pending(symbol)  # finish the older releases now instead of in the background
            c = research_store.company(symbol)
            done = sum(r["status"] == "done" for r in c["releases"])
            print(f"{t:<12} {c['name']}: {done}/{len(c['releases'])} releases in {time.time() - started:.0f}s")
        except Exception as exc:
            print(f"{t:<12} skipped: {exc}")


if __name__ == "__main__":
    main()
