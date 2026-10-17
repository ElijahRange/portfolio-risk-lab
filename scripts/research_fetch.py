"""Fill the Company Research database ahead of time (e.g. before a demo).

Run inside the dashboard container, from the repo root:
    podman exec dashboard .venv/bin/python scripts/research_fetch.py               # every portfolio stock
    podman exec dashboard .venv/bin/python scripts/research_fetch.py AAPL MSFT     # specific tickers
    podman exec dashboard .venv/bin/python scripts/research_fetch.py IBM --only 2024Q1 2023Q4
    podman exec dashboard .venv/bin/python scripts/research_fetch.py --reanalyse   # redo NLP, no API calls

Calls already stored are skipped, so it is safe to re-run after the free plan's daily limit resets.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import config, data, transcripts  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="defaults to the stocks in the portfolio file")
    p.add_argument("--quarters", type=int, default=config.RESEARCH_QUARTERS, help="latest calls per company")
    p.add_argument("--only", nargs="+", metavar="YYYYQN", help="fetch just these fiscal quarters")
    p.add_argument("--reanalyse", action="store_true", help="re-run the NLP on stored calls only")
    args = p.parse_args()

    if args.reanalyse:
        transcripts.reanalyse_all()
        print("Re-analysed every stored call.")
        return
    tickers = args.tickers or [t for t in data.load_holdings()["ticker"] if t not in config.CASH_TICKERS]
    for t in tickers:
        r = transcripts.ingest(t, quarters=args.quarters, only=args.only)
        print(f"{r['symbol']:<12} fetched {r['fetched'] or '-'}  already had {r['skipped'] or '-'}  "
              f"no transcript {r['missing'] or '-'}" + (f"\n             STOPPED: {r['error']}" if r["error"] else ""))
        if r["error"] and ("limit" in r["error"].lower() or "rate" in r["error"].lower()):
            print("Alpha Vantage limit reached; run this again later to continue.")
            break


if __name__ == "__main__":
    main()
