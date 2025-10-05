"""Background daemon to refresh metrics continuously."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

from persistence import StockRepository, append_delisted, read_delisted
from main import fetch_one


class SimpleDaemon:
    def __init__(self, tickers: List[str], sleep_seconds: int = 60, db_path: str = "stocks.db") -> None:
        self.tickers = tickers
        self.sleep_seconds = sleep_seconds
        self.repo = StockRepository(db_path)
        self.consecutive_failures = 0

    def run(self) -> None:
        print(f"Starting daemon for {len(self.tickers)} tickers")
        print(f"Sleep interval: {self.sleep_seconds} seconds")

        for ticker in self.tickers:
            if not self._fetch_and_save(ticker):
                break
            time.sleep(self.sleep_seconds)

        while True:
            try:
                oldest = self.repo.get_oldest_ticker() or self.tickers[0]
                self._fetch_and_save(oldest)
                time.sleep(self.sleep_seconds)
            except KeyboardInterrupt:
                print("\nDaemon stopped")
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Error: {exc}")
                time.sleep(self.sleep_seconds)

    def _fetch_and_save(self, ticker: str) -> bool:
        """Fetch and save ticker. Returns True if successful, False if rate limited."""
        result = fetch_one(ticker)

        # Check for rate limit errors
        if result.errors:
            error_msgs = [e.message.lower() for e in result.errors]

            # Detect rate limiting
            if any('rate' in msg or 'limit' in msg or '429' in msg or 'too many' in msg
                   for msg in error_msgs):
                self.consecutive_failures += 1
                self._handle_rate_limit()
                return False

        if result.errors:
            messages = [err.message.lower() for err in result.errors]
            if any("delisted" in msg or "no data found" in msg for msg in messages):
                append_delisted(ticker)
                print(f"⚠️  {ticker} appears delisted; recorded in delisted_list.txt")
                return False

        if result.metrics:
            df = pd.DataFrame([result.metrics.model_dump()])
            df.set_index("ticker", inplace=True)
            self.repo.save_metrics(df)
            print(f"✓ {ticker}")
            self.consecutive_failures = 0  # Reset on success
            return True

        print(f"✗ {ticker}")
        return False

    def _handle_rate_limit(self) -> None:
        """Exponential backoff when rate limited."""
        if self.consecutive_failures == 1:
            wait = 30 * 60  # 30 minutes
            print("\n⚠️  RATE LIMITED! Waiting 30 minutes...")
            print(f"   Time now: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Will retry at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + wait))}")
        elif self.consecutive_failures == 2:
            wait = 60 * 60  # 1 hour
            print("\n⚠️  STILL RATE LIMITED! Waiting 1 hour...")
            print(f"   Time now: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Will retry at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + wait))}")
        else:
            print(f"\n🛑 RATE LIMITED {self.consecutive_failures} times!")
            print("   Manual intervention needed. Restart script when ready.")
            print("   Script will auto-resume from oldest ticker.")
            sys.exit(1)

        time.sleep(wait)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinanceBro metrics daemon")
    parser.add_argument("tickers_file", type=Path, help="Path to file with one ticker per line")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds to sleep between updates")
    parser.add_argument("--db", type=str, default="stocks.db", help="SQLite database file")
    return parser.parse_args(argv)


def load_tickers(path: Path) -> List[str]:
    with path.open() as fh:
        tickers = [
            line.strip()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]
    delisted = read_delisted()
    return [ticker for ticker in tickers if ticker.upper() not in delisted]


def main(argv: List[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tickers = load_tickers(args.tickers_file)
    if not tickers:
        print("Ticker file is empty")
        sys.exit(1)
    daemon = SimpleDaemon(tickers, sleep_seconds=args.sleep, db_path=args.db)
    daemon.run()


if __name__ == "__main__":
    main()
