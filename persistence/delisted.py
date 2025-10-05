"""Helpers for tracking delisted tickers."""

from __future__ import annotations

from pathlib import Path
from typing import Set


DEFAULT_DELISTED_FILE = Path("delisted_list.txt")


def read_delisted(path: Path = DEFAULT_DELISTED_FILE) -> Set[str]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {line.strip().upper() for line in fh if line.strip()}


def append_delisted(ticker: str, path: Path = DEFAULT_DELISTED_FILE) -> None:
    ticker_upper = ticker.upper()
    existing = read_delisted(path)
    if ticker_upper in existing:
        return
    with path.open("a") as fh:
        fh.write(f"{ticker_upper}\n")
