"""SQLite repository for storing computed stock metrics."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


METRIC_COLUMNS = [
    "price",
    "mkt_cap",
    "pb",
    "pe",
    "ps",
    "ev_ebitda",
    "pcf",
    "dividend_yield_ttm",
    "buyback_yield_ttm",
    "shareholder_yield_ttm",
    "mom_6m",
    "vc2_score",
    "vc2_abs_score",
    "price_timestamp",
]


class StockRepository:
    """Lightweight repository persisting metrics snapshots to SQLite."""

    def __init__(self, db_path: str | Path = "stocks.db") -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_metrics (
                    ticker TEXT PRIMARY KEY,
                    price REAL,
                    mkt_cap REAL,
                    pb REAL,
                    pe REAL,
                    ps REAL,
                    ev_ebitda REAL,
                    pcf REAL,
                    dividend_yield_ttm REAL,
                    buyback_yield_ttm REAL,
                    shareholder_yield_ttm REAL,
                    mom_6m REAL,
                    vc2_score REAL,
                    vc2_abs_score REAL,
                    price_timestamp TEXT,
                    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_metrics(self, df: pd.DataFrame) -> None:
        """Upsert metrics for provided tickers."""

        if df.empty:
            return

        with sqlite3.connect(self._db_path) as conn:
            insert_cols = ["ticker"] + METRIC_COLUMNS
            placeholders = ", ".join("?" for _ in insert_cols)
            update_assignments = ", ".join(
                f"{col}=excluded.{col}"
                for col in METRIC_COLUMNS
            )
            sql = (
                f"INSERT INTO stock_metrics ({', '.join(insert_cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(ticker) DO UPDATE SET {update_assignments}, "
                "last_updated_at=CURRENT_TIMESTAMP"
            )

            for ticker, row in df.iterrows():
                row_dict = row.to_dict()
                values = [ticker] + [row_dict.get(col) for col in METRIC_COLUMNS]
                conn.execute(sql, values)
            conn.commit()

    def get_oldest_ticker(self) -> Optional[str]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT ticker FROM stock_metrics ORDER BY last_updated_at ASC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def get_all_metrics(self) -> pd.DataFrame:
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM stock_metrics ORDER BY ticker",
                conn,
                index_col="ticker",
            )
