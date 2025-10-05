"""Thin wrapper around yfinance to ease testing and substitution."""

from __future__ import annotations

from typing import Any

import yfinance as yf


class YFinanceClient:
    """Provide typed accessors for yfinance data used in the pipeline."""

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._ticker = yf.Ticker(symbol)

    @property
    def symbol(self) -> str:
        return self._symbol

    def price_history(self, days: int, *, auto_adjust: bool = True, actions: bool = False):
        return self._ticker.history(period=f"{days}d", auto_adjust=auto_adjust, actions=actions)

    def dividends(self):
        return self._ticker.dividends

    def shares_full(self, *, start: Any = None):
        return self._ticker.get_shares_full(start=start)

    def ttm_income_statement(self):
        return getattr(self._ticker, "ttm_income_stmt", None)

    def ttm_cashflow(self):
        return getattr(self._ticker, "ttm_cashflow", None)

    def income_statement(self, freq: str = "yearly"):
        try:
            return self._ticker.get_income_stmt(freq=freq)
        except Exception:
            fallback = "income_stmt" if freq == "yearly" else "quarterly_income_stmt"
            return getattr(self._ticker, fallback, None)

    def cashflow(self, freq: str = "yearly"):
        try:
            return self._ticker.get_cashflow(freq=freq)
        except Exception:
            fallback = "cashflow" if freq == "yearly" else "quarterly_cashflow"
            return getattr(self._ticker, fallback, None)

    def balance_sheet(self, freq: str = "yearly"):
        try:
            return self._ticker.get_balance_sheet(freq=freq)
        except Exception:
            fallback = "balance_sheet" if freq == "yearly" else "quarterly_balance_sheet"
            return getattr(self._ticker, fallback, None)

    def raw_ticker(self) -> yf.Ticker:
        """Expose the underlying yfinance object when direct access is unavoidable."""
        return self._ticker
