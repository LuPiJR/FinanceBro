"""Financial metric calculations extracted from the CLI orchestration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from constants import MOMENTUM_MONTHS


class FinancialCalculator:
    """Provide reusable calculations for valuation and momentum metrics."""

    def market_cap(self, price: Optional[float], shares: Optional[float]) -> Optional[float]:
        if price is None or shares is None:
            return None
        return price * shares

    def enterprise_value(
        self,
        market_cap: Optional[float],
        total_debt: Optional[float],
        cash: Optional[float],
    ) -> Optional[float]:
        if market_cap is None:
            return None
        debt_component = 0 if total_debt is None else total_debt
        cash_component = 0 if cash is None else cash
        return market_cap + debt_component - cash_component

    def price_to_earnings(
        self,
        price: Optional[float],
        net_income: Optional[float],
        shares: Optional[float],
    ) -> Optional[float]:
        if not self._is_positive(price, net_income, shares):
            return None
        eps = net_income / shares
        if eps <= 0:
            return None
        return price / eps

    def price_to_sales(
        self,
        market_cap: Optional[float],
        revenue: Optional[float],
    ) -> Optional[float]:
        if market_cap is None or revenue is None or revenue <= 0:
            return None
        return market_cap / revenue

    def ev_to_ebitda(
        self,
        enterprise_value: Optional[float],
        ebitda: Optional[float],
    ) -> Optional[float]:
        if enterprise_value is None or ebitda is None or ebitda <= 0:
            return None
        return enterprise_value / ebitda

    def price_to_cashflow(
        self,
        price: Optional[float],
        operating_cashflow: Optional[float],
        shares: Optional[float],
    ) -> Optional[float]:
        if not self._is_positive(price, operating_cashflow, shares):
            return None
        cash_flow_per_share = operating_cashflow / shares
        if cash_flow_per_share <= 0:
            return None
        return price / cash_flow_per_share

    def price_to_book(self, price: Optional[float], book_value_per_share: Optional[float]) -> Optional[float]:
        if price is None or book_value_per_share is None or book_value_per_share <= 0:
            return None
        return price / book_value_per_share

    def buyback_yield_from_shares(
        self,
        current_shares: Optional[float],
        previous_shares: Optional[float],
    ) -> Optional[float]:
        if not self._is_positive(current_shares, previous_shares):
            return None
        numerator = -(current_shares - previous_shares)
        denominator = (current_shares + previous_shares) / 2
        if denominator <= 0:
            return None
        return numerator / denominator

    def buyback_yield_from_cashflow(
        self,
        net_stock_issuance: Optional[float],
        market_cap: Optional[float],
    ) -> Optional[float]:
        if net_stock_issuance is None or market_cap is None or market_cap <= 0:
            return None
        try:
            return -(net_stock_issuance / market_cap)
        except (TypeError, ZeroDivisionError):
            return None

    def select_buyback_yield(
        self,
        share_based: Optional[float],
        cashflow_based: Optional[float],
    ) -> Optional[float]:
        if cashflow_based is not None and np.isfinite(cashflow_based):
            return cashflow_based
        if share_based is not None and np.isfinite(share_based):
            return share_based
        return None

    def shareholder_yield(
        self,
        dividend_yield: Optional[float],
        buyback_yield: Optional[float],
    ) -> Optional[float]:
        div_component = dividend_yield if dividend_yield is not None else 0
        buyback_component = buyback_yield if buyback_yield is not None else 0
        if div_component == 0 and buyback_component == 0:
            if dividend_yield is None and buyback_yield is None:
                return None
        return div_component + buyback_component

    def momentum(self, close_series: pd.Series | None, months: int = MOMENTUM_MONTHS) -> Optional[float]:
        if close_series is None or close_series.empty:
            return None
        clean_series = close_series.dropna()
        if len(clean_series) < 2:
            return None
        last_idx = clean_series.index[-1]
        base_idx = last_idx - pd.DateOffset(months=months)
        base_price = clean_series.asof(base_idx)
        if pd.isna(base_price):
            base_price = clean_series.iloc[0]
        if pd.isna(base_price) or base_price <= 0:
            return None
        return float(clean_series.iloc[-1] / base_price - 1.0)

    @staticmethod
    def dividend_yield(dividends_ttm: Optional[float], price: Optional[float]) -> Optional[float]:
        if dividends_ttm is None or price is None or price <= 0:
            return None
        return dividends_ttm / price

    @staticmethod
    def as_snapshot(history: pd.DataFrame | None) -> Optional[float]:
        if history is None or history.empty or "Close" not in history:
            return None
        close_series = history["Close"].dropna()
        if close_series.empty:
            return None
        return float(close_series.iloc[-1])

    @staticmethod
    def _is_positive(*values: Optional[float]) -> bool:
        return all(value is not None and value > 0 for value in values)
