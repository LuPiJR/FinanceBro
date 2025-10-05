"""Domain-level data structures for FinanceBro."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(slots=True)
class PriceSnapshot:
    history: pd.DataFrame | None
    close: Optional[float]
    timestamp: Optional[str]


@dataclass(slots=True)
class ShareCounts:
    current: Optional[float]
    previous: Optional[float]


@dataclass(slots=True)
class DividendInfo:
    series: pd.Series | None
    ttm_total: Optional[float]
    dividend_yield: Optional[float]
