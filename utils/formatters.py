"""Helpers for serialising and formatting payloads/metrics."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from constants import (
    DIVIDEND_PAYLOAD_LIMIT,
    PRICE_HISTORY_PAYLOAD_LIMIT,
    STATEMENT_PAYLOAD_LIMIT,
)
from .math_utils import sanitize_float


def serialize_scalar(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.generic, float, int, np.integer, bool)):
        return sanitize_float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serialize_scalar(v) for v in value]
    return value


def frame_payload(frame: pd.DataFrame, limit: int = PRICE_HISTORY_PAYLOAD_LIMIT) -> List[Dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    trimmed = frame.tail(limit).reset_index()
    records = trimmed.to_dict(orient="records")
    return [
        {key: serialize_scalar(val) for key, val in record.items()}
        for record in records
    ]


def series_payload(series: pd.Series, limit: int = DIVIDEND_PAYLOAD_LIMIT) -> Dict[str, Any]:
    if series is None or len(series) == 0:
        return {}
    trimmed = series.tail(limit)
    data = trimmed.to_dict()
    return {str(key): serialize_scalar(val) for key, val in data.items()}


def statement_payload(frame: pd.DataFrame, limit: int = STATEMENT_PAYLOAD_LIMIT) -> Dict[str, Dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    max_cols = min(limit, frame.shape[1])
    trimmed = frame.iloc[:, :max_cols]
    payload = trimmed.to_dict()
    return {
        str(column): {str(idx): serialize_scalar(val) for idx, val in values.items()}
        for column, values in payload.items()
    }


def format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "None"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)
