"""Numeric helper functions."""

from typing import Any, Optional

import numpy as np
import pandas as pd


def safe_div(a: Any, b: Any):
    """Divide two values, returning NaN for invalid results."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    if isinstance(out, pd.DataFrame):
        return out.replace([np.inf, -np.inf], np.nan)
    if isinstance(out, np.ndarray):
        return np.where(np.isfinite(out), out, np.nan)
    if np.isscalar(out):
        return np.nan if not np.isfinite(out) else float(out)
    return out


def sanitize_float(value: Any) -> Optional[float]:
    """Coerce different numeric types to clean floats, returning None for invalid values."""
    if value is None:
        return None
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    if isinstance(value, pd.Index):
        if len(value) == 0:
            return None
        value = value[0]
    if isinstance(value, (np.ndarray, list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    if isinstance(value, (np.generic, float, int, np.integer)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, bool):
        return float(value)
    return None
