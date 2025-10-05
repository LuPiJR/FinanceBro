"""Value Composite 2 scoring utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from constants import ABS_VC2_THRESHOLDS


def compute_relative_vc2(
    df: pd.DataFrame,
    component_directions: Dict[str, bool],
) -> pd.Series:
    """Return a percentile-like VC2 score where higher is better."""

    ranks: Dict[str, pd.Series] = {}
    for column, ascending in component_directions.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if not series.notna().any():
            continue
        component_ranks = series.rank(method="min", ascending=ascending)
        if component_ranks.notna().any():
            worst_rank = component_ranks.max() + 1
            component_ranks = component_ranks.fillna(worst_rank)
        ranks[column] = component_ranks

    if not ranks:
        return pd.Series(dtype=float)

    rank_df = pd.DataFrame(ranks)
    vc2_raw = rank_df.sum(axis=1)

    max_raw = vc2_raw.max()
    min_raw = vc2_raw.min()
    if np.isclose(max_raw, min_raw):
        return pd.Series(50.0, index=vc2_raw.index)

    return (max_raw - vc2_raw) / (max_raw - min_raw) * 100.0


def absolute_vc2_score(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    """Compute pass-rate vs absolute VC2 thresholds."""

    total = 0
    passes = 0

    for key, (operator, threshold) in ABS_VC2_THRESHOLDS.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            continue
        total += 1
        if operator == "<=":
            passes += value <= threshold
        elif operator == ">=":
            passes += value >= threshold

    if total == 0:
        return None
    return (passes / total) * 100.0
