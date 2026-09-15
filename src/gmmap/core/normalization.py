"""Sequential score normalization used by gmMAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_zscore(df: pd.DataFrame, axis: int) -> pd.DataFrame:
    mean = df.mean(axis=axis)
    std = df.std(axis=axis, ddof=0).replace(0, np.nan)
    if axis == 0:
        out = df.sub(mean, axis=1).div(std, axis=1)
    else:
        out = df.sub(mean, axis=0).div(std, axis=0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sequential_normalize(raw_scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return S1-S4 score matrices.

    S1: raw scores.
    S2: trait-wise z-score across observations.
    S3: row-wise z-score on S2.
    S4: trait-wise z-score on S3.
    """
    s1 = raw_scores.copy()
    s2 = _safe_zscore(s1, axis=0)
    s3 = _safe_zscore(s2, axis=1)
    s4 = _safe_zscore(s3, axis=0)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4}
