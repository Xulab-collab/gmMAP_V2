"""gmMAP-perturb utilities."""

from __future__ import annotations

import pandas as pd


def compute_perturb_delta(
    activity_scores: pd.DataFrame,
    inhibition_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Δ = Activity - Inhibition for matched metabolites/signatures."""
    common_obs = activity_scores.index.intersection(inhibition_scores.index)
    common_cols = activity_scores.columns.intersection(inhibition_scores.columns)
    return activity_scores.loc[common_obs, common_cols] - inhibition_scores.loc[common_obs, common_cols]


def summarize_perturb_by_branch(
    delta_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    branch_key: str,
    pseudotime_key: str | None = None,
) -> pd.DataFrame:
    """Summarize perturbation Δ scores by branch and optionally pseudotime bins."""
    common = delta_scores.index.intersection(metadata.index)
    df = delta_scores.loc[common].copy()
    meta = metadata.loc[common]
    df[branch_key] = meta[branch_key].astype(str)
    group_cols = [branch_key]
    if pseudotime_key is not None:
        df["pseudotime_bin"] = pd.qcut(meta[pseudotime_key], q=10, duplicates="drop")
        group_cols.append("pseudotime_bin")
    return df.groupby(group_cols, observed=False).mean(numeric_only=True).reset_index()
