"""Pseudotime association utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def correlate_with_pseudotime(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    pseudotime_key: str,
    traits: list[str] | None = None,
    group_key: str | None = None,
) -> pd.DataFrame:
    """Spearman correlation between metabolite scores and pseudotime."""
    traits = list(scores.columns) if traits is None else traits
    common = scores.index.intersection(metadata.index)
    scores = scores.loc[common, traits]
    meta = metadata.loc[common]
    if pseudotime_key not in meta.columns:
        raise ValueError(f"Metadata does not contain pseudotime column: {pseudotime_key}")

    rows: list[dict[str, object]] = []
    groups = [("all", meta.index)] if group_key is None else list(meta.groupby(group_key).groups.items())
    for group, idx in groups:
        pt = pd.to_numeric(meta.loc[idx, pseudotime_key], errors="coerce")
        for trait in traits:
            y = pd.to_numeric(scores.loc[idx, trait], errors="coerce")
            mask = pt.notna() & y.notna() & np.isfinite(pt) & np.isfinite(y)
            if mask.sum() < 10:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(pt.loc[mask], y.loc[mask])
            rows.append({"group": group, "trait": trait, "n": int(mask.sum()), "rho": rho, "pvalue": p})

    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["pvalue"].notna()
        out["qvalue"] = np.nan
        out.loc[mask, "qvalue"] = multipletests(out.loc[mask, "pvalue"], method="fdr_bh")[1]
    return out.sort_values(["qvalue", "trait"], na_position="last")
