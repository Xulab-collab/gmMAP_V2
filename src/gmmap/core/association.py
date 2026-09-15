"""Cell-type association statistics for gmMAP scores."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def auroc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute AUROC using rank statistics."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(score)
    rank_sum_pos = ranks[y_true].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auc_pvalue_normal(auc: float, n_pos: int, n_neg: int) -> float:
    """Approximate two-sided p-value for AUROC deviation from 0.5.

    This is a lightweight fallback. Replace with DeLong/bootstrap code in production if needed.
    """
    if not np.isfinite(auc) or n_pos <= 0 or n_neg <= 0:
        return np.nan
    q1 = auc / (2 - auc) if auc < 1 else 1.0
    q2 = 2 * auc**2 / (1 + auc) if auc > 0 else 0.0
    se = math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
    if se == 0 or not np.isfinite(se):
        return np.nan
    z = (auc - 0.5) / se
    return float(2 * stats.norm.sf(abs(z)))


def bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction preserving NA values."""
    out = pd.Series(np.nan, index=p.index, dtype=float)
    mask = p.notna() & np.isfinite(p.to_numpy(dtype=float, na_value=np.nan))
    if mask.any():
        out.loc[mask] = multipletests(p.loc[mask].astype(float), method="fdr_bh")[1]
    return out


def one_vs_rest_association(
    up_scores: pd.DataFrame,
    down_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    celltype_key: str,
    min_cells_in_type: int = 20,
) -> pd.DataFrame:
    """Run one-vs-rest AUROC association analysis for each trait and cell type."""
    common_obs = up_scores.index.intersection(down_scores.index).intersection(metadata.index)
    up_scores = up_scores.loc[common_obs]
    down_scores = down_scores.loc[common_obs]
    metadata = metadata.loc[common_obs]

    celltypes = metadata[celltype_key].astype(str)
    rows: list[dict[str, object]] = []
    for ct, idx in celltypes.groupby(celltypes).groups.items():
        y = (celltypes == ct).to_numpy()
        n_pos = int(y.sum())
        n_neg = int((~y).sum())
        if n_pos < min_cells_in_type or n_neg < min_cells_in_type:
            continue
        for trait in up_scores.columns.intersection(down_scores.columns):
            auc_up = auroc_rank(y, up_scores[trait].to_numpy())
            auc_down = auroc_rank(y, down_scores[trait].to_numpy())
            effect_up = auc_up - 0.5 if np.isfinite(auc_up) else np.nan
            effect_down = auc_down - 0.5 if np.isfinite(auc_down) else np.nan
            delta = effect_up - effect_down if np.isfinite(effect_up) and np.isfinite(effect_down) else np.nan
            p_up = auc_pvalue_normal(auc_up, n_pos, n_neg)
            p_down = auc_pvalue_normal(auc_down, n_pos, n_neg)
            # Delta p-value is a conservative placeholder; replace with DeLong/bootstrap if available.
            p_delta = max(p_up, p_down) if np.isfinite(p_up) and np.isfinite(p_down) else np.nan
            if np.isfinite(delta) and delta >= 0:
                direction = "up-dominant"
                final_side = "UP"
                final_auc = auc_up
                final_effect = effect_up
                final_p = p_up
            else:
                direction = "down-dominant"
                final_side = "DOWN"
                final_auc = auc_down
                final_effect = -effect_down if np.isfinite(effect_down) else np.nan
                final_p = p_down
            rows.append(
                {
                    "celltype": ct,
                    "trait": trait,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "AUROC_up": auc_up,
                    "effect_up": effect_up,
                    "p_up": p_up,
                    "AUROC_down": auc_down,
                    "effect_down": effect_down,
                    "p_down": p_down,
                    "AUROC_effect_delta": delta,
                    "p_delta": p_delta,
                    "direction": direction,
                    "final_side": final_side,
                    "final_AUROC": final_auc,
                    "final_effect": final_effect,
                    "final_p": final_p,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for col in ["p_up", "p_down", "p_delta", "final_p"]:
        result[col.replace("p_", "q_") if col != "final_p" else "final_q"] = bh_fdr(result[col])
    result["final_q_trait_across_celltypes"] = np.nan
    for trait, idx in result.groupby("trait").groups.items():
        result.loc[idx, "final_q_trait_across_celltypes"] = bh_fdr(result.loc[idx, "final_p"])
    return result.sort_values(["celltype", "final_q", "trait"])


def empirical_null_threshold(
    assoc: pd.DataFrame,
    q_col: str = "final_q",
    effect_col: str = "final_effect",
    bg_q: float = 0.20,
    quantile: float = 0.95,
) -> float:
    """Estimate empirical-null effect threshold from non-significant associations."""
    if assoc.empty or q_col not in assoc or effect_col not in assoc:
        return float("nan")
    bg = assoc.loc[assoc[q_col] > bg_q, effect_col].abs().dropna()
    if bg.empty:
        return float("nan")
    return float(bg.quantile(quantile))
