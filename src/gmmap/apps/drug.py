"""gmMAP-drug reversal scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rank_drug_reversal(
    disease_signature: pd.DataFrame,
    drug_signatures: pd.DataFrame,
    trait_col: str = "trait",
    disease_effect_col: str = "effect",
    drug_col: str = "drug",
    drug_effect_col: str = "effect",
) -> pd.DataFrame:
    """Rank drugs by negative correlation to a disease metabolite signature.

    Larger `final_rank_score` indicates stronger predicted reversal.
    """
    disease = disease_signature[[trait_col, disease_effect_col]].dropna().copy()
    disease = disease.rename(columns={disease_effect_col: "disease_effect"})
    rows: list[dict[str, object]] = []
    for drug, df_drug in drug_signatures.groupby(drug_col):
        tmp = disease.merge(
            df_drug[[trait_col, drug_effect_col]].rename(columns={drug_effect_col: "drug_effect"}),
            on=trait_col,
            how="inner",
        ).dropna()
        if tmp.shape[0] < 5:
            rho, p = np.nan, np.nan
            score = np.nan
        else:
            rho, p = stats.spearmanr(tmp["disease_effect"], tmp["drug_effect"])
            score = -rho
        rows.append(
            {
                "drug": drug,
                "n_overlap_traits": int(tmp.shape[0]),
                "spearman_rho": rho,
                "pvalue": p,
                "final_rank_score": score,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("final_rank_score", ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_percentile"] = 1 - (out["rank"] - 1) / max(len(out) - 1, 1)
    return out
