"""gmMAP scoring methods."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from gmmap.core.normalization import sequential_normalize
from gmmap.core.signatures import TraitSignature


def _matrix_to_dataframe(expr, genes: list[str] | pd.Index, obs_names: list[str] | pd.Index | None = None) -> pd.DataFrame:
    if sparse.issparse(expr):
        expr = expr.toarray()
    obs_names = [f"cell_{i}" for i in range(expr.shape[0])] if obs_names is None else obs_names
    return pd.DataFrame(expr, index=pd.Index(obs_names, name="cell"), columns=pd.Index(genes, name="gene"))


def expression_from_adata(adata, layer: str | None = None) -> pd.DataFrame:
    """Extract expression matrix from AnnData as cells/spots x genes."""
    x = adata.layers[layer] if layer else adata.X
    return _matrix_to_dataframe(x, genes=adata.var_names, obs_names=adata.obs_names)


def score_smrs(expr: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted mean expression score: X·w / sum(|w|)."""
    common = expr.columns.intersection(weights.index)
    if len(common) == 0:
        return pd.Series(0.0, index=expr.index)
    w = weights.loc[common].astype(float)
    denom = np.abs(w).sum()
    if denom == 0:
        return pd.Series(0.0, index=expr.index)
    return pd.Series(expr.loc[:, common].to_numpy() @ w.to_numpy() / denom, index=expr.index)


def score_waucell(expr: pd.DataFrame, weights: pd.Series, top_frac: float = 0.05) -> pd.Series:
    """Weighted AUCell-like rank enrichment score.

    Genes are ranked within each cell/spot. Signature genes occurring in the top expression-ranked
    fraction receive higher contributions. This implementation is intentionally lightweight; replace it
    with the exact production implementation if your manuscript uses a custom wAUCell variant.
    """
    common = expr.columns.intersection(weights.index)
    if len(common) == 0:
        return pd.Series(0.0, index=expr.index)

    n_genes = expr.shape[1]
    top_k = max(1, int(np.ceil(n_genes * top_frac)))
    w = weights.loc[common].astype(float)
    w = w / (np.abs(w).sum() or 1.0)

    ranks = expr.rank(axis=1, method="average", ascending=False)
    sig_ranks = ranks.loc[:, common]
    contribution = (top_k - sig_ranks + 1).clip(lower=0) / top_k
    scores = contribution.to_numpy() @ w.to_numpy()
    return pd.Series(scores, index=expr.index)


def score_signatures(
    expr: pd.DataFrame,
    signatures: Mapping[str, TraitSignature],
    method: str = "waucell",
    top_frac: float = 0.05,
    stage: str = "S3",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Score UP, DOWN and NET metabolite programs and return selected stage matrices.

    Returns
    -------
    up_stage, down_stage, net_stage, all_matrices
    """
    method = method.lower()
    if method not in {"waucell", "smrs"}:
        raise ValueError("method must be 'waucell' or 'smrs'")

    score_fn = score_waucell if method == "waucell" else score_smrs
    up_raw: dict[str, pd.Series] = {}
    down_raw: dict[str, pd.Series] = {}

    for trait, sig in signatures.items():
        if method == "waucell":
            up_raw[trait] = score_fn(expr, sig.up, top_frac=top_frac)
            down_raw[trait] = score_fn(expr, sig.down, top_frac=top_frac)
        else:
            up_raw[trait] = score_fn(expr, sig.up)
            down_raw[trait] = score_fn(expr, sig.down)

    up = pd.DataFrame(up_raw, index=expr.index)
    down = pd.DataFrame(down_raw, index=expr.index)
    net = up - down

    up_norm = {f"UP_{k}": v for k, v in sequential_normalize(up).items()}
    down_norm = {f"DOWN_{k}": v for k, v in sequential_normalize(down).items()}
    net_norm = {f"NET_{k}": v for k, v in sequential_normalize(net).items()}
    all_matrices = {**up_norm, **down_norm, **net_norm}

    stage = stage.upper()
    return all_matrices[f"UP_{stage}"], all_matrices[f"DOWN_{stage}"], all_matrices[f"NET_{stage}"], all_matrices
