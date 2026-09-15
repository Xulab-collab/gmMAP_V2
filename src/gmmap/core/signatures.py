"""Metabolite-gene program construction from MAGMA/GWAS gene Z matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TraitSignature:
    """UP/DOWN weighted gene program for one metabolite trait."""

    trait: str
    up: pd.Series
    down: pd.Series


def standardize_gene_index(magma_z: pd.DataFrame, gene_col: str | None = None) -> pd.DataFrame:
    """Return a gene-indexed MAGMA Z matrix."""
    df = magma_z.copy()
    if gene_col is not None and gene_col in df.columns:
        df = df.set_index(gene_col)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def build_trait_signature(
    z_matrix: pd.DataFrame,
    trait: str,
    top_n: int = 1000,
    min_valid_genes: int = 200,
) -> TraitSignature | None:
    """Build UP and DOWN weighted gene signatures for a single trait.

    UP genes are the strongest positive Z genes. DOWN genes are the strongest negative Z genes,
    stored as positive magnitudes for downstream weighted enrichment.
    """
    if trait not in z_matrix.columns:
        raise KeyError(f"Trait not found in MAGMA Z matrix: {trait}")

    z = z_matrix[trait].replace([np.inf, -np.inf], np.nan).dropna()
    if z.shape[0] < min_valid_genes:
        return None

    up = z[z > 0].sort_values(ascending=False).head(top_n)
    down = (-z[z < 0]).sort_values(ascending=False).head(top_n)
    if min(len(up), len(down)) < min_valid_genes:
        return None
    return TraitSignature(trait=trait, up=up, down=down)


def build_signatures(
    z_matrix: pd.DataFrame,
    traits: list[str] | None = None,
    top_n: int = 1000,
    min_valid_genes: int = 200,
) -> dict[str, TraitSignature]:
    """Build UP/DOWN signatures for multiple metabolite traits."""
    traits = list(z_matrix.columns) if traits is None else traits
    signatures: dict[str, TraitSignature] = {}
    for trait in traits:
        sig = build_trait_signature(z_matrix, trait, top_n=top_n, min_valid_genes=min_valid_genes)
        if sig is not None:
            signatures[trait] = sig
    return signatures


def signature_table(signatures: dict[str, TraitSignature]) -> pd.DataFrame:
    """Convert signatures to a long table: trait, side, gene, weight."""
    rows: list[dict[str, object]] = []
    for trait, sig in signatures.items():
        for side, series in (("UP", sig.up), ("DOWN", sig.down)):
            for gene, weight in series.items():
                rows.append({"trait": trait, "side": side, "gene": gene, "weight": float(weight)})
    return pd.DataFrame(rows)


def signatures_from_table(table: pd.DataFrame) -> dict[str, TraitSignature]:
    """Rebuild TraitSignature objects from a long signature table."""
    required = {"trait", "side", "gene", "weight"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Signature table is missing columns: {sorted(missing)}")

    signatures: dict[str, TraitSignature] = {}
    for trait, df_trait in table.groupby("trait", sort=False):
        up_df = df_trait[df_trait["side"].str.upper() == "UP"]
        down_df = df_trait[df_trait["side"].str.upper() == "DOWN"]
        up = pd.Series(up_df["weight"].to_numpy(float), index=up_df["gene"].astype(str))
        down = pd.Series(down_df["weight"].to_numpy(float), index=down_df["gene"].astype(str))
        signatures[str(trait)] = TraitSignature(trait=str(trait), up=up, down=down)
    return signatures
