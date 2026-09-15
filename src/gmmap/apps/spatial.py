"""Spatial projection utilities."""

from __future__ import annotations

import pandas as pd


def export_spatial_trait_table(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    trait: str,
    x_key: str = "x_image",
    y_key: str = "y_image",
    sample_key: str | None = "Sample",
) -> pd.DataFrame:
    """Create a spatial plotting table for one metabolite trait."""
    if trait not in scores.columns:
        raise KeyError(f"Trait not found in score table: {trait}")
    common = scores.index.intersection(metadata.index)
    cols = [x_key, y_key] + ([sample_key] if sample_key and sample_key in metadata.columns else [])
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing spatial columns: {missing}")
    out = metadata.loc[common, cols].copy()
    out["trait"] = trait
    out["gmmap_score"] = scores.loc[common, trait]
    return out.reset_index(names="obs_id")


def summarize_by_region(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    region_key: str,
    traits: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize gmMAP scores by anatomical region or tissue label."""
    traits = list(scores.columns) if traits is None else traits
    common = scores.index.intersection(metadata.index)
    df = scores.loc[common, traits].copy()
    df[region_key] = metadata.loc[common, region_key].astype(str)
    return df.groupby(region_key)[traits].agg(["mean", "median", "std", "count"])
