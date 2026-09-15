"""Configuration objects for gmMAP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GmmapConfig:
    """Default parameters used by the gmMAP pipeline."""

    top_n_genes: int = 1000
    min_valid_genes: int = 200
    min_cells_in_type: int = 20
    top_frac_waucell: float = 0.05
    stage: str = "S3"
    n_jobs: int = 1
    delong_max_total_cells: int = 50000
    boot_n: int = 200
    q_cutoff: float = 0.05
    bg_q_cutoff: float = 0.20
    emp_null_quantile: float = 0.95
    specificity_gap_min: float = 0.03
