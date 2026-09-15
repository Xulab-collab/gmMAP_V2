"""gmMAP-flux: relative metabolic-flow potential utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_flux_modules(path: str | Path) -> list[dict[str, Any]]:
    """Load metabolic modules from YAML."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict) and "modules" in data:
        return list(data["modules"])
    if isinstance(data, list):
        return data
    raise ValueError("Flux module YAML must be a list or contain a 'modules' list.")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_flux_potential(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    modules: list[dict[str, Any]],
    celltype_key: str,
) -> pd.DataFrame:
    """Compute a lightweight relative gmMAP-flux potential per cell type.

    Expected module fields:
    - name
    - input_trait
    - output_trait
    - ratio_trait optional
    - ratio_sign optional, default 1
    - enzyme_support optional numeric 0-1
    - compartment_support optional numeric 0-1
    - cofactor_support optional numeric 0-1
    """
    common = scores.index.intersection(metadata.index)
    scores = scores.loc[common]
    meta = metadata.loc[common]

    rows: list[dict[str, object]] = []
    for ct, idx in meta.groupby(celltype_key).groups.items():
        ct_scores = scores.loc[idx]
        for module in modules:
            name = str(module.get("name", "unnamed_module"))
            input_trait = module.get("input_trait")
            output_trait = module.get("output_trait")
            ratio_trait = module.get("ratio_trait")
            ratio_sign = float(module.get("ratio_sign", 1.0))
            if input_trait not in ct_scores or output_trait not in ct_scores:
                continue
            s_in = float(ct_scores[input_trait].mean())
            s_out = float(ct_scores[output_trait].mean())
            delta_s = s_out - s_in
            ratio_term = 0.0
            if ratio_trait and ratio_trait in ct_scores:
                ratio_term = ratio_sign * float(ct_scores[ratio_trait].mean())
            raw_activation = delta_s + ratio_term
            enzyme = float(module.get("enzyme_support", 1.0))
            compartment = float(module.get("compartment_support", 1.0))
            cofactor = float(module.get("cofactor_support", 1.0))
            feasibility = enzyme * compartment * cofactor
            potential = _sigmoid(raw_activation) * feasibility
            rows.append(
                {
                    "celltype": ct,
                    "module": name,
                    "input_trait": input_trait,
                    "output_trait": output_trait,
                    "ratio_trait": ratio_trait,
                    "s_in": s_in,
                    "s_out": s_out,
                    "delta_s": delta_s,
                    "ratio_term": ratio_term,
                    "raw_activation": raw_activation,
                    "enzyme_support": enzyme,
                    "compartment_support": compartment,
                    "cofactor_support": cofactor,
                    "feasibility": feasibility,
                    "flux_potential": potential,
                }
            )
    return pd.DataFrame(rows)
