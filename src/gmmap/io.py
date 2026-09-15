"""Input/output helpers for gmMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path, index_col: int | str | None = None) -> pd.DataFrame:
    """Read CSV/TSV automatically from suffix."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    sep = "\t" if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz") else ","
    return pd.read_csv(path, sep=sep, index_col=index_col)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    sep = "\t" if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz") else ","
    df.to_csv(path, sep=sep, index=index)


def read_h5ad(path: str | Path) -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Please install anndata: python -m pip install anndata") from exc
    return ad.read_h5ad(path)


def dense_matrix(x: Any):
    """Convert sparse matrix-like objects to dense numpy arrays."""
    if hasattr(x, "toarray"):
        return x.toarray()
    return x
