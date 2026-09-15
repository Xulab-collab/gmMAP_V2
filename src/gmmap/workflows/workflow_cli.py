"""Console wrappers for the full gmMAP manuscript workflows.

The original research scripts were written as standalone files.  This module exposes
stable command-line entry points while preserving the original implementation in
``gmmap.workflows`` modules.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from typing import Any


def _parse_scalar(value: str) -> Any:
    """Parse CLI override values while keeping ordinary strings safe."""
    if value is None:
        return None
    text = str(value)
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _apply_overrides(module: Any, mapping: dict[str, Any]) -> None:
    for key, value in mapping.items():
        if value is not None:
            setattr(module, key, value)


def _apply_set_overrides(module: Any, overrides: list[str] | None) -> None:
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --set override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        setattr(module, key.strip(), _parse_scalar(value.strip()))


def _common_config_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--h5ad", required=True, help="Input AnnData .h5ad file.")
    p.add_argument("--magma-z", required=True, help="Gene-by-trait MAGMA/gmMAP Z matrix.")
    p.add_argument("--celltype-key", required=True, help="obs column used for one-vs-rest association.")
    p.add_argument("--prefix", default="Meta")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--top-n-genes", type=int, default=None)
    p.add_argument("--min-valid-genes", type=int, default=None)
    p.add_argument("--min-cells-in-type", type=int, default=None)
    p.add_argument("--top-frac-waucell", type=float, default=None)
    p.add_argument("--methods-to-run", default=None, help="Comma-separated list, e.g. wAUCell,scMRS.")
    p.add_argument("--stages-to-run", default=None, help="Comma-separated list, e.g. S1_raw,S3_geneSetZ_cellZ.")
    p.add_argument("--skip-existing", default=None, choices=["true", "false"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any global variable in the workflow module.")
    return p


def _run_scheme_like(module_name: str, description: str) -> None:
    args = _common_config_parser(description).parse_args()
    module = importlib.import_module(f"gmmap.workflows.{module_name}")
    _apply_overrides(module, {
        "H5AD_PATH": args.h5ad,
        "ZMAT_PATH": args.magma_z,
        "CELLTYPE_KEY": args.celltype_key,
        "PREFIX": args.prefix,
        "OUT_DIR": args.outdir,
        "N_JOBS": args.n_jobs,
        "TOP_N_GENES": args.top_n_genes,
        "MIN_VALID_GENES": args.min_valid_genes,
        "MIN_CELLS_IN_TYPE": args.min_cells_in_type,
        "TOP_FRAC_WAUCELL": args.top_frac_waucell,
        "METHODS_TO_RUN": _csv(args.methods_to_run),
        "STAGES_TO_RUN": _csv(args.stages_to_run),
        "SKIP_EXISTING": None if args.skip_existing is None else args.skip_existing == "true",
    })
    _apply_set_overrides(module, args.set)
    module.main()


def scheme_c_main() -> None:
    _run_scheme_like("main_scheme_c", "Run the full gmMAP Scheme C AUROC Up/Down/Net pipeline.")


def spatial_auroc_main() -> None:
    _run_scheme_like("spatial_auroc_waucell", "Run the full gmMAP spatial AUROC/wAUCell workflow.")


def pseudotime_pyucell_main() -> None:
    p = argparse.ArgumentParser(description="Run the full gmMAP-pseudotime pyUCell workflow.")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--magma-z", required=True)
    p.add_argument("--celltype-col", required=True)
    p.add_argument("--pseudotime-col", required=True)
    p.add_argument("--prefix", default="Meta")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-jobs-ucell", type=int, default=None)
    p.add_argument("--n-jobs-cor", type=int, default=None)
    p.add_argument("--top-n-genes", type=int, default=None)
    p.add_argument("--min-valid-genes", type=int, default=None)
    p.add_argument("--min-cells-per-lineage", type=int, default=None)
    p.add_argument("--target-traits", default=None, help="Comma-separated trait subset.")
    p.add_argument("--lineages", default=None, help="Python literal dict override for LINEAGES.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    module = importlib.import_module("gmmap.workflows.pseudotime_pyucell")
    _apply_overrides(module, {
        "H5AD_PATH": args.h5ad,
        "ZMAT_PATH": args.magma_z,
        "CELLTYPE_COL": args.celltype_col,
        "PSEUDOTIME_COL": args.pseudotime_col,
        "PREFIX": args.prefix,
        "OUT_DIR": args.outdir,
        "N_JOBS_UCELL": args.n_jobs_ucell,
        "N_JOBS_COR": args.n_jobs_cor,
        "TOP_N_GENES": args.top_n_genes,
        "MIN_VALID_GENES": args.min_valid_genes,
        "MIN_CELLS_PER_LINEAGE": args.min_cells_per_lineage,
        "TARGET_TRAITS": _csv(args.target_traits),
        "LINEAGES": None if args.lineages is None else _parse_scalar(args.lineages),
    })
    _apply_set_overrides(module, args.set)
    module.main()


def pseudotime_plot_main() -> None:
    p = argparse.ArgumentParser(description="Plot gmMAP-pseudotime metabolite trajectories.")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--celltype-col", required=True)
    p.add_argument("--pseudotime-col", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--target-traits", default=None)
    p.add_argument("--lineages", default=None, help="Python literal dict override for LINEAGES.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    module = importlib.import_module("gmmap.workflows.pseudotime_plot")
    _apply_overrides(module, {
        "H5AD_PATH": args.h5ad,
        "SCORES_PATH": args.scores,
        "CELLTYPE_COL": args.celltype_col,
        "PSEUDOTIME_COL": args.pseudotime_col,
        "OUT_DIR": args.outdir,
        "TARGET_TRAITS": _csv(args.target_traits),
        "LINEAGES": None if args.lineages is None else _parse_scalar(args.lineages),
    })
    _apply_set_overrides(module, args.set)
    module.main()


def perturb_full_main() -> None:
    p = argparse.ArgumentParser(description="Run the full gmMAP-perturb workflow.")
    p.add_argument("--h5ad", required=True)
    p.add_argument("--magma-z", required=True)
    p.add_argument("--celltype-col", required=True)
    p.add_argument("--pseudotime-col", required=True)
    p.add_argument("--umap-key", default=None)
    p.add_argument("--prefix", default="Meta")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-jobs-ucell", type=int, default=None)
    p.add_argument("--n-jobs-cor", type=int, default=None)
    p.add_argument("--top-n-genes", type=int, default=None)
    p.add_argument("--min-valid-genes", type=int, default=None)
    p.add_argument("--target-traits", default=None)
    p.add_argument("--manual-plot-traits", default=None)
    p.add_argument("--trajectory-config", default=None, help="Python literal dict override for TRAJECTORY_CONFIG.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    module = importlib.import_module("gmmap.workflows.perturb_full")
    _apply_overrides(module, {
        "H5AD_PATH": args.h5ad,
        "ZMAT_PATH": args.magma_z,
        "CELLTYPE_COL": args.celltype_col,
        "PSEUDOTIME_COL": args.pseudotime_col,
        "UMAP_KEY": args.umap_key,
        "PREFIX": args.prefix,
        "OUT_DIR": args.outdir,
        "N_JOBS_UCELL": args.n_jobs_ucell,
        "N_JOBS_COR": args.n_jobs_cor,
        "TOP_N_GENES": args.top_n_genes,
        "MIN_VALID_GENES": args.min_valid_genes,
        "TARGET_TRAITS": _csv(args.target_traits),
        "MANUAL_PLOT_TRAITS": _csv(args.manual_plot_traits),
        "TRAJECTORY_CONFIG": None if args.trajectory_config is None else _parse_scalar(args.trajectory_config),
    })
    _apply_set_overrides(module, args.set)
    module.main()


def _forward_to_argparse_module(module_name: str) -> None:
    module = importlib.import_module(f"gmmap.workflows.{module_name}")
    module.main()


def flux_full_main() -> None:
    _forward_to_argparse_module("flux_model")


def flux_plot_main() -> None:
    _forward_to_argparse_module("flux_plot_metabolism")


def spatial_identify_main() -> None:
    _forward_to_argparse_module("spatial_identify_and_plot")


def spatial_violin_batch_main() -> None:
    _forward_to_argparse_module("spatial_violin_alltraits_batch")


def spatial_violin_similarity_main() -> None:
    _forward_to_argparse_module("spatial_violin_similarity")


def spatial_groupdiff_main() -> None:
    _forward_to_argparse_module("spatial_target_group_diff")


def drug_rank_full_main() -> None:
    _forward_to_argparse_module("drug_batch_reversal")


def drug_plot_main() -> None:
    _forward_to_argparse_module("drug_plot_pancancer")


def resources_main() -> None:
    from importlib.resources import files

    base = files("gmmap") / "resources"
    print("gmMAP bundled resources:")
    print(f"  {base / 'drug' / 'custom_cmap_metabolic_drugs_level.csv.gz'}")
    print(f"  {base / 'flux' / 'module_ratio_template.csv'}")
    print(f"  {base / 'flux' / 'Module_round7.xlsx'}")
