"""Command-line interface for gmMAP."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from gmmap._version import __version__
from gmmap.apps.drug import rank_drug_reversal
from gmmap.apps.flux import compute_flux_potential, load_flux_modules
from gmmap.apps.perturb import compute_perturb_delta, summarize_perturb_by_branch
from gmmap.apps.pseudotime import correlate_with_pseudotime
from gmmap.apps.spatial import export_spatial_trait_table, summarize_by_region
from gmmap.core.association import empirical_null_threshold, one_vs_rest_association
from gmmap.core.scoring import expression_from_adata, score_signatures
from gmmap.core.signatures import build_signatures, signature_table, signatures_from_table, standardize_gene_index
from gmmap.io import ensure_dir, read_h5ad, read_table, write_table
from gmmap.logging_utils import setup_logging


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def build_score_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score gmMAP metabolite-gene programs.")
    _add_common(p)
    p.add_argument("--h5ad", required=True, help="Input AnnData .h5ad file.")
    p.add_argument("--magma-z", required=True, help="Gene x trait MAGMA Z matrix CSV/TSV.")
    p.add_argument("--gene-col", default=None, help="Gene column in MAGMA table if genes are not the index.")
    p.add_argument("--celltype-key", default=None, help="Metadata column copied into obs_metadata.csv.")
    p.add_argument("--layer", default=None, help="AnnData layer to use instead of X.")
    p.add_argument("--traits", nargs="*", default=None, help="Optional subset of traits.")
    p.add_argument("--top-n-genes", type=int, default=1000)
    p.add_argument("--min-valid-genes", type=int, default=200)
    p.add_argument("--score-method", choices=["waucell", "smrs"], default="waucell")
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--stage", default="S3", choices=["S1", "S2", "S3", "S4"])
    p.add_argument("--background", choices=["none", "matched"], default="none",
                   help="matched: separate scDRS-inspired calibration; does not apply S1-S4")
    p.add_argument("--n-ctrl", type=int, default=1000)
    p.add_argument("--n-mean-bins", type=int, default=20)
    p.add_argument("--n-var-bins", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--exclude-target-controls", action="store_true")
    p.add_argument("--variance-alpha", type=float, default=0.0,
                   help="Matched mode only: weight / (expression SD + epsilon)^alpha; 0 preserves core score")
    p.add_argument("--cell-batch-size", type=int, default=128)
    p.add_argument("--sample-key", default=None, help="Optional donor ID for group contrasts in matched mode")
    p.add_argument("--save-control-scores", action="store_true")
    p.add_argument("--background-memory-gb", type=float, default=2.0)
    p.add_argument("--input-is-log1p", action="store_true",
                   help="Confirm matched-mode input is library-normalized nonnegative log1p expression")
    p.add_argument("--out", required=True, help="Output directory.")
    return p


def run_score(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    logging.info("Reading AnnData: %s", args.h5ad)
    adata = read_h5ad(args.h5ad)

    logging.info("Reading MAGMA Z matrix: %s", args.magma_z)
    z = standardize_gene_index(read_table(args.magma_z, index_col=None if args.gene_col else 0), gene_col=args.gene_col)
    if args.background == "matched":
        if not args.input_is_log1p:
            raise ValueError("Matched mode requires --input-is-log1p after verifying normalized log1p input")
        if args.random_seed < 0:
            raise ValueError("random seed must be nonnegative")
        from gmmap.core.background import run_matched_score
        logging.warning("Matched mode uses positive MAGMA association strength, no DOWN/NET; --stage is not applied")
        run_matched_score(
            adata, z, out / "matched_background", layer=args.layer, traits=args.traits,
            top_n=args.top_n_genes, min_genes=args.min_valid_genes, n_ctrl=args.n_ctrl,
            n_mean_bins=args.n_mean_bins, n_var_bins=args.n_var_bins,
            seed=args.random_seed, exclude_target=args.exclude_target_controls,
            method=args.score_method, top_frac=args.top_frac,
            variance_alpha=args.variance_alpha, batch_size=args.cell_batch_size,
            sample_key=args.sample_key, group_key=args.celltype_key,
            save_controls=args.save_control_scores, memory_gb=args.background_memory_gb,
        )
        logging.info("Done: %s", out / "matched_background")
        return
    expr = expression_from_adata(adata, layer=args.layer)
    logging.info("Building signatures")
    signatures = build_signatures(z, traits=args.traits, top_n=args.top_n_genes, min_valid_genes=args.min_valid_genes)
    if not signatures:
        raise RuntimeError("No valid trait signatures were constructed. Check gene overlap and min_valid_genes.")
    write_table(signature_table(signatures), out / "gmmap_signatures_long.csv", index=False)

    logging.info("Scoring %d traits using %s", len(signatures), args.score_method)
    up, down, net, all_matrices = score_signatures(
        expr,
        signatures,
        method=args.score_method,
        top_frac=args.top_frac,
        stage=args.stage,
    )
    write_table(up, out / f"gmmap_scores_UP_{args.stage}.csv")
    write_table(down, out / f"gmmap_scores_DOWN_{args.stage}.csv")
    write_table(net, out / f"gmmap_scores_NET_{args.stage}.csv")
    write_table(net, out / f"gmmap_scores_{args.stage}.csv")

    obs = adata.obs.copy()
    if args.celltype_key and args.celltype_key not in obs.columns:
        raise ValueError(f"celltype_key not found in adata.obs: {args.celltype_key}")
    write_table(obs, out / "obs_metadata.csv")
    logging.info("Done: %s", out)


def build_association_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run gmMAP one-vs-rest association analysis.")
    _add_common(p)
    p.add_argument("--up-scores", default=None, help="UP score CSV/TSV. If omitted, infer from --scores by replacing NET with UP is not attempted.")
    p.add_argument("--down-scores", default=None, help="DOWN score CSV/TSV.")
    p.add_argument("--scores", default=None, help="NET score table; accepted for compatibility but UP/DOWN are recommended.")
    p.add_argument("--metadata", required=True)
    p.add_argument("--celltype-key", required=True)
    p.add_argument("--min-cells-in-type", type=int, default=20)
    p.add_argument("--out", required=True)
    return p


def run_association(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    metadata = read_table(args.metadata, index_col=0)
    if args.up_scores and args.down_scores:
        up = read_table(args.up_scores, index_col=0)
        down = read_table(args.down_scores, index_col=0)
    elif args.scores:
        logging.warning("Only --scores was provided; using the same table as UP and negative scores as DOWN fallback.")
        net = read_table(args.scores, index_col=0)
        up = net
        down = -net
    else:
        raise ValueError("Provide --up-scores and --down-scores, or provide --scores as a fallback.")
    assoc = one_vs_rest_association(up, down, metadata, args.celltype_key, min_cells_in_type=args.min_cells_in_type)
    threshold = empirical_null_threshold(assoc)
    write_table(assoc, out / "gmmap_association.csv", index=False)
    pd.DataFrame([{"empirical_null_abs_effect_threshold_q95": threshold}]).to_csv(out / "empirical_null_threshold.csv", index=False)
    logging.info("Done: %s", out)


def build_spatial_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export spatial gmMAP trait tables or region summaries.")
    _add_common(p)
    p.add_argument("--h5ad", default=None, help="Optional h5ad for metadata.")
    p.add_argument("--metadata", default=None, help="Metadata CSV/TSV if --h5ad is not used.")
    p.add_argument("--scores", required=True)
    p.add_argument("--trait", default=None)
    p.add_argument("--region-key", default=None)
    p.add_argument("--sample-key", default="Sample")
    p.add_argument("--x-key", default="x_image")
    p.add_argument("--y-key", default="y_image")
    p.add_argument("--out", required=True)
    return p


def run_spatial(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    scores = read_table(args.scores, index_col=0)
    if args.h5ad:
        metadata = read_h5ad(args.h5ad).obs
    elif args.metadata:
        metadata = read_table(args.metadata, index_col=0)
    else:
        raise ValueError("Provide --h5ad or --metadata.")

    if args.trait:
        table = export_spatial_trait_table(scores, metadata, args.trait, args.x_key, args.y_key, args.sample_key)
        write_table(table, out / "spatial_trait_table.csv", index=False)
    if args.region_key:
        summary = summarize_by_region(scores, metadata, args.region_key)
        write_table(summary, out / "spatial_region_summary.csv")
    logging.info("Done: %s", out)


def build_pseudotime_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pseudotime-metabolite association.")
    _add_common(p)
    p.add_argument("--scores", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--pseudotime-key", required=True)
    p.add_argument("--traits", nargs="*", default=None)
    p.add_argument("--group-key", default=None)
    p.add_argument("--out", required=True)
    return p


def run_pseudotime(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    scores = read_table(args.scores, index_col=0)
    metadata = read_table(args.metadata, index_col=0)
    result = correlate_with_pseudotime(scores, metadata, args.pseudotime_key, args.traits, args.group_key)
    write_table(result, out / "gmmap_pseudotime_spearman.csv", index=False)
    logging.info("Done: %s", out)


def build_flux_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run gmMAP-flux module scoring.")
    _add_common(p)
    p.add_argument("--scores", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--modules", required=True)
    p.add_argument("--celltype-key", required=True)
    p.add_argument("--out", required=True)
    return p


def run_flux(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    scores = read_table(args.scores, index_col=0)
    metadata = read_table(args.metadata, index_col=0)
    modules = load_flux_modules(args.modules)
    result = compute_flux_potential(scores, metadata, modules, args.celltype_key)
    write_table(result, out / "gmmap_flux_potential.csv", index=False)
    logging.info("Done: %s", out)


def build_perturb_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run gmMAP-perturb delta scoring.")
    _add_common(p)
    p.add_argument("--activity-scores", required=True)
    p.add_argument("--inhibition-scores", required=True)
    p.add_argument("--metadata", default=None)
    p.add_argument("--branch-key", default=None)
    p.add_argument("--pseudotime-key", default=None)
    p.add_argument("--out", required=True)
    return p


def run_perturb(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    activity = read_table(args.activity_scores, index_col=0)
    inhibition = read_table(args.inhibition_scores, index_col=0)
    delta = compute_perturb_delta(activity, inhibition)
    write_table(delta, out / "gmmap_perturb_delta.csv")
    if args.metadata and args.branch_key:
        metadata = read_table(args.metadata, index_col=0)
        summary = summarize_perturb_by_branch(delta, metadata, args.branch_key, args.pseudotime_key)
        write_table(summary, out / "gmmap_perturb_branch_summary.csv", index=False)
    logging.info("Done: %s", out)


def build_drug_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run gmMAP-drug reversal ranking.")
    _add_common(p)
    p.add_argument("--disease-signature", required=True)
    p.add_argument("--drug-signatures", required=True)
    p.add_argument("--trait-col", default="trait")
    p.add_argument("--disease-effect-col", default="effect")
    p.add_argument("--drug-col", default="drug")
    p.add_argument("--drug-effect-col", default="effect")
    p.add_argument("--out", required=True)
    return p


def run_drug(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    out = ensure_dir(args.out)
    disease = read_table(args.disease_signature)
    drugs = read_table(args.drug_signatures)
    ranking = rank_drug_reversal(
        disease,
        drugs,
        trait_col=args.trait_col,
        disease_effect_col=args.disease_effect_col,
        drug_col=args.drug_col,
        drug_effect_col=args.drug_effect_col,
    )
    write_table(ranking, out / "gmmap_drug_reversal_ranking.csv", index=False)
    logging.info("Done: %s", out)


def _dispatch() -> None:
    """Dispatch gmmap subcommands without duplicating parser definitions."""
    import sys

    from gmmap.core.bulk_smrs import main as bulk_smrs_main

    commands = {
        "bulk-smrs": bulk_smrs_main,
        "score": score_main,
        "association": association_main,
        "spatial": spatial_main,
        "pseudotime": pseudotime_main,
        "flux": flux_main,
        "perturb": perturb_main,
        "drug": drug_main,
    }
    if len(sys.argv) <= 1 or sys.argv[1] in {"-h", "--help"}:
        print("gmMAP command-line interface")
        print("\nUsage: gmmap <command> [options]")
        print("\nCommands:")
        for name in commands:
            print(f"  {name}")
        print("\nExamples:")
        print("  gmmap score --help")
        print("  gmmap association --help")
        return
    if sys.argv[1] in {"--version", "-V"}:
        print(f"gmmap {__version__}")
        return
    command = sys.argv[1]
    if command not in commands:
        raise SystemExit(f"Unknown gmmap command: {command}. Use 'gmmap --help'.")
    sys.argv = [f"gmmap-{command}"] + sys.argv[2:]
    commands[command]()


def main() -> None:
    _dispatch()


def score_main() -> None:
    args = build_score_parser().parse_args()
    run_score(args)


def association_main() -> None:
    args = build_association_parser().parse_args()
    run_association(args)


def spatial_main() -> None:
    args = build_spatial_parser().parse_args()
    run_spatial(args)


def pseudotime_main() -> None:
    args = build_pseudotime_parser().parse_args()
    run_pseudotime(args)


def flux_main() -> None:
    args = build_flux_parser().parse_args()
    run_flux(args)


def perturb_main() -> None:
    args = build_perturb_parser().parse_args()
    run_perturb(args)


def drug_main() -> None:
    args = build_drug_parser().parse_args()
    run_drug(args)
