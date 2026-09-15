#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch rank drugs by reversing group-differential trait/metabolite score patterns
across many cell-type differential result files.

Input
-----
1. A folder containing trait differential result files, for example:
   TraitScore_GroupDiff_allCelltypes/
     trait_score_diff_cSCC_Epi_cSCC_Tumor_vs_Adjacent.csv
     trait_score_diff_Myeloid_cSCC_Tumor_vs_Adjacent.csv
     trait_score_diff_Tcell_cSCC_Tumor_vs_Adjacent.csv

2. custom CMap drug-level result:
   custom_CMap_ALL_drug_level.csv.gz

Output
------
For each cell type:
  per_celltype/<analysis_id>/
    drug_reverse_ranking.csv
    drug_reverse_ranking_trait_contributions.csv.gz
    matched_trait_drug_input_table.csv.gz
    drug_reverse_ranking_summary.txt

Merged outputs:
  ALL_celltype_drug_reversal_rankings.csv.gz
  ALL_celltype_drug_reversal_trait_contributions.csv.gz
  ALL_celltype_drug_reversal_TOP.xlsx
  drug_reversal_summary_across_celltypes.csv
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.+-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def pick_col(df, candidates):
    if df is None or df.empty:
        return None

    lower = {str(c).lower(): c for c in df.columns}
    for x in candidates:
        if str(x).lower() in lower:
            return lower[str(x).lower()]
    return None


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q

    pv = p[ok]
    n = len(pv)

    order = np.argsort(pv)
    ranked = pv[order]

    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)

    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked

    q[ok] = q_valid
    return q


def read_table(path):
    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".csv.gz"):
        df = pd.read_csv(path, compression="infer", low_memory=False)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    elif path.endswith(".tsv.gz"):
        df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def infer_analysis_id_from_file(path):
    base = os.path.basename(path)
    base = re.sub(r"\.csv(\.gz)?$", "", base)
    return safe_name(base)


def infer_metadata_from_trait_diff(df, file_path):
    """
    Infer celltype/group information from columns if possible.
    """

    analysis_id = infer_analysis_id_from_file(file_path)

    celltype = "NA"
    group1 = "NA"
    group2 = "NA"
    group_col = "NA"

    if "celltype" in df.columns:
        vals = df["celltype"].dropna().astype(str).unique()
        if len(vals) == 1:
            celltype = vals[0]

    if "group1" in df.columns:
        vals = df["group1"].dropna().astype(str).unique()
        if len(vals) == 1:
            group1 = vals[0]

    if "group2" in df.columns:
        vals = df["group2"].dropna().astype(str).unique()
        if len(vals) == 1:
            group2 = vals[0]

    if "group_col" in df.columns:
        vals = df["group_col"].dropna().astype(str).unique()
        if len(vals) == 1:
            group_col = vals[0]

    if celltype == "NA":
        # Try to parse from filename:
        # trait_score_diff_<celltype>_<group1>_vs_<group2>
        base = re.sub(r"\.csv(\.gz)?$", "", os.path.basename(file_path))
        base2 = base.replace("trait_score_diff_", "")

        m = re.match(r"(.+?)_(.+?)_vs_(.+)$", base2)
        if m:
            celltype = m.group(1)
            if group1 == "NA":
                group1 = m.group(2)
            if group2 == "NA":
                group2 = m.group(3)

    return {
        "analysis_id": analysis_id,
        "celltype": celltype,
        "group1": group1,
        "group2": group2,
        "group_col": group_col,
    }


def standardize_by_trait(df, trait_col, score_col):
    def _zscore(s):
        x = pd.to_numeric(s, errors="coerce")
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)

        if not np.isfinite(sd) or sd == 0:
            return x * 0.0

        return (x - mu) / sd

    return df.groupby(trait_col, dropna=False)[score_col].transform(_zscore)


def load_cmap_drug(args):
    print(f"[1] Reading custom CMap drug-level result: {args.cmap_drug}")
    df = read_table(args.cmap_drug)

    if args.cmap_trait_col not in df.columns:
        raise KeyError(
            f"{args.cmap_trait_col} not found in CMap drug file.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if args.score_col not in df.columns:
        raise KeyError(
            f"{args.score_col} not found in CMap drug file.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    drug_col = args.drug_col

    if drug_col is None or str(drug_col).lower() == "auto":
        drug_col = pick_col(
            df,
            [
                "drug_name",
                "drug",
                "pert_iname",
                "compound_name",
                "compound",
                "group1",
                "case_group",
                "signature_id"
            ]
        )

    if drug_col is None or drug_col not in df.columns:
        raise KeyError(
            "Cannot infer drug column. Please provide --drug_col.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"    drug column: {drug_col}")
    print(f"    trait column: {args.cmap_trait_col}")
    print(f"    score column: {args.score_col}")

    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")

    df = df.dropna(subset=[args.cmap_trait_col, drug_col, args.score_col])
    df = df[np.isfinite(df[args.score_col])].copy()

    if args.cmap_q_col is not None and str(args.cmap_q_col).lower() != "none":
        if args.cmap_q_col in df.columns and args.cmap_q_max < 1:
            df[args.cmap_q_col] = pd.to_numeric(df[args.cmap_q_col], errors="coerce")
            before = df.shape[0]
            df = df[df[args.cmap_q_col] <= args.cmap_q_max].copy()
            print(
                f"    filter {args.cmap_q_col} <= {args.cmap_q_max}: "
                f"{before:,} -> {df.shape[0]:,}"
            )

    if args.min_abs_drug_score > 0:
        before = df.shape[0]
        df = df[df[args.score_col].abs() >= args.min_abs_drug_score].copy()
        print(
            f"    filter abs({args.score_col}) >= {args.min_abs_drug_score}: "
            f"{before:,} -> {df.shape[0]:,}"
        )

    df["trait"] = df[args.cmap_trait_col].astype(str)
    df["drug"] = df[drug_col].astype(str)
    df["drug_score_raw"] = pd.to_numeric(df[args.score_col], errors="coerce")

    df = df[
        df["trait"].notna() &
        df["drug"].notna() &
        df["drug_score_raw"].notna() &
        np.isfinite(df["drug_score_raw"])
    ].copy()

    if args.standardize_drug_by_trait:
        print("    standardize drug scores within each trait: YES")
        df["drug_score_used"] = standardize_by_trait(
            df,
            trait_col="trait",
            score_col="drug_score_raw"
        )
    else:
        print("    standardize drug scores within each trait: NO")
        df["drug_score_used"] = df["drug_score_raw"]

    df = df[np.isfinite(df["drug_score_used"])].copy()

    if df.empty:
        raise RuntimeError("No CMap drug rows left after filtering.")

    print(f"    CMap rows used: {df.shape[0]:,}")
    print(f"    traits in CMap table: {df['trait'].nunique():,}")
    print(f"    drugs in CMap table: {df['drug'].nunique():,}")

    return df


def load_trait_diff_one_file(path, args):
    df = read_table(path)

    if args.trait_col not in df.columns:
        raise KeyError(
            f"{args.trait_col} not found in trait_diff file: {path}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if args.effect_col not in df.columns:
        raise KeyError(
            f"{args.effect_col} not found in trait_diff file: {path}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    meta = infer_metadata_from_trait_diff(df, path)

    df[args.effect_col] = pd.to_numeric(df[args.effect_col], errors="coerce")

    df = df.dropna(subset=[args.trait_col, args.effect_col])
    df = df[np.isfinite(df[args.effect_col])].copy()

    if args.q_col is not None and str(args.q_col).lower() != "none":
        if args.q_col in df.columns and args.q_max < 1:
            df[args.q_col] = pd.to_numeric(df[args.q_col], errors="coerce")
            before = df.shape[0]
            df = df[df[args.q_col] <= args.q_max].copy()
            print(f"      filter {args.q_col} <= {args.q_max}: {before:,} -> {df.shape[0]:,}")
        elif args.q_col not in df.columns:
            print(f"      Warning: q_col {args.q_col} not found; q-value filter skipped.")

    if args.p_col is not None and str(args.p_col).lower() != "none":
        if args.p_col in df.columns and args.p_max < 1:
            df[args.p_col] = pd.to_numeric(df[args.p_col], errors="coerce")
            before = df.shape[0]
            df = df[df[args.p_col] <= args.p_max].copy()
            print(f"      filter {args.p_col} <= {args.p_max}: {before:,} -> {df.shape[0]:,}")

    if args.min_abs_effect > 0:
        before = df.shape[0]
        df = df[df[args.effect_col].abs() >= args.min_abs_effect].copy()
        print(
            f"      filter abs({args.effect_col}) >= {args.min_abs_effect}: "
            f"{before:,} -> {df.shape[0]:,}"
        )

    if df.empty:
        return pd.DataFrame(), meta

    # One row per trait
    df = (
        df.sort_values(args.effect_col, key=lambda x: np.abs(x), ascending=False)
          .drop_duplicates(args.trait_col, keep="first")
          .copy()
    )

    df["trait"] = df[args.trait_col].astype(str)
    df["group_effect"] = pd.to_numeric(df[args.effect_col], errors="coerce")
    df["group_effect_sign"] = np.sign(df["group_effect"])

    df = df[
        df["group_effect"].notna() &
        np.isfinite(df["group_effect"]) &
        (df["group_effect_sign"] != 0)
    ].copy()

    return df, meta


def compute_reversal_scores(merged, args):
    rows = []
    contrib_rows = []

    mode_factor = -1.0 if args.mode == "reverse" else 1.0

    for drug, sub in merged.groupby("drug", dropna=False):
        sub = sub.copy()

        if sub.shape[0] < args.min_traits_per_drug:
            continue

        effect = sub["group_effect"].to_numpy(dtype=float)
        drug_score = sub["drug_score_used"].to_numpy(dtype=float)

        finite = np.isfinite(effect) & np.isfinite(drug_score)

        if finite.sum() < args.min_traits_per_drug:
            continue

        effect = effect[finite]
        drug_score = drug_score[finite]
        sub = sub.iloc[np.where(finite)[0]].copy()

        weights = np.abs(effect)

        if args.max_weight is not None and args.max_weight > 0:
            weights = np.minimum(weights, args.max_weight)

        if not np.isfinite(weights).any() or np.nansum(weights) <= 0:
            continue

        effect_sign = np.sign(effect)
        drug_sign = np.sign(drug_score)

        reverse_mask = effect_sign * drug_sign < 0
        mimic_mask = effect_sign * drug_sign > 0

        n_traits = len(effect)
        n_reverse = int(np.sum(reverse_mask))
        n_mimic = int(np.sum(mimic_mask))
        n_neutral = int(n_traits - n_reverse - n_mimic)

        reverse_fraction = n_reverse / n_traits
        mimic_fraction = n_mimic / n_traits

        per_trait_fit = mode_factor * effect_sign * drug_score

        weighted_fit = float(np.nansum(weights * per_trait_fit) / np.nansum(weights))
        mean_fit = float(np.nanmean(per_trait_fit))
        median_fit = float(np.nanmedian(per_trait_fit))

        denom = np.sqrt(np.nansum(effect ** 2) * np.nansum(drug_score ** 2))
        cosine_fit = (
            float(mode_factor * np.nansum(effect * drug_score) / denom)
            if denom > 0 else np.nan
        )

        try:
            direction_success = n_reverse if args.mode == "reverse" else n_mimic
            p_direction = binomtest(
                direction_success,
                n_traits,
                p=0.5,
                alternative="greater"
            ).pvalue
        except Exception:
            p_direction = np.nan

        direction_fraction_for_rank = reverse_fraction if args.mode == "reverse" else mimic_fraction

        rows.append({
            "drug": drug,
            "mode": args.mode,
            "n_traits_used": n_traits,
            "n_reverse_direction": n_reverse,
            "n_mimic_direction": n_mimic,
            "n_neutral_direction": n_neutral,
            "reverse_direction_fraction": reverse_fraction,
            "mimic_direction_fraction": mimic_fraction,
            "weighted_fit_score": weighted_fit,
            "mean_fit_score": mean_fit,
            "median_fit_score": median_fit,
            "cosine_fit_score": cosine_fit,
            "direction_binom_p": p_direction,
            "mean_abs_group_effect": float(np.nanmean(np.abs(effect))),
            "mean_abs_drug_score_raw": float(np.nanmean(np.abs(sub["drug_score_raw"].to_numpy(dtype=float)))),
            "mean_abs_drug_score_used": float(np.nanmean(np.abs(drug_score))),
            "direction_fraction_for_rank": direction_fraction_for_rank,
        })

        sub["per_trait_fit_score"] = per_trait_fit
        sub["per_trait_weight"] = weights
        sub["per_trait_weighted_contribution"] = weights * per_trait_fit
        sub["is_reverse_direction"] = reverse_mask
        sub["is_mimic_direction"] = mimic_mask
        sub["drug_for_ranking"] = drug

        contrib_rows.append(sub)

    rank = pd.DataFrame(rows)

    if rank.empty:
        return rank, pd.DataFrame()

    rank["direction_binom_q"] = bh_fdr(rank["direction_binom_p"].to_numpy(dtype=float))

    rank["cosine_fit_score_filled"] = rank["cosine_fit_score"].fillna(0.0)
    rank["weighted_fit_score_filled"] = rank["weighted_fit_score"].fillna(0.0)
    rank["direction_fraction_filled"] = rank["direction_fraction_for_rank"].fillna(0.0)

    rank["final_rank_score"] = (
        rank["cosine_fit_score_filled"] *
        np.sqrt(rank["n_traits_used"].astype(float)) *
        rank["direction_fraction_filled"]
    )

    sort_cols = [
        "final_rank_score",
        "weighted_fit_score",
        "cosine_fit_score",
        "direction_fraction_for_rank",
        "n_traits_used",
        "direction_binom_q",
    ]

    sort_cols = [c for c in sort_cols if c in rank.columns]

    ascending_map = {
        "final_rank_score": False,
        "weighted_fit_score": False,
        "cosine_fit_score": False,
        "direction_fraction_for_rank": False,
        "n_traits_used": False,
        "direction_binom_q": True,
    }

    rank = rank.sort_values(
        sort_cols,
        ascending=[ascending_map[c] for c in sort_cols]
    ).reset_index(drop=True)

    rank["rank"] = np.arange(1, rank.shape[0] + 1)

    contrib = (
        pd.concat(contrib_rows, axis=0, ignore_index=True)
        if len(contrib_rows) > 0 else pd.DataFrame()
    )

    if not contrib.empty:
        rank_key = rank[["drug", "rank", "final_rank_score"]].copy()
        rank_key = rank_key.rename(columns={"drug": "drug_for_ranking"})

        contrib = contrib.merge(
            rank_key,
            on="drug_for_ranking",
            how="left"
        )

        contrib = contrib.sort_values(
            ["rank", "per_trait_weighted_contribution"],
            ascending=[True, False]
        )

    return rank, contrib


def summarize_drugs_across_celltypes(all_rank):
    if all_rank.empty:
        return pd.DataFrame()

    df = all_rank.copy()

    summary = (
        df.groupby("drug", dropna=False)
          .agg(
              n_celltype_analyses=("analysis_id", "nunique"),
              n_total_rank_rows=("drug", "count"),
              median_final_rank_score=("final_rank_score", "median"),
              mean_final_rank_score=("final_rank_score", "mean"),
              max_final_rank_score=("final_rank_score", "max"),
              median_weighted_fit_score=("weighted_fit_score", "median"),
              median_cosine_fit_score=("cosine_fit_score", "median"),
              median_reverse_direction_fraction=("reverse_direction_fraction", "median"),
              median_n_traits_used=("n_traits_used", "median"),
              best_rank=("rank", "min"),
          )
          .reset_index()
    )

    top10 = (
        df[df["rank"] <= 10]
        .groupby("drug", dropna=False)
        .size()
        .rename("n_top10_celltypes")
        .reset_index()
    )

    top20 = (
        df[df["rank"] <= 20]
        .groupby("drug", dropna=False)
        .size()
        .rename("n_top20_celltypes")
        .reset_index()
    )

    top50 = (
        df[df["rank"] <= 50]
        .groupby("drug", dropna=False)
        .size()
        .rename("n_top50_celltypes")
        .reset_index()
    )

    summary = summary.merge(top10, on="drug", how="left")
    summary = summary.merge(top20, on="drug", how="left")
    summary = summary.merge(top50, on="drug", how="left")

    for c in ["n_top10_celltypes", "n_top20_celltypes", "n_top50_celltypes"]:
        summary[c] = summary[c].fillna(0).astype(int)

    summary = summary.sort_values(
        [
            "n_top10_celltypes",
            "n_top20_celltypes",
            "median_final_rank_score",
            "mean_final_rank_score",
            "best_rank"
        ],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    summary["global_drug_rank"] = np.arange(1, summary.shape[0] + 1)

    return summary


def process_one_trait_diff_file(path, cmap, args, out_base):
    print("\n" + "=" * 100)
    print(f"[Trait diff] {path}")

    trait_diff, meta = load_trait_diff_one_file(path, args)

    if trait_diff.empty:
        return None, None, {
            **meta,
            "file": path,
            "reason": "no_trait_after_filter"
        }

    print(f"    traits after filtering: {trait_diff.shape[0]:,}")

    keep_diff_cols = [
        "trait",
        "group_effect",
        "group_effect_sign",
        "mean_diff",
        "z_wilcoxon",
        "p_wilcoxon",
        "q_wilcoxon",
        "q_wilcoxon_global",
        "z_welch_approx",
        "p_ttest",
        "q_ttest",
        "q_ttest_global",
        "log2FC",
        "pseudo_log2FC_shifted",
        "signed_log2FC_absmean",
        "direction",
        "celltype",
        "group1",
        "group2",
        "celltype_key",
        "group_col",
    ]

    keep_diff_cols = [c for c in keep_diff_cols if c in trait_diff.columns]

    merged = cmap.merge(
        trait_diff[keep_diff_cols],
        on="trait",
        how="inner"
    )

    print(f"    matched rows: {merged.shape[0]:,}")
    print(f"    matched traits: {merged['trait'].nunique() if not merged.empty else 0:,}")
    print(f"    matched drugs: {merged['drug'].nunique() if not merged.empty else 0:,}")

    if merged.empty:
        return None, None, {
            **meta,
            "file": path,
            "reason": "no_matched_traits_with_cmap"
        }

    rank, contrib = compute_reversal_scores(merged, args)

    if rank.empty:
        return None, None, {
            **meta,
            "file": path,
            "reason": "no_drug_ranking_generated"
        }

    analysis_id = meta["analysis_id"]
    celltype = meta["celltype"]
    group1 = meta["group1"]
    group2 = meta["group2"]

    rank["analysis_id"] = analysis_id
    rank["celltype"] = celltype
    rank["group1"] = group1
    rank["group2"] = group2
    rank["trait_diff_file"] = os.path.abspath(path)

    if not contrib.empty:
        contrib["analysis_id"] = analysis_id
        contrib["celltype"] = celltype
        contrib["group1"] = group1
        contrib["group2"] = group2
        contrib["trait_diff_file"] = os.path.abspath(path)

    one_outdir = os.path.join(out_base, "per_celltype", analysis_id)
    os.makedirs(one_outdir, exist_ok=True)

    rank_path = os.path.join(one_outdir, "drug_reverse_ranking.csv")
    contrib_path = os.path.join(one_outdir, "drug_reverse_ranking_trait_contributions.csv.gz")
    merged_path = os.path.join(one_outdir, "matched_trait_drug_input_table.csv.gz")
    summary_path = os.path.join(one_outdir, "drug_reverse_ranking_summary.txt")

    rank.to_csv(rank_path, index=False)
    contrib.to_csv(contrib_path, index=False, compression="gzip")
    merged.to_csv(merged_path, index=False, compression="gzip")

    with open(summary_path, "w") as f:
        f.write("Drug reversal ranking summary\n")
        f.write("=============================\n")
        f.write(f"analysis_id: {analysis_id}\n")
        f.write(f"celltype: {celltype}\n")
        f.write(f"group1: {group1}\n")
        f.write(f"group2: {group2}\n")
        f.write(f"trait_diff_file: {path}\n")
        f.write(f"traits_after_filter: {trait_diff.shape[0]}\n")
        f.write(f"matched_rows: {merged.shape[0]}\n")
        f.write(f"matched_traits: {merged['trait'].nunique()}\n")
        f.write(f"matched_drugs: {merged['drug'].nunique()}\n")
        f.write(f"ranked_drugs: {rank.shape[0]}\n")
        f.write(f"mode: {args.mode}\n")
        f.write(f"effect_col: {args.effect_col}\n")
        f.write(f"q_col: {args.q_col}\n")
        f.write(f"q_max: {args.q_max}\n")
        f.write(f"score_col: {args.score_col}\n")
        f.write(f"standardize_drug_by_trait: {args.standardize_drug_by_trait}\n")

    return rank, contrib, None


def main():
    parser = argparse.ArgumentParser(
        description="Batch drug reversal ranking for all celltype trait-diff result files."
    )

    parser.add_argument("--trait_diff_dir", required=True)
    parser.add_argument(
        "--pattern",
        default="trait_score_diff_*.csv",
        help="Input trait-diff file pattern. Use '*.csv' or '*.csv.gz' as needed."
    )
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--cmap_drug", required=True)
    parser.add_argument("--outdir", default="Drug_Reversal_Batch_outputs")

    parser.add_argument("--trait_col", default="trait")
    parser.add_argument("--effect_col", default="z_wilcoxon")
    parser.add_argument("--q_col", default="q_wilcoxon")
    parser.add_argument("--q_max", type=float, default=0.05)
    parser.add_argument("--p_col", default=None)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--min_abs_effect", type=float, default=0.0)

    parser.add_argument("--cmap_trait_col", default="trait")
    parser.add_argument("--drug_col", default="auto")
    parser.add_argument("--score_col", default="median_score")
    parser.add_argument("--cmap_q_col", default=None)
    parser.add_argument("--cmap_q_max", type=float, default=1.0)
    parser.add_argument("--min_abs_drug_score", type=float, default=0.0)

    parser.add_argument(
        "--mode",
        choices=["reverse", "mimic"],
        default="reverse"
    )
    parser.add_argument("--min_traits_per_drug", type=int, default=3)
    parser.add_argument("--max_weight", type=float, default=10.0)
    parser.add_argument("--standardize_drug_by_trait", action="store_true")

    parser.add_argument("--top_n_excel_per_celltype", type=int, default=50)
    parser.add_argument("--top_n_excel_global", type=int, default=500)
    parser.add_argument("--excel_max_rows", type=int, default=1000000)

    parser.add_argument(
        "--skip_allcelltypes_merged",
        action="store_true",
        help="Skip files containing ALLCELLTYPES in filename."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cmap = load_cmap_drug(args)

    if args.recursive:
        files = sorted(
            glob.glob(
                os.path.join(args.trait_diff_dir, "**", args.pattern),
                recursive=True
            )
        )
    else:
        files = sorted(glob.glob(os.path.join(args.trait_diff_dir, args.pattern)))

    # Exclude significant-only / summary / skipped / merged files by default
    clean_files = []
    for f in files:
        b = os.path.basename(f)

        if "_significant_" in b:
            continue
        if "summary" in b:
            continue
        if "skipped" in b:
            continue

        if args.skip_allcelltypes_merged and "ALLCELLTYPES" in b:
            continue

        clean_files.append(f)

    files = clean_files

    if len(files) == 0:
        raise FileNotFoundError(
            f"No trait-diff files found in {args.trait_diff_dir} with pattern {args.pattern}"
        )

    print(f"\n[2] Found trait-diff files: {len(files):,}")

    all_rank_list = []
    all_contrib_list = []
    failed = []

    for i, path in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] Processing {os.path.basename(path)}")

        try:
            rank, contrib, fail = process_one_trait_diff_file(
                path=path,
                cmap=cmap,
                args=args,
                out_base=args.outdir
            )

            if fail is not None:
                failed.append(fail)
                print(f"    Failed: {fail['reason']}")
                continue

            all_rank_list.append(rank)

            if contrib is not None and not contrib.empty:
                all_contrib_list.append(contrib)

            print(f"    OK: ranked drugs = {rank.shape[0]:,}")

        except Exception as e:
            failed.append({
                "analysis_id": infer_analysis_id_from_file(path),
                "celltype": "NA",
                "group1": "NA",
                "group2": "NA",
                "file": path,
                "reason": str(e)
            })
            print(f"    ERROR: {e}")
            continue

    if len(all_rank_list) == 0:
        failed_path = os.path.join(args.outdir, "failed_trait_diff_files.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        raise RuntimeError(f"No valid ranking generated. Failed table: {failed_path}")

    all_rank = pd.concat(all_rank_list, axis=0, ignore_index=True)

    all_rank_path = os.path.join(
        args.outdir,
        "ALL_celltype_drug_reversal_rankings.csv.gz"
    )
    all_rank.to_csv(all_rank_path, index=False, compression="gzip")

    if len(all_contrib_list) > 0:
        all_contrib = pd.concat(all_contrib_list, axis=0, ignore_index=True)
    else:
        all_contrib = pd.DataFrame()

    all_contrib_path = os.path.join(
        args.outdir,
        "ALL_celltype_drug_reversal_trait_contributions.csv.gz"
    )
    all_contrib.to_csv(all_contrib_path, index=False, compression="gzip")

    drug_summary = summarize_drugs_across_celltypes(all_rank)

    drug_summary_path = os.path.join(
        args.outdir,
        "drug_reversal_summary_across_celltypes.csv"
    )
    drug_summary.to_csv(drug_summary_path, index=False)

    if len(failed) > 0:
        failed_path = os.path.join(args.outdir, "failed_trait_diff_files.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
    else:
        failed_path = None

    xlsx_path = os.path.join(args.outdir, "ALL_celltype_drug_reversal_TOP.xlsx")

    # Excel top tables
    top_per_celltype = (
        all_rank.sort_values(["analysis_id", "rank"], ascending=[True, True])
        .groupby("analysis_id", group_keys=False)
        .head(args.top_n_excel_per_celltype)
        .reset_index(drop=True)
    )

    top_global = all_rank.sort_values(
        ["final_rank_score", "weighted_fit_score", "cosine_fit_score"],
        ascending=[False, False, False]
    ).head(args.top_n_excel_global)

    with pd.ExcelWriter(xlsx_path) as writer:
        top_per_celltype.head(args.excel_max_rows).to_excel(
            writer,
            sheet_name="top_per_celltype",
            index=False
        )
        top_global.head(args.excel_max_rows).to_excel(
            writer,
            sheet_name="top_global_rows",
            index=False
        )
        drug_summary.head(args.excel_max_rows).to_excel(
            writer,
            sheet_name="drug_summary",
            index=False
        )
        if failed_path is not None:
            pd.DataFrame(failed).head(args.excel_max_rows).to_excel(
                writer,
                sheet_name="failed_files",
                index=False
            )

    print("\nDone.")
    print(f"Output directory: {args.outdir}")
    print(f"All rankings: {all_rank_path}")
    print(f"All trait contributions: {all_contrib_path}")
    print(f"Drug summary across celltypes: {drug_summary_path}")
    print(f"Excel TOP: {xlsx_path}")

    if failed_path is not None:
        print(f"Failed files: {failed_path}")

    print("\nTop 20 cross-celltype summary drugs:")
    print(drug_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
