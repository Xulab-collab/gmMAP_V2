#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pan-cancer epithelial metabolism + drug reversal plotting
=========================================================

This script makes CNS-style figures for each cell type with:
  1) Volcano plot of metabolite/trait differences between two groups
  2) Horizontal bar plot of top predicted reversing drugs

Key display changes in this version
-----------------------------------
- Volcano labels use readable trait names instead of raw GCST IDs
- By default only the top 6 upregulated metabolites are labeled
- Larger fonts and cleaner spacing
- Volcano y-axis is automatically capped to avoid being too tall
- Drug prediction panels also use larger labels and cleaner layout

Main outputs
------------
1. Multi-page PDF:
   <outdir>/PanCancer_epithelial_metabolism_drugs_combined.pdf

2. Per-celltype PDFs and PNGs:
   <outdir>/per_celltype/<celltype>.pdf
   <outdir>/per_celltype/<celltype>.png

3. Summary tables:
   <outdir>/plot_summary_labeled_traits.csv
   <outdir>/plot_summary_shown_drugs.csv
"""

import os
import re
import math
import textwrap
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


# -----------------------------------
# Utilities
# -----------------------------------
def nice_label(x):
    x = str(x)
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def wrap_label(x, width=22):
    return "\n".join(textwrap.wrap(nice_label(x), width=width, break_long_words=False))


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.+-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


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


def auto_pick(df, candidates, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    if required:
        raise KeyError(
            f"Cannot find any of columns: {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None


def ensure_numeric(df, cols):
    for c in cols:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 13.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "legend.fontsize": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# -----------------------------------
# Trait table
# -----------------------------------
def build_trait_label_map(args):
    if args.trait_name_table is None:
        return None

    meta = read_table(args.trait_name_table)

    if args.trait_map_id_col == "auto":
        id_col = auto_pick(meta, ["trait", "GCST", "trait_id", "gwas_id", "Trait"], required=False)
    else:
        id_col = args.trait_map_id_col if args.trait_map_id_col in meta.columns else None

    if args.trait_map_name_col == "auto":
        name_col = auto_pick(
            meta,
            ["reportedTrait", "reported_trait", "trait_name", "traitLabel", "trait_label", "metabolite", "name"],
            required=False
        )
    else:
        name_col = args.trait_map_name_col if args.trait_map_name_col in meta.columns else None

    if id_col is None or name_col is None:
        return None

    meta2 = meta[[id_col, name_col]].copy()
    meta2.columns = ["trait_raw", "trait_label"]
    meta2["trait_raw"] = meta2["trait_raw"].astype(str)
    meta2["trait_label"] = meta2["trait_label"].astype(str)
    meta2 = meta2.dropna(subset=["trait_raw", "trait_label"]).drop_duplicates("trait_raw")
    return meta2.set_index("trait_raw")["trait_label"].to_dict()


def prepare_trait_table(df, args):
    celltype_col = args.celltype_col
    if celltype_col == "auto":
        celltype_col = auto_pick(df, ["celltype", "cell_type", "celltype_plot", "cluster", "analysis_unit"])

    trait_col = args.trait_col
    if trait_col == "auto":
        trait_col = auto_pick(df, ["trait", "metabolite", "reportedTrait", "trait_id"])

    effect_col = args.effect_col
    if effect_col == "auto":
        effect_col = auto_pick(
            df,
            ["mean_diff", "z_wilcoxon", "signed_log2FC_absmean", "pseudo_log2FC_shifted", "log2FC"]
        )

    q_col = args.q_col
    if q_col == "auto":
        q_col = auto_pick(
            df,
            ["q_wilcoxon_global", "q_wilcoxon", "q_ttest_global", "q_ttest", "p_wilcoxon", "p_ttest", "pvalue", "p"]
        )

    group1_col = auto_pick(df, ["group1"], required=False)
    group2_col = auto_pick(df, ["group2"], required=False)

    label_col = None
    if args.trait_label_col != "auto" and args.trait_label_col in df.columns:
        label_col = args.trait_label_col
    elif args.trait_label_col == "auto":
        label_col = auto_pick(
            df,
            ["reportedTrait", "reported_trait", "trait_name", "traitLabel", "trait_label", "metabolite_name", "name"],
            required=False
        )

    out = df.copy()
    out = ensure_numeric(out, [effect_col, q_col])

    out["celltype_plot"] = out[celltype_col].astype(str)
    out["trait_plot"] = out[trait_col].astype(str)
    out["effect_plot"] = pd.to_numeric(out[effect_col], errors="coerce")
    out["q_plot"] = pd.to_numeric(out[q_col], errors="coerce")
    out = out.dropna(subset=["celltype_plot", "trait_plot", "effect_plot", "q_plot"]).copy()
    out["q_plot"] = out["q_plot"].clip(lower=1e-300, upper=1.0)
    out["neglog10q_raw"] = -np.log10(out["q_plot"])

    if label_col is not None:
        out["trait_label_plot"] = out[label_col].fillna(out["trait_plot"]).astype(str)
    else:
        out["trait_label_plot"] = out["trait_plot"].astype(str)

    ext_map = build_trait_label_map(args)
    if ext_map is not None:
        out["trait_label_plot"] = out["trait_plot"].map(ext_map).fillna(out["trait_label_plot"])

    if group1_col is not None:
        out["group1_plot"] = out[group1_col].astype(str)
    else:
        out["group1_plot"] = args.group1_name if args.group1_name else "Tumor"

    if group2_col is not None:
        out["group2_plot"] = out[group2_col].astype(str)
    else:
        out["group2_plot"] = args.group2_name if args.group2_name else "Adjacent"

    return out, celltype_col, trait_col, effect_col, q_col


def compute_effect_cutoff(sub, args):
    if args.effect_cutoff != "auto":
        return float(args.effect_cutoff)

    bg = sub[sub["q_plot"] > args.null_q_min]
    if bg.shape[0] >= args.min_background_points:
        cutoff = np.nanquantile(np.abs(bg["effect_plot"].to_numpy(dtype=float)), args.effect_quantile)
    else:
        cutoff = np.nanquantile(np.abs(sub["effect_plot"].to_numpy(dtype=float)), args.effect_quantile_fallback)

    cutoff = max(float(cutoff), args.min_effect_cutoff)
    return cutoff


def add_trait_display_metrics(df, args):
    rows = []
    for ct, sub in df.groupby("celltype_plot", sort=False):
        sub = sub.copy()
        effect_cut = compute_effect_cutoff(sub, args)
        # Volcano y-axis upper limit.
        # If --y_limit is provided, all volcano plots use this fixed upper limit.
        # Otherwise, use a robust data-driven cap controlled by --y_cap_quantile
        # and --max_neglog10q.
        if args.y_limit is not None and args.y_limit > 0:
            ycap = float(args.y_limit)
        else:
            ycap = np.nanquantile(sub["neglog10q_raw"].to_numpy(dtype=float), args.y_cap_quantile)
            ycap = max(ycap, -np.log10(args.q_cutoff) * 1.25, 5.0)
            ycap = min(ycap, args.max_neglog10q)

        sub["effect_cutoff_plot"] = effect_cut
        sub["sig_plot"] = (sub["q_plot"] <= args.q_cutoff) & (np.abs(sub["effect_plot"]) >= effect_cut)
        sub["neglog10q_cap"] = ycap
        sub["q_capped_plot"] = sub["neglog10q_raw"] > ycap
        sub["neglog10q_plot"] = np.minimum(sub["neglog10q_raw"], ycap)
        rows.append(sub)
    return pd.concat(rows, axis=0, ignore_index=True)


# -----------------------------------
# Drug table
# -----------------------------------
def extract_celltype_from_analysis_id(x):
    x = str(x)
    # removes common wrappers from batch drug-ranking outputs
    x = re.sub(r"^trait_score_diff_", "", x)
    x = re.sub(r"\.csv(\.gz)?$", "", x)
    x = re.sub(r"_(Tumor|Normal|Adjacent|Control|Case)_vs_(Tumor|Normal|Adjacent|Control|Case)$", "", x)
    return x


def prepare_drug_table(df, args):
    # resolve celltype
    if args.drug_celltype_col != "auto":
        celltype_col = args.drug_celltype_col
    else:
        celltype_col = auto_pick(df, ["celltype", "cell_type", "celltype_plot", "analysis_id"], required=True)

    # resolve drug name
    drug_col = args.drug_col
    if drug_col == "auto":
        drug_col = auto_pick(df, ["drug", "drug_name", "pert_iname", "compound"], required=True)

    # resolve score
    score_col = args.drug_score_col
    if score_col == "auto":
        score_col = auto_pick(
            df,
            ["final_rank_score", "mean_fit", "median_fit", "drug_score", "score"],
            required=True
        )

    q_col = args.drug_q_col
    if q_col == "auto":
        q_col = auto_pick(df, ["min_qvalue", "qvalue", "q", "drug_q"], required=False)

    rp_col = args.rank_percentile_col
    if rp_col == "auto":
        rp_col = auto_pick(df, ["rank_percentile", "percentile", "rank_pct"], required=False)

    rank_col = auto_pick(df, ["rank", "drug_rank"], required=False)
    rev_col = auto_pick(df, ["reverse_direction_fraction", "reverse_frac"], required=False)

    out = df.copy()
    out = ensure_numeric(out, [score_col, q_col, rp_col, rank_col, rev_col] if rev_col else [score_col, q_col, rp_col, rank_col])

    out["drug_plot"] = out[drug_col].astype(str)
    out["drug_score_plot"] = pd.to_numeric(out[score_col], errors="coerce")
    out["drug_q_plot"] = pd.to_numeric(out[q_col], errors="coerce") if q_col is not None else np.nan
    out["rank_percentile_plot"] = pd.to_numeric(out[rp_col], errors="coerce") if rp_col is not None else np.nan
    out["rank_plot"] = pd.to_numeric(out[rank_col], errors="coerce") if rank_col is not None else np.nan
    out["reverse_direction_fraction"] = pd.to_numeric(out[rev_col], errors="coerce") if rev_col is not None else np.nan

    if celltype_col == "analysis_id":
        out["celltype_plot"] = out[celltype_col].astype(str).map(extract_celltype_from_analysis_id)
    else:
        out["celltype_plot"] = out[celltype_col].astype(str)

    out = out.dropna(subset=["celltype_plot", "drug_plot", "drug_score_plot"]).copy()
    return out


# -----------------------------------
# Plotting
# -----------------------------------
def label_priority(df):
    return (
        -np.log10(np.clip(df["q_plot"].astype(float), 1e-300, 1.0))
        * (np.abs(df["effect_plot"]) + 1e-9)
    )


def annotate_top_traits(ax, sub, n_label=6):
    if n_label <= 0 or sub.empty:
        return []

    label_df = sub.copy()
    label_df = label_df[
        np.isfinite(label_df["effect_plot"]) &
        np.isfinite(label_df["neglog10q_plot"])
    ].copy()

    if label_df.empty:
        return []

    label_df["label_priority"] = label_priority(label_df)
    label_df = label_df.sort_values("label_priority", ascending=False).head(n_label)

    offsets = [(12, 12), (12, -12), (-12, 12), (-12, -12), (16, 0), (-16, 0)]
    labels = []

    for i, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        ha = "left" if dx >= 0 else "right"
        label_text = nice_label(row["trait_label_plot"])
        labels.append(label_text)

        ax.annotate(
            label_text,
            xy=(row["effect_plot"], row["neglog10q_plot"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.0,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.60,
                color="0.45",
                shrinkA=0,
                shrinkB=2
            ),
            zorder=5,
            clip_on=False,
        )
    return labels


def plot_volcano(ax, sub, celltype_name, args):
    if sub.empty:
        ax.text(0.5, 0.5, "No trait data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return []

    nonsig = sub[~sub["sig_plot"]].copy()
    up = sub[sub["sig_plot"] & (sub["effect_plot"] > 0)].copy()
    down = sub[sub["sig_plot"] & (sub["effect_plot"] < 0)].copy()
    capped = sub[sub["q_capped_plot"]].copy()

    color_nonsig = "#CBD0D8"
    color_up = "#C23B22"
    color_down = "#3266B0"
    color_cap = "#111111"

    ax.scatter(
        nonsig["effect_plot"], nonsig["neglog10q_plot"],
        s=args.volcano_point_size, alpha=0.50, linewidths=0, color=color_nonsig, rasterized=True
    )
    ax.scatter(
        down["effect_plot"], down["neglog10q_plot"],
        s=args.volcano_point_size + 5, alpha=0.88, linewidths=0, color=color_down, rasterized=True
    )
    ax.scatter(
        up["effect_plot"], up["neglog10q_plot"],
        s=args.volcano_point_size + 5, alpha=0.88, linewidths=0, color=color_up, rasterized=True
    )

    if capped.shape[0] > 0:
        ax.scatter(
            capped["effect_plot"], capped["neglog10q_plot"],
            s=12, alpha=0.70, linewidths=0, color=color_cap, marker="v", rasterized=True
        )

    effect_cutoff = float(sub["effect_cutoff_plot"].median())
    ax.axvline(0, color="0.35", lw=1.0, linestyle="-")
    ax.axvline(effect_cutoff, color="0.45", lw=0.9, linestyle="--")
    ax.axvline(-effect_cutoff, color="0.45", lw=0.9, linestyle="--")
    ax.axhline(-np.log10(args.q_cutoff), color="0.5", lw=0.9, linestyle="--")

    x = sub["effect_plot"].to_numpy(dtype=float)
    if args.x_limit is not None and args.x_limit > 0:
        xmax = float(args.x_limit)
    else:
        xmax = np.nanquantile(np.abs(x), args.x_quantile) if len(x) > 0 else 1.0
        xmax = max(xmax * 1.10, effect_cutoff * 1.55, args.min_x_limit)
    ax.set_xlim(-xmax, xmax)

    y_cap = float(sub["neglog10q_cap"].median())
    ymax = max(6.0, y_cap * 1.03)
    ax.set_ylim(0, ymax)

    g1 = sub["group1_plot"].dropna().astype(str).iloc[0] if sub["group1_plot"].notna().any() else "Group1"
    g2 = sub["group2_plot"].dropna().astype(str).iloc[0] if sub["group2_plot"].notna().any() else "Group2"

    ax.set_xlabel(f"Metabolite score difference ({g1} - {g2})", fontsize=12.4)
    ax.set_ylabel("-log10(q value)", fontsize=12.4)
    ax.set_title(nice_label(celltype_name), loc="left", pad=10, fontsize=13.2)

    ax.text(0.03, 0.965, f"{g2} higher", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.4, color=color_down, fontweight="bold")
    ax.text(0.97, 0.965, f"{g1} higher", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.4, color=color_up, fontweight="bold")

    sig_text = (
        f"q <= {args.q_cutoff:g}\n"
        f"|effect| >= {effect_cutoff:.3g}\n"
        f"Sig: {int(sub['sig_plot'].sum())}\n"
        f"{g1} up: {up.shape[0]}\n"
        f"{g2} up: {down.shape[0]}"
    )
    ax.text(
        0.985, 0.045, sig_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.9,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.86", alpha=0.95)
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=10.8)
    despine(ax)

    # Label top upregulated metabolites only by default
    label_sub = up.copy() if args.label_only_up else sub[sub["sig_plot"]].copy()
    labels = annotate_top_traits(ax, label_sub, n_label=args.n_label_traits)
    return labels


def select_top_drugs(sub, args):
    sub = sub.copy()
    sub = sub[np.isfinite(sub["drug_score_plot"])].copy()

    if args.only_positive_drug_score:
        sub = sub[sub["drug_score_plot"] > 0].copy()

    if args.min_rank_percentile > 0 and "rank_percentile_plot" in sub.columns:
        sub = sub[
            sub["rank_percentile_plot"].isna() |
            (sub["rank_percentile_plot"] >= args.min_rank_percentile)
        ].copy()

    if args.drug_q_max is not None and "drug_q_plot" in sub.columns:
        sub = sub[sub["drug_q_plot"].isna() | (sub["drug_q_plot"] <= args.drug_q_max)].copy()

    if sub.empty:
        return sub

    if "rank_plot" in sub.columns and sub["rank_plot"].notna().any():
        sub = sub.sort_values(["rank_plot", "drug_score_plot"], ascending=[True, False])
    else:
        sub = sub.sort_values("drug_score_plot", ascending=False)

    sub = sub.head(args.top_n_drugs).copy()
    sub = sub.sort_values("drug_score_plot", ascending=True)
    return sub


def plot_drug_bars(ax, sub, args):
    sub2 = select_top_drugs(sub, args)

    if sub2.empty:
        ax.text(0.5, 0.5, "No drug ranking data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return sub2, []

    if "rank_percentile_plot" in sub2.columns and sub2["rank_percentile_plot"].notna().any():
        rp = sub2["rank_percentile_plot"].fillna(0.5).clip(0, 1).to_numpy()
        cmap = plt.cm.viridis
        colors = [cmap(0.16 + 0.78 * x) for x in rp]
    else:
        colors = ["#2F6F6E"] * sub2.shape[0]

    y = np.arange(sub2.shape[0])
    ax.barh(
        y,
        sub2["drug_score_plot"].to_numpy(dtype=float),
        color=colors,
        edgecolor="none",
        height=0.58
    )

    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, width=args.drug_label_width) for x in sub2["drug_plot"].tolist()])
    ax.axvline(0, color="0.35", lw=1.0)

    vals = sub2["drug_score_plot"].to_numpy(dtype=float)
    xmax = np.nanmax(np.abs(vals)) if len(vals) > 0 else 1.0
    xmax = max(xmax, 0.15)
    xmin = min(0, np.nanmin(vals) if len(vals) > 0 else 0)
    span = xmax - xmin
    if span <= 0:
        span = xmax

    ax.set_xlim(xmin - 0.05 * span, xmax + 0.44 * span)
    ax.set_xlabel("Drug reversal score", fontsize=12.4)
    ax.set_title("Top predicted reversing drugs", loc="left", pad=10, fontsize=13.2)

    shown_drugs = []
    for yi, (_, row) in enumerate(sub2.iterrows()):
        val = float(row["drug_score_plot"])
        rp = row["rank_percentile_plot"] if "rank_percentile_plot" in row.index else np.nan
        txt = f"{val:.2f}"
        if pd.notna(rp):
            txt += f"  RP={rp:.2f}"

        ha = "left" if val >= 0 else "right"
        x_text = val + 0.022 * span if val >= 0 else val - 0.022 * span
        ax.text(x_text, yi, txt, va="center", ha=ha, fontsize=8.8, color="black")
        shown_drugs.append(str(row["drug_plot"]))

    text_bits = [f"n = {sub2.shape[0]}"]
    if "drug_q_plot" in sub2.columns and sub2["drug_q_plot"].notna().any():
        text_bits.append(f"best q = {sub2['drug_q_plot'].min():.1e}")
    if "reverse_direction_fraction" in sub2.columns and sub2["reverse_direction_fraction"].notna().any():
        text_bits.append(f"median reverse = {np.nanmedian(sub2['reverse_direction_fraction']):.2f}")

    ax.text(
        1.0, 1.075, " | ".join(text_bits), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.0, color="0.25", clip_on=False
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=10.8)
    ax.tick_params(axis="y", pad=4)
    despine(ax)

    return sub2, shown_drugs


def plot_one_celltype(trait_sub, drug_sub, celltype_name, args):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(args.single_fig_width, args.single_fig_height),
        gridspec_kw={"width_ratios": [1.05, 1.05]}
    )

    labels = plot_volcano(axes[0], trait_sub, celltype_name, args)
    shown_drug_df, shown_drugs = plot_drug_bars(axes[1], drug_sub, args)

    fig.suptitle(
        "Pan-cancer epithelial metabolic rewiring and predicted reversing drugs",
        x=0.02, y=0.985, ha="left", fontsize=14.2, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, labels, shown_drug_df


def plot_page(celltypes, trait_df, drug_df, args):
    n = len(celltypes)
    fig_h = max(3.8 * n, args.combined_row_height * n)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(args.combined_fig_width, fig_h),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.05, 1.05]}
    )

    labeled_rows = []
    drug_rows = []

    for i, ct in enumerate(celltypes):
        trait_sub = trait_df[trait_df["celltype_plot"] == ct].copy()
        drug_sub = drug_df[drug_df["celltype_plot"] == ct].copy()

        labels = plot_volcano(axes[i, 0], trait_sub, ct, args)
        shown_drug_df, shown_drugs = plot_drug_bars(axes[i, 1], drug_sub, args)

        labeled_rows.append({
            "celltype": ct,
            "labeled_traits": "; ".join(labels)
        })

        for d in shown_drugs:
            drug_rows.append({"celltype": ct, "drug": d})

    fig.suptitle(
        "Pan-cancer epithelial metabolic rewiring and predicted reversing drugs",
        x=0.02, y=0.995, ha="left", fontsize=14.4, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig, labeled_rows, drug_rows


# -----------------------------------
# Main
# -----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot pan-cancer epithelial metabolite rewiring and drug reversal.")
    parser.add_argument("--trait_diff", required=True, help="Trait-difference table.")
    parser.add_argument("--drug_rank", required=True, help="Drug-ranking table.")
    parser.add_argument("--outdir", default="PanCancer_Epithelial_Metabolism_Drugs_v3")

    # trait table
    parser.add_argument("--celltype_col", default="auto")
    parser.add_argument("--trait_col", default="auto")
    parser.add_argument("--effect_col", default="auto")
    parser.add_argument("--q_col", default="auto")
    parser.add_argument("--group1_name", default=None)
    parser.add_argument("--group2_name", default=None)

    # trait label mapping
    parser.add_argument("--trait_label_col", default="auto",
                        help="Readable trait-name column already present in --trait_diff.")
    parser.add_argument("--trait_name_table", default=None,
                        help="Optional external trait metadata table for mapping GCST IDs to trait names.")
    parser.add_argument("--trait_map_id_col", default="auto",
                        help="Trait-ID column in --trait_name_table.")
    parser.add_argument("--trait_map_name_col", default="auto",
                        help="Readable trait-name column in --trait_name_table.")

    # drug table
    parser.add_argument("--drug_celltype_col", default="auto")
    parser.add_argument("--drug_col", default="auto")
    parser.add_argument("--drug_score_col", default="auto")
    parser.add_argument("--drug_q_col", default="auto")
    parser.add_argument("--rank_percentile_col", default="auto")

    # significance / volcano behavior
    parser.add_argument("--q_cutoff", type=float, default=0.05)
    parser.add_argument("--effect_cutoff", default="auto",
                        help='Fixed numeric cutoff or "auto".')
    parser.add_argument("--null_q_min", type=float, default=0.20)
    parser.add_argument("--effect_quantile", type=float, default=0.95)
    parser.add_argument("--effect_quantile_fallback", type=float, default=0.90)
    parser.add_argument("--min_background_points", type=int, default=20)
    parser.add_argument("--min_effect_cutoff", type=float, default=0.03)

    parser.add_argument("--y_cap_quantile", type=float, default=0.98)
    parser.add_argument("--max_neglog10q", type=float, default=25.0)
    parser.add_argument("--y_limit", type=float, default=None,
                        help="Manual upper limit for volcano y-axis, i.e. displayed -log10(q). "
                             "For example, --y_limit 20 fixes all volcano plots to 0-20.")
    parser.add_argument("--x_quantile", type=float, default=0.99)
    parser.add_argument("--min_x_limit", type=float, default=0.20)
    parser.add_argument("--x_limit", type=float, default=None)

    parser.add_argument("--n_label_traits", type=int, default=6)
    parser.add_argument("--label_only_up", action="store_true", default=True,
                        help="Label only upregulated significant metabolites. Default behavior is ON.")

    # drug behavior
    parser.add_argument("--top_n_drugs", type=int, default=6)
    parser.add_argument("--only_positive_drug_score", action="store_true", default=True)
    parser.add_argument("--min_rank_percentile", type=float, default=0.0)
    parser.add_argument("--drug_q_max", type=float, default=None)
    parser.add_argument("--drug_label_width", type=int, default=22)

    # celltype filter
    parser.add_argument("--celltype_list", default=None,
                        help="Comma-separated celltypes to include.")
    parser.add_argument("--celltype_regex", default=None,
                        help="Regex to filter celltypes.")

    # figure
    parser.add_argument("--volcano_point_size", type=float, default=9.0)
    parser.add_argument("--single_fig_width", type=float, default=13.8)
    parser.add_argument("--single_fig_height", type=float, default=5.0)
    parser.add_argument("--combined_fig_width", type=float, default=15.8)
    parser.add_argument("--combined_row_height", type=float, default=4.05)
    parser.add_argument("--celltypes_per_page", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=450)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_dir = os.path.join(args.outdir, "per_celltype")
    os.makedirs(per_dir, exist_ok=True)

    apply_pub_style()

    print("[1] Reading trait-difference table ...")
    trait_raw = read_table(args.trait_diff)
    trait_df, _, _, _, _ = prepare_trait_table(trait_raw, args)
    trait_df = add_trait_display_metrics(trait_df, args)

    print("[2] Reading drug-ranking table ...")
    drug_raw = read_table(args.drug_rank)
    drug_df = prepare_drug_table(drug_raw, args)

    # celltype filtering
    celltypes = sorted(set(trait_df["celltype_plot"].astype(str)))
    if args.celltype_list:
        keep = {x.strip() for x in str(args.celltype_list).split(",") if x.strip()}
        celltypes = [x for x in celltypes if x in keep]
    if args.celltype_regex:
        pat = re.compile(args.celltype_regex, flags=re.IGNORECASE)
        celltypes = [x for x in celltypes if bool(pat.search(x))]

    if len(celltypes) == 0:
        raise RuntimeError("No celltypes left after filtering.")

    print(f"[3] Plotting {len(celltypes)} cell types ...")

    # Per-celltype figures
    summary_labels = []
    summary_drugs = []

    for ct in celltypes:
        trait_sub = trait_df[trait_df["celltype_plot"] == ct].copy()
        drug_sub = drug_df[drug_df["celltype_plot"] == ct].copy()

        fig, labels, shown_drug_df = plot_one_celltype(trait_sub, drug_sub, ct, args)
        pdf_path = os.path.join(per_dir, f"{safe_name(ct)}.pdf")
        png_path = os.path.join(per_dir, f"{safe_name(ct)}.png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        summary_labels.append({"celltype": ct, "labeled_traits": "; ".join(labels)})
        if shown_drug_df is not None and shown_drug_df.shape[0] > 0:
            for _, row in shown_drug_df.iterrows():
                summary_drugs.append({
                    "celltype": ct,
                    "drug": row["drug_plot"],
                    "drug_score": row["drug_score_plot"]
                })

    # Combined multi-page PDF
    combined_pdf = os.path.join(args.outdir, "PanCancer_epithelial_metabolism_drugs_combined.pdf")
    with PdfPages(combined_pdf) as pdf:
        for start in range(0, len(celltypes), args.celltypes_per_page):
            batch = celltypes[start:start + args.celltypes_per_page]
            fig, _, _ = plot_page(batch, trait_df, drug_df, args)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Save summary tables
    pd.DataFrame(summary_labels).to_csv(
        os.path.join(args.outdir, "plot_summary_labeled_traits.csv"),
        index=False
    )
    pd.DataFrame(summary_drugs).to_csv(
        os.path.join(args.outdir, "plot_summary_shown_drugs.csv"),
        index=False
    )

    print("Done.")
    print(f"Combined PDF: {combined_pdf}")
    print(f"Per-celltype directory: {per_dir}")


if __name__ == "__main__":
    main()
