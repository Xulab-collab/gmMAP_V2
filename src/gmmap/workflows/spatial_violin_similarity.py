#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing for spatial metabolite mapping results:
1) violin plots for each selected metabolite across top N tissues
2) tissue-tissue clustering / similarity heatmap based on metabolite features

Input:
- h5ad
- Meta_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz
- cellTraitMatrices directory
- trait_meta
- selected table from previous spatial workflow
  e.g. Tissue_top_metabolites_spatial_S3_optimized_net_1/tables/top_positive_levels_per_tissue.csv

"""

import os
import re
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
try:
    import h5py
except ImportError:
    h5py = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-style font settings.
# Keep PDF text editable in Illustrator/Inkscape.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import zscore


# -----------------------------
# utilities
# -----------------------------
def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def safe_name(x, max_len=180):
    x = str(x)
    x = re.sub(r"[\\/:*?\"<>|\s]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if len(x) > max_len:
        x = x[:max_len].rstrip("_")
    return x if x else "NA"

def clean_trait_label(x):
    x = str(x)
    x = re.sub(r"\s+(levels?|ratio)\s*$", "", x, flags=re.I).strip()
    return x

def infer_obs_group_col(obs, stats_groups, tissue_col=None):
    if tissue_col and tissue_col in obs.columns:
        return tissue_col

    stats_set = set(map(str, pd.Series(stats_groups).dropna().unique()))
    preferred = [
        "Subregion", "Organ", "Organ_Full_Name", "organ_tissue",
        "celltype", "cell_type", "cell_ontology_class",
        "clusters", "sample", "batch"
    ]

    best_col, best_overlap = None, -1
    for col in list(dict.fromkeys(preferred + list(obs.columns))):
        if col not in obs.columns:
            continue
        vals = set(map(str, pd.Series(obs[col]).dropna().unique()))
        if not vals:
            continue
        overlap = len(stats_set.intersection(vals))
        if overlap > best_overlap:
            best_col, best_overlap = col, overlap

    return best_col

def load_trait_meta(path):
    meta = pd.read_csv(path)
    if "trait" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "trait"})

    name_col = None
    for c in ["reportedTrait", "reported_trait", "trait_name", "name", "metabolite", "label"]:
        if c in meta.columns:
            name_col = c
            break

    if name_col is None:
        meta["reportedTrait"] = meta["trait"].astype(str)
        name_col = "reportedTrait"

    type_col = None
    for c in meta.columns:
        if c in ["trait", name_col]:
            continue
        vals = meta[c].dropna().astype(str).str.lower()
        if vals.empty:
            continue
        hit = vals.str.contains(r"\blevels?\b|\bratio\b", regex=True).mean()
        if hit > 0.2:
            type_col = c
            break

    out = meta[["trait", name_col]].copy().rename(columns={name_col: "reportedTrait"})
    out["trait"] = out["trait"].astype(str)
    out["reportedTrait"] = out["reportedTrait"].astype(str)

    if type_col is not None:
        vals = meta[type_col].astype(str).str.lower()
        out["trait_type"] = np.where(
            vals.str.contains("ratio", regex=False), "ratio",
            np.where(vals.str.contains("level", regex=False), "levels", "unknown")
        )
    else:
        rt = out["reportedTrait"].str.lower()
        out["trait_type"] = np.where(
            rt.str.contains(r"\bratio\b", regex=True), "ratio",
            np.where(rt.str.contains(r"\blevels?\b", regex=True), "levels", "unknown")
        )

    out["trait_display"] = out["reportedTrait"].map(clean_trait_label)
    out.loc[out["trait_display"].eq(""), "trait_display"] = out.loc[out["trait_display"].eq(""), "trait"]
    return out.drop_duplicates("trait")

def add_trait_meta(df, trait_meta):
    df = df.copy()
    df["trait"] = df["trait"].astype(str)
    out = df.merge(trait_meta, on="trait", how="left")
    out["reportedTrait"] = out["reportedTrait"].fillna(out["trait"])
    out["trait_display"] = out["trait_display"].fillna(out["trait"])
    out["trait_type"] = out["trait_type"].fillna("unknown")
    return out

def choose_q_col(df, requested=None):
    if requested and requested in df.columns:
        return requested
    candidates = [
        "final_q_celltype_across_traits",
        "final_q_trait_across_celltypes",
        "q_delta_celltype_across_traits",
        "q_delta_trait_across_celltypes",
        "final_p",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("No usable q/p column found in scheme_long.")

def compute_empirical_null_tau(df, q_col, bg_q=0.20, quantile=0.95):
    tau_by_type = {}
    for ttype, sub in df.groupby("trait_type"):
        bg = pd.to_numeric(sub.loc[pd.to_numeric(sub[q_col], errors="coerce") > bg_q, "final_effect"], errors="coerce")
        bg = bg.abs()
        bg = bg[np.isfinite(bg)]
        tau = float(np.quantile(bg, quantile)) if bg.size > 0 else 0.0
        tau_by_type[ttype] = tau

    bg_all = pd.to_numeric(df.loc[pd.to_numeric(df[q_col], errors="coerce") > bg_q, "final_effect"], errors="coerce")
    bg_all = bg_all.abs()
    bg_all = bg_all[np.isfinite(bg_all)]
    global_tau = float(np.quantile(bg_all, quantile)) if bg_all.size > 0 else 0.0
    return tau_by_type, global_tau

def _max_other_values_by_trait(df, value_col, trait_col="trait"):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(trait_col).groups.items():
        idx = list(idx)
        vals = pd.to_numeric(df.loc[idx, value_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            continue

        finite_vals = vals[finite]
        max1 = np.max(finite_vals)
        n_max1 = np.sum(finite_vals == max1)
        if n_max1 >= 2:
            other = np.where(finite, max1, np.nan)
        else:
            others = finite_vals[finite_vals != max1]
            max2 = np.max(others) if others.size > 0 else np.nan
            other = np.where(vals == max1, max2, max1)
            other[~finite] = np.nan
        out.loc[idx] = other
    return out

def add_specificity_gap_columns(df):
    df = df.copy()
    eff = pd.to_numeric(df["final_effect"], errors="coerce").astype(float)

    df["positive_effect_strength"] = np.where(eff > 0, eff, 0.0)
    df["negative_effect_strength"] = np.where(eff < 0, -eff, 0.0)
    df["abs_final_effect"] = np.abs(eff)

    df["other_max_positive_effect"] = _max_other_values_by_trait(df, "positive_effect_strength").fillna(0.0)
    df["other_max_negative_effect"] = _max_other_values_by_trait(df, "negative_effect_strength").fillna(0.0)
    df["other_max_abs_effect"] = _max_other_values_by_trait(df, "abs_final_effect").fillna(0.0)

    df["specificity_gap_positive"] = df["positive_effect_strength"] - df["other_max_positive_effect"]
    df["specificity_gap_negative"] = df["negative_effect_strength"] - df["other_max_negative_effect"]
    df["specificity_gap_abs"] = df["abs_final_effect"] - df["other_max_abs_effect"]

    df["specificity_gap_directional"] = np.where(
        eff >= 0,
        df["specificity_gap_positive"],
        df["specificity_gap_negative"],
    )
    return df

def apply_filters(
    stats,
    q_col,
    trait_types=("levels",),
    plot_direction="positive",
    q_cutoff=0.05,
    exclude_unknown_x=False,
    use_empirical_null=False,
    emp_null_bg_q=0.20,
    emp_null_quantile=0.95,
    use_specificity_gap=False,
    specificity_gap_min=0.0,
    specificity_gap_mode="positive",
):
    df = stats.copy()
    df["final_effect"] = pd.to_numeric(df["final_effect"], errors="coerce")
    df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
    df = add_specificity_gap_columns(df)

    if specificity_gap_mode == "absolute":
        df["specificity_gap_used"] = df["specificity_gap_abs"]
    elif specificity_gap_mode == "directional":
        df["specificity_gap_used"] = df["specificity_gap_directional"]
    else:
        df["specificity_gap_used"] = df["specificity_gap_positive"]

    if exclude_unknown_x:
        is_unknown = (
            df["trait"].astype(str).str.upper().str.startswith("X-") |
            df["trait_display"].astype(str).str.upper().str.startswith("X-") |
            df["reportedTrait"].astype(str).str.upper().str.startswith("X-")
        )
        df = df.loc[~is_unknown].copy()

    if trait_types:
        allowed = set([x.strip() for x in trait_types])
        df = df.loc[df["trait_type"].isin(allowed)].copy()

    tau_by_type, global_tau = compute_empirical_null_tau(df, q_col=q_col, bg_q=emp_null_bg_q, quantile=emp_null_quantile)
    if use_empirical_null:
        df["empirical_null_tau"] = df["trait_type"].map(tau_by_type).fillna(global_tau)
    else:
        df["empirical_null_tau"] = 0.0

    df["passes_specificity_gap"] = (
        (not use_specificity_gap) |
        (pd.to_numeric(df["specificity_gap_used"], errors="coerce") >= float(specificity_gap_min))
    )

    df["abs_final_effect"] = df["final_effect"].abs()
    df["is_significant"] = (
        np.isfinite(df["final_effect"]) &
        np.isfinite(df[q_col]) &
        (df[q_col] <= q_cutoff) &
        (df["abs_final_effect"] >= df["empirical_null_tau"]) &
        (df["passes_specificity_gap"])
    )

    df = df.loc[df["is_significant"]].copy()

    plot_direction = str(plot_direction).lower()
    if plot_direction == "positive":
        df = df.loc[df["final_effect"] > 0].copy()
        df = df.sort_values(["celltype", "final_effect"], ascending=[True, False])
    elif plot_direction == "negative":
        df = df.loc[df["final_effect"] < 0].copy()
        df["sort_val"] = df["final_effect"].abs()
        df = df.sort_values(["celltype", "sort_val"], ascending=[True, False])
    elif plot_direction == "abs":
        df["sort_val"] = df["final_effect"].abs()
        df = df.sort_values(["celltype", "sort_val"], ascending=[True, False])
    else:
        raise ValueError("plot_direction must be one of: positive, negative, abs")

    return df

def score_path(score_dir, prefix, method, kind, stage):
    return Path(score_dir) / f"{prefix}_{method}_{kind}_{stage}.csv.gz"

def read_score_subset(path, obs_names, traits):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score matrix not found: {path}")

    traits = [str(t) for t in traits]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    idx_col = header[0]
    header_set = set(header)
    keep_traits = [t for t in traits if t in header_set]
    usecols = [idx_col] + keep_traits

    df = pd.read_csv(path, usecols=usecols, index_col=0)
    df.index = df.index.astype(str)
    df = df.reindex(pd.Index(obs_names.astype(str)), copy=False)
    return df

def build_pair_score_for_tissue(trait, tissue, score_mode, stats_sub, scores):
    trait = str(trait)
    tissue = str(tissue)
    score_mode = str(score_mode).lower()

    if score_mode in ["net", "signed_net"]:
        return pd.to_numeric(scores["net"][trait], errors="coerce").to_numpy(dtype=float)

    elif score_mode == "up":
        return pd.to_numeric(scores["up"][trait], errors="coerce").to_numpy(dtype=float)

    elif score_mode == "down":
        return pd.to_numeric(scores["down"][trait], errors="coerce").to_numpy(dtype=float)

    elif score_mode == "finalside":
        sub = stats_sub.loc[(stats_sub["trait"].astype(str) == trait) & (stats_sub["celltype"].astype(str) == tissue)]
        if sub.empty:
            # fallback to net if no exact pair
            return pd.to_numeric(scores["net"][trait], errors="coerce").to_numpy(dtype=float)

        side = str(sub.iloc[0].get("final_side", "UP")).upper()
        if side.startswith("DOWN"):
            return pd.to_numeric(scores["down"][trait], errors="coerce").to_numpy(dtype=float)
        else:
            return pd.to_numeric(scores["up"][trait], errors="coerce").to_numpy(dtype=float)

    else:
        raise ValueError("score_mode must be one of: net, signed_net, up, down, finalside")

def get_clean_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "clean_redblue",
        ["#2C7BB6", "#F7F7F7", "#D7191C"],
        N=256,
    )



def get_tissue_color_map():
    """
    Tissue colors approximating the reference legend supplied by the user.
    Keys can be full labels or distinctive prefixes.
    """
    return {
        "Muscle Tissue": "#bccb75",
        "Bone Marrow Tissue": "#f57f17",
        "Periportal": "#8e443c",
        "Small Intestine Tissue": "#df8e90",
        "Pericentral": "#d9b56d",
        "Colon Tissue": "#9c604d",
        "Brown Fat Tissue": "#7dc96d",
        "Other": "#9c7a31",
        "Cerebral Cortex": "#b9a7d0",
        "Alveoli": "#2c7fb8",
        "Renal Cortex": "#d6616b",
        "Heart Tissue": "#cec96d",
        "Pancreas Tissue": "#e0b04a",
        "Fiber Tracts": "#e7aac7",
        "Midbrain/Hindbrain": "#a1be4d",
        "Outer Stripe of Outer Medulla": "#bc9b2d",
        "Hypodermis": "#27b0c0",
        "Gastric Glands (Parietal Cell Rich)": "#b6ae16",
        "Caudoputamen": "#d92928",
        "Thymic Cortex": "#bf67b9",
        "Midbrain": "#8aa146",
        "Cerebellar Cortex (Mol. Layer)": "#8d63bb",
        "Foveolar Epithelium": "#969696",
        "Gastric Glands (Chief Cell Rich)": "#c8c8c8",
        "Inner Stripe of Outer Medulla": "#5858a6",
        "Red Pulp": "#b04a4a",
        "Marginal Zone": "#8f95d8",
        "Cerebellar Cortex (Gran. Layer)": "#f08a84",
        "Lymph Node Tissue": "#6670d1",
        "Spinal Cord": "#80457b",
        "Meninges": "#66883b",
        "Thymic Medulla": "#cf9ad6",
        "Dermis": "#c09a90",
        "Bronchioles": "#2ea337",
        "Epidermis": "#d676c2",
        "White Pulp": "#5ea5cf",
        "Thalamus": "#a54c94",
        "Ventricle": "#3881bd",
        "Hippocampus": "#83c9d8",
        "Bronchi": "#f0a359",
        "Blood Vessel": "#9bb8dc",
        "Inner Medulla": "#3d3b83",
    }


def pick_tissue_color(tissue, color_map=None, fallback_cycle=None):
    tissue = str(tissue)
    if color_map is None:
        color_map = get_tissue_color_map()
    if tissue in color_map:
        return color_map[tissue]
    for k, v in color_map.items():
        if tissue.startswith(k) or k.startswith(tissue) or k in tissue or tissue in k:
            return v
    if fallback_cycle is None:
        fallback_cycle = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    idx = abs(hash(tissue)) % len(fallback_cycle)
    return fallback_cycle[idx]


def choose_focus_tissue_for_trait(trait, selected_table, filtered_sub=None, plot_direction="positive"):
    """
    Choose the tissue name used in the violin filename/title so it matches
    the corresponding spatial plot naming convention:
        focus_tissue__metabolite__trait__score_mode__topN_tissues_violin.pdf

    Priority:
    1) selected_table rows for this trait, ranked by rank_in_tissue when available
    2) otherwise ranked by final_effect or abs(final_effect)
    3) filtered_sub top row
    """
    trait = str(trait)

    if selected_table is not None and "trait" in selected_table.columns and "celltype" in selected_table.columns:
        st = selected_table.loc[selected_table["trait"].astype(str).eq(trait)].copy()
        if not st.empty:
            if "rank_in_tissue" in st.columns:
                st["rank_in_tissue"] = pd.to_numeric(st["rank_in_tissue"], errors="coerce")
                st = st.sort_values("rank_in_tissue", ascending=True)
            elif "final_effect" in st.columns:
                st["final_effect"] = pd.to_numeric(st["final_effect"], errors="coerce")
                if str(plot_direction).lower() in ["negative", "abs"]:
                    st["_rank_val_"] = st["final_effect"].abs()
                    st = st.sort_values("_rank_val_", ascending=False)
                else:
                    st = st.sort_values("final_effect", ascending=False)
            return str(st.iloc[0]["celltype"])

    if filtered_sub is not None and not filtered_sub.empty and "celltype" in filtered_sub.columns:
        return str(filtered_sub.iloc[0]["celltype"])

    return "NA_focus"


def plot_one_violin(
    out_png,
    trait_display,
    trait_id,
    tissue_order,
    data_list,
    effect_list,
    q_list,
    score_mode="net",
    title_extra="",
    focus_tissue=None,
    tissue_color_map=None,
    jitter_frac=0.20,
    max_scatter_points_per_tissue=1800,
    invert_y_axis=False,
    reverse_tissue_order=False,
):
    """
    Draw a CNS-style violin plot with:
    - one color per tissue, matched to the supplied reference legend
    - semi-transparent colored violin bodies
    - jittered hollow scatter points
    - compact blue box overlays with black median line
    """
    if tissue_color_map is None:
        tissue_color_map = get_tissue_color_map()

    # Optional display-only reversal of the x-axis tissue order.
    # This does not change the selected tissues or any statistics.
    if reverse_tissue_order:
        tissue_order = list(tissue_order)[::-1]
        data_list = list(data_list)[::-1]
        effect_list = list(effect_list)[::-1]
        q_list = list(q_list)[::-1]

    fig_w = max(6.2, 0.78 * len(tissue_order) + 2.6)
    fig_h = 5.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    positions = np.arange(1, len(tissue_order) + 1)
    rng = np.random.default_rng(0)

    for pos, tissue, vals in zip(positions, tissue_order, data_list):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        col = pick_tissue_color(tissue, tissue_color_map)

        vp = ax.violinplot(
            [vals],
            positions=[pos],
            widths=0.82,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_linewidth(0.9)
            body.set_alpha(0.22)

        if vals.size > max_scatter_points_per_tissue:
            vals_plot = rng.choice(vals, size=max_scatter_points_per_tissue, replace=False)
        else:
            vals_plot = vals
        xj = pos + rng.uniform(-jitter_frac, jitter_frac, size=vals_plot.size)
        ax.scatter(
            xj, vals_plot,
            s=10,
            facecolors='none',
            edgecolors=col,
            linewidths=0.8,
            alpha=0.35,
            zorder=2,
        )

        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor('#2C7FB8')
            patch.set_edgecolor('#2C7FB8')
            patch.set_alpha(0.92)
            patch.set_linewidth(1.0)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_color('#2C7FB8')
            line.set_linewidth(1.0)
        for line in bp["medians"]:
            line.set_color('black')
            line.set_linewidth(1.6)

    # X-axis labels: tissue names only.
    # Remove effect, q-value and sample-size annotations for a cleaner CNS-style panel.
    labels = [str(t) for t in tissue_order]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    score_label = {
        'up': 'wAUC score (up genes)',
        'down': 'wAUC score (down genes)',
        'net': 'wAUC score (net)',
        'signed_net': 'wAUC score (signed net)',
        'finalside': 'wAUC score (selected side)',
    }.get(str(score_mode).lower(), f'{score_mode} score')
    ax.set_ylabel(score_label, fontsize=14)

    if focus_tissue is not None and str(focus_tissue) not in ["", "NA", "NA_focus"]:
        ax.set_title(f"{trait_display}\nSpatial focus: {focus_tissue}", fontsize=20, pad=18)
    else:
        ax.set_title(f"{trait_display}".strip(), fontsize=20, pad=18)

    ax.axhline(0, color='#1f4e79', linewidth=2.0, linestyle=(0, (3, 3)), alpha=0.9, zorder=1)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for sp in ['left', 'bottom', 'top', 'right']:
        ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(axis='y', labelsize=12, width=1.0, length=4)
    ax.tick_params(axis='x', labelsize=12, width=1.0, length=4)
    ax.grid(False)

    all_vals = np.concatenate([np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))] for x in data_list if len(x)])
    if all_vals.size:
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        pad = max(0.35, 0.05 * (ymax - ymin + 1e-6))
        ax.set_ylim(ymin - pad, ymax + pad)

    if title_extra:
        ax.text(0.5, 1.02, title_extra, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=12)

    # Optional display-only score-axis reversal.
    # Useful when the final panel appears vertically opposite to the desired convention.
    if invert_y_axis:
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_tissue_similarity_heatmap(
    mat,
    out_png,
    out_pdf=None,
    title="Tissue similarity based on metabolite features",
    cmap=None,
    cluster_method="average",
    reverse_order=False,
):
    # mat: tissue x features
    mat = mat.copy()
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    # correlation across tissues
    corr = pd.DataFrame(
        np.corrcoef(mat.values),
        index=mat.index,
        columns=mat.index
    ).fillna(0.0)

    # hierarchical clustering on 1 - corr
    dist = 1 - corr
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=cluster_method)
    order = leaves_list(Z)
    if reverse_order:
        order = order[::-1]

    corr_ord = corr.iloc[order, order]

    # optional cluster color strip
    clus = fcluster(Z, t=4, criterion="maxclust")
    clus_ord = clus[order]
    cluster_palette = ["#3C8DBC", "#8B80A8", "#C05C7A", "#D9A441", "#59B3A9", "#6C757D"]
    cluster_colors = [cluster_palette[(c - 1) % len(cluster_palette)] for c in clus_ord]

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "corrmap",
            ["#3288BD", "#66C2A5", "#D9F0A3", "#FEE08B", "#F46D43", "#B30059"],
            N=256
        )

    fig = plt.figure(figsize=(9.5, 8.4))
    gs = gridspec.GridSpec(
        nrows=3, ncols=3,
        width_ratios=[1.5, 0.18, 7.2],
        height_ratios=[1.5, 0.18, 7.2],
        wspace=0.02, hspace=0.02
    )

    ax_dendro_col = fig.add_subplot(gs[0, 2])
    ax_dendro_row = fig.add_subplot(gs[2, 0])
    ax_colbar = fig.add_subplot(gs[1, 2])
    ax_rowbar = fig.add_subplot(gs[2, 1])
    ax_heat = fig.add_subplot(gs[2, 2])

    # top dendrogram
    dendrogram(Z, ax=ax_dendro_col, color_threshold=None, no_labels=True)
    ax_dendro_col.set_xticks([])
    ax_dendro_col.set_yticks([])
    for sp in ax_dendro_col.spines.values():
        sp.set_visible(False)

    # left dendrogram
    dendrogram(Z, ax=ax_dendro_row, orientation="left", color_threshold=None, no_labels=True)
    ax_dendro_row.set_xticks([])
    ax_dendro_row.set_yticks([])
    for sp in ax_dendro_row.spines.values():
        sp.set_visible(False)

    # cluster bars
    col_rgb = np.array([matplotlib.colors.to_rgb(c) for c in cluster_colors])[None, :, :]
    row_rgb = np.array([matplotlib.colors.to_rgb(c) for c in cluster_colors])[:, None, :]
    ax_colbar.imshow(col_rgb, aspect="auto")
    ax_rowbar.imshow(row_rgb, aspect="auto")
    ax_colbar.set_axis_off()
    ax_rowbar.set_axis_off()

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax_heat.imshow(corr_ord.values, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax_heat.set_xticks(np.arange(corr_ord.shape[1]))
    ax_heat.set_yticks(np.arange(corr_ord.shape[0]))
    ax_heat.set_xticklabels(corr_ord.columns, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticklabels(corr_ord.index, fontsize=8)
    ax_heat.tick_params(length=0)

    # cell borders
    ax_heat.set_xticks(np.arange(-.5, corr_ord.shape[1], 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, corr_ord.shape[0], 1), minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_target_traits(x):
    """
    Parse user-provided GCST/trait IDs robustly.

    Accepted examples:
      --target_traits GCST90201371
      --target_traits GCST90201371,GCST90200417
      --target_traits "['GCST90201371']"

    This cleans brackets/quotes so the script does not search for
    the literal string "['GCST90201371']".
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple, set)):
        raw = []
        for v in x:
            raw.extend(parse_target_traits(v))
        return list(dict.fromkeys(raw))

    s = str(x).strip()
    if not s:
        return []

    for ch in ["[", "]", "(", ")", "{", "}", "\"", "'"]:
        s = s.replace(ch, "")

    parts = re.split(r"[,;\\s|]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"(GCST\\d+)", p, flags=re.I)
        if m:
            p = m.group(1).upper()
        out.append(p)

    return list(dict.fromkeys(out))


def build_target_trait_stats(
    stats,
    target_traits,
    trait_types=None,
    target_direction="positive",
    q_cutoff=0.05,
    target_use_q_filter=False,
    target_top_tissues=10,
):
    """
    Build a filtered stats table for target GCST traits.

    This mode bypasses selected_table and selects tissues directly for each target trait:
    - positive: final_effect > 0, sorted descending
    - negative: final_effect < 0, sorted by abs(final_effect)
    - abs: sorted by abs(final_effect)
    - all: no direction filter, sorted by final_effect descending
    """
    target_traits = parse_target_traits(target_traits)
    if not target_traits:
        raise ValueError("--target_traits is empty.")

    df = stats.copy()
    df["trait"] = df["trait"].astype(str)
    df["celltype"] = df["celltype"].astype(str)
    df["final_effect"] = pd.to_numeric(df["final_effect"], errors="coerce")
    if "q_value_used" in df.columns:
        df["q_value_used"] = pd.to_numeric(df["q_value_used"], errors="coerce")

    sub = df.loc[df["trait"].isin(target_traits)].copy()
    missing = sorted(set(target_traits) - set(sub["trait"].astype(str).unique()))
    if missing:
        print(f"[warn] target traits not found after method/stage filtering: {missing}")

    if sub.empty:
        trait_examples = sorted(df["trait"].astype(str).dropna().unique().tolist())[:10]
        raise ValueError(
            "No rows found for --target_traits after method/stage filtering. "
            "Please check the GCST IDs and --method/--stage.\\n"
            f"Cleaned target_traits = {target_traits}\\n"
            f"Example trait IDs in current filtered table = {trait_examples}"
        )

    if trait_types:
        allowed = set([str(x).strip() for x in trait_types if str(x).strip()])
        if "trait_type" in sub.columns:
            sub = sub.loc[sub["trait_type"].astype(str).isin(allowed)].copy()

    if target_use_q_filter:
        if "q_value_used" not in sub.columns:
            raise KeyError("target_use_q_filter requires q_value_used column.")
        sub = sub.loc[pd.to_numeric(sub["q_value_used"], errors="coerce") <= float(q_cutoff)].copy()

    if sub.empty:
        raise ValueError("No target-trait rows left after trait_type/q filtering.")

    direction = str(target_direction).lower()
    out = []
    for trait, g in sub.groupby("trait", sort=False):
        g = g.copy()
        if direction == "positive":
            g = g.loc[g["final_effect"] > 0].copy()
            g = g.sort_values("final_effect", ascending=False)
        elif direction == "negative":
            g = g.loc[g["final_effect"] < 0].copy()
            g["_rank_val_"] = g["final_effect"].abs()
            g = g.sort_values("_rank_val_", ascending=False)
        elif direction == "abs":
            g["_rank_val_"] = g["final_effect"].abs()
            g = g.sort_values("_rank_val_", ascending=False)
        elif direction == "all":
            g = g.sort_values("final_effect", ascending=False)
        else:
            raise ValueError("target_direction must be one of: positive, negative, abs, all")

        if g.empty:
            print(f"[warn] target trait {trait} has no tissue rows after target_direction={direction}.")
            continue

        if target_top_tissues is not None and int(target_top_tissues) > 0:
            g = g.head(int(target_top_tissues)).copy()

        g["rank_in_target_trait"] = np.arange(1, g.shape[0] + 1)
        out.append(g)

    if not out:
        raise ValueError("No target-trait rows available for violin plotting.")

    return pd.concat(out, axis=0, ignore_index=True)



def _decode_h5ad_values(x):
    """Decode bytes/object arrays from h5py to Python strings when needed."""
    arr = np.asarray(x)
    if arr.dtype.kind == "S":
        return arr.astype(str)
    if arr.dtype.kind == "O":
        return np.array([
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            for v in arr
        ], dtype=object)
    return arr


def _read_h5ad_obs_elem(elem):
    """
    Minimal reader for common AnnData /obs elements without constructing the full AnnData object.
    Supports normal datasets and categorical groups with codes/categories.
    """
    if h5py is None:
        raise ImportError("h5py is required for fast h5ad obs reading.")

    if isinstance(elem, h5py.Dataset):
        return _decode_h5ad_values(elem[()])

    if isinstance(elem, h5py.Group):
        enc = elem.attrs.get("encoding-type", b"")
        if isinstance(enc, bytes):
            enc = enc.decode("utf-8")

        # Standard AnnData categorical encoding.
        if "codes" in elem and "categories" in elem:
            codes = np.asarray(elem["codes"][()])
            cats = _read_h5ad_obs_elem(elem["categories"])
            out = np.empty(codes.shape[0], dtype=object)
            valid = codes >= 0
            out[~valid] = np.nan
            if np.any(valid):
                out[valid] = np.asarray(cats, dtype=object)[codes[valid]]
            return out

        # Nullable arrays in newer AnnData versions.
        if "values" in elem and "mask" in elem:
            vals = _read_h5ad_obs_elem(elem["values"]).astype(object)
            mask = np.asarray(elem["mask"][()]).astype(bool)
            vals[mask] = np.nan
            return vals

        # String arrays may be stored under "values".
        if "values" in elem:
            return _read_h5ad_obs_elem(elem["values"])

    raise TypeError(f"Unsupported h5ad obs element: {type(elem)}")


def _get_h5ad_obs_index_key(obs_group):
    idx_key = obs_group.attrs.get("_index", None)
    if idx_key is None:
        idx_key = obs_group.attrs.get("index", None)
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode("utf-8")
    if idx_key is None:
        for k in ["_index", "index"]:
            if k in obs_group:
                return k
        raise KeyError("Cannot find obs index key in h5ad /obs group.")
    return str(idx_key)


def list_h5ad_obs_columns(h5ad_path):
    """List obs columns from h5ad without reading the whole AnnData."""
    if h5py is None:
        raise ImportError("h5py is required for fast h5ad obs reading.")
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        idx_key = _get_h5ad_obs_index_key(obs)
        col_order = obs.attrs.get("column-order", None)
        if col_order is not None:
            cols = [
                c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                for c in list(col_order)
            ]
            cols = [c for c in cols if c in obs and c != idx_key]
        else:
            cols = [c for c in obs.keys() if c != idx_key]
    return cols


def read_h5ad_obs_columns_fast(h5ad_path, columns):
    """
    Read only obs index and selected obs columns from h5ad.
    This avoids sc.read_h5ad(...).obs.copy(), which is very slow for huge h5ad files
    with many categorical obs columns.
    """
    if h5py is None:
        raise ImportError("h5py is required for fast h5ad obs reading.")

    columns = [str(c) for c in columns if c is not None]
    with h5py.File(h5ad_path, "r") as f:
        obs_g = f["obs"]
        idx_key = _get_h5ad_obs_index_key(obs_g)
        idx = _decode_h5ad_values(obs_g[idx_key][()]).astype(str)

        data = {}
        missing = []
        for col in columns:
            if col in obs_g:
                data[col] = _read_h5ad_obs_elem(obs_g[col])
            else:
                missing.append(col)

    if missing:
        print(f"[warn] requested obs columns not found in h5ad: {missing}")

    return pd.DataFrame(data, index=pd.Index(idx.astype(str), name=idx_key))


def read_h5ad_obs_for_violin_fast(h5ad_path, stats_groups, tissue_col=None):
    """
    Fast obs reader for this violin script. It only needs:
      - obs_names, to align score matrices
      - one tissue/celltype annotation column, to split violin values by tissue

    If --tissue_col is provided, only that column is read.
    If --tissue_col is omitted, only a small set of candidate tissue columns is read
    and matched to stats['celltype'].
    """
    preferred = [
        "Subregion", "Organ", "Organ_Full_Name", "organ_tissue",
        "celltype", "cell_type", "cell_ontology_class",
        "clusters", "sample", "batch"
    ]

    obs_cols = list_h5ad_obs_columns(h5ad_path)

    if tissue_col:
        if tissue_col not in obs_cols:
            raise KeyError(
                f"--tissue_col '{tissue_col}' was not found in h5ad obs. "
                f"Available examples: {obs_cols[:30]}"
            )
        obs = read_h5ad_obs_columns_fast(h5ad_path, [tissue_col])
        return obs, tissue_col

    candidate_cols = [c for c in preferred if c in obs_cols]
    if len(candidate_cols) == 0:
        raise KeyError(
            "Could not infer tissue column quickly because none of the preferred columns exist. "
            "Please specify --tissue_col. "
            f"Available obs columns examples: {obs_cols[:50]}"
        )

    obs_small = read_h5ad_obs_columns_fast(h5ad_path, candidate_cols)
    inferred = infer_obs_group_col(obs_small, stats_groups, tissue_col=None)
    if inferred is None:
        raise ValueError(
            "Could not infer tissue/celltype column from candidate obs columns. "
            "Please specify --tissue_col explicitly. "
            f"Candidate columns tested: {candidate_cols}"
        )
    return obs_small[[inferred]].copy(), inferred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--scheme_long", required=True)
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--trait_meta", required=True)
    ap.add_argument("--selected_table", default=None,
                    help="Optional previous spatial output table, e.g. top_positive_levels_per_tissue.csv. Not required when --target_traits is set.")
    ap.add_argument("--target_traits", default=None,
                    help="Comma-separated target metabolite/trait IDs, e.g. GCST90200223,GCST90200417. If set, export violin plots for these GCST traits directly.")
    ap.add_argument("--target_direction", default="positive", choices=["positive", "negative", "abs", "all"],
                    help="How to rank/filter tissues for target traits. positive=final_effect>0, negative=final_effect<0, abs=absolute effect, all=no direction filter.")
    ap.add_argument("--target_top_tissues", type=int, default=None,
                    help="Top N tissues per target trait for violin plots. If omitted, uses --top_tissues_violin.")
    ap.add_argument("--target_use_q_filter", action="store_true",
                    help="When --target_traits is set, additionally require q_value_used <= --q_cutoff.")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--method", default="wAUCell")
    ap.add_argument("--stage", default="S3_geneSetZ_cellZ")
    ap.add_argument("--prefix", default="Meta")

    ap.add_argument("--q_col", default=None)
    ap.add_argument("--q_cutoff", type=float, default=0.05)
    ap.add_argument("--exclude_unknown_x", action="store_true")
    ap.add_argument("--use_empirical_null", action="store_true")
    ap.add_argument("--emp_null_bg_q", type=float, default=0.20)
    ap.add_argument("--emp_null_quantile", type=float, default=0.95)
    ap.add_argument("--use_specificity_gap", action="store_true")
    ap.add_argument("--specificity_gap_min", type=float, default=0.0)
    ap.add_argument("--specificity_gap_mode", default="positive",
                    choices=["positive", "directional", "absolute"])
    ap.add_argument("--plot_direction", default="positive",
                    choices=["positive", "negative", "abs"])
    ap.add_argument("--plot_trait_types", default="levels")

    ap.add_argument("--score_mode", default="net",
                    choices=["net", "signed_net", "up", "down", "finalside"])

    ap.add_argument("--tissue_col", default=None)
    ap.add_argument("--top_tissues_violin", type=int, default=10)
    ap.add_argument("--violin_max_points_per_tissue", type=int, default=4000)
    ap.add_argument("--invert_violin_y", action="store_true",
                    help="Display-only option: invert the violin plot score y-axis.")
    ap.add_argument("--reverse_violin_tissue_order", action="store_true",
                    help="Display-only option: reverse the left-to-right tissue order in violin plots.")
    ap.add_argument("--reverse_similarity_order", action="store_true",
                    help="Display-only option: reverse row/column order in the tissue similarity heatmap.")

    ap.add_argument("--similarity_feature_set", default="selected",
                    choices=["selected", "filtered_all"])
    ap.add_argument("--min_tissues_for_similarity", type=int, default=3)

    args = ap.parse_args()

    print(f"[display] invert_violin_y={args.invert_violin_y}, "
          f"reverse_violin_tissue_order={args.reverse_violin_tissue_order}, "
          f"reverse_similarity_order={args.reverse_similarity_order}")

    mkdir(args.outdir)
    violin_dir = Path(args.outdir) / "violin_top_tissues"
    cluster_dir = Path(args.outdir) / "tissue_similarity"
    mkdir(violin_dir)
    mkdir(cluster_dir)

    print("[1/6] load data...")
    trait_meta = load_trait_meta(args.trait_meta)
    stats = pd.read_csv(args.scheme_long)
    stats["trait"] = stats["trait"].astype(str)
    stats["celltype"] = stats["celltype"].astype(str)
    stats = add_trait_meta(stats, trait_meta)

    stats = stats.loc[
        (stats["method"].astype(str) == str(args.method)) &
        (stats["stage"].astype(str) == str(args.stage))
    ].copy()

    q_col = choose_q_col(stats, args.q_col)
    stats["q_value_used"] = pd.to_numeric(stats[q_col], errors="coerce")

    print("[obs] fast reading only obs index and tissue annotation from h5ad...")
    try:
        obs, tissue_col = read_h5ad_obs_for_violin_fast(
            h5ad_path=args.h5ad,
            stats_groups=stats["celltype"],
            tissue_col=args.tissue_col,
        )
    except Exception as e:
        raise RuntimeError(
            "Fast h5ad obs reading failed. Most commonly, you need to specify the correct "
            "--tissue_col, for example --tissue_col Subregion or --tissue_col Organ. "
            f"Original error: {e}"
        )

    if tissue_col is None:
        raise ValueError("Could not infer tissue/celltype column from h5ad obs. Please specify --tissue_col")

    print(f"[obs] using tissue column: {tissue_col}")
    print(f"[obs] loaded obs shape: {obs.shape}")

    print("[2/6] load selected table or target GCST traits...")
    trait_types = [x.strip() for x in str(args.plot_trait_types).split(",") if x.strip()]

    if args.target_traits:
        selected_table = None
        selected_traits = parse_target_traits(args.target_traits)
        target_top_n = args.target_top_tissues if args.target_top_tissues is not None else args.top_tissues_violin

        print(f"[target] raw --target_traits: {args.target_traits}")
        print(f"[target] cleaned GCST/trait IDs: {selected_traits}")
        filtered_stats = build_target_trait_stats(
            stats=stats,
            target_traits=selected_traits,
            trait_types=trait_types,
            target_direction=args.target_direction,
            q_cutoff=args.q_cutoff,
            target_use_q_filter=args.target_use_q_filter,
            target_top_tissues=target_top_n,
        )

        target_rows_path = Path(args.outdir) / "TARGET_TRAITS_rows_used_for_violin.csv"
        filtered_stats.to_csv(target_rows_path, index=False)
        print(f"[save] {target_rows_path}  n={filtered_stats.shape[0]}")

    else:
        if args.selected_table is None:
            raise ValueError("Please provide --selected_table, or use --target_traits GCSTxxxx for direct GCST violin plotting.")

        selected_table = pd.read_csv(args.selected_table)
        selected_table["trait"] = selected_table["trait"].astype(str)
        if "trait_display" not in selected_table.columns:
            selected_table = add_trait_meta(selected_table, trait_meta)
        if "celltype" not in selected_table.columns:
            raise ValueError("selected_table must contain 'celltype' column")
        selected_table["celltype"] = selected_table["celltype"].astype(str)

        print("[3/6] build filtered stats...")
        filtered_stats = apply_filters(
            stats=stats,
            q_col=q_col,
            trait_types=trait_types,
            plot_direction=args.plot_direction,
            q_cutoff=args.q_cutoff,
            exclude_unknown_x=args.exclude_unknown_x,
            use_empirical_null=args.use_empirical_null,
            emp_null_bg_q=args.emp_null_bg_q,
            emp_null_quantile=args.emp_null_quantile,
            use_specificity_gap=args.use_specificity_gap,
            specificity_gap_min=args.specificity_gap_min,
            specificity_gap_mode=args.specificity_gap_mode,
        )

        # restrict to the metabolites that were actually selected in previous spatial run
        selected_traits = selected_table["trait"].astype(str).unique().tolist()
        filtered_stats = filtered_stats.loc[filtered_stats["trait"].isin(selected_traits)].copy()

        if filtered_stats.empty:
            raise ValueError("No filtered_stats left after filtering. Check your parameters.")

    print("[4/6] preload score matrices...")
    needed_kinds = {"net"}
    if args.score_mode in ["up", "finalside"]:
        needed_kinds.add("up")
    if args.score_mode in ["down", "finalside"]:
        needed_kinds.add("down")
    if args.score_mode == "signed_net":
        needed_kinds.add("net")

    scores = {}
    for kind in sorted(needed_kinds):
        p = score_path(args.score_dir, args.prefix, args.method, kind, args.stage)
        print("[read score]", p)
        scores[kind] = read_score_subset(p, obs.index.astype(str), selected_traits)

    print("[5/6] violin plots...")
    violin_index_rows = []
    for trait in selected_traits:
        sub = filtered_stats.loc[filtered_stats["trait"].astype(str) == str(trait)].copy()
        if sub.empty:
            continue

        if args.target_traits:
            # Already filtered/ranked by build_target_trait_stats; preserve the target ordering.
            if "rank_in_target_trait" in sub.columns:
                sub = sub.sort_values("rank_in_target_trait", ascending=True)
            else:
                sub = sub.sort_values("final_effect", ascending=False)
            target_top_n = args.target_top_tissues if args.target_top_tissues is not None else args.top_tissues_violin
            sub = sub.head(target_top_n).copy()
        else:
            if args.plot_direction == "negative":
                sub["rank_val"] = sub["final_effect"].abs()
                sub = sub.sort_values("rank_val", ascending=False)
            elif args.plot_direction == "abs":
                sub["rank_val"] = sub["final_effect"].abs()
                sub = sub.sort_values("rank_val", ascending=False)
            else:
                sub = sub.sort_values("final_effect", ascending=False)

            sub = sub.head(args.top_tissues_violin).copy()
        tissue_order = sub["celltype"].astype(str).tolist()

        data_list = []
        eff_list = []
        q_list = []
        keep_tissues = []

        for _, row in sub.iterrows():
            tissue = str(row["celltype"])
            score_vec = build_pair_score_for_tissue(
                trait=trait,
                tissue=tissue,
                score_mode=args.score_mode,
                stats_sub=stats,
                scores=scores
            )
            mask = obs[tissue_col].astype(str).eq(tissue).to_numpy()
            vals = pd.to_numeric(pd.Series(score_vec).loc[mask], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                continue
            if vals.size > args.violin_max_points_per_tissue:
                rng = np.random.default_rng(0)
                vals = rng.choice(vals, size=args.violin_max_points_per_tissue, replace=False)

            data_list.append(vals)
            eff_list.append(float(row["final_effect"]))
            q_list.append(float(row["q_value_used"]) if np.isfinite(row["q_value_used"]) else np.nan)
            keep_tissues.append(tissue)

        if len(data_list) < 2:
            continue

        trait_display = str(sub["trait_display"].iloc[0])

        # Make violin filename consistent with spatial plot names:
        # focus_tissue__metabolite__trait__score_mode__topN_tissues_violin.pdf
        focus_tissue = choose_focus_tissue_for_trait(
            trait=trait,
            selected_table=selected_table,
            filtered_sub=sub,
            plot_direction=(args.target_direction if args.target_traits else args.plot_direction),
        )
        out_pdf = violin_dir / (
            f"{safe_name(focus_tissue)}__{safe_name(trait_display)}__"
            f"{safe_name(trait)}__{safe_name(args.score_mode)}__"
            f"top{len(keep_tissues)}_tissues_violin.pdf"
        )

        plot_one_violin(
            out_png=out_pdf,
            trait_display=trait_display,
            trait_id=trait,
            tissue_order=keep_tissues,
            data_list=data_list,
            effect_list=eff_list,
            q_list=q_list,
            score_mode=args.score_mode,
            title_extra=f"(top {len(keep_tissues)} tissues)",
            focus_tissue=focus_tissue,
            invert_y_axis=args.invert_violin_y,
            reverse_tissue_order=args.reverse_violin_tissue_order,
        )

        violin_index_rows.append({
            "trait": trait,
            "trait_display": trait_display,
            "focus_tissue_for_spatial_name": focus_tissue,
            "score_mode": args.score_mode,
            "n_tissues": len(keep_tissues),
            "plot_path": str(out_pdf)
        })

    pd.DataFrame(violin_index_rows).to_csv(Path(args.outdir) / "violin_plot_index.csv", index=False)

    print("[6/6] tissue similarity clustering...")
    if args.similarity_feature_set == "selected":
        feature_traits = selected_traits
    else:
        feature_traits = filtered_stats["trait"].astype(str).unique().tolist()

    sim_df = filtered_stats.loc[filtered_stats["trait"].isin(feature_traits)].copy()
    mat = sim_df.pivot_table(index="celltype", columns="trait_display", values="final_effect", aggfunc="mean")
    mat = mat.loc[mat.notna().sum(axis=1) >= args.min_tissues_for_similarity].copy()
    mat = mat.fillna(0.0)

    if mat.shape[0] >= 3 and mat.shape[1] >= 3:
        out_pdf = cluster_dir / "tissue_similarity_clustered_correlation.pdf"
        plot_tissue_similarity_heatmap(
            mat=mat,
            out_png=out_pdf,
            out_pdf=None,
            title="Tissue similarity based on metabolite features",
            reverse_order=args.reverse_similarity_order,
        )
        mat.to_csv(cluster_dir / "tissue_by_metabolite_feature_matrix.csv")
    else:
        print("[skip] too few tissues/features for similarity heatmap")

    print("[done] outputs saved to:", args.outdir)


if __name__ == "__main__":
    main()
