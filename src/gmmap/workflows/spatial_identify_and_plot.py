#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identify top tissue-associated metabolites and visualize metabolite scores in spatial coordinates.

Typical use:
python identify_and_spatial_plot_tissue_metabolites.py \
  --h5ad /path/to/input \
  --scheme_long Meta_SchemeC_AUROC_UpDownNet_outputs/Meta_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz \
  --score_dir Meta_SchemeC_AUROC_UpDownNet_outputs/cellTraitMatrices \
  --trait_meta /path/to/input \
  --outdir Tissue_top_metabolites_spatial \
  --method wAUCell \
  --stage S3_geneSetZ_cellZ \
  --top_n_per_tissue 5 \
  --spatial_top_n 1 \
  --q_cutoff 0.05 \
  --use_empirical_null

Input files from your SchemeC workflow:
  1) ALL_stats_long: tissue/celltype × trait AUROC/final_effect/q table
  2) cellTraitMatrices: per spot/cell trait scores, e.g.
       Meta_wAUCell_up_S3_geneSetZ_cellZ.csv.gz
       Meta_wAUCell_down_S3_geneSetZ_cellZ.csv.gz
       Meta_wAUCell_net_S3_geneSetZ_cellZ.csv.gz
  3) trait metadata: trait -> reportedTrait and levels/ratio class

Main outputs:
  - tables/top_positive_levels_per_tissue.csv
  - tables/top_negative_levels_per_tissue.csv
  - tables/top_abs_levels_per_tissue.csv
  - tables/top_positive_ratio_per_tissue.csv
  - tables/top_negative_ratio_per_tissue.csv
  - tables/top_abs_ratio_per_tissue.csv
  - heatmaps/*.pdf
  - spatial_plots/*.pdf, including a combined multi-page PDF and one PDF per spatial plot

"""

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize


# -----------------------------
# Basic utilities
# -----------------------------
def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_name(x, max_len=160):
    x = str(x)
    x = re.sub(r"[\\/:*?\"<>|\s]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if len(x) > max_len:
        x = x[:max_len].rstrip("_")
    return x if x else "NA"


def pick_existing_col(df, preferred=None, candidates=None, required=False, label="column"):
    if preferred and preferred in df.columns:
        return preferred
    if candidates:
        for c in candidates:
            if c in df.columns:
                return c
    if required:
        raise KeyError(f"Cannot find {label}. preferred={preferred}, candidates={candidates}")
    return None


def get_clean_diverging_cmap(cmap="gmetmap_redblue"):
    """
    Clean blue-white-red colormap for metabolite-score spatial plots.

    Why this is needed:
    - Some default diverging maps have very dark high-end colors that can look black.
    - Values outside robust plotting limits should still be shown as red/blue rather
      than becoming visually black or confusing.
    - Missing values are set to light gray.
    """
    if cmap in ["gmetmap_redblue", "paper_redblue", "clean_redblue"]:
        cmap_obj = LinearSegmentedColormap.from_list(
            "gmetmap_redblue",
            ["#2166AC", "#F7F7F7", "#D73027"],
            N=256,
        )
    else:
        cmap_obj = mpl.colormaps.get_cmap(cmap).copy()

    # Explicitly control special colors so the upper end never turns black.
    cmap_obj.set_bad("#D9D9D9")     # NaN / missing
    cmap_obj.set_under("#2166AC")   # below vmin
    cmap_obj.set_over("#D73027")    # above vmax
    return cmap_obj


def get_score_norm(vals, symmetric=True, upper_q=0.995, lower_q=0.005):
    """Robust normalization centered at zero for signed metabolite scores."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return Normalize(vmin=-1, vmax=1), -1, 1

    if symmetric:
        vmax = np.nanquantile(np.abs(vals), upper_q)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(vals))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = -float(vmax)
        vmax = float(vmax)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        vmin, vmax = np.nanquantile(vals, [lower_q, upper_q])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=False)
    return norm, float(vmin), float(vmax)



def _safe_quantile(x, q, default=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    q = min(max(float(q), 0.0), 1.0)
    return float(np.nanquantile(x, q))


def get_focus_contrast_norm(
    plot_obs,
    sel_mask,
    group_col=None,
    focus_group=None,
    score_col="_score_",
    mode="robust",
    symmetric=True,
    robust_upper_q=0.995,
    robust_lower_q=0.005,
    bg_q=0.97,
    focus_q=0.995,
):
    """
    Automatically set color limits to make the focus tissue/region stand out.

    mode="robust":
        Original behavior. The color center is 0 for signed scores.

    mode="focus_positive":
        Positive-focus contrast. The white/red transition is set to the
        upper quantile of non-focus spots/cells. Therefore, only scores higher
        than most non-focus tissue become visibly red.

        vcenter = quantile(non-focus score, bg_q), clipped to be >= 0
        vmax    = quantile(focus score, focus_q)
        vmin    = lower robust quantile of all selected scores

    mode="focus_auto":
        Automatically chooses positive or negative focus contrast according to
        whether the focus tissue has a stronger positive or negative tail.
    """
    mode = str(mode).lower()
    vals_all = pd.to_numeric(plot_obs.loc[sel_mask, score_col], errors="coerce").to_numpy(dtype=float)
    vals_all = vals_all[np.isfinite(vals_all)]
    if vals_all.size == 0:
        return Normalize(vmin=-1, vmax=1), -1.0, 1.0, 0.0, "robust_empty"

    if mode == "robust" or group_col is None or focus_group is None or group_col not in plot_obs.columns:
        norm, vmin, vmax = get_score_norm(vals_all, symmetric=symmetric, upper_q=robust_upper_q, lower_q=robust_lower_q)
        return norm, vmin, vmax, 0.0 if symmetric else float((vmin + vmax) / 2.0), "robust"

    selected_obs = plot_obs.loc[sel_mask].copy()
    focus_mask = selected_obs[group_col].astype(str).eq(str(focus_group)).to_numpy()
    focus_vals = pd.to_numeric(selected_obs.loc[focus_mask, score_col], errors="coerce").to_numpy(dtype=float)
    other_vals = pd.to_numeric(selected_obs.loc[~focus_mask, score_col], errors="coerce").to_numpy(dtype=float)
    focus_vals = focus_vals[np.isfinite(focus_vals)]
    other_vals = other_vals[np.isfinite(other_vals)]

    # If focus/non-focus separation cannot be estimated, fall back safely.
    if focus_vals.size < 5 or other_vals.size < 5:
        norm, vmin, vmax = get_score_norm(vals_all, symmetric=symmetric, upper_q=robust_upper_q, lower_q=robust_lower_q)
        return norm, vmin, vmax, 0.0 if symmetric else float((vmin + vmax) / 2.0), "robust_fallback"

    # Decide direction automatically when requested.
    if mode == "focus_auto":
        pos_tail = _safe_quantile(focus_vals, focus_q, 0.0) - max(0.0, _safe_quantile(other_vals, bg_q, 0.0))
        neg_tail = min(0.0, _safe_quantile(other_vals, 1.0 - bg_q, 0.0)) - _safe_quantile(focus_vals, 1.0 - focus_q, 0.0)
        mode = "focus_negative" if neg_tail > pos_tail else "focus_positive"

    eps = 1e-6
    if mode == "focus_positive":
        # Put the white point at the non-focus upper background. This makes
        # non-focus mild positive scores look neutral, and reserves red for
        # focus-enriched positive values.
        vcenter = max(0.0, _safe_quantile(other_vals, bg_q, 0.0))
        vmax = _safe_quantile(focus_vals, focus_q, np.nan)
        if (not np.isfinite(vmax)) or vmax <= vcenter + eps:
            vmax = _safe_quantile(vals_all, robust_upper_q, np.nan)
        if (not np.isfinite(vmax)) or vmax <= vcenter + eps:
            vmax = float(np.nanmax(vals_all))
        if (not np.isfinite(vmax)) or vmax <= vcenter + eps:
            vmax = vcenter + 1.0

        vmin = _safe_quantile(vals_all, robust_lower_q, np.nan)
        if (not np.isfinite(vmin)) or vmin >= vcenter - eps:
            vmin = min(float(np.nanmin(vals_all)), 0.0)
        if vmin >= vcenter - eps:
            vmin = vcenter - max(abs(vmax - vcenter), 1.0)

        norm = TwoSlopeNorm(vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax))
        return norm, float(vmin), float(vmax), float(vcenter), "focus_positive"

    if mode == "focus_negative":
        # Mirror logic for negative focus enrichment: blue is reserved for
        # focus-specific low/negative values.
        vcenter = min(0.0, _safe_quantile(other_vals, 1.0 - bg_q, 0.0))
        vmin = _safe_quantile(focus_vals, 1.0 - focus_q, np.nan)
        if (not np.isfinite(vmin)) or vmin >= vcenter - eps:
            vmin = _safe_quantile(vals_all, robust_lower_q, np.nan)
        if (not np.isfinite(vmin)) or vmin >= vcenter - eps:
            vmin = float(np.nanmin(vals_all))
        if (not np.isfinite(vmin)) or vmin >= vcenter - eps:
            vmin = vcenter - 1.0

        vmax = _safe_quantile(vals_all, robust_upper_q, np.nan)
        if (not np.isfinite(vmax)) or vmax <= vcenter + eps:
            vmax = max(float(np.nanmax(vals_all)), 0.0)
        if vmax <= vcenter + eps:
            vmax = vcenter + max(abs(vcenter - vmin), 1.0)

        norm = TwoSlopeNorm(vmin=float(vmin), vcenter=float(vcenter), vmax=float(vmax))
        return norm, float(vmin), float(vmax), float(vcenter), "focus_negative"

    raise ValueError("color_scale_mode must be one of: robust, focus_positive, focus_negative, focus_auto")


def select_top_score_outline_points(sub, score_col="_score_", frac=0.05, mode="positive"):
    """
    Select spots/cells to outline according to metabolite score.

    Parameters
    ----------
    sub : DataFrame
        One spatial panel/sample. Must contain score_col.
    score_col : str
        Column containing plotted metabolite score.
    frac : float
        Top fraction to outline. 0.05 means top 5%.
    mode : {"positive", "high", "absolute"}
        positive : outline the highest positive scores only. If no positive score exists, outline nothing.
        high     : outline the highest raw scores, even if all values are negative.
        absolute : outline the largest absolute scores regardless of sign.
    """
    if sub is None or sub.empty:
        return sub.iloc[0:0].copy()

    frac = float(frac)
    if (not np.isfinite(frac)) or frac <= 0:
        return sub.iloc[0:0].copy()
    frac = min(frac, 1.0)

    v = pd.to_numeric(sub[score_col], errors="coerce")
    finite = np.isfinite(v.to_numpy(dtype=float))
    if finite.sum() == 0:
        return sub.iloc[0:0].copy()

    mode = str(mode).lower()
    if mode == "positive":
        candidate = sub.loc[finite & (v > 0)].copy()
        if candidate.empty:
            return sub.iloc[0:0].copy()
        rank_value = pd.to_numeric(candidate[score_col], errors="coerce")
    elif mode == "absolute":
        candidate = sub.loc[finite].copy()
        rank_value = pd.to_numeric(candidate[score_col], errors="coerce").abs()
    elif mode == "high":
        candidate = sub.loc[finite].copy()
        rank_value = pd.to_numeric(candidate[score_col], errors="coerce")
    else:
        raise ValueError(f"Unknown top_score_outline_by: {mode}. Use positive, high, or absolute.")

    if candidate.empty:
        return sub.iloc[0:0].copy()

    n_top = int(np.ceil(candidate.shape[0] * frac))
    n_top = max(1, min(n_top, candidate.shape[0]))

    # nlargest is faster and avoids numerical issues around tied quantile thresholds.
    top_idx = rank_value.nlargest(n_top).index
    return candidate.loc[top_idx].copy()



def _convex_hull_xy(points):
    """
    Compute a 2D convex hull with Andrew's monotonic chain algorithm.
    Returns hull points in order. No scipy dependency is required.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return pts

    # Remove duplicates and sort by x, then y.
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return hull


def draw_focus_dashed_boundary(
    ax,
    focus_df,
    x_col,
    y_col,
    color="black",
    linewidth=0.8,
    alpha=0.95,
    linestyle="--",
    max_points=20000,
    pad_frac=0.0,
    zorder=5,
):
    """
    Draw a dashed convex-hull boundary around the focus tissue/organ in one panel.

    This outlines the anatomical focus region rather than individual high-score spots.
    If too few points are available, falls back to a dashed bounding box.
    """
    if focus_df is None or focus_df.empty:
        return

    xy = focus_df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.shape[0] < 2:
        return

    if xy.shape[0] > int(max_points):
        rng = np.random.default_rng(3)
        idx = rng.choice(xy.shape[0], size=int(max_points), replace=False)
        xy = xy[idx]

    if xy.shape[0] >= 3:
        hull = _convex_hull_xy(xy)
        if hull.shape[0] >= 3:
            hull = np.vstack([hull, hull[0]])
            ax.plot(
                hull[:, 0], hull[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
                rasterized=False,
            )
            return

    # Fallback: dashed bounding box when hull cannot be computed.
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    dx = (xmax - xmin) * float(pad_frac)
    dy = (ymax - ymin) * float(pad_frac)
    xs = [xmin - dx, xmax + dx, xmax + dx, xmin - dx, xmin - dx]
    ys = [ymin - dy, ymin - dy, ymax + dy, ymax + dy, ymin - dy]
    ax.plot(
        xs, ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        rasterized=False,
    )


def infer_coordinate_cols(obs, x_col=None, y_col=None):
    if x_col and y_col:
        if x_col not in obs.columns or y_col not in obs.columns:
            raise KeyError(f"Coordinate columns not found: {x_col}, {y_col}")
        return x_col, y_col

    pairs = [
        ("x_plotting", "y_plotting"),
        ("x_image", "y_image"),
        ("x_scaled_image", "y_scaled_image"),
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
    ]
    for x, y in pairs:
        if x in obs.columns and y in obs.columns:
            return x, y

    raise KeyError(
        "Cannot infer spatial coordinate columns. "
        "Please provide --x_col and --y_col. "
        f"Available obs columns: {list(obs.columns)[:50]}"
    )


def infer_obs_group_col(obs, stats_groups, tissue_col=None):
    """
    Infer which adata.obs column corresponds to stats_long['celltype'].
    This is used only for highlighting/faceting spatial plots.
    """
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

    if best_overlap <= 0:
        warnings.warn(
            "No adata.obs column overlaps with stats_long['celltype']; "
            "spatial plots will not outline the selected tissue/celltype. "
            "Use --tissue_col if you know the correct column."
        )
        return tissue_col if tissue_col in obs.columns else None

    print(f"[infer] Using adata.obs['{best_col}'] for tissue/celltype highlighting; overlap={best_overlap}")
    return best_col


def infer_sample_col(obs, sample_col=None):
    if sample_col and sample_col in obs.columns:
        return sample_col
    for c in ["sample", "Sample", "Sample_barcode", "sampleid", "batch", "section", "slide"]:
        if c in obs.columns:
            return c
    return None


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
            print(f"[infer] Using significance column: {c}")
            return c
    raise KeyError("No usable q/p column found in stats_long.")


# -----------------------------
# Trait metadata
# -----------------------------
def load_trait_meta(path):
    meta = pd.read_csv(path)
    if "trait" not in meta.columns:
        # fall back to first column
        meta = meta.rename(columns={meta.columns[0]: "trait"})

    name_col = None
    for c in ["reportedTrait", "reported_trait", "trait_name", "name", "metabolite", "label"]:
        if c in meta.columns:
            name_col = c
            break
    if name_col is None:
        meta["reportedTrait"] = meta["trait"].astype(str)
        name_col = "reportedTrait"

    # Find a column whose values contain "levels" or "ratio".
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

    out = meta[["trait", name_col]].copy()
    out = out.rename(columns={name_col: "reportedTrait"})
    out["trait"] = out["trait"].astype(str)
    out["reportedTrait"] = out["reportedTrait"].astype(str)

    if type_col is not None:
        vals = meta[type_col].astype(str).str.lower()
        out["trait_type"] = np.where(
            vals.str.contains("ratio", regex=False), "ratio",
            np.where(vals.str.contains("level", regex=False), "levels", "unknown")
        )
    else:
        # infer from reportedTrait suffix
        rt = out["reportedTrait"].str.lower()
        out["trait_type"] = np.where(
            rt.str.contains(r"\bratio\b", regex=True), "ratio",
            np.where(rt.str.contains(r"\blevels?\b", regex=True), "levels", "unknown")
        )

    # Display name: remove suffix " levels" / " ratio"
    out["trait_display"] = (
        out["reportedTrait"]
        .str.replace(r"\s+(levels?|ratio)\s*$", "", regex=True, case=False)
        .str.strip()
    )
    out.loc[out["trait_display"].eq(""), "trait_display"] = out.loc[out["trait_display"].eq(""), "trait"]

    return out.drop_duplicates("trait")


def add_trait_meta(stats, trait_meta):
    stats = stats.copy()
    stats["trait"] = stats["trait"].astype(str)
    out = stats.merge(trait_meta, on="trait", how="left")
    out["trait_display"] = out["trait_display"].fillna(out["trait"])
    out["reportedTrait"] = out["reportedTrait"].fillna(out["trait"])
    out["trait_type"] = out["trait_type"].fillna("unknown")
    return out


def is_unknown_x(df):
    s1 = df["trait"].astype(str).str.upper()
    s2 = df["trait_display"].astype(str).str.upper()
    s3 = df["reportedTrait"].astype(str).str.upper()
    return s1.str.startswith("X-") | s2.str.startswith("X-") | s3.str.startswith("X-")


# -----------------------------
# Top metabolite selection
# -----------------------------
def compute_empirical_null_tau(df, q_col, bg_q=0.20, quantile=0.95):
    """
    Empirical-null effect threshold:
    tau = quantile(|final_effect| among background pairs with q > bg_q)
    calculated separately for levels/ratio/unknown when possible.
    """
    tau_by_type = {}
    for ttype, sub in df.groupby("trait_type"):
        bg = sub.loc[pd.to_numeric(sub[q_col], errors="coerce") > bg_q, "final_effect"]
        bg = pd.to_numeric(bg, errors="coerce").abs()
        bg = bg[np.isfinite(bg)]
        tau = float(np.quantile(bg, quantile)) if bg.size > 0 else 0.0
        tau_by_type[ttype] = tau

    all_bg = df.loc[pd.to_numeric(df[q_col], errors="coerce") > bg_q, "final_effect"]
    all_bg = pd.to_numeric(all_bg, errors="coerce").abs()
    all_bg = all_bg[np.isfinite(all_bg)]
    global_tau = float(np.quantile(all_bg, quantile)) if all_bg.size > 0 else 0.0

    return tau_by_type, global_tau



def _max_other_values_by_trait(df, value_col, trait_col="trait"):
    """
    For each row, compute max(value_col) among the same trait but other celltypes.
    If a trait has only one valid row, returns NaN for that row.
    """
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
            max2 = np.max(finite_vals[finite_vals != max1]) if finite_vals.size > 1 else np.nan
            other = np.where(vals == max1, max2, max1)
            other[~finite] = np.nan

        out.loc[idx] = other
    return out


def add_specificity_gap_columns(df):
    """
    Add trait-level specificity metrics across celltypes/tissues.

    Positive specificity:
        specificity_gap_positive = max(final_effect, 0) - max_other(max(final_effect, 0))
    Negative specificity:
        specificity_gap_negative = max(-final_effect, 0) - max_other(max(-final_effect, 0))
    Absolute specificity:
        specificity_gap_abs      = abs(final_effect) - max_other(abs(final_effect))

    A larger positive value means the trait is more specific to this tissue/celltype
    than to any other tissue/celltype in the same trait.
    """
    df = df.copy()
    eff = pd.to_numeric(df["final_effect"], errors="coerce").astype(float)

    df["positive_effect_strength"] = np.where(eff > 0, eff, 0.0)
    df["negative_effect_strength"] = np.where(eff < 0, -eff, 0.0)
    df["abs_final_effect"] = np.abs(eff)

    df["other_max_positive_effect"] = _max_other_values_by_trait(df, "positive_effect_strength")
    df["other_max_negative_effect"] = _max_other_values_by_trait(df, "negative_effect_strength")
    df["other_max_abs_effect"] = _max_other_values_by_trait(df, "abs_final_effect")

    # If there is no other valid tissue for the same trait, use 0 as background.
    df["other_max_positive_effect"] = df["other_max_positive_effect"].fillna(0.0)
    df["other_max_negative_effect"] = df["other_max_negative_effect"].fillna(0.0)
    df["other_max_abs_effect"] = df["other_max_abs_effect"].fillna(0.0)

    df["specificity_gap_positive"] = df["positive_effect_strength"] - df["other_max_positive_effect"]
    df["specificity_gap_negative"] = df["negative_effect_strength"] - df["other_max_negative_effect"]
    df["specificity_gap_abs"] = df["abs_final_effect"] - df["other_max_abs_effect"]

    df["specificity_gap_directional"] = np.where(
        eff >= 0,
        df["specificity_gap_positive"],
        df["specificity_gap_negative"],
    )
    return df


def make_top_tables(df, q_col, top_n=5, q_cutoff=0.05, use_empirical_null=True,
                    emp_bg_q=0.20, emp_quantile=0.95,
                    use_specificity_gap=False, specificity_gap_min=0.03,
                    specificity_gap_mode="directional"):
    df = df.copy()
    df["final_effect"] = pd.to_numeric(df["final_effect"], errors="coerce")
    df[q_col] = pd.to_numeric(df[q_col], errors="coerce")

    # Add cross-tissue / cross-celltype specificity metrics before filtering.
    df = add_specificity_gap_columns(df)

    if specificity_gap_mode not in ["directional", "absolute", "positive"]:
        raise ValueError("specificity_gap_mode must be one of: directional, absolute, positive")

    if specificity_gap_mode == "absolute":
        df["specificity_gap_used"] = df["specificity_gap_abs"]
    elif specificity_gap_mode == "positive":
        df["specificity_gap_used"] = df["specificity_gap_positive"]
    else:
        df["specificity_gap_used"] = df["specificity_gap_directional"]

    df["passes_specificity_gap"] = (
        (not use_specificity_gap)
        | (pd.to_numeric(df["specificity_gap_used"], errors="coerce") >= float(specificity_gap_min))
    )

    tau_by_type, global_tau = compute_empirical_null_tau(
        df, q_col=q_col, bg_q=emp_bg_q, quantile=emp_quantile
    )

    if use_empirical_null:
        df["empirical_null_tau"] = df["trait_type"].map(tau_by_type).fillna(global_tau)
    else:
        df["empirical_null_tau"] = 0.0

    df["abs_final_effect"] = df["final_effect"].abs()
    df["is_significant"] = (
        np.isfinite(df["final_effect"])
        & np.isfinite(df[q_col])
        & (df[q_col] <= q_cutoff)
        & (df["abs_final_effect"] >= df["empirical_null_tau"])
        & (df["passes_specificity_gap"])
    )

    # Significant table
    sig = df.loc[df["is_significant"]].copy()

    tables = {}
    for ttype in ["levels", "ratio", "unknown"]:
        sub_sig = sig.loc[sig["trait_type"].eq(ttype)].copy()

        pos = (
            sub_sig.loc[sub_sig["final_effect"] > 0]
            .sort_values(["celltype", "final_effect"], ascending=[True, False])
            .groupby("celltype", group_keys=False)
            .head(top_n)
            .copy()
        )
        if not pos.empty:
            pos["rank_in_tissue"] = pos.groupby("celltype")["final_effect"].rank(
                method="first", ascending=False
            ).astype(int)

        neg = (
            sub_sig.loc[sub_sig["final_effect"] < 0]
            .assign(abs_final_effect=lambda x: x["final_effect"].abs())
            .sort_values(["celltype", "abs_final_effect"], ascending=[True, False])
            .groupby("celltype", group_keys=False)
            .head(top_n)
            .copy()
        )
        if not neg.empty:
            neg["rank_in_tissue"] = neg.groupby("celltype")["abs_final_effect"].rank(
                method="first", ascending=False
            ).astype(int)

        ab = (
            sub_sig.assign(abs_final_effect=lambda x: x["final_effect"].abs())
            .sort_values(["celltype", "abs_final_effect"], ascending=[True, False])
            .groupby("celltype", group_keys=False)
            .head(top_n)
            .copy()
        )
        if not ab.empty:
            ab["rank_in_tissue"] = ab.groupby("celltype")["abs_final_effect"].rank(
                method="first", ascending=False
            ).astype(int)

        tables[(ttype, "positive")] = pos
        tables[(ttype, "negative")] = neg
        tables[(ttype, "abs")] = ab

        # Fallback top tables without q/effect threshold, useful when a tissue has no significant pair
        sub_all = df.loc[df["trait_type"].eq(ttype)].copy()
        if use_specificity_gap:
            sub_all = sub_all.loc[sub_all["passes_specificity_gap"]].copy()
        fallback = (
            sub_all.assign(abs_final_effect=lambda x: x["final_effect"].abs())
            .sort_values(["celltype", "abs_final_effect"], ascending=[True, False])
            .groupby("celltype", group_keys=False)
            .head(top_n)
            .copy()
        )
        if not fallback.empty:
            fallback["rank_in_tissue"] = fallback.groupby("celltype")["abs_final_effect"].rank(
                method="first", ascending=False
            ).astype(int)
        tables[(ttype, "fallback_abs_all")] = fallback

    params = {
        "q_col": q_col,
        "q_cutoff": q_cutoff,
        "use_empirical_null": bool(use_empirical_null),
        "empirical_null_bg_q": emp_bg_q,
        "empirical_null_quantile": emp_quantile,
        "tau_by_type": tau_by_type,
        "global_tau": global_tau,
        "use_specificity_gap": bool(use_specificity_gap),
        "specificity_gap_min": float(specificity_gap_min),
        "specificity_gap_mode": specificity_gap_mode,
    }
    return df, sig, tables, params


# -----------------------------
# Heatmap
# -----------------------------
def plot_effect_heatmap(top_df, full_df, out_pdf, title, max_traits=80):
    if top_df is None or top_df.empty:
        print(f"[skip] Empty heatmap: {title}")
        return

    traits = list(dict.fromkeys(top_df["trait"].astype(str).tolist()))
    if len(traits) > max_traits:
        # keep traits with largest max absolute effect in the full table
        tmp = (
            full_df.loc[full_df["trait"].isin(traits)]
            .assign(abs_effect=lambda x: pd.to_numeric(x["final_effect"], errors="coerce").abs())
            .groupby("trait")["abs_effect"].max()
            .sort_values(ascending=False)
            .head(max_traits)
        )
        traits = tmp.index.tolist()

    sub = full_df.loc[full_df["trait"].isin(traits)].copy()
    if sub.empty:
        return

    row_order = sorted(sub["celltype"].astype(str).unique())
    trait_label = (
        sub[["trait", "trait_display"]]
        .drop_duplicates("trait")
        .set_index("trait")["trait_display"]
        .to_dict()
    )

    mat = sub.pivot_table(index="celltype", columns="trait", values="final_effect", aggfunc="mean")
    mat = mat.reindex(index=row_order, columns=traits)

    data = mat.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return

    vmax = np.nanquantile(np.abs(finite), 0.98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.1

    fig_w = max(7.5, min(28, 0.28 * len(traits) + 3.0))
    fig_h = max(4.5, min(32, 0.25 * len(row_order) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_obj = get_clean_diverging_cmap("gmetmap_redblue")
    norm = TwoSlopeNorm(vmin=-float(vmax), vcenter=0.0, vmax=float(vmax))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap_obj, norm=norm)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Metabolites")
    ax.set_ylabel("Tissue / region")

    ax.set_xticks(np.arange(len(traits)))
    ax.set_xticklabels([trait_label.get(t, t) for t in traits], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=6)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="both")
    cb.set_label("Final effect", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[save] {out_pdf}")


# -----------------------------
# Score matrix reading
# -----------------------------
def score_path(score_dir, prefix, method, kind, stage):
    return Path(score_dir) / f"{prefix}_{method}_{kind}_{stage}.csv.gz"


def read_score_subset(path, obs_names, traits):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score matrix not found: {path}")

    traits = [str(t) for t in traits]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if len(header) == 0:
        raise ValueError(f"Empty score matrix header: {path}")

    idx_col = header[0]
    header_set = set(header)
    keep_traits = [t for t in traits if t in header_set]
    missing = sorted(set(traits) - set(keep_traits))
    if missing:
        warnings.warn(f"{len(missing)} selected traits are missing in {path.name}; examples: {missing[:5]}")

    usecols = [idx_col] + keep_traits
    df = pd.read_csv(path, usecols=usecols, index_col=0)
    df.index = df.index.astype(str)
    df = df.reindex(pd.Index(obs_names.astype(str), name=df.index.name))
    return df


def build_pair_score(row, mode, scores):
    trait = str(row["trait"])
    side = str(row.get("final_side", "UP")).upper()
    effect = float(row.get("final_effect", np.nan))

    if mode == "net":
        return scores["net"][trait]
    if mode == "up":
        return scores["up"][trait]
    if mode == "down":
        return scores["down"][trait]
    if mode == "signed_net":
        v = scores["net"][trait].copy()
        # Align negative final effects so high color means stronger tissue-associated direction.
        if np.isfinite(effect) and effect < 0:
            v = -v
        return v
    if mode == "finalside":
        if side == "DOWN":
            return -scores["down"][trait]
        return scores["up"][trait]
    raise ValueError(f"Unknown score mode: {mode}")


# -----------------------------
# Spatial plotting
# -----------------------------
def plot_one_spatial(obs, score, x_col, y_col, title, out_png=None, pdf=None,
                     group_col=None, focus_group=None, sample_col=None, max_panels=4,
                     point_size=1.2, cmap="gmetmap_redblue", symmetric=True,
                     outline_focus=True, focus_dashed_boundary=False,
                     focus_dashed_boundary_color="black",
                     focus_dashed_boundary_linewidth=0.8,
                     focus_dashed_boundary_alpha=0.95,
                     focus_dashed_boundary_linestyle="--",
                     top_score_outline=False, top_score_frac=0.05,
                     top_score_outline_by="positive", top_score_outline_color="black",
                     top_score_outline_linewidth=0.25, top_score_outline_alpha=0.95,
                     color_scale_mode="robust", color_bg_q=0.97, color_focus_q=0.995,
                     color_robust_upper_q=0.995, color_robust_lower_q=0.005,
                     invert_y=True, dpi=450, max_points_per_panel=None):
    plot_obs = obs.copy()
    plot_obs["_score_"] = pd.to_numeric(pd.Series(score, index=plot_obs.index), errors="coerce")
    plot_obs = plot_obs[np.isfinite(plot_obs["_score_"])].copy()
    if plot_obs.empty:
        return None

    # Choose panels/samples
    if sample_col and sample_col in plot_obs.columns:
        if group_col and focus_group is not None and group_col in plot_obs.columns:
            counts = (
                plot_obs.assign(_is_focus_=plot_obs[group_col].astype(str).eq(str(focus_group)))
                .groupby(sample_col)["_is_focus_"].sum()
                .sort_values(ascending=False)
            )
            selected = counts[counts > 0].head(max_panels).index.tolist()
            if len(selected) == 0:
                selected = plot_obs[sample_col].value_counts().head(max_panels).index.tolist()
        else:
            selected = plot_obs[sample_col].value_counts().head(max_panels).index.tolist()
    else:
        selected = [None]

    # Color limits from selected panels
    sel_mask = np.ones(plot_obs.shape[0], dtype=bool)
    if selected != [None] and sample_col:
        sel_mask = plot_obs[sample_col].isin(selected).to_numpy()
    vals = plot_obs.loc[sel_mask, "_score_"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    # Use a clean colormap and an optional focus-aware normalization.
    # focus_positive mode sets the white/red transition to the non-focus upper background,
    # so red is reserved for values that exceed most non-focus spots/cells.
    cmap_obj = get_clean_diverging_cmap(cmap)
    norm, vmin, vmax, vcenter, color_mode_used = get_focus_contrast_norm(
        plot_obs=plot_obs,
        sel_mask=sel_mask,
        group_col=group_col,
        focus_group=focus_group,
        score_col="_score_",
        mode=color_scale_mode,
        symmetric=symmetric,
        robust_upper_q=color_robust_upper_q,
        robust_lower_q=color_robust_lower_q,
        bg_q=color_bg_q,
        focus_q=color_focus_q,
    )

    n = len(selected)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig_w = 3.2 * ncols + 0.7
    fig_h = 3.1 * nrows + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    last_sc = None
    for ax_i, sample in enumerate(selected):
        ax = axes_flat[ax_i]
        if sample is None:
            sub = plot_obs
            panel_title = "all spots"
        else:
            sub = plot_obs.loc[plot_obs[sample_col].eq(sample)].copy()
            panel_title = str(sample)

        if max_points_per_panel is not None and sub.shape[0] > max_points_per_panel:
            sub = sub.sample(n=max_points_per_panel, random_state=0)

        last_sc = ax.scatter(
            sub[x_col], sub[y_col],
            c=sub["_score_"],
            s=point_size,
            linewidths=0,
            cmap=cmap_obj,
            norm=norm,
            rasterized=True
        )

        if outline_focus and group_col and focus_group is not None and group_col in sub.columns:
            focus = sub.loc[sub[group_col].astype(str).eq(str(focus_group))]
            if not focus.empty:
                # Outline only a random subset when too many focus points, to keep PDFs light.
                if focus.shape[0] > 12000:
                    focus = focus.sample(n=12000, random_state=1)
                ax.scatter(
                    focus[x_col], focus[y_col],
                    s=max(point_size * 2.2, 1.6),
                    facecolors="none",
                    edgecolors="#4D4D4D",
                    linewidths=0.05,
                    alpha=0.25,
                    rasterized=True
                )

        if focus_dashed_boundary and group_col and focus_group is not None and group_col in sub.columns:
            focus_boundary = sub.loc[sub[group_col].astype(str).eq(str(focus_group))]
            if not focus_boundary.empty:
                draw_focus_dashed_boundary(
                    ax=ax,
                    focus_df=focus_boundary,
                    x_col=x_col,
                    y_col=y_col,
                    color=focus_dashed_boundary_color,
                    linewidth=focus_dashed_boundary_linewidth,
                    alpha=focus_dashed_boundary_alpha,
                    linestyle=focus_dashed_boundary_linestyle,
                    zorder=5,
                )

        if top_score_outline:
            top_spots = select_top_score_outline_points(
                sub,
                score_col="_score_",
                frac=top_score_frac,
                mode=top_score_outline_by,
            )
            if not top_spots.empty:
                # Keep files light if an extremely large number of spots are selected.
                if top_spots.shape[0] > 12000:
                    top_spots = top_spots.sample(n=12000, random_state=2)
                ax.scatter(
                    top_spots[x_col], top_spots[y_col],
                    s=max(point_size * 4.2, 3.0),
                    facecolors="none",
                    edgecolors=top_score_outline_color,
                    linewidths=top_score_outline_linewidth,
                    alpha=top_score_outline_alpha,
                    rasterized=True,
                    zorder=4,
                )

        ax.set_title(panel_title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if invert_y:
            ax.invert_yaxis()

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=9, y=0.985)

    # Leave a dedicated blank strip on the right and place the colorbar there,
    # so the colorbar does not overlap the rightmost spatial panel.
    fig.subplots_adjust(left=0.015, right=0.865, bottom=0.035, top=0.88,
                        wspace=0.10, hspace=0.22)

    if last_sc is not None:
        cax = fig.add_axes([0.895, 0.16, 0.012, 0.66])
        cb = fig.colorbar(last_sc, cax=cax, extend="both")
        cb.set_label("Metabolite score", fontsize=8, labelpad=8)
        cb.ax.tick_params(labelsize=7, length=2)
        if str(color_mode_used).startswith("focus"):
            cb.ax.axhline(vcenter, color="black", linewidth=0.4, alpha=0.7)
            cb.ax.text(1.65, vcenter, f"bg q={color_bg_q:g}",
                       transform=cb.ax.get_yaxis_transform(),
                       va="center", ha="left", fontsize=5.5)

    if pdf is not None:
        pdf.savefig(fig, dpi=dpi)
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def make_spatial_plots(plot_df, obs, score_dir, outdir, prefix, method, stage,
                       x_col, y_col, group_col=None, sample_col=None,
                       score_mode="net", spatial_top_n=1, max_plots=120,
                       point_size=1.2, max_panels=4, cmap="gmetmap_redblue",
                       invert_y=True, outline_focus=True,
                       focus_dashed_boundary=False,
                       focus_dashed_boundary_color="black",
                       focus_dashed_boundary_linewidth=0.8,
                       focus_dashed_boundary_alpha=0.95,
                       focus_dashed_boundary_linestyle="--",
                       top_score_outline=False,
                       top_score_frac=0.05, top_score_outline_by="positive",
                       top_score_outline_color="black", top_score_outline_linewidth=0.25,
                       top_score_outline_alpha=0.95,
                       color_scale_mode="robust", color_bg_q=0.97, color_focus_q=0.995,
                       color_robust_upper_q=0.995, color_robust_lower_q=0.005,
                       max_points_per_panel=None, dpi=450):
    if plot_df is None or plot_df.empty:
        print("[skip] No selected pairs for spatial plots.")
        return

    mkdir(outdir)
    plot_df = plot_df.copy()
    plot_df["celltype"] = plot_df["celltype"].astype(str)
    plot_df["trait"] = plot_df["trait"].astype(str)

    # keep top N per tissue in plot_df order
    if "rank_in_tissue" in plot_df.columns:
        plot_df = plot_df.sort_values(["celltype", "rank_in_tissue"])
    plot_df = plot_df.groupby("celltype", group_keys=False).head(spatial_top_n)
    plot_df = plot_df.head(max_plots)

    selected_traits = sorted(plot_df["trait"].unique())

    needed_kinds = {"net"}
    if score_mode in ["up", "finalside"]:
        needed_kinds.add("up")
    if score_mode in ["down", "finalside"]:
        needed_kinds.add("down")
    if score_mode == "signed_net":
        needed_kinds.add("net")

    scores = {}
    for kind in sorted(needed_kinds):
        p = score_path(score_dir, prefix, method, kind, stage)
        print(f"[read] {p}")
        scores[kind] = read_score_subset(p, obs.index.astype(str), selected_traits)

    pdf_path = Path(outdir) / f"spatial_top_pairs_{score_mode}.pdf"
    index_rows = []
    with PdfPages(pdf_path) as pdf:
        for k, row in plot_df.iterrows():
            trait = str(row["trait"])
            if not all(trait in scores[kind].columns for kind in scores):
                continue

            display = str(row.get("trait_display", trait))
            ttype = str(row.get("trait_type", "unknown"))
            focus = str(row["celltype"])
            eff = row.get("final_effect", np.nan)
            qv = row.get("q_value_used", row.get("final_q_celltype_across_traits", np.nan))
            side = row.get("final_side", "")

            score = build_pair_score(row, score_mode, scores)

            title = (
                f"{display} [{ttype}] | focus: {focus}\n"
                f"final_effect={float(eff):.4g}, q={float(qv):.3g}, side={side}, score_mode={score_mode}"
                if np.isfinite(pd.to_numeric(eff, errors='coerce')) else
                f"{display} [{ttype}] | focus: {focus} | score_mode={score_mode}"
            )

            single_pdf = Path(outdir) / f"{safe_name(focus)}__{safe_name(display)}__{safe_name(trait)}__{score_mode}.pdf"
            plot_one_spatial(
                obs=obs,
                score=score,
                x_col=x_col,
                y_col=y_col,
                title=title,
                out_png=single_pdf,
                pdf=pdf,
                group_col=group_col,
                focus_group=focus,
                sample_col=sample_col,
                max_panels=max_panels,
                point_size=point_size,
                cmap=cmap,
                symmetric=(score_mode in ["net", "signed_net", "finalside"]),
                outline_focus=outline_focus,
                focus_dashed_boundary=focus_dashed_boundary,
                focus_dashed_boundary_color=focus_dashed_boundary_color,
                focus_dashed_boundary_linewidth=focus_dashed_boundary_linewidth,
                focus_dashed_boundary_alpha=focus_dashed_boundary_alpha,
                focus_dashed_boundary_linestyle=focus_dashed_boundary_linestyle,
                top_score_outline=top_score_outline,
                top_score_frac=top_score_frac,
                top_score_outline_by=top_score_outline_by,
                top_score_outline_color=top_score_outline_color,
                top_score_outline_linewidth=top_score_outline_linewidth,
                top_score_outline_alpha=top_score_outline_alpha,
                color_scale_mode=color_scale_mode,
                color_bg_q=color_bg_q,
                color_focus_q=color_focus_q,
                color_robust_upper_q=color_robust_upper_q,
                color_robust_lower_q=color_robust_lower_q,
                invert_y=invert_y,
                dpi=dpi,
                max_points_per_panel=max_points_per_panel
            )

            index_rows.append({
                "celltype": focus,
                "trait": trait,
                "trait_display": display,
                "trait_type": ttype,
                "final_effect": eff,
                "q_value_used": qv,
                "final_side": side,
                "score_mode": score_mode,
                "pdf": str(single_pdf)
            })

    pd.DataFrame(index_rows).to_csv(Path(outdir) / "spatial_plot_index.csv", index=False)
    print(f"[save] {pdf_path}")
    print(f"[save] {Path(outdir) / 'spatial_plot_index.csv'}")



def parse_target_traits(x):
    """Parse GCST IDs robustly."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            out.extend(parse_target_traits(v))
        return list(dict.fromkeys(out))
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


def make_target_trait_plot_df(
    full_df: pd.DataFrame,
    target_traits,
    direction="positive",
    target_top_tissues=10,
    target_q_cutoff=None,
    target_use_q_filter=False,
    target_trait_types=None,
):
    """Build plot_df for user-specified GCST/trait IDs."""
    target_traits = parse_target_traits(target_traits)
    if not target_traits:
        raise ValueError("--target_traits is empty.")

    df = full_df.copy()
    df["trait"] = df["trait"].astype(str)
    df["celltype"] = df["celltype"].astype(str)
    df["final_effect"] = pd.to_numeric(df["final_effect"], errors="coerce")

    sub = df.loc[df["trait"].isin(target_traits)].copy()
    missing = sorted(set(target_traits) - set(sub["trait"].astype(str).unique()))
    if missing:
        warnings.warn(f"Requested target traits not found after method/stage filtering: {missing}")

    if sub.empty:
        examples = sorted(df["trait"].astype(str).dropna().unique().tolist())[:10]
        raise ValueError(
            "No rows found for --target_traits after method/stage filtering. "
            "Please check the GCST IDs and --method/--stage.\n"
            f"Cleaned target_traits = {target_traits}\n"
            f"Example trait IDs in current filtered table = {examples}"
        )

    if target_trait_types and "trait_type" in sub.columns:
        allowed = set([str(x).strip() for x in target_trait_types if str(x).strip()])
        sub = sub.loc[sub["trait_type"].astype(str).isin(allowed)].copy()

    if target_use_q_filter and target_q_cutoff is not None and "q_value_used" in sub.columns:
        qv = pd.to_numeric(sub["q_value_used"], errors="coerce")
        sub = sub.loc[qv <= float(target_q_cutoff)].copy()

    if sub.empty:
        raise ValueError("No target-trait rows left after optional trait_type/q filtering.")

    direction = str(direction).lower()
    out = []
    for trait, g in sub.groupby("trait", sort=False):
        g = g.copy()
        if direction == "positive":
            g = g.loc[g["final_effect"] > 0].copy()
            g = g.sort_values("final_effect", ascending=False)
        elif direction == "negative":
            g = g.loc[g["final_effect"] < 0].copy()
            g = g.assign(_rank_abs_=g["final_effect"].abs()).sort_values("_rank_abs_", ascending=False)
        elif direction == "abs":
            g = g.assign(_rank_abs_=g["final_effect"].abs()).sort_values("_rank_abs_", ascending=False)
        elif direction == "all":
            g = g.sort_values("final_effect", ascending=False)
        else:
            raise ValueError("--target_direction must be one of: positive, negative, abs, all")

        if g.empty:
            warnings.warn(f"Target trait {trait} has no rows after target_direction={direction}.")
            continue

        if target_top_tissues is not None and int(target_top_tissues) > 0:
            g = g.head(int(target_top_tissues)).copy()

        g["rank_in_tissue"] = 1
        g["rank_in_target_trait"] = np.arange(1, g.shape[0] + 1)
        g["target_trait_mode"] = True
        out.append(g)

    if not out:
        raise ValueError("No target-trait rows available for plotting.")
    return pd.concat(out, axis=0, ignore_index=True)



# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Identify each tissue's top associated metabolites and visualize their spatial metabolite scores."
    )
    ap.add_argument("--h5ad", required=True, help="Spatial AnnData h5ad file.")
    ap.add_argument("--scheme_long", required=True, help="Meta_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz")
    ap.add_argument("--score_dir", required=True, help="Directory containing cellTraitMatrices.")
    ap.add_argument("--trait_meta", required=True, help="meta_category_levels_ratio.csv")
    ap.add_argument("--outdir", default="Tissue_top_metabolites_spatial")
    ap.add_argument("--prefix", default="Meta")
    ap.add_argument("--method", default="wAUCell")
    ap.add_argument("--stage", default="S3_geneSetZ_cellZ")

    ap.add_argument("--tissue_col", default=None, help="adata.obs column matching stats_long['celltype']; auto-inferred if omitted.")
    ap.add_argument("--sample_col", default=None, help="Column used to facet spatial sections; auto-inferred if omitted.")
    ap.add_argument("--x_col", default=None)
    ap.add_argument("--y_col", default=None)

    ap.add_argument("--q_col", default=None, help="q/p column for significance; auto-inferred if omitted.")
    ap.add_argument("--q_cutoff", type=float, default=0.05)
    ap.add_argument("--top_n_per_tissue", type=int, default=5)
    ap.add_argument("--exclude_unknown_x", action="store_true", help="Exclude traits whose name starts with X-.")
    ap.add_argument("--use_empirical_null", action="store_true", help="Require |final_effect| >= empirical-null tau.")
    ap.add_argument("--emp_null_bg_q", type=float, default=0.20)
    ap.add_argument("--emp_null_quantile", type=float, default=0.95)
    ap.add_argument("--use_specificity_gap", action="store_true",
                    help="Require selected metabolites to be tissue/celltype-specific compared with other tissues/celltypes of the same trait.")
    ap.add_argument("--specificity_gap_min", type=float, default=0.03,
                    help="Minimum specificity gap. Example: focus effect must exceed the strongest other tissue by this value.")
    ap.add_argument("--specificity_gap_mode", default="directional", choices=["directional", "absolute", "positive"],
                    help="directional: positive uses positive gap and negative uses negative gap; absolute: uses abs effect gap; positive: only positive-effect specificity.")

    ap.add_argument("--make_heatmaps", action="store_true", default=True)
    ap.add_argument("--heatmap_max_traits", type=int, default=80)

    ap.add_argument("--make_spatial", action="store_true", default=True)
    ap.add_argument("--plot_trait_types", default="levels,ratio", help="Comma-separated: levels,ratio,unknown")
    ap.add_argument("--plot_direction", default="positive", choices=["positive", "negative", "abs", "fallback_abs_all"])
    ap.add_argument("--spatial_score_mode", default="net", choices=["net", "signed_net", "finalside", "up", "down"])
    ap.add_argument("--target_traits", default=None,
                    help="Comma-separated target metabolite/trait IDs, e.g. GCST90200223,GCST90200417. If set, plot these traits directly.")
    ap.add_argument("--target_top_tissues", type=int, default=10,
                    help="When --target_traits is set, keep top N tissues/celltypes per target trait. Use 0 or negative to keep all.")
    ap.add_argument("--target_direction", default="positive", choices=["positive", "negative", "abs", "all"],
                    help="How to rank/filter tissues for --target_traits.")
    ap.add_argument("--target_use_q_filter", action="store_true",
                    help="When --target_traits is set, additionally require q_value_used <= --q_cutoff.")
    ap.add_argument("--spatial_top_n", type=int, default=1, help="Number of top metabolites plotted per tissue.")
    ap.add_argument("--max_spatial_plots", type=int, default=120)
    ap.add_argument("--max_sample_panels", type=int, default=4)
    ap.add_argument("--point_size", type=float, default=1.2)
    ap.add_argument("--cmap", default="gmetmap_redblue",
                    help="Colormap for spatial plots. Use gmetmap_redblue for clean red-white-blue colors, or any matplotlib cmap such as coolwarm/RdBu_r.")
    ap.add_argument("--color_scale_mode", default="robust",
                    choices=["robust", "focus_positive", "focus_negative", "focus_auto"],
                    help="Color normalization. robust=global robust scale; focus_positive=white/red transition is non-focus upper background; focus_auto chooses positive/negative focus contrast automatically.")
    ap.add_argument("--color_bg_q", type=float, default=0.97,
                    help="For focus color scaling, non-focus quantile used as the white/red background threshold. Default 0.97.")
    ap.add_argument("--color_focus_q", type=float, default=0.995,
                    help="For focus color scaling, focus quantile used as vmax/vmin. Default 0.995.")
    ap.add_argument("--color_robust_upper_q", type=float, default=0.995,
                    help="Robust upper quantile for normal color scaling and fallback. Default 0.995.")
    ap.add_argument("--color_robust_lower_q", type=float, default=0.005,
                    help="Robust lower quantile for normal color scaling and fallback. Default 0.005.")
    ap.add_argument("--no_invert_y", action="store_true",
                    help="Do not apply the default y-axis inversion.")
    ap.add_argument("--flip_y_axis", action="store_true",
                    help="Flip the y-axis relative to the default behavior. Use this if the exported spatial map is upside down.")
    ap.add_argument("--no_outline_focus", action="store_true",
                    help="Do not draw outlines around all spots/cells from the focus tissue/region/celltype.")
    ap.add_argument("--focus_dashed_boundary", action="store_true",
                    help="Draw a black dashed boundary around the focus tissue/organ in each spatial panel.")
    ap.add_argument("--focus_dashed_boundary_color", default="black",
                    help="Color of the dashed focus-organ boundary. Default: black.")
    ap.add_argument("--focus_dashed_boundary_linewidth", type=float, default=0.8,
                    help="Line width of the dashed focus-organ boundary. Default: 0.8.")
    ap.add_argument("--focus_dashed_boundary_alpha", type=float, default=0.95,
                    help="Alpha of the dashed focus-organ boundary. Default: 0.95.")
    ap.add_argument("--focus_dashed_boundary_linestyle", default="--",
                    help="Line style of the focus-organ boundary. Default: --.")
    ap.add_argument("--top_score_outline", action="store_true",
                    help="Draw an outline around top-scoring spots/cells in each displayed spatial panel.")
    ap.add_argument("--top_score_frac", type=float, default=0.05,
                    help="Fraction of top score spots/cells to outline per panel. Default: 0.05 = top 5%.")
    ap.add_argument("--top_score_outline_by", default="positive", choices=["positive", "high", "absolute"],
                    help="How to define top score spots/cells: positive=top positive scores only; high=highest raw scores; absolute=largest absolute scores.")
    ap.add_argument("--top_score_outline_color", default="black",
                    help="Outline color for top-scoring spots/cells. Default: black.")
    ap.add_argument("--top_score_outline_linewidth", type=float, default=0.25,
                    help="Outline linewidth for top-scoring spots/cells.")
    ap.add_argument("--top_score_outline_alpha", type=float, default=0.95,
                    help="Outline alpha for top-scoring spots/cells.")
    ap.add_argument("--max_points_per_panel", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=450)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    heatmap_dir = outdir / "heatmaps"
    spatial_dir = outdir / "spatial_plots"
    mkdir(outdir)
    mkdir(tables_dir)
    mkdir(heatmap_dir)
    mkdir(spatial_dir)

    print("[1/6] Loading SchemeC long table...")
    stats = pd.read_csv(args.scheme_long)
    required = ["celltype", "trait", "final_effect"]
    missing_req = [c for c in required if c not in stats.columns]
    if missing_req:
        raise KeyError(f"Missing required columns in scheme_long: {missing_req}")

    stats["celltype"] = stats["celltype"].astype(str)
    stats["trait"] = stats["trait"].astype(str)

    if "method" in stats.columns:
        stats = stats.loc[stats["method"].astype(str).eq(args.method)].copy()
    if "stage" in stats.columns:
        stats = stats.loc[stats["stage"].astype(str).eq(args.stage)].copy()
    if stats.empty:
        raise ValueError(f"No rows left after filtering method={args.method}, stage={args.stage}")

    q_col = choose_q_col(stats, args.q_col)

    print("[2/6] Loading metabolite trait metadata...")
    trait_meta = load_trait_meta(args.trait_meta)
    stats = add_trait_meta(stats, trait_meta)

    if args.exclude_unknown_x:
        before = stats.shape[0]
        stats = stats.loc[~is_unknown_x(stats)].copy()
        print(f"[filter] exclude X-* unknown metabolites: {before} -> {stats.shape[0]} rows")

    stats["q_value_used"] = pd.to_numeric(stats[q_col], errors="coerce")
    stats["final_effect"] = pd.to_numeric(stats["final_effect"], errors="coerce")

    print("[3/6] Selecting top metabolites per tissue...")
    full_with_sig, sig_df, tables, params = make_top_tables(
        stats,
        q_col=q_col,
        top_n=args.top_n_per_tissue,
        q_cutoff=args.q_cutoff,
        use_empirical_null=args.use_empirical_null,
        emp_bg_q=args.emp_null_bg_q,
        emp_quantile=args.emp_null_quantile,
        use_specificity_gap=args.use_specificity_gap,
        specificity_gap_min=args.specificity_gap_min,
        specificity_gap_mode=args.specificity_gap_mode,
    )

    params.update({
        "h5ad": args.h5ad,
        "scheme_long": args.scheme_long,
        "score_dir": args.score_dir,
        "trait_meta": args.trait_meta,
        "method": args.method,
        "stage": args.stage,
        "top_n_per_tissue": args.top_n_per_tissue,
        "exclude_unknown_x": bool(args.exclude_unknown_x),
        "use_specificity_gap": bool(args.use_specificity_gap),
        "specificity_gap_min": args.specificity_gap_min,
        "specificity_gap_mode": args.specificity_gap_mode,
        "no_invert_y": bool(args.no_invert_y),
        "flip_y_axis": bool(args.flip_y_axis),
        "focus_dashed_boundary": bool(args.focus_dashed_boundary),
        "focus_dashed_boundary_color": args.focus_dashed_boundary_color,
        "focus_dashed_boundary_linewidth": args.focus_dashed_boundary_linewidth,
    })
    with open(outdir / "selection_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    full_with_sig.to_csv(tables_dir / "all_pairs_with_trait_meta_and_significance.csv.gz", index=False, compression="gzip")
    sig_df.to_csv(tables_dir / "all_significant_pairs.csv.gz", index=False, compression="gzip")

    # Save top tables
    for (ttype, direction), tab in tables.items():
        out_csv = tables_dir / f"top_{direction}_{ttype}_per_tissue.csv"
        tab.to_csv(out_csv, index=False)
        print(f"[save] {out_csv}  n={tab.shape[0]}")

    # Also save a concise top1 level table for the user's main question
    top1_levels = tables.get(("levels", "positive"), pd.DataFrame()).copy()
    if not top1_levels.empty:
        top1_levels = top1_levels.sort_values(["celltype", "rank_in_tissue"]).groupby("celltype", group_keys=False).head(1)
        top1_levels.to_csv(tables_dir / "MOST_RELATED_metabolite_levels_TOP1_positive_per_tissue.csv", index=False)
        print(f"[save] {tables_dir / 'MOST_RELATED_metabolite_levels_TOP1_positive_per_tissue.csv'}")

    print("[4/6] Drawing summary heatmaps...")
    if args.make_heatmaps:
        for ttype in ["levels", "ratio"]:
            for direction in ["positive", "negative", "abs"]:
                tab = tables.get((ttype, direction), pd.DataFrame())
                plot_effect_heatmap(
                    top_df=tab,
                    full_df=full_with_sig,
                    out_pdf=heatmap_dir / f"heatmap_top_{direction}_{ttype}_final_effect.pdf",
                    title=f"Top {direction} {ttype} metabolites per tissue ({args.method}, {args.stage})",
                    max_traits=args.heatmap_max_traits,
                )

    print("[5/6] Loading AnnData obs for spatial visualization...")
    adata = sc.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)

    x_col, y_col = infer_coordinate_cols(obs, args.x_col, args.y_col)
    group_col = infer_obs_group_col(obs, full_with_sig["celltype"], args.tissue_col)
    sample_col = infer_sample_col(obs, args.sample_col)
    print(f"[spatial] coordinates: x={x_col}, y={y_col}")
    print(f"[spatial] group_col={group_col}, sample_col={sample_col}")

    print("[6/6] Drawing spatial metabolite-score plots...")
    if args.make_spatial:
        if args.target_traits:
            print(f"[target] raw --target_traits: {args.target_traits}")
            target_ids = parse_target_traits(args.target_traits)
            print(f"[target] cleaned GCST/trait IDs: {target_ids}")

            plot_trait_types = [x.strip() for x in args.plot_trait_types.split(",") if x.strip()]
            top_n_target = args.target_top_tissues if args.target_top_tissues and args.target_top_tissues > 0 else None

            plot_df = make_target_trait_plot_df(
                full_df=full_with_sig,
                target_traits=target_ids,
                direction=args.target_direction,
                target_top_tissues=top_n_target,
                target_q_cutoff=args.q_cutoff,
                target_use_q_filter=args.target_use_q_filter,
                target_trait_types=plot_trait_types,
            )

            target_table = tables_dir / "TARGET_TRAITS_rows_used_for_spatial_plotting.csv"
            plot_df.to_csv(target_table, index=False)
            print(f"[save] {target_table}  n={plot_df.shape[0]}")

            spatial_top_n_for_call = 999999
        else:
            plot_trait_types = [x.strip() for x in args.plot_trait_types.split(",") if x.strip()]
            plot_tabs = []
            for ttype in plot_trait_types:
                tab = tables.get((ttype, args.plot_direction), pd.DataFrame()).copy()
                if tab.empty:
                    print(f"[warn] No significant {args.plot_direction} table for {ttype}; using fallback_abs_all.")
                    tab = tables.get((ttype, "fallback_abs_all"), pd.DataFrame()).copy()
                if not tab.empty:
                    tab["plot_trait_type_request"] = ttype
                    plot_tabs.append(tab)

            if len(plot_tabs) > 0:
                plot_df = pd.concat(plot_tabs, axis=0, ignore_index=True)
                spatial_top_n_for_call = args.spatial_top_n
            else:
                plot_df = pd.DataFrame()
                spatial_top_n_for_call = args.spatial_top_n

        if plot_df is not None and not plot_df.empty:
            make_spatial_plots(
                plot_df=plot_df,
                obs=obs,
                score_dir=args.score_dir,
                outdir=spatial_dir,
                prefix=args.prefix,
                method=args.method,
                stage=args.stage,
                x_col=x_col,
                y_col=y_col,
                group_col=group_col,
                sample_col=sample_col,
                score_mode=args.spatial_score_mode,
                spatial_top_n=spatial_top_n_for_call,
                max_plots=args.max_spatial_plots,
                point_size=args.point_size,
                max_panels=args.max_sample_panels,
                cmap=args.cmap,
                # Default behavior is invert_y=True unless --no_invert_y is used.
                # --flip_y_axis toggles that behavior, so it flips the exported spatial map.
                invert_y=((not args.no_invert_y) ^ bool(args.flip_y_axis)),
                outline_focus=(not args.no_outline_focus),
                focus_dashed_boundary=args.focus_dashed_boundary,
                focus_dashed_boundary_color=args.focus_dashed_boundary_color,
                focus_dashed_boundary_linewidth=args.focus_dashed_boundary_linewidth,
                focus_dashed_boundary_alpha=args.focus_dashed_boundary_alpha,
                focus_dashed_boundary_linestyle=args.focus_dashed_boundary_linestyle,
                top_score_outline=args.top_score_outline,
                top_score_frac=args.top_score_frac,
                top_score_outline_by=args.top_score_outline_by,
                top_score_outline_color=args.top_score_outline_color,
                top_score_outline_linewidth=args.top_score_outline_linewidth,
                top_score_outline_alpha=args.top_score_outline_alpha,
                color_scale_mode=args.color_scale_mode,
                color_bg_q=args.color_bg_q,
                color_focus_q=args.color_focus_q,
                color_robust_upper_q=args.color_robust_upper_q,
                color_robust_lower_q=args.color_robust_lower_q,
                max_points_per_panel=args.max_points_per_panel,
                dpi=args.dpi,
            )
        else:
            print("[skip] No tables available for spatial plotting.")

    print("\nDone.")
    print(f"Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
