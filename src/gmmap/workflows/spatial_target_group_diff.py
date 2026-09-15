#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differential analysis of a specified metabolite score within a specified tissue.

This script is designed for gMetMap / SchemeC spatial or single-cell outputs.
It compares one metabolite trait score between user-defined groups in one tissue/organ.

Main functions:
1) Read only required obs columns from h5ad using h5py, avoiding slow full AnnData loading.
2) Read selected metabolite score from cellTraitMatrices.
3) Filter cells/spots by target tissue.
4) Compare score distributions between groups from an obs column.
5) Export CNS-style violin/box/jitter plot as PDF with statistical annotations.
6) Export group summary and pairwise statistics as CSV.

Example:
python plot_target_metabolite_tissue_group_diff.py \
  --h5ad /path/to/input \
  --scheme_long Meta_SchemeC_AUROC_UpDownNet_outputs/Meta_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz \
  --score_dir Meta_SchemeC_AUROC_UpDownNet_outputs/cellTraitMatrices \
  --trait_meta /path/to/input \
  --outdir Diff_GCST90201371_SmallIntestine_group \
  --method wAUCell \
  --stage S3_geneSetZ_cellZ \
  --target_traits GCST90201371 \
  --target_tissue "Small Intestine Tissue" \
  --tissue_col Subregion \
  --group_col condition \
  --score_mode finalside \
  --groups Control,Treated \
  --reference_group Control
"""

import argparse
import itertools
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# -----------------------------
# Global plot style
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# Utilities
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


def parse_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            out.extend(parse_list(v))
        return list(dict.fromkeys(out))
    s = str(x).strip()
    if not s:
        return []
    for ch in ["[", "]", "(", ")", "{", "}", "\"", "'"]:
        s = s.replace(ch, "")
    return [v.strip() for v in re.split(r"[,;|]+", s) if v.strip()]


def parse_target_traits(x):
    vals = parse_list(x)
    out = []
    for v in vals:
        m = re.search(r"(GCST\d+)", v, flags=re.I)
        out.append(m.group(1).upper() if m else v)
    return list(dict.fromkeys(out))


def clean_trait_label(x):
    x = str(x)
    x = re.sub(r"\s+(levels?|ratio)\s*$", "", x, flags=re.I).strip()
    return x


def format_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.3g}"


def p_to_stars(p):
    if p is None or not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return q
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(y) - np.mean(x)) / np.sqrt(pooled)


def rank_biserial_from_u(u, n1, n2):
    if n1 <= 0 or n2 <= 0:
        return np.nan
    # Positive means second group tends to be larger than first group when u is computed for x vs y.
    auc = u / (n1 * n2)
    return 1 - 2 * auc


# -----------------------------
# h5ad fast obs reader
# -----------------------------
def _decode_h5ad_values(x):
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
    if h5py is None:
        raise ImportError("h5py is required for fast h5ad obs reading.")

    if isinstance(elem, h5py.Dataset):
        return _decode_h5ad_values(elem[()])

    if isinstance(elem, h5py.Group):
        if "codes" in elem and "categories" in elem:
            codes = np.asarray(elem["codes"][()])
            cats = _read_h5ad_obs_elem(elem["categories"])
            out = np.empty(codes.shape[0], dtype=object)
            valid = codes >= 0
            out[~valid] = np.nan
            if np.any(valid):
                out[valid] = np.asarray(cats, dtype=object)[codes[valid]]
            return out

        if "values" in elem and "mask" in elem:
            vals = _read_h5ad_obs_elem(elem["values"]).astype(object)
            mask = np.asarray(elem["mask"][()]).astype(bool)
            vals[mask] = np.nan
            return vals

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
        raise KeyError(f"Requested obs columns not found in h5ad: {missing}")
    return pd.DataFrame(data, index=pd.Index(idx.astype(str), name=idx_key))


def infer_tissue_col_from_h5ad(h5ad_path, stats_groups):
    preferred = [
        "Subregion", "Organ", "Organ_Full_Name", "organ_tissue",
        "celltype", "cell_type", "cell_ontology_class",
        "clusters", "sample", "batch"
    ]
    obs_cols = list_h5ad_obs_columns(h5ad_path)
    candidates = [c for c in preferred if c in obs_cols]
    if not candidates:
        raise KeyError(
            "Cannot infer tissue column. Please specify --tissue_col. "
            f"Available obs columns examples: {obs_cols[:50]}"
        )
    obs_small = read_h5ad_obs_columns_fast(h5ad_path, candidates)
    stats_set = set(map(str, pd.Series(stats_groups).dropna().unique()))
    best_col, best_overlap = None, -1
    for col in candidates:
        vals = set(map(str, pd.Series(obs_small[col]).dropna().unique()))
        overlap = len(stats_set.intersection(vals))
        if overlap > best_overlap:
            best_col, best_overlap = col, overlap
    if best_overlap <= 0:
        raise ValueError(
            "Could not infer tissue_col by overlap with scheme_long['celltype']. "
            f"Candidate columns tested: {candidates}. Please specify --tissue_col."
        )
    return best_col


# -----------------------------
# Trait metadata and scores
# -----------------------------
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
    return None


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
        raise KeyError(f"Selected traits are missing in {path.name}: {missing}")

    usecols = [idx_col] + keep_traits
    df = pd.read_csv(path, usecols=usecols, index_col=0)
    df.index = df.index.astype(str)
    df = df.reindex(pd.Index(obs_names.astype(str), name=df.index.name))
    return df


def build_score_for_trait(trait, tissue, score_mode, stats_sub, scores):
    trait = str(trait)
    tissue = str(tissue)
    mode = str(score_mode).lower()

    if mode in ["net", "signed_net"]:
        return pd.to_numeric(scores["net"][trait], errors="coerce")
    if mode == "up":
        return pd.to_numeric(scores["up"][trait], errors="coerce")
    if mode == "down":
        return pd.to_numeric(scores["down"][trait], errors="coerce")
    if mode == "finalside":
        sub = stats_sub.loc[
            (stats_sub["trait"].astype(str) == trait) &
            (stats_sub["celltype"].astype(str) == tissue)
        ]
        if sub.empty:
            warnings.warn(f"No exact trait-tissue row for finalside: {trait}, {tissue}. Fallback to net.")
            return pd.to_numeric(scores["net"][trait], errors="coerce")
        side = str(sub.iloc[0].get("final_side", "UP")).upper()
        if side.startswith("DOWN"):
            return -pd.to_numeric(scores["down"][trait], errors="coerce")
        return pd.to_numeric(scores["up"][trait], errors="coerce")
    raise ValueError("score_mode must be one of: net, signed_net, up, down, finalside")


# -----------------------------
# Statistics
# -----------------------------
def summarize_by_group(df, group_col, score_col="_score_"):
    rows = []
    for g, sub in df.groupby(group_col, sort=False):
        v = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({
            "group": g,
            "n": int(v.size),
            "mean": float(np.mean(v)) if v.size else np.nan,
            "median": float(np.median(v)) if v.size else np.nan,
            "sd": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
            "sem": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
            "q25": float(np.quantile(v, 0.25)) if v.size else np.nan,
            "q75": float(np.quantile(v, 0.75)) if v.size else np.nan,
        })
    return pd.DataFrame(rows)


def compute_global_tests(df, group_col, score_col="_score_"):
    data = []
    for _, sub in df.groupby(group_col, sort=False):
        v = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            data.append(v)

    rows = []
    if len(data) >= 2:
        try:
            kw = scipy_stats.kruskal(*data)
            rows.append({"test": "Kruskal-Wallis", "statistic": float(kw.statistic), "p": float(kw.pvalue)})
        except Exception as e:
            rows.append({"test": "Kruskal-Wallis", "statistic": np.nan, "p": np.nan, "error": str(e)})

        try:
            an = scipy_stats.f_oneway(*data)
            rows.append({"test": "One-way ANOVA", "statistic": float(an.statistic), "p": float(an.pvalue)})
        except Exception as e:
            rows.append({"test": "One-way ANOVA", "statistic": np.nan, "p": np.nan, "error": str(e)})

    return pd.DataFrame(rows)


def compute_pairwise_tests(df, group_col, score_col="_score_", reference_group=None, comparisons="reference"):
    groups = list(pd.Series(df[group_col]).dropna().astype(str).unique())
    group_to_vals = {}
    for g in groups:
        v = pd.to_numeric(df.loc[df[group_col].astype(str).eq(g), score_col], errors="coerce").to_numpy(dtype=float)
        group_to_vals[g] = v[np.isfinite(v)]

    if reference_group is not None:
        reference_group = str(reference_group)
        if reference_group not in group_to_vals:
            raise ValueError(f"--reference_group '{reference_group}' is not found in selected data groups: {groups}")

    pairs = []
    if str(comparisons).lower() == "all" or reference_group is None:
        pairs = list(itertools.combinations(groups, 2))
    else:
        pairs = [(reference_group, g) for g in groups if g != reference_group]

    rows = []
    for g1, g2 in pairs:
        x = group_to_vals[g1]
        y = group_to_vals[g2]
        row = {"group1": g1, "group2": g2, "n1": int(len(x)), "n2": int(len(y))}
        row["mean1"] = float(np.mean(x)) if len(x) else np.nan
        row["mean2"] = float(np.mean(y)) if len(y) else np.nan
        row["median1"] = float(np.median(x)) if len(x) else np.nan
        row["median2"] = float(np.median(y)) if len(y) else np.nan
        row["delta_mean_group2_minus_group1"] = row["mean2"] - row["mean1"] if np.isfinite(row["mean1"]) and np.isfinite(row["mean2"]) else np.nan
        row["delta_median_group2_minus_group1"] = row["median2"] - row["median1"] if np.isfinite(row["median1"]) and np.isfinite(row["median2"]) else np.nan

        if len(x) >= 2 and len(y) >= 2:
            try:
                tt = scipy_stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                row["welch_t"] = float(tt.statistic)
                row["welch_p"] = float(tt.pvalue)
            except Exception:
                row["welch_t"] = np.nan
                row["welch_p"] = np.nan

            try:
                mw = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
                row["mannwhitney_u"] = float(mw.statistic)
                row["mannwhitney_p"] = float(mw.pvalue)
                row["rank_biserial_group2_larger"] = float(rank_biserial_from_u(float(mw.statistic), len(x), len(y)))
            except Exception:
                row["mannwhitney_u"] = np.nan
                row["mannwhitney_p"] = np.nan
                row["rank_biserial_group2_larger"] = np.nan

            row["cohens_d_group2_minus_group1"] = float(cohens_d(x, y))
        else:
            row["welch_t"] = np.nan
            row["welch_p"] = np.nan
            row["mannwhitney_u"] = np.nan
            row["mannwhitney_p"] = np.nan
            row["rank_biserial_group2_larger"] = np.nan
            row["cohens_d_group2_minus_group1"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["mannwhitney_q"] = bh_fdr(out["mannwhitney_p"].to_numpy(dtype=float))
        out["welch_q"] = bh_fdr(out["welch_p"].to_numpy(dtype=float))
    return out


# -----------------------------
# Plotting
# -----------------------------
def make_group_palette(groups):
    base = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#8CD17D",
        "#B6992D", "#499894", "#D37295", "#B07AA1", "#59A14F"
    ]
    return {g: base[i % len(base)] for i, g in enumerate(groups)}


def add_pvalue_brackets(ax, positions, y_base, y_step, pairwise, group_order,
                        p_col="mannwhitney_q", max_brackets=6, fontsize=11):
    if pairwise is None or pairwise.empty:
        return
    pos_map = {g: i + 1 for i, g in enumerate(group_order)}
    shown = pairwise.copy()
    shown["_p_for_plot_"] = pd.to_numeric(shown[p_col], errors="coerce")
    shown = shown.sort_values("_p_for_plot_", ascending=True).head(max_brackets)

    y = y_base
    for _, r in shown.iterrows():
        g1, g2 = str(r["group1"]), str(r["group2"])
        if g1 not in pos_map or g2 not in pos_map:
            continue
        x1, x2 = pos_map[g1], pos_map[g2]
        if x1 == x2:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        p = r.get(p_col, np.nan)
        label = f"{p_to_stars(p)}\nq={format_p(p)}"
        ax.plot([x1, x1, x2, x2], [y, y + y_step * 0.22, y + y_step * 0.22, y], color="black", lw=0.8)
        ax.text((x1 + x2) / 2, y + y_step * 0.26, label, ha="center", va="bottom", fontsize=fontsize)
        y += y_step


def plot_group_diff_violin(
    plot_df,
    out_pdf,
    trait_display,
    trait_id,
    target_tissue,
    group_col,
    group_order,
    score_mode,
    global_tests,
    pairwise,
    reference_group=None,
    show_points=True,
    max_points_per_group=1800,
    fig_min_width=4.0,
    fig_width_per_group=0.55,
    fig_width_base=1.7,
    fig_height=4.6,
    title_fontsize=18,
    axis_label_fontsize=16,
    tick_fontsize=15,
    pvalue_fontsize=11,
    xtick_rotation=35,
):
    groups = list(group_order)
    palette = make_group_palette(groups)
    positions = np.arange(1, len(groups) + 1)
    rng = np.random.default_rng(0)

    # Compact panel: narrower width and slightly shorter height, while keeping labels large.
    fig_w = max(float(fig_min_width), float(fig_width_per_group) * len(groups) + float(fig_width_base))
    fig_h = float(fig_height)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data_list = []
    for g in groups:
        vals = pd.to_numeric(plot_df.loc[plot_df[group_col].astype(str).eq(str(g)), "_score_"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data_list.append(vals)

    for pos, g, vals in zip(positions, groups, data_list):
        if vals.size == 0:
            continue
        col = palette[g]
        vp = ax.violinplot([vals], positions=[pos], widths=0.68, showmeans=False, showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_edgecolor(col)
            body.set_linewidth(0.9)
            body.set_alpha(0.22)

        if show_points:
            vals_plot = vals
            if vals_plot.size > max_points_per_group:
                vals_plot = rng.choice(vals_plot, size=max_points_per_group, replace=False)
            xj = pos + rng.uniform(-0.14, 0.14, size=vals_plot.size)
            ax.scatter(xj, vals_plot, s=8, facecolors="none", edgecolors=col, linewidths=0.65, alpha=0.35, zorder=2)

        bp = ax.boxplot([vals], positions=[pos], widths=0.26, patch_artist=True, showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
            patch.set_alpha(0.85)
            patch.set_linewidth(1.0)
        for line in bp["whiskers"] + bp["caps"]:
            line.set_color(col)
            line.set_linewidth(1.0)
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.6)

    score_label = {
        "up": "wAUC score (up genes)",
        "down": "wAUC score (down genes)",
        "net": "wAUC score (net)",
        "signed_net": "wAUC score (signed net)",
        "finalside": "wAUC score (selected side)",
    }.get(str(score_mode).lower(), f"{score_mode} score")

    # Global test label
    global_label = ""
    if global_tests is not None and not global_tests.empty:
        kw = global_tests.loc[global_tests["test"].eq("Kruskal-Wallis")]
        if not kw.empty:
            global_label = f"Kruskal-Wallis p={format_p(float(kw.iloc[0]['p']))}"

    title = f"{trait_display}\n{target_tissue} | {group_col}"
    if global_label:
        title += f" | {global_label}"

    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.set_ylabel(score_label, fontsize=axis_label_fontsize)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=xtick_rotation, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize, width=1.0, length=4)
    ax.tick_params(axis="x", labelsize=tick_fontsize, width=1.0, length=4)
    ax.margins(x=0.03)
    ax.axhline(0, color="#1f4e79", linewidth=1.5, linestyle=(0, (3, 3)), alpha=0.8, zorder=1)

    for sp in ["left", "bottom", "top", "right"]:
        ax.spines[sp].set_linewidth(1.0)

    all_vals = np.concatenate([v for v in data_list if v.size > 0]) if any(v.size > 0 for v in data_list) else np.array([])
    if all_vals.size > 0:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        yr = ymax - ymin if ymax > ymin else 1.0
        y_base = ymax + yr * 0.055
        y_step = yr * 0.085
        max_br = min(6, max(1, len(groups) - 1 if reference_group else 6))
        add_pvalue_brackets(
            ax=ax,
            positions=positions,
            y_base=y_base,
            y_step=y_step,
            pairwise=pairwise,
            group_order=groups,
            p_col="mannwhitney_q",
            max_brackets=max_br,
            fontsize=pvalue_fontsize,
        )
        ax.set_ylim(ymin - yr * 0.06, y_base + y_step * (max_br + 0.65))

    fig.tight_layout(pad=0.35)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Differential analysis of a target metabolite score between groups within a specified tissue.")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--scheme_long", required=True)
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--trait_meta", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--method", default="wAUCell")
    ap.add_argument("--stage", default="S3_geneSetZ_cellZ")
    ap.add_argument("--prefix", default="Meta")
    ap.add_argument("--q_col", default=None)

    ap.add_argument("--target_traits", required=True, help="Comma-separated GCST trait IDs, e.g. GCST90201371")
    ap.add_argument("--target_tissue", required=True, help="Tissue/organ name matched to scheme_long['celltype'] and h5ad obs tissue_col.")
    ap.add_argument("--tissue_col", default=None, help="obs column containing tissue/organ annotation. Auto-inferred if omitted.")
    ap.add_argument("--group_col", required=True, help="obs column defining groups to compare, e.g. condition, genotype, treatment.")
    ap.add_argument("--groups", default=None, help="Optional comma-separated group order/subset, e.g. Control,Treated.")
    ap.add_argument("--reference_group", default=None, help="Optional reference group for pairwise comparisons.")
    ap.add_argument("--comparisons", default="reference", choices=["reference", "all"], help="Pairwise comparison mode.")
    ap.add_argument("--score_mode", default="finalside", choices=["net", "signed_net", "up", "down", "finalside"])
    ap.add_argument("--min_cells_per_group", type=int, default=10)
    ap.add_argument("--max_points_per_group", type=int, default=4000)
    ap.add_argument("--violin_max_points_per_group", type=int, default=1800)
    ap.add_argument("--no_points", action="store_true", help="Do not draw jittered points on violin plots.")

    # Compact figure layout controls
    ap.add_argument("--fig_min_width", type=float, default=4.0,
                    help="Minimum figure width in inches. Default: 4.0 for compact plots.")
    ap.add_argument("--fig_width_per_group", type=float, default=0.55,
                    help="Additional figure width per group in inches. Default: 0.55.")
    ap.add_argument("--fig_width_base", type=float, default=1.7,
                    help="Base figure width in inches before adding group-dependent width. Default: 1.7.")
    ap.add_argument("--fig_height", type=float, default=4.6,
                    help="Figure height in inches. Default: 4.6.")
    ap.add_argument("--title_fontsize", type=float, default=18,
                    help="Title font size. Default: 18.")
    ap.add_argument("--axis_label_fontsize", type=float, default=16,
                    help="X/Y axis label font size. Default: 16.")
    ap.add_argument("--tick_fontsize", type=float, default=15,
                    help="Tick label font size. Default: 15.")
    ap.add_argument("--pvalue_fontsize", type=float, default=11,
                    help="P-value annotation font size. Default: 11.")
    ap.add_argument("--xtick_rotation", type=float, default=35,
                    help="X tick label rotation angle. Default: 35.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    mkdir(outdir)
    plot_dir = outdir / "plots"
    table_dir = outdir / "tables"
    mkdir(plot_dir)
    mkdir(table_dir)

    target_traits = parse_target_traits(args.target_traits)
    if not target_traits:
        raise ValueError("--target_traits is empty after parsing.")

    print("[1/5] Load scheme_long and trait metadata...")
    trait_meta = load_trait_meta(args.trait_meta)
    stats = pd.read_csv(args.scheme_long)
    required = ["celltype", "trait", "final_effect"]
    missing = [c for c in required if c not in stats.columns]
    if missing:
        raise KeyError(f"Missing required columns in scheme_long: {missing}")

    stats["trait"] = stats["trait"].astype(str)
    stats["celltype"] = stats["celltype"].astype(str)
    if "method" in stats.columns:
        stats = stats.loc[stats["method"].astype(str).eq(str(args.method))].copy()
    if "stage" in stats.columns:
        stats = stats.loc[stats["stage"].astype(str).eq(str(args.stage))].copy()
    if stats.empty:
        raise ValueError(f"No rows left after filtering method={args.method}, stage={args.stage}")

    q_col = choose_q_col(stats, args.q_col)
    if q_col is not None:
        stats["q_value_used"] = pd.to_numeric(stats[q_col], errors="coerce")
    else:
        stats["q_value_used"] = np.nan
    stats = add_trait_meta(stats, trait_meta)

    found_traits = sorted(set(target_traits).intersection(set(stats["trait"].astype(str).unique())))
    missing_traits = sorted(set(target_traits) - set(found_traits))
    if missing_traits:
        print(f"[warn] target traits not found after method/stage filtering: {missing_traits}")
    if not found_traits:
        examples = sorted(stats["trait"].astype(str).drop_duplicates().head(20).tolist())
        raise ValueError(f"No target traits found. Examples in current table: {examples}")

    print("[2/5] Read h5ad obs columns fast...")
    if args.tissue_col is None:
        tissue_col = infer_tissue_col_from_h5ad(args.h5ad, stats["celltype"])
    else:
        tissue_col = args.tissue_col

    obs = read_h5ad_obs_columns_fast(args.h5ad, [tissue_col, args.group_col])
    obs.index = obs.index.astype(str)
    print(f"[obs] loaded obs shape: {obs.shape}; tissue_col={tissue_col}; group_col={args.group_col}")

    print("[3/5] Read score matrices...")
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
        scores[kind] = read_score_subset(p, obs.index.astype(str), found_traits)

    print("[4/5] Differential analysis and plotting...")
    all_group_summaries = []
    all_pairwise = []
    all_global = []
    plot_index = []

    for trait in found_traits:
        trait_rows = stats.loc[stats["trait"].astype(str).eq(str(trait))].copy()
        exact = trait_rows.loc[trait_rows["celltype"].astype(str).eq(str(args.target_tissue))].copy()
        if exact.empty:
            print(f"[warn] No exact target_tissue row for trait={trait}, tissue={args.target_tissue}; still plotting obs-filtered tissue if present.")
            stats_for_score = trait_rows
        else:
            stats_for_score = exact

        trait_display = str(trait_rows["trait_display"].dropna().iloc[0]) if "trait_display" in trait_rows.columns and not trait_rows["trait_display"].dropna().empty else trait
        trait_type = str(trait_rows["trait_type"].dropna().iloc[0]) if "trait_type" in trait_rows.columns and not trait_rows["trait_type"].dropna().empty else "unknown"

        score = build_score_for_trait(
            trait=trait,
            tissue=args.target_tissue,
            score_mode=args.score_mode,
            stats_sub=stats,
            scores=scores,
        )

        df = obs.copy()
        df["_score_"] = pd.to_numeric(pd.Series(score, index=df.index), errors="coerce")
        df[tissue_col] = df[tissue_col].astype(str)
        df[args.group_col] = df[args.group_col].astype(str)

        df = df.loc[df[tissue_col].eq(str(args.target_tissue))].copy()
        df = df.loc[np.isfinite(df["_score_"])].copy()
        if df.empty:
            print(f"[warn] No cells/spots left for trait={trait}, tissue={args.target_tissue}.")
            continue

        if args.groups:
            group_order = parse_list(args.groups)
            df = df.loc[df[args.group_col].isin(group_order)].copy()
        else:
            group_order = df[args.group_col].value_counts().index.astype(str).tolist()

        # Drop groups below min cell threshold.
        counts = df[args.group_col].value_counts()
        group_order = [g for g in group_order if counts.get(g, 0) >= args.min_cells_per_group]
        df = df.loc[df[args.group_col].isin(group_order)].copy()

        if len(group_order) < 2:
            print(f"[warn] Fewer than 2 groups with >= {args.min_cells_per_group} cells for trait={trait}. Skipping.")
            continue

        # Optional downsampling for tables/plot memory while preserving group comparison.
        if args.max_points_per_group and args.max_points_per_group > 0:
            tmp = []
            rng = np.random.default_rng(0)
            for g in group_order:
                sub = df.loc[df[args.group_col].eq(g)]
                if sub.shape[0] > args.max_points_per_group:
                    sub = sub.sample(n=args.max_points_per_group, random_state=0)
                tmp.append(sub)
            df_plot = pd.concat(tmp, axis=0)
        else:
            df_plot = df

        summary = summarize_by_group(df_plot, args.group_col)
        summary.insert(0, "trait", trait)
        summary.insert(1, "trait_display", trait_display)
        summary.insert(2, "trait_type", trait_type)
        summary.insert(3, "target_tissue", args.target_tissue)
        summary.insert(4, "score_mode", args.score_mode)
        all_group_summaries.append(summary)

        global_tests = compute_global_tests(df_plot, args.group_col)
        if not global_tests.empty:
            global_tests.insert(0, "trait", trait)
            global_tests.insert(1, "trait_display", trait_display)
            global_tests.insert(2, "target_tissue", args.target_tissue)
            global_tests.insert(3, "score_mode", args.score_mode)
            all_global.append(global_tests)

        pairwise = compute_pairwise_tests(
            df_plot,
            group_col=args.group_col,
            reference_group=args.reference_group,
            comparisons=args.comparisons,
        )
        if not pairwise.empty:
            pairwise.insert(0, "trait", trait)
            pairwise.insert(1, "trait_display", trait_display)
            pairwise.insert(2, "target_tissue", args.target_tissue)
            pairwise.insert(3, "score_mode", args.score_mode)
            all_pairwise.append(pairwise)

        out_pdf = plot_dir / (
            f"{safe_name(args.target_tissue)}__{safe_name(trait_display)}__"
            f"{safe_name(trait)}__{safe_name(args.group_col)}__{safe_name(args.score_mode)}__diff_violin.pdf"
        )
        plot_group_diff_violin(
            plot_df=df_plot,
            out_pdf=out_pdf,
            trait_display=trait_display,
            trait_id=trait,
            target_tissue=args.target_tissue,
            group_col=args.group_col,
            group_order=group_order,
            score_mode=args.score_mode,
            global_tests=global_tests,
            pairwise=pairwise,
            reference_group=args.reference_group,
            show_points=(not args.no_points),
            max_points_per_group=args.violin_max_points_per_group,
            fig_min_width=args.fig_min_width,
            fig_width_per_group=args.fig_width_per_group,
            fig_width_base=args.fig_width_base,
            fig_height=args.fig_height,
            title_fontsize=args.title_fontsize,
            axis_label_fontsize=args.axis_label_fontsize,
            tick_fontsize=args.tick_fontsize,
            pvalue_fontsize=args.pvalue_fontsize,
            xtick_rotation=args.xtick_rotation,
        )

        # Save per-trait raw values for reproducibility.
        values_path = table_dir / (
            f"{safe_name(args.target_tissue)}__{safe_name(trait_display)}__"
            f"{safe_name(trait)}__{safe_name(args.group_col)}__values.csv.gz"
        )
        df_plot[[tissue_col, args.group_col, "_score_"]].to_csv(values_path, index=True, compression="gzip")

        plot_index.append({
            "trait": trait,
            "trait_display": trait_display,
            "trait_type": trait_type,
            "target_tissue": args.target_tissue,
            "group_col": args.group_col,
            "score_mode": args.score_mode,
            "n_cells_or_spots": int(df_plot.shape[0]),
            "n_groups": int(len(group_order)),
            "groups": ",".join(map(str, group_order)),
            "plot_pdf": str(out_pdf),
            "values_csv": str(values_path),
        })

    print("[5/5] Save summary tables...")
    if all_group_summaries:
        pd.concat(all_group_summaries, ignore_index=True).to_csv(table_dir / "group_summary.csv", index=False)
    if all_global:
        pd.concat(all_global, ignore_index=True).to_csv(table_dir / "global_tests.csv", index=False)
    if all_pairwise:
        pd.concat(all_pairwise, ignore_index=True).to_csv(table_dir / "pairwise_tests.csv", index=False)
    pd.DataFrame(plot_index).to_csv(outdir / "diff_violin_plot_index.csv", index=False)

    print("[done] outputs saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
