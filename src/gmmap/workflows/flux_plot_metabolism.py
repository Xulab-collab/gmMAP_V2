#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_metabolic_flow_round13_CNS_style.py

CNS-style visualization for outputs from
metabolic_flow_model_wAUCellS3_ratio_enzyme_constrained_round13(_mt).py

Expected files in --input-dir (supports .csv.gz or .csv):
Subpathway level:
    subpathway_activation_matrix
    subpathway_direction_matrix
    subpathway_confidence_matrix
    subpathway_signed_flux_matrix
    subpathway_ratio_support_matrix
    subpathway_enzyme_capacity_matrix
    subpathway_compartment_support_matrix
    subpathway_cofactor_support_matrix
    subpathway_feasibility_matrix
    subpathway_mean_s3_matrix
    subpathway_delta_s3_matrix

Module level:
    module_activation_matrix
    module_direction_matrix
    module_confidence_matrix
    module_signed_flux_matrix
    module_ratio_support_matrix
    module_enzyme_capacity_matrix
    module_compartment_support_matrix
    module_cofactor_support_matrix
    module_feasibility_matrix
    module_mean_s3_matrix
    module_delta_s3_matrix
    module_annotation_used

Outputs:
    Fig1A_subpathway_flux_heatmap.pdf/png
    Fig1B_subpathway_bubble.pdf/png
    Fig2_celltype_top_subpathways.pdf/png
    Fig3_activation_direction_phaseplot.pdf/png
    Fig4_constraint_landscape.pdf/png
    Fig5_selected_subpathway_module_decomposition_<NAME>.pdf/png
    Fig6_flux_vs_constraints_<CELLTYPE>.pdf/png
    plotting_summary.txt

Usage:
python plot_metabolic_flow_round13_CNS_style.py \
  --input-dir metabolic_flow_round13_out \
  --outdir metabolic_flow_round13_figures

Optional:
  --selected-celltypes "Fibroblast,Macrophage,T cell"
  --selected-subpathways "Tryptophan/Kynurenine Branch,Carnitine Pool"
  --topk 8
  --module-topn 40
  --cluster-rows
  --cluster-cols
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["savefig.bbox"] = "tight"


# ----------------------------
# helpers
# ----------------------------

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def parse_comma_list(x):
    if x is None or str(x).strip() == "":
        return None
    # Support comma, semicolon and newline separated values.
    parts = re.split(r"[,;\n\r]+", str(x))
    return [i.strip() for i in parts if i.strip()]


def parse_name_file(path):
    """Read one name per line, or comma/semicolon-separated names from a text file."""
    if path is None or str(path).strip() == "":
        return None
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return parse_comma_list(txt)


def norm_name(x):
    """Normalize names for forgiving matching."""
    s = str(x).strip().lower()
    s = s.replace("；", ";").replace("，", ",")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return s


def celltype_prefix(x):
    """
    Return the abbreviation before ':'.
    Example: 'ErPrT: early proximal tubule' -> 'erprt'
    """
    return norm_name(str(x).split(":")[0])


def resolve_celltypes(requested, available, mode="auto", strict=False):
    """
    Resolve user-provided cell type names to exact column names.

    mode:
      exact    : exact name after whitespace normalization
      prefix   : match abbreviation before ':'; e.g. ErPrT, DTLH, CnT
      contains : substring match
      regex    : regular-expression search
      auto     : exact -> prefix -> contains, in this order
    """
    available = list(available)
    if requested is None or len(requested) == 0:
        return None, [], []

    resolved = []
    unmatched = []
    ambiguous = []

    norm_to_avail = {}
    for a in available:
        norm_to_avail.setdefault(norm_name(a), []).append(a)

    prefix_to_avail = {}
    for a in available:
        prefix_to_avail.setdefault(celltype_prefix(a), []).append(a)

    def add_unique(matches):
        for m in matches:
            if m not in resolved:
                resolved.append(m)

    for raw in requested:
        q = str(raw).strip()
        qn = norm_name(q)
        matches = []

        if mode in ["exact", "auto"]:
            matches = norm_to_avail.get(qn, [])

        if len(matches) == 0 and mode in ["prefix", "auto"]:
            # Allow 'ErPrT' to match 'ErPrT: early proximal tubule'
            matches = prefix_to_avail.get(qn, [])
            # Also allow exact visible prefix with punctuation removed
            if len(matches) == 0:
                matches = [a for a in available if celltype_prefix(a) == qn]

        if len(matches) == 0 and mode in ["contains", "auto"]:
            matches = [a for a in available if qn in norm_name(a)]

        if len(matches) == 0 and mode == "regex":
            try:
                pat = re.compile(q, flags=re.IGNORECASE)
                matches = [a for a in available if pat.search(a)]
            except re.error as e:
                raise ValueError(f"Invalid regex for --selected-celltypes: {q!r}. Regex error: {e}")

        if len(matches) == 0:
            unmatched.append(q)
        elif len(matches) > 1:
            ambiguous.append((q, matches))
            # Keep all matches in non-strict mode; this is useful for terms like 'NPC'.
            if not strict:
                add_unique(matches)
        else:
            add_unique(matches)

    if strict and (unmatched or ambiguous):
        msg = []
        if unmatched:
            msg.append("Unmatched cell types: " + ", ".join(unmatched))
        if ambiguous:
            msg.append("Ambiguous cell types:\n" + "\n".join([f"  {q}: {ms}" for q, ms in ambiguous]))
        msg.append("Available cell types:\n  " + "\n  ".join(available))
        raise ValueError("\n".join(msg))

    return resolved, unmatched, ambiguous


def sanitize_filename(s):
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:120]

def find_file(input_dir, stem):
    for ext in [".csv.gz", ".csv"]:
        p = os.path.join(input_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def read_matrix(input_dir, stem):
    p = find_file(input_dir, stem)
    if p is None:
        existing = sorted(os.listdir(input_dir)) if os.path.isdir(input_dir) else []
        raise FileNotFoundError(
            f"Missing file for stem '{stem}' in {input_dir}\n"
            f"Expected: {stem}.csv.gz or {stem}.csv\n"
            f"Existing files:\n - " + "\n - ".join(existing[:200])
        )
    return pd.read_csv(p, index_col=0), p

def read_optional_table(input_dir, stem):
    p = find_file(input_dir, stem)
    if p is None:
        return None, None
    return pd.read_csv(p), p

def filter_df(df, rows=None, cols=None):
    out = df.copy()
    if rows is not None:
        keep_rows = [r for r in rows if r in out.index]
        out = out.loc[keep_rows]
    if cols is not None:
        keep_cols = [c for c in cols if c in out.columns]
        out = out[keep_cols]
    return out

def robust_vmax(df, q=0.98, floor=1e-6):
    vals = df.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    vmax = float(np.quantile(np.abs(vals), q))
    if (not np.isfinite(vmax)) or (vmax <= 0):
        vmax = float(np.max(np.abs(vals))) if vals.size else 1.0
    return max(vmax, floor)

def robust_nonneg_vmax(df, q=0.98, floor=1e-6):
    vals = df.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    if vals.size == 0:
        return 1.0
    vmax = float(np.quantile(vals, q))
    if (not np.isfinite(vmax)) or (vmax <= 0):
        vmax = float(np.max(vals)) if vals.size else 1.0
    return max(vmax, floor)

def maybe_cluster(df, cluster_rows=False, cluster_cols=False):
    out = df.copy()
    if HAVE_SCIPY and cluster_rows and out.shape[0] > 2:
        row_data = out.fillna(0).to_numpy(dtype=float)
        try:
            row_order = leaves_list(linkage(pdist(row_data), method="average"))
            out = out.iloc[row_order, :]
        except Exception:
            pass
    if HAVE_SCIPY and cluster_cols and out.shape[1] > 2:
        col_data = out.fillna(0).T.to_numpy(dtype=float)
        try:
            col_order = leaves_list(linkage(pdist(col_data), method="average"))
            out = out.iloc[:, col_order]
        except Exception:
            pass
    return out

def save_fig(fig, out_pdf, out_png):
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def choose_top_subpathways(flux_df, topn=30):
    if flux_df.empty:
        return []
    rank = flux_df.abs().mean(axis=1).sort_values(ascending=False)
    return rank.head(min(topn, len(rank))).index.tolist()


def clean_subpathway_label(label, max_chars=34):
    """
    Make a compact label for phase-plot annotations.

    Many subpathway names contain bilingual text separated by '|'.
    For dense phase plots, the English part is usually more readable.
    """
    s = pretty_subpathway_label(label)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Branch", "Br.")
    s = s.replace("Related", "Rel.")
    s = s.replace("Intermediates", "Interm.")
    s = s.replace("Acylcarnitines", "AcylCar.")
    s = s.replace("Containing", "Cont.")
    s = s.replace("Dicarboxylic", "Dicarb.")
    s = s.replace("Glutamate/Glutamine", "Glu/Gln")
    s = s.replace("Oxaloacetate/Aspartate", "OAA/Asp")
    s = s.replace("Betaine-Sarcosine-Dimethylglycine", "Betaine/Sar/DMG")
    s = s.replace("Methionine Cycle/Sulfur-Cont.", "Met/Sulfur")
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars - 1].rstrip() + "…"
    return s


def pretty_subpathway_label(label):
    """
    Clean pathway labels for display in figures.
    Examples:
      "Polyol/Uronic Acid Branch | /" -> "Polyol/Uronic Acid Branch"
      "BCAA α-Keto Acid Branch | BCAA α-" -> "BCAA α-Keto Acid Branch"
    """
    s = str(label)
    s = s.split("|")[0].strip()
    s = re.sub(r"[\/|]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def pretty_subpathway_labels(labels):
    return [pretty_subpathway_label(x) for x in labels]


def select_top_bottom_labels(flux_s, topn=5):
    """
    Select top positive and bottom negative subpathways by signed flux.

    Returns labels from the original index. If a cell type has fewer positive
    or negative points, it returns as many as available.
    """
    s = pd.to_numeric(flux_s, errors="coerce").dropna()
    if s.empty or topn <= 0:
        return []
    pos = s[s > 0].sort_values(ascending=False).head(topn)
    neg = s[s < 0].sort_values(ascending=True).head(topn)
    keep = list(pos.index) + list(neg.index)
    if len(keep) == 0:
        keep = s.reindex(s.abs().sort_values(ascending=False).head(topn * 2).index).index.tolist()
    return keep


def annotate_phase_points(ax, x, y, c, labels=None, topn=5, max_chars=34,
                          fontsize=6.0, draw_arrows=True):
    """
    Add labels to the phase plot for the top and bottom signed-flux subpathways.
    """
    keep = select_top_bottom_labels(c, topn=topn)
    if len(keep) == 0:
        return

    ax.margins(x=0.12, y=0.12)

    for k, name in enumerate(keep):
        if name not in x.index or name not in y.index:
            continue
        xv, yv = x.loc[name], y.loc[name]
        if pd.isna(xv) or pd.isna(yv):
            continue
        sign = 1 if float(c.loc[name]) >= 0 else -1
        dx = 6 + 2 * (k % 3)
        dy = sign * (6 + 3 * (k % 4))
        if k % 2 == 1:
            dx = -dx
        ha = "left" if dx > 0 else "right"
        arrowprops = dict(arrowstyle="-", lw=0.35, color="0.35", alpha=0.8) if draw_arrows else None
        ax.annotate(
            clean_subpathway_label(name, max_chars=max_chars),
            xy=(float(xv), float(yv)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=fontsize,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.75", lw=0.25, alpha=0.72),
            arrowprops=arrowprops,
            zorder=5,
            clip_on=False,
        )


# ----------------------------
# FAO pathway filtering
# ----------------------------

FAO_CORE_PATTERNS = [
    # Most directly related to fatty-acid transport and beta-oxidation intermediates
    "carnitine pool",
    "long-chain acylcarnitines",
    "medium-chain acylcarnitines",
    "short-chain acylcarnitines",
    "dicarboxylic acid/hydroxyacylcarnitine",
    "dicarboxylic fatty acid",
    "free fatty acids",
    "fatty acid derivatives",
    "ketone body",
]

GLYCOLYSIS_PATTERNS = [
    # Up to 3 key glycolysis-associated branches
    "hexose/disaccharide",
    "glycerate/triose",
    "lactate/pyruvate related",
]

FAO_SUPPORT_PATTERNS = [
    # Not FAO itself, but downstream mitochondrial oxidative support
    "citrate/aconitate",
    "α-ketoglutarate",
    "alpha-ketoglutarate",
    "succinyl/succinate",
    "succinate",
    "malate branch",
    "fumarate/maleate",
    "oxaloacetate/aspartate",
    "reductive tca",
    "pyruvate entry",
]

FAO_LIPID_REMODELING_PATTERNS = [
    # Lipid remodelling/signalling; keep optional because these are not direct FAO evidence
    "lysophospholipids",
    "lipid mediator",
    "ethanolamide",
    "ceramide",
    "sphingosine",
    "sphingomyelin",
    "glycosylceramide",
    "head group precursors",
]


def match_patterns(label, patterns):
    s = pretty_subpathway_label(label).lower()
    s = s.replace("α", "alpha")
    return any(p.lower().replace("α", "alpha") in s for p in patterns)


def select_pathways_by_preset(index, preset="all", extra_patterns=None):
    """
    Select subpathway names by biological preset.

    preset:
      all            : no pathway filtering
      fao_core       : FAO/carnitine/acylcarnitine/fatty-acid branches only
      fao_support    : FAO core + TCA/pyruvate/mitochondrial oxidative support
                         + up to 3 glycolysis-associated branches
      fao_extended   : FAO support + lipid-remodelling branches
      custom_keyword : use --pathway-keywords only
    """
    index = list(index)
    preset = str(preset).strip().lower()
    if preset in ["", "all", "none"]:
        return None

    patterns = []
    if preset == "fao_core":
        patterns = FAO_CORE_PATTERNS
    elif preset == "fao_support":
        patterns = FAO_CORE_PATTERNS + FAO_SUPPORT_PATTERNS + GLYCOLYSIS_PATTERNS
    elif preset == "fao_extended":
        patterns = FAO_CORE_PATTERNS + FAO_SUPPORT_PATTERNS + GLYCOLYSIS_PATTERNS + FAO_LIPID_REMODELING_PATTERNS
    elif preset == "custom_keyword":
        patterns = []
    else:
        raise ValueError(
            f"Unknown --pathway-preset: {preset}. "
            "Choose from: all, fao_core, fao_support, fao_extended, custom_keyword."
        )

    if extra_patterns:
        patterns = patterns + list(extra_patterns)

    selected = [x for x in index if match_patterns(x, patterns)]
    return selected


# ----------------------------
# plotting functions
# ----------------------------

def plot_subpathway_flux_heatmap(
    flux_df, outdir, cluster_rows=False, cluster_cols=False,
    row_height=0.20, col_width=0.28,
    min_height=4.5, min_width=5.8,
    extra_height=1.0, extra_width=1.2,
    title_fontsize=11.0, axis_label_fontsize=9.5,
    xtick_fontsize=8.0, ytick_fontsize=7.2,
    cbar_label_fontsize=9.0, cbar_tick_fontsize=8.0
):
    """
    Fig1A: compact heatmap for subpathway-level signed flow potential.

    Figure size is automatically derived from the number of rows/columns and
    font sizes, but kept intentionally compact.
    """
    df = maybe_cluster(flux_df, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    df = df.copy()
    df.index = pretty_subpathway_labels(df.index)
    vmax = robust_vmax(df)

    # Compact size, lightly adjusted by font size.
    h = max(min_height, row_height * df.shape[0] + extra_height + 0.05 * ytick_fontsize)
    w = max(min_width, col_width * df.shape[1] + extra_width + 0.06 * xtick_fontsize)

    fig, ax = plt.subplots(figsize=(w, h))
    if HAVE_SNS:
        hm = sns.heatmap(
            df, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.12, linecolor="#F2F2F2",
            cbar_kws={"label": "Signed flow potential"},
            ax=ax
        )
        # Format colorbar
        if hm.collections:
            cbar = hm.collections[0].colorbar
            if cbar is not None:
                cbar.set_label("Signed flow potential", fontsize=cbar_label_fontsize)
                cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    else:
        im = ax.imshow(df.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=xtick_fontsize)
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index, fontsize=ytick_fontsize)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Signed flow potential", fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_tick_fontsize)

    ax.set_title("Subpathway-level signed metabolic flow potential", fontsize=title_fontsize, pad=6)
    ax.set_xlabel("Cell type", fontsize=axis_label_fontsize)
    ax.set_ylabel("Subpathway", fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)

    save_fig(fig,
             os.path.join(outdir, "Fig1A_subpathway_flux_heatmap.pdf"),
             os.path.join(outdir, "Fig1A_subpathway_flux_heatmap.png"))


def plot_subpathway_bubble(flux_df, conf_df, feas_df, outdir):
    rows = flux_df.index.tolist()
    row_labels = pretty_subpathway_labels(rows)
    cols = flux_df.columns.tolist()
    vmax = robust_vmax(flux_df)
    X, Y, C, S, E = [], [], [], [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = flux_df.loc[r, c]
            conf = conf_df.loc[r, c] if (r in conf_df.index and c in conf_df.columns) else np.nan
            feas = feas_df.loc[r, c] if (r in feas_df.index and c in feas_df.columns) else np.nan
            if pd.isna(val):
                continue
            X.append(j); Y.append(i); C.append(val)
            conf_num = 0.0 if pd.isna(conf) else float(np.clip(conf, 0, 1))
            feas_num = 0.0 if pd.isna(feas) else float(np.clip(feas, 0, 1))
            S.append(18 + 180 * conf_num)
            E.append(feas_num)
    h = max(5.5, 0.28 * len(rows) + 1.4)
    w = max(7.0, 0.35 * len(cols) + 2.3)
    fig, ax = plt.subplots(figsize=(w, h))
    sc = ax.scatter(X, Y, c=C, s=S, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    edgecolors="black", linewidths=0.25, alpha=0.95)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Subpathway")
    ax.set_title("Subpathway flow landscape: color = signed flow, size = confidence", fontsize=12, pad=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Signed flow potential")
    # size legend
    for lab, size in [(0.2, 18 + 180 * 0.2), (0.5, 18 + 180 * 0.5), (0.9, 18 + 180 * 0.9)]:
        ax.scatter([], [], s=size, c="white", edgecolors="black", linewidths=0.4, label=f"{lab:.1f}")
    ax.legend(title="Confidence", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig,
             os.path.join(outdir, "Fig1B_subpathway_bubble.pdf"),
             os.path.join(outdir, "Fig1B_subpathway_bubble.png"))

def plot_top_subpathways(
    flux_df, outdir, topk=8,
    panel_width=4.4, panel_height=3.8,
    ytick_fontsize=7.5, title_fontsize=10.5,
    axis_label_fontsize=9.0, xtick_fontsize=8.0,
    suptitle_fontsize=12.0, x_nbins=4,
    x_pad_frac=0.04
):
    """
    Fig2: compact bar plots of top activated/suppressed subpathways by cell type.

    Updated in this version:
    - all panels use the same x-axis range
    - the global x-axis range is determined from the overall minimum and
      maximum values among the displayed top positive and top negative
      subpathways across all selected cell types
    """
    ncols = min(3, flux_df.shape[1])
    nrows = int(np.ceil(flux_df.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_width*ncols, panel_height*nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    # Precompute the plotted values for each cell type and the global x-axis range.
    plot_data = {}
    global_vals = []
    for ct in flux_df.columns:
        s = flux_df[ct].dropna().sort_values()
        neg = s.head(topk)
        pos = s.tail(topk)
        plot_s = pd.concat([neg, pos]).drop_duplicates()
        plot_data[ct] = plot_s
        if len(plot_s) > 0:
            global_vals.extend(plot_s.values.tolist())

    if len(global_vals) == 0:
        global_min, global_max = -1.0, 1.0
    else:
        global_min = float(np.nanmin(global_vals))
        global_max = float(np.nanmax(global_vals))
        if not np.isfinite(global_min):
            global_min = -1.0
        if not np.isfinite(global_max):
            global_max = 1.0
        if global_min == global_max:
            delta = max(abs(global_min), 1e-3) * 0.05 + 1e-3
            global_min -= delta
            global_max += delta

    # Keep zero included and add small padding. Use the same range for all panels.
    x_left = min(global_min, 0.0)
    x_right = max(global_max, 0.0)
    span = x_right - x_left
    pad = span * x_pad_frac if span > 0 else 0.01
    x_left -= pad
    x_right += pad

    for idx, ct in enumerate(flux_df.columns):
        ax = axes.ravel()[idx]
        ax.axis("on")
        plot_s = plot_data[ct]
        colors = ["#3B6FB6" if v < 0 else "#C84E4E" for v in plot_s.values]

        ax.barh(np.arange(len(plot_s)), plot_s.values, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(np.arange(len(plot_s)))
        ax.set_yticklabels(pretty_subpathway_labels(plot_s.index), fontsize=ytick_fontsize)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(ct, fontsize=title_fontsize)
        ax.set_xlabel("Signed flow potential", fontsize=axis_label_fontsize)

        # Shared x-axis range across all panels
        ax.set_xlim(x_left, x_right)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_nbins))
        ax.tick_params(axis="x", labelsize=xtick_fontsize)

    fig.suptitle("Top activated and suppressed subpathways by cell type", fontsize=suptitle_fontsize, y=1.01)
    save_fig(fig,
             os.path.join(outdir, "Fig2_celltype_top_subpathways.pdf"),
             os.path.join(outdir, "Fig2_celltype_top_subpathways.png"))


def plot_phaseplot(act_df, dir_df, flux_df, conf_df, outdir,
                   label_topn=5, label_max_chars=34, label_fontsize=6.0,
                   draw_label_arrows=True):
    """
    Activation-direction phase plot.

    New in this version:
    - For each cell type, annotate the top N positive and bottom N negative
      subpathways/metabolic branches ranked by signed flow potential.
    """
    ncols = min(3, act_df.shape[1])
    nrows = int(np.ceil(act_df.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.1*ncols, 5.1*nrows), squeeze=False)
    vmax = robust_vmax(flux_df)
    sc = None
    for ax in axes.ravel():
        ax.axis("off")
    common_idx = act_df.index.intersection(dir_df.index).intersection(flux_df.index).intersection(conf_df.index)
    for idx, ct in enumerate(act_df.columns):
        ax = axes.ravel()[idx]
        ax.axis("on")
        x = act_df.loc[common_idx, ct].astype(float)
        y = dir_df.loc[common_idx, ct].astype(float)
        c = flux_df.loc[common_idx, ct].astype(float)
        s = conf_df.loc[common_idx, ct].astype(float).fillna(0)
        size = 18 + 150 * s.clip(0, 1)
        sc = ax.scatter(x, y, c=c, s=size, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                        edgecolors="black", linewidths=0.25, alpha=0.9, zorder=2)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

        annotate_phase_points(
            ax=ax, x=x, y=y, c=c,
            topn=label_topn,
            max_chars=label_max_chars,
            fontsize=label_fontsize,
            draw_arrows=draw_label_arrows,
        )

        ax.set_title(ct, fontsize=11)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Direction")
    fig.suptitle(
        f"Activation–direction phase space of subpathways\n"
        f"Labels: top {label_topn} positive and bottom {label_topn} negative signed-flow branches per cell type",
        fontsize=13, y=1.01
    )
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Signed flow potential")
    save_fig(fig,
             os.path.join(outdir, "Fig3_activation_direction_phaseplot.pdf"),
             os.path.join(outdir, "Fig3_activation_direction_phaseplot.png"))

def plot_constraint_landscape(enzyme_df, comp_df, cof_df, flux_df, outdir, cluster_rows=False, cluster_cols=False, topn=30):
    selected = choose_top_subpathways(flux_df, topn=topn)
    enz = filter_df(enzyme_df, rows=selected)
    comp = filter_df(comp_df, rows=selected, cols=enz.columns.tolist())
    cof = filter_df(cof_df, rows=selected, cols=enz.columns.tolist())
    enz = maybe_cluster(enz, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    comp = comp.loc[enz.index, enz.columns]
    cof = cof.loc[enz.index, enz.columns]

    enz = enz.copy(); comp = comp.copy(); cof = cof.copy()
    clean_rows = pretty_subpathway_labels(enz.index)
    enz.index = clean_rows
    comp.index = clean_rows
    cof.index = clean_rows

    vmax = max(robust_nonneg_vmax(enz), robust_nonneg_vmax(comp), robust_nonneg_vmax(cof))
    h = max(6.0, 0.22 * enz.shape[0] + 1.2)
    w = max(12.5, 0.36 * enz.shape[1] * 3 + 1.8)
    fig, axes = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={"wspace": 0.18})
    mats = [(enz, "Enzyme capacity"), (comp, "Compartment support"), (cof, "Cofactor support")]
    for ax, (df, title) in zip(axes, mats):
        if HAVE_SNS:
            sns.heatmap(df, cmap="YlOrRd", vmin=0, vmax=vmax,
                        linewidths=0.15, linecolor="#F2F2F2",
                        cbar_kws={"label": title}, ax=ax)
        else:
            im = ax.imshow(df.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(df.columns, rotation=90)
            ax.set_yticks(np.arange(df.shape[0]))
            ax.set_yticklabels(df.index)
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(title)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Cell type")
        ax.set_ylabel("Subpathway" if title == "Enzyme capacity" else "")
        ax.tick_params(axis="y", labelsize=8)
    save_fig(fig,
             os.path.join(outdir, "Fig4_constraint_landscape.pdf"),
             os.path.join(outdir, "Fig4_constraint_landscape.png"))

def plot_module_decomposition(module_flux_df, module_conf_df, ann, selected_subpathways, outdir, module_topn=40):
    if ann is None or ann.empty or "module_id" not in ann.columns:
        return []
    sp_col = "subpathway_en" if "subpathway_en" in ann.columns else None
    if sp_col is None and "subpathway_cn" in ann.columns:
        sp_col = "subpathway_cn"
    if sp_col is None:
        return []

    made = []
    for sp in selected_subpathways:
        mods = ann.loc[ann[sp_col].astype(str) == str(sp), "module_id"].astype(str).tolist()
        mods = [m for m in mods if m in module_flux_df.index]
        if len(mods) == 0:
            continue
        df = module_flux_df.loc[mods]
        rank = df.abs().mean(axis=1).sort_values(ascending=False)
        keep = rank.head(min(module_topn, len(rank))).index.tolist()
        df = df.loc[keep]
        conf = module_conf_df.loc[df.index, df.columns] if (not module_conf_df.empty) else pd.DataFrame(index=df.index, columns=df.columns)
        vmax = robust_vmax(df)
        h = max(4.0, 0.28 * df.shape[0] + 1.3)
        w = max(7.0, 0.33 * df.shape[1] + 2.0)
        fig, axes = plt.subplots(1, 2, figsize=(w * 1.7, h), gridspec_kw={"wspace": 0.18})
        if HAVE_SNS:
            sns.heatmap(df, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
                        linewidths=0.15, linecolor="#F2F2F2",
                        cbar_kws={"label": "Signed flow potential"}, ax=axes[0])
            sns.heatmap(conf, cmap="YlOrRd", vmin=0, vmax=robust_nonneg_vmax(conf),
                        linewidths=0.15, linecolor="#F2F2F2",
                        cbar_kws={"label": "Confidence"}, ax=axes[1])
        else:
            im0 = axes[0].imshow(df.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            axes[0].set_xticks(np.arange(df.shape[1])); axes[0].set_xticklabels(df.columns, rotation=90)
            axes[0].set_yticks(np.arange(df.shape[0])); axes[0].set_yticklabels(df.index)
            fig.colorbar(im0, ax=axes[0]).set_label("Signed flow potential")
            vmax1 = robust_nonneg_vmax(conf)
            im1 = axes[1].imshow(conf.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax1)
            axes[1].set_xticks(np.arange(conf.shape[1])); axes[1].set_xticklabels(conf.columns, rotation=90)
            axes[1].set_yticks(np.arange(conf.shape[0])); axes[1].set_yticklabels(conf.index)
            fig.colorbar(im1, ax=axes[1]).set_label("Confidence")
        axes[0].set_title(f"{sp}: module signed flow", fontsize=12)
        axes[1].set_title(f"{sp}: module confidence", fontsize=12)
        axes[0].set_xlabel("Cell type"); axes[0].set_ylabel("Module")
        axes[1].set_xlabel("Cell type"); axes[1].set_ylabel("")
        base = sanitize_filename(sp)
        save_fig(fig,
                 os.path.join(outdir, f"Fig5_selected_subpathway_module_decomposition_{base}.pdf"),
                 os.path.join(outdir, f"Fig5_selected_subpathway_module_decomposition_{base}.png"))
        made.append(sp)
    return made

def plot_flux_vs_constraints(flux_df, enz_df, comp_df, cof_df, outdir, selected_celltypes=None, topn=40):
    made = []
    celltypes = selected_celltypes if selected_celltypes is not None else flux_df.columns.tolist()[:4]
    for ct in celltypes:
        if ct not in flux_df.columns:
            continue
        common_idx = flux_df.index.intersection(enz_df.index).intersection(comp_df.index).intersection(cof_df.index)
        tmp = pd.DataFrame({
            "flux": flux_df.loc[common_idx, ct],
            "enzyme": enz_df.loc[common_idx, ct],
            "compartment": comp_df.loc[common_idx, ct],
            "cofactor": cof_df.loc[common_idx, ct],
        }).dropna()
        if tmp.empty:
            continue
        tmp["importance"] = tmp["flux"].abs()
        tmp = tmp.sort_values("importance", ascending=False).head(min(topn, len(tmp)))
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw={"wspace": 0.28})
        pairs = [("enzyme", "Enzyme capacity"), ("compartment", "Compartment support"), ("cofactor", "Cofactor support")]
        for ax, (col, title) in zip(axes, pairs):
            ax.scatter(tmp[col], tmp["flux"], s=35, alpha=0.85, edgecolors="black", linewidths=0.3)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel(title)
            ax.set_ylabel("Signed flow potential")
            ax.set_title(f"{ct}: flux vs {title.lower()}", fontsize=11)
        save_fig(fig,
                 os.path.join(outdir, f"Fig6_flux_vs_constraints_{sanitize_filename(ct)}.pdf"),
                 os.path.join(outdir, f"Fig6_flux_vs_constraints_{sanitize_filename(ct)}.png"))
        made.append(ct)
    return made


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--selected-celltypes", default="",
                    help=("Cell types to plot. Accepts full names or abbreviations before ':'. "
                          "Examples: 'ErPrT,DTLH,CnT' or "
                          "'ErPrT: early proximal tubule,DTLH: distal tubule/loop of Henle'."))
    ap.add_argument("--selected-celltype-file", default="",
                    help="Optional text file with one selected cell type per line.")
    ap.add_argument("--celltype-match-mode", default="auto",
                    choices=["auto", "exact", "prefix", "contains", "regex"],
                    help=("How to match --selected-celltypes to matrix columns. "
                          "auto = exact, then prefix abbreviation, then contains."))
    ap.add_argument("--strict-celltype-match", action="store_true",
                    help="Raise an error if any selected cell type is unmatched or ambiguous.")
    ap.add_argument("--list-celltypes", action="store_true",
                    help="Print available cell types in the input matrix and exit.")
    ap.add_argument("--selected-subpathways", default="",
                    help="Optional explicit subpathways to plot. Applied together with --pathway-preset if both are provided.")
    ap.add_argument("--pathway-preset", default="all",
                    choices=["all", "fao_core", "fao_support", "fao_extended", "custom_keyword"],
                    help=("Filter subpathways by preset. "
                          "fao_core = carnitine/acylcarnitine/fatty-acid/ketone branches; "
                          "fao_support = fao_core plus TCA/pyruvate oxidative-support branches "
                          "and 3 glycolysis-related branches "
                          "(Hexose/Disaccharide, Glycerate/Triose, Lactate/Pyruvate Related); "
                          "fao_extended = fao_support plus lipid-remodelling branches."))
    ap.add_argument("--pathway-keywords", default="",
                    help=("Extra pathway keywords for filtering, comma/semicolon/newline separated. "
                          "Use with --pathway-preset custom_keyword for a fully custom subset."))
    ap.add_argument("--list-subpathways", action="store_true",
                    help="Print available subpathways in the input matrix and exit.")
    ap.add_argument("--topk", type=int, default=8)

    # Fig1A compact heatmap controls
    ap.add_argument("--fig1a-row-height", type=float, default=0.20,
                    help="Height contribution per subpathway row for Fig1A.")
    ap.add_argument("--fig1a-col-width", type=float, default=0.28,
                    help="Width contribution per cell-type column for Fig1A.")
    ap.add_argument("--fig1a-min-height", type=float, default=4.5,
                    help="Minimum figure height for Fig1A.")
    ap.add_argument("--fig1a-min-width", type=float, default=5.8,
                    help="Minimum figure width for Fig1A.")
    ap.add_argument("--fig1a-title-fontsize", type=float, default=11.0,
                    help="Title font size for Fig1A.")
    ap.add_argument("--fig1a-axis-label-fontsize", type=float, default=9.5,
                    help="Axis-label font size for Fig1A.")
    ap.add_argument("--fig1a-xtick-fontsize", type=float, default=8.0,
                    help="X-axis tick font size for Fig1A.")
    ap.add_argument("--fig1a-ytick-fontsize", type=float, default=7.2,
                    help="Y-axis tick font size for Fig1A.")
    ap.add_argument("--fig1a-cbar-label-fontsize", type=float, default=9.0,
                    help="Colorbar label font size for Fig1A.")
    ap.add_argument("--fig1a-cbar-tick-fontsize", type=float, default=8.0,
                    help="Colorbar tick font size for Fig1A.")

    # Fig2 compact layout controls
    ap.add_argument("--fig2-panel-width", type=float, default=4.4,
                    help="Width of each small panel in Fig2.")
    ap.add_argument("--fig2-panel-height", type=float, default=3.8,
                    help="Height of each small panel in Fig2.")
    ap.add_argument("--fig2-ytick-fontsize", type=float, default=7.5,
                    help="Y-axis pathway-label font size for Fig2.")
    ap.add_argument("--fig2-title-fontsize", type=float, default=10.5,
                    help="Per-panel title font size for Fig2.")
    ap.add_argument("--fig2-axis-label-fontsize", type=float, default=9.0,
                    help="Axis-label font size for Fig2.")
    ap.add_argument("--fig2-xtick-fontsize", type=float, default=8.0,
                    help="X-axis tick font size for Fig2.")
    ap.add_argument("--fig2-suptitle-fontsize", type=float, default=12.0,
                    help="Main title font size for Fig2.")
    ap.add_argument("--fig2-x-nbins", type=int, default=4,
                    help="Maximum number of major x ticks for Fig2.")
    ap.add_argument("--fig2-x-pad-frac", type=float, default=0.04,
                    help="Fractional padding added to the symmetric x-axis range in Fig2.")
    ap.add_argument("--module-topn", type=int, default=40)
    ap.add_argument("--top-subpathways-for-constraint", type=int, default=30)
    ap.add_argument("--phase-label-topn", type=int, default=5,
                    help="Annotate top N positive and bottom N negative signed-flow subpathways in Fig3 for each cell type.")
    ap.add_argument("--phase-label-max-chars", type=int, default=34,
                    help="Maximum characters for each Fig3 subpathway label.")
    ap.add_argument("--phase-label-fontsize", type=float, default=6.0,
                    help="Font size for Fig3 subpathway labels.")
    ap.add_argument("--no-phase-label-arrows", action="store_true",
                    help="Disable small leader lines from Fig3 labels to the corresponding points.")
    ap.add_argument("--cluster-rows", action="store_true")
    ap.add_argument("--cluster-cols", action="store_true")
    args = ap.parse_args()

    mkdir(args.outdir)

    sub_flux, p_sub_flux = read_matrix(args.input_dir, "subpathway_signed_flux_matrix")
    sub_act, _ = read_matrix(args.input_dir, "subpathway_activation_matrix")
    sub_dir, _ = read_matrix(args.input_dir, "subpathway_direction_matrix")
    sub_conf, _ = read_matrix(args.input_dir, "subpathway_confidence_matrix")
    sub_feas, _ = read_matrix(args.input_dir, "subpathway_feasibility_matrix")
    sub_enz, _ = read_matrix(args.input_dir, "subpathway_enzyme_capacity_matrix")
    sub_comp, _ = read_matrix(args.input_dir, "subpathway_compartment_support_matrix")
    sub_cof, _ = read_matrix(args.input_dir, "subpathway_cofactor_support_matrix")

    mod_flux, _ = read_matrix(args.input_dir, "module_signed_flux_matrix")
    mod_conf, _ = read_matrix(args.input_dir, "module_confidence_matrix")
    ann, p_ann = read_optional_table(args.input_dir, "module_annotation_used")

    if args.list_celltypes:
        print("Available cell types:")
        for c in sub_flux.columns:
            print(c)
        return

    if args.list_subpathways:
        print("Available subpathways:")
        for sp in sub_flux.index:
            print(sp)
        return

    requested_celltypes = []
    x1 = parse_comma_list(args.selected_celltypes)
    x2 = parse_name_file(args.selected_celltype_file)
    if x1:
        requested_celltypes.extend(x1)
    if x2:
        requested_celltypes.extend(x2)
    if len(requested_celltypes) == 0:
        requested_celltypes = None

    selected_celltypes, unmatched_celltypes, ambiguous_celltypes = resolve_celltypes(
        requested_celltypes,
        available=sub_flux.columns.tolist(),
        mode=args.celltype_match_mode,
        strict=args.strict_celltype_match,
    )
    if requested_celltypes is not None:
        print("Requested cell types:")
        for x in requested_celltypes:
            print(f"  - {x}")
        print("Resolved cell types used for plotting:")
        for x in selected_celltypes:
            print(f"  - {x}")
        if unmatched_celltypes:
            print("WARNING: unmatched cell types were ignored:")
            for x in unmatched_celltypes:
                print(f"  - {x}")
        if ambiguous_celltypes:
            print("WARNING: ambiguous cell type patterns matched multiple columns:")
            for q, ms in ambiguous_celltypes:
                print(f"  - {q}: {', '.join(ms)}")

    explicit_subpathways = parse_comma_list(args.selected_subpathways)
    pathway_keywords = parse_comma_list(args.pathway_keywords)

    preset_subpathways = select_pathways_by_preset(
        sub_flux.index,
        preset=args.pathway_preset,
        extra_patterns=pathway_keywords,
    )

    if explicit_subpathways is not None and preset_subpathways is not None:
        # Use the intersection when both explicit subpathways and preset filtering are given.
        selected_subpathways = [sp for sp in explicit_subpathways if sp in set(preset_subpathways)]
    elif explicit_subpathways is not None:
        selected_subpathways = explicit_subpathways
    else:
        selected_subpathways = preset_subpathways

    if selected_subpathways is not None:
        print(f"Pathway preset: {args.pathway_preset}")
        print("Subpathways selected for plotting:")
        for sp in selected_subpathways:
            print(f"  - {sp}")

    # filter
    sub_flux = filter_df(sub_flux, rows=selected_subpathways, cols=selected_celltypes)
    if sub_flux.empty:
        msg = [
            "After filtering, subpathway_signed_flux_matrix is empty.",
            "Check selected celltypes/subpathways/pathway preset.",
            f"Pathway preset: {args.pathway_preset}",
            f"Pathway keywords: {args.pathway_keywords if args.pathway_keywords else 'None'}",
        ]
        if requested_celltypes is not None:
            msg.append("Requested cell types: " + ", ".join(requested_celltypes))
            msg.append("Resolved cell types: " + (", ".join(selected_celltypes) if selected_celltypes else "None"))
            msg.append("Available cell types:\n  " + "\n  ".join(map(str, sub_act.columns.tolist())))
        raise ValueError("\n".join(msg))
    sub_act = filter_df(sub_act, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_dir = filter_df(sub_dir, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_conf = filter_df(sub_conf, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_feas = filter_df(sub_feas, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_enz = filter_df(sub_enz, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_comp = filter_df(sub_comp, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())
    sub_cof = filter_df(sub_cof, rows=sub_flux.index.tolist(), cols=sub_flux.columns.tolist())

    if selected_subpathways is None:
        selected_subpathways = choose_top_subpathways(sub_flux, topn=6)

    plot_subpathway_flux_heatmap(
        sub_flux, args.outdir,
        cluster_rows=args.cluster_rows, cluster_cols=args.cluster_cols,
        row_height=args.fig1a_row_height,
        col_width=args.fig1a_col_width,
        min_height=args.fig1a_min_height,
        min_width=args.fig1a_min_width,
        title_fontsize=args.fig1a_title_fontsize,
        axis_label_fontsize=args.fig1a_axis_label_fontsize,
        xtick_fontsize=args.fig1a_xtick_fontsize,
        ytick_fontsize=args.fig1a_ytick_fontsize,
        cbar_label_fontsize=args.fig1a_cbar_label_fontsize,
        cbar_tick_fontsize=args.fig1a_cbar_tick_fontsize,
    )
    plot_subpathway_bubble(sub_flux, sub_conf, sub_feas, args.outdir)
    plot_top_subpathways(
        sub_flux, args.outdir, topk=args.topk,
        panel_width=args.fig2_panel_width,
        panel_height=args.fig2_panel_height,
        ytick_fontsize=args.fig2_ytick_fontsize,
        title_fontsize=args.fig2_title_fontsize,
        axis_label_fontsize=args.fig2_axis_label_fontsize,
        xtick_fontsize=args.fig2_xtick_fontsize,
        suptitle_fontsize=args.fig2_suptitle_fontsize,
        x_nbins=args.fig2_x_nbins,
        x_pad_frac=args.fig2_x_pad_frac,
    )
    plot_phaseplot(
        sub_act, sub_dir, sub_flux, sub_conf, args.outdir,
        label_topn=args.phase_label_topn,
        label_max_chars=args.phase_label_max_chars,
        label_fontsize=args.phase_label_fontsize,
        draw_label_arrows=(not args.no_phase_label_arrows),
    )
    plot_constraint_landscape(sub_enz, sub_comp, sub_cof, sub_flux, args.outdir,
                              cluster_rows=args.cluster_rows, cluster_cols=args.cluster_cols,
                              topn=args.top_subpathways_for_constraint)

    selected_made = plot_module_decomposition(mod_flux, mod_conf, ann, selected_subpathways, args.outdir,
                                              module_topn=args.module_topn)
    ct_for_scatter = selected_celltypes if selected_celltypes is not None else list(sub_flux.columns[:4])
    ct_scatter_made = plot_flux_vs_constraints(sub_flux, sub_enz, sub_comp, sub_cof, args.outdir,
                                               selected_celltypes=ct_for_scatter, topn=40)

    summary_lines = [
        "CNS-style round13 plotting summary",
        f"Input directory: {args.input_dir}",
        f"Output directory: {args.outdir}",
        f"Read: {p_sub_flux}",
        f"Annotation: {p_ann}",
        f"Subpathways plotted: {sub_flux.shape[0]}",
        f"Cell types plotted: {sub_flux.shape[1]}",
        f"Selected cell types requested: {', '.join(requested_celltypes) if requested_celltypes else 'All'}",
        f"Selected cell types resolved: {', '.join(selected_celltypes) if selected_celltypes else 'All'}",
        f"Unmatched cell types ignored: {', '.join(unmatched_celltypes) if unmatched_celltypes else 'None'}",
        f"Pathway preset: {args.pathway_preset}",
        f"Pathway keywords: {args.pathway_keywords if args.pathway_keywords else 'None'}",
        f"Selected subpathways: {', '.join(selected_subpathways) if selected_subpathways else 'All'}",
        f"Module decomposition panels: {', '.join(selected_made) if selected_made else 'None'}",
        f"Flux-vs-constraints scatter panels: {', '.join(ct_scatter_made) if ct_scatter_made else 'None'}",
        "",
        "Figure set:",
        "  Fig1A_subpathway_flux_heatmap",
        "  Fig1B_subpathway_bubble",
        "  Fig2_celltype_top_subpathways",
        "  Fig3_activation_direction_phaseplot",
        "  Fig4_constraint_landscape",
        "  Fig5_selected_subpathway_module_decomposition_*",
        "  Fig6_flux_vs_constraints_*",
    ]
    with open(os.path.join(args.outdir, "plotting_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
