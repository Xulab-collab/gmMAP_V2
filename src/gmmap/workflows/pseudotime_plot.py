#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import spearmanr
from difflib import get_close_matches

# 可选：LOWESS 平滑
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False


# =========================
# Global font settings: Arial + non-bold
# =========================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["axes.labelweight"] = "normal"
mpl.rcParams["axes.titleweight"] = "normal"
mpl.rcParams["figure.titleweight"] = "normal"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# =========================
# Config
# =========================
H5AD_PATH = "data/input.h5ad"

SCORES_PATH = "results/gmmap_scores.csv.gz"

# trait -> reportedTrait 映射文件
MAPPING_PATH = "data/trait_metadata.csv"

# 是否把 score matrix 的列名转成 reportedTrait
USE_REPORTEDTRAIT_NAMES = True

# 是否把重命名后的矩阵另存为新文件
SAVE_RENAMED_SCORE_MATRIX = True
RENAMED_SCORES_PATH = (
    "Meta_pyUCell_pseudotime_outputs/cellTraitMatrices/"
    "Meta_pyUCell_signed_scores.reportedTrait.csv.gz"
)

CELLTYPE_COL = "knn_cell_type"
PSEUDOTIME_COL = "monocle_pseudotime"

OUT_DIR = "Meta_pyUCell_pseudotime_outputs/metabolite_pseudotime_plots"

# 可选 "DT", "PT", "Podo"，或 None（全体细胞）
LINEAGE_NAME = "PT"

# 这里既可以写 reportedTrait，也可以写原始 trait
METABOLITES = [
    "Nicotinamide levels",
    "4-hydroxyphenylacetoylcarnitine levels",
    "3-Hydroxybutyrate levels",
    "Lignoceroylcarnitine (C24) levels",
    "Cerotoylcarnitine (C26) levels",
    "Heptenedioate (C7:1-DC) levels",
    "Eicosenedioate (C20:1-DC) levels",
    "Palmitoleate (16:1n7) levels",
    "Dihomo-linolenate (20:3n3 or n6) levels",
    "Retinol (Vitamin A) levels",
    "Carnitine to ergothioneine ratio",
    "Suberate (C8-DC) levels",
    "Tetradecadienedioate (C14:2-DC) levels",
    "3-hydroxyhexanoate levels",
    "Glutarate (C5-DC) levels",
    "Acetylcarnitine (c2) levels",
    "Decanoylcarnitine (C10) levels",
    "Laurylcarnitine levels",
    "Stearoylcarnitine (C18) levels",
    "Adenosine 5'-diphosphate (ADP) to glucose ratio",
    "Adenosine 5'-diphosphate (ADP) to choline ratio",
    "Adenosine 5'-diphosphate (ADP) to glycerol 3-phosphate ratio",
    "Adenosine 5'-diphosphate (ADP) levels",
    "Pyruvate levels",
    "Phosphate to acetoacetate ratio"
]
# 绘图参数
POINT_SIZE = 14
POINT_ALPHA = 0.55
SMOOTH_FRAC = 0.22
FIG_WIDTH = 7.2
PANEL_HEIGHT = 2.3
DPI = 300

# 输出控制
SAVE_COMBINED_PLOT = True
SAVE_INDIVIDUAL_PDFS = True
SAVE_INDIVIDUAL_PNGS = False

LINEAGES = {
    "DT": [
        "NPCd: nephron progenitor cells d",
        "RVCSBa: renal vesicle/comma-shapedbody a",
        "RVCSB b: renal vesicle/comma-shapedbody b",
        "SSBm/d: s-shaped body medial/distal",
        "DTLH: distal tubule/loop of Henle",
    ],
    "PT": [
        "NPCd: nephron progenitor cells d",
        "RVCSBa: renal vesicle/comma-shapedbody a",
        "SSBpr: s-shaped body proximal precursor cells",
        "ErPrT: early proximal tubule",
    ],
    "Podo": [
        "NPCd: nephron progenitor cells d",
        "RVCSBa: renal vesicle/comma-shapedbody a",
        "RVCSB b: renal vesicle/comma-shapedbody b",
        "SSBpod: s-shaped body podocyte precursor cells",
        "Pod: podocyte",
    ],
}


# =========================
# Helpers
# =========================
def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3g}"


def smooth_xy(x, y, frac=0.22):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size < 5:
        order = np.argsort(x)
        return x[order], y[order]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if HAS_LOWESS:
        sm = lowess(y, x, frac=frac, it=0, return_sorted=True)
        return sm[:, 0], sm[:, 1]

    win = max(7, int(np.ceil(len(x) * frac)))
    if win % 2 == 0:
        win += 1
    ys = pd.Series(y).rolling(
        win, center=True, min_periods=max(3, win // 3)
    ).median()
    ys = ys.interpolate(limit_direction="both").to_numpy()
    return x, ys


def get_palette(categories):
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i % 9) for i in range(len(categories))]
    return {c: colors[i] for i, c in enumerate(categories)}


def make_unique_names(names):
    seen = {}
    out = []
    for x in names:
        if x not in seen:
            seen[x] = 1
            out.append(x)
        else:
            seen[x] += 1
            out.append(f"{x}__{seen[x]}")
    return out


def sanitize_filename(x, max_len=120):
    x = str(x)
    bad_chars = ['/', '\\', ' ', '(', ')', ':', ';', ',', '[', ']', '{', '}', '+', '*', '?', '"', "'", '|', '<', '>']
    for ch in bad_chars:
        x = x.replace(ch, "_")
    while "__" in x:
        x = x.replace("__", "_")
    x = x.strip("_")
    if len(x) > max_len:
        x = x[:max_len]
    return x


def load_trait_mapping(mapping_path):
    mp = pd.read_csv(mapping_path)

    required = {"trait", "reportedTrait"}
    missing = required - set(mp.columns)
    if missing:
        raise KeyError(f"Mapping file missing required columns: {sorted(missing)}")

    mp = mp[["trait", "reportedTrait"]].copy()
    mp["trait"] = mp["trait"].astype(str)
    mp["reportedTrait"] = mp["reportedTrait"].astype(str)

    mp = mp.replace({
        "reportedTrait": {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan
        }
    })
    mp = mp.dropna(subset=["trait", "reportedTrait"])
    mp = mp.drop_duplicates(subset=["trait"], keep="first")

    trait2reported = dict(zip(mp["trait"], mp["reportedTrait"]))
    return trait2reported, mp


def rename_score_df_columns(score_df, mapping_path, save_path=None, verbose=True):
    trait2reported, mapping_df = load_trait_mapping(mapping_path)

    old_cols = list(score_df.columns)
    mapped_cols = [trait2reported.get(col, col) for col in old_cols]
    final_cols = make_unique_names(mapped_cols)

    renamed_df = score_df.copy()
    renamed_df.columns = final_cols

    trait_to_final = dict(zip(old_cols, final_cols))

    rename_info = {
        "old_cols": old_cols,
        "mapped_cols": mapped_cols,
        "final_cols": final_cols,
        "trait_to_reported": trait2reported,
        "trait_to_final": trait_to_final,
        "mapping_df": mapping_df,
    }

    if verbose:
        n_changed = sum(a != b for a, b in zip(old_cols, final_cols))
        print(f">>> Column renaming finished: {n_changed}/{len(old_cols)} columns renamed")

        not_mapped = [c for c in old_cols if c not in trait2reported]
        if len(not_mapped) > 0:
            print(f">>> {len(not_mapped)} columns had no mapping and were kept unchanged")
            print(">>> Unmapped examples:", not_mapped[:10])

        dup_reported = mapping_df["reportedTrait"][
            mapping_df["reportedTrait"].duplicated()
        ].unique().tolist()
        if len(dup_reported) > 0:
            print(">>> Warning: duplicated reportedTrait found in mapping file.")
            print(">>> Auto-suffixed examples:", dup_reported[:10])

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        renamed_df.to_csv(save_path, compression="gzip")
        if verbose:
            print(f">>> Saved renamed score matrix to: {save_path}")

    return renamed_df, rename_info


def resolve_requested_metabolites(metabolites, score_columns, rename_info=None):
    score_columns = list(score_columns)
    resolved = []
    unresolved = []

    trait_to_final = {}
    if rename_info is not None:
        trait_to_final = rename_info.get("trait_to_final", {})

    for m in metabolites:
        if m in score_columns:
            resolved.append(m)
        elif m in trait_to_final and trait_to_final[m] in score_columns:
            resolved.append(trait_to_final[m])
        else:
            unresolved.append(m)

    if len(unresolved) > 0:
        msg = ["以下代谢物在分数矩阵中不存在："]
        for m in unresolved:
            candidates = get_close_matches(m, score_columns, n=5, cutoff=0.3)
            if len(candidates) > 0:
                msg.append(f"  - {m}   |  可能想找：{', '.join(candidates)}")
            else:
                msg.append(f"  - {m}")
        raise KeyError("\n".join(msg))

    return resolved


def load_data():
    adata = sc.read_h5ad(H5AD_PATH)
    score_df = pd.read_csv(SCORES_PATH, index_col=0)

    adata.obs_names = adata.obs_names.astype(str)
    score_df.index = score_df.index.astype(str)

    if CELLTYPE_COL not in adata.obs.columns:
        raise KeyError(f"{CELLTYPE_COL} not found in adata.obs")
    if PSEUDOTIME_COL not in adata.obs.columns:
        raise KeyError(f"{PSEUDOTIME_COL} not found in adata.obs")

    common_cells = adata.obs_names.intersection(score_df.index)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between h5ad.obs_names and score_df.index")

    obs = adata.obs.loc[common_cells, [CELLTYPE_COL, PSEUDOTIME_COL]].copy()
    score_df = score_df.loc[common_cells].copy()

    rename_info = None
    if USE_REPORTEDTRAIT_NAMES:
        save_path = RENAMED_SCORES_PATH if SAVE_RENAMED_SCORE_MATRIX else None
        score_df, rename_info = rename_score_df_columns(
            score_df=score_df,
            mapping_path=MAPPING_PATH,
            save_path=save_path,
            verbose=True
        )

    return adata, obs, score_df, rename_info


def subset_lineage(obs, score_df, lineage_name=None):
    obs = obs.copy()

    if lineage_name is None:
        mask = np.isfinite(obs[PSEUDOTIME_COL].values)
        obs_sub = obs.loc[mask].copy()
        score_sub = score_df.loc[obs_sub.index].copy()
        cat_order = list(pd.unique(obs_sub[CELLTYPE_COL].astype(str)))
        return obs_sub, score_sub, cat_order

    if lineage_name not in LINEAGES:
        raise KeyError(f"lineage_name must be one of {list(LINEAGES.keys())} or None")

    lineage_celltypes = LINEAGES[lineage_name]
    mask = obs[CELLTYPE_COL].isin(lineage_celltypes) & np.isfinite(obs[PSEUDOTIME_COL].values)

    obs_sub = obs.loc[mask].copy()
    score_sub = score_df.loc[obs_sub.index].copy()

    obs_sub[CELLTYPE_COL] = pd.Categorical(
        obs_sub[CELLTYPE_COL],
        categories=lineage_celltypes,
        ordered=True
    )
    obs_sub = obs_sub.sort_values([PSEUDOTIME_COL, CELLTYPE_COL])
    score_sub = score_sub.loc[obs_sub.index]

    return obs_sub, score_sub, lineage_celltypes


def build_plot_df(obs_sub, score_sub, metabolites, rename_info=None):
    resolved_metabolites = resolve_requested_metabolites(
        metabolites=metabolites,
        score_columns=score_sub.columns,
        rename_info=rename_info
    )
    plot_df = obs_sub.join(score_sub[resolved_metabolites], how="inner")
    plot_df = plot_df.loc[np.isfinite(plot_df[PSEUDOTIME_COL].values)].copy()
    return plot_df, resolved_metabolites


def compute_correlations(plot_df, metabolites):
    rows = []
    for met in metabolites:
        x = plot_df[PSEUDOTIME_COL].to_numpy(dtype=float)
        y = plot_df[met].to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            rho, p = np.nan, np.nan
            n = int(ok.sum())
        else:
            rho, p = spearmanr(x[ok], y[ok], nan_policy="omit")
            n = int(ok.sum())

        rows.append({
            "metabolite": met,
            "rho": rho,
            "pvalue": p,
            "n_cells": n
        })

    return pd.DataFrame(rows)


def add_right_strip(ax, label):
    ax.text(
        1.01, 0.5, label,
        transform=ax.transAxes,
        rotation=-90,
        va="center", ha="left",
        fontsize=10,
        fontweight="normal",
        fontfamily="Arial",
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="#E6E6E6",
            edgecolor="#BDBDBD"
        )
    )


def plot_single_metabolite(
    plot_df,
    metabolite,
    lineage_name,
    cat_order,
    out_dir,
    point_size=14,
    point_alpha=0.55,
    smooth_frac=0.22,
    dpi=300,
    save_pdf=True,
    save_png=False,
):
    os.makedirs(out_dir, exist_ok=True)

    palette = get_palette(cat_order)

    x_all = plot_df[PSEUDOTIME_COL].to_numpy(dtype=float)
    y_all = plot_df[metabolite].to_numpy(dtype=float)
    ct_all = plot_df[CELLTYPE_COL].astype(str).to_numpy()

    ok_all = np.isfinite(x_all) & np.isfinite(y_all)

    if ok_all.sum() < 5:
        rho, p = np.nan, np.nan
    else:
        rho, p = spearmanr(x_all[ok_all], y_all[ok_all], nan_policy="omit")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    for ct in cat_order:
        idx = ok_all & (ct_all == str(ct))
        if idx.sum() == 0:
            continue
        ax.scatter(
            x_all[idx],
            y_all[idx],
            s=point_size,
            alpha=point_alpha,
            color=palette[ct],
            edgecolors="none",
            label=str(ct)
        )

    xs, ys = smooth_xy(x_all[ok_all], y_all[ok_all], frac=smooth_frac)
    ax.plot(xs, ys, color="black", linewidth=2.2, zorder=5)

    label = f"Spearman rho = {rho:.3f}\nP = {format_p_value(p)}"
    ax.text(
        0.98, 0.95, label,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        fontweight="normal",
        fontfamily="Arial",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.88
        )
    )

    title = f"{metabolite} along pseudotime"
    if lineage_name is not None:
        title += f" ({lineage_name} lineage)"

    ax.set_title(
        title,
        fontsize=13,
        fontweight="normal",
        fontfamily="Arial"
    )
    ax.set_xlabel(
        "Pseudotime",
        fontsize=11,
        fontweight="normal",
        fontfamily="Arial"
    )
    ax.set_ylabel(
        "signed UCell",
        fontsize=11,
        fontweight="normal",
        fontfamily="Arial"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        leg = ax.legend(
            handles, labels,
            loc="best",
            frameon=False,
            fontsize=9
        )
        for txt in leg.get_texts():
            txt.set_fontfamily("Arial")
            txt.set_fontweight("normal")

    plt.tight_layout()

    safe_met = sanitize_filename(metabolite)
    prefix = "metabolite_pseudotime"
    if lineage_name is not None:
        prefix += f"_{sanitize_filename(lineage_name)}"
    prefix += f"_{safe_met}"

    pdf_path = os.path.join(out_dir, f"{prefix}.pdf")
    png_path = os.path.join(out_dir, f"{prefix}.png")

    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f">>> Saved individual PDF: {pdf_path}")

    if save_png:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f">>> Saved individual PNG: {png_path}")

    plt.close(fig)

    return {
        "metabolite": metabolite,
        "rho": rho,
        "pvalue": p,
        "pdf_path": pdf_path if save_pdf else None,
        "png_path": png_path if save_png else None,
    }


def plot_metabolites(plot_df, metabolites, lineage_name, cat_order, out_prefix):
    os.makedirs(OUT_DIR, exist_ok=True)

    corr_df = compute_correlations(plot_df, metabolites)

    if SAVE_COMBINED_PLOT:
        n_panels = len(metabolites)
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(FIG_WIDTH, PANEL_HEIGHT * n_panels + 1.1),
            sharex=True
        )
        if n_panels == 1:
            axes = [axes]

        palette = get_palette(cat_order)

        for i, met in enumerate(metabolites):
            ax = axes[i]

            x_all = plot_df[PSEUDOTIME_COL].to_numpy(dtype=float)
            y_all = plot_df[met].to_numpy(dtype=float)
            ct_all = plot_df[CELLTYPE_COL].astype(str).to_numpy()

            ok_all = np.isfinite(x_all) & np.isfinite(y_all)

            for ct in cat_order:
                idx = ok_all & (ct_all == str(ct))
                if idx.sum() == 0:
                    continue
                ax.scatter(
                    x_all[idx], y_all[idx],
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA,
                    color=palette[ct],
                    edgecolors="none",
                    label=str(ct) if i == 0 else None
                )

            xs, ys = smooth_xy(x_all[ok_all], y_all[ok_all], frac=SMOOTH_FRAC)
            ax.plot(xs, ys, color="black", linewidth=2.0, zorder=5)

            row = corr_df.loc[corr_df["metabolite"] == met].iloc[0]
            label = f"Spearman rho = {row['rho']:.3f}\nP = {format_p_value(row['pvalue'])}"
            ax.text(
                0.98, 0.95, label,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=10,
                fontweight="normal",
                fontfamily="Arial",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85
                )
            )

            add_right_strip(ax, met)

            ax.set_ylabel(
                "signed UCell",
                fontsize=11,
                fontweight="normal",
                fontfamily="Arial"
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)

        axes[-1].set_xlabel(
            "Pseudotime",
            fontsize=11,
            fontweight="normal",
            fontfamily="Arial"
        )

        title = "Metabolite dynamics along pseudotime"
        if lineage_name is not None:
            title += f" ({lineage_name} lineage)"

        fig.suptitle(
            title,
            fontsize=14,
            fontweight="normal",
            fontfamily="Arial",
            y=0.995
        )

        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            leg = fig.legend(
                handles, labels,
                loc="lower center",
                ncol=min(3, len(labels)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.005),
                fontsize=9
            )
            for txt in leg.get_texts():
                txt.set_fontfamily("Arial")
                txt.set_fontweight("normal")

        plt.tight_layout(rect=[0, 0.04, 0.95, 0.97])

        png_path = os.path.join(OUT_DIR, f"{out_prefix}.png")
        pdf_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")

        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f">>> Saved combined plot: {png_path}")
        print(f">>> Saved combined plot: {pdf_path}")

    stats_path = os.path.join(OUT_DIR, f"{out_prefix}_correlation_stats.csv")
    corr_df.to_csv(stats_path, index=False)
    print(f">>> Saved stats: {stats_path}")

    if SAVE_INDIVIDUAL_PDFS or SAVE_INDIVIDUAL_PNGS:
        single_dir = os.path.join(OUT_DIR, f"{out_prefix}_individual")
        os.makedirs(single_dir, exist_ok=True)

        single_rows = []
        for met in metabolites:
            res = plot_single_metabolite(
                plot_df=plot_df,
                metabolite=met,
                lineage_name=lineage_name,
                cat_order=cat_order,
                out_dir=single_dir,
                point_size=POINT_SIZE,
                point_alpha=POINT_ALPHA,
                smooth_frac=SMOOTH_FRAC,
                dpi=DPI,
                save_pdf=SAVE_INDIVIDUAL_PDFS,
                save_png=SAVE_INDIVIDUAL_PNGS,
            )
            single_rows.append(res)

        single_stats = pd.DataFrame(single_rows)
        single_stats_path = os.path.join(single_dir, "individual_plot_paths_and_stats.csv")
        single_stats.to_csv(single_stats_path, index=False)
        print(f">>> Saved individual plot summary: {single_stats_path}")


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(">>> Load h5ad and signed UCell score matrix")
    _, obs, score_df, rename_info = load_data()

    print(">>> Subset lineage")
    obs_sub, score_sub, cat_order = subset_lineage(
        obs=obs,
        score_df=score_df,
        lineage_name=LINEAGE_NAME
    )

    print(f">>> Cells used: {obs_sub.shape[0]}")
    if obs_sub.shape[0] == 0:
        raise ValueError("No cells left after lineage/pseudotime filtering.")

    print(">>> Build plot dataframe")
    plot_df, resolved_metabolites = build_plot_df(
        obs_sub=obs_sub,
        score_sub=score_sub,
        metabolites=METABOLITES,
        rename_info=rename_info
    )

    print(">>> Metabolites used in plotting:")
    for m0, m1 in zip(METABOLITES, resolved_metabolites):
        if m0 == m1:
            print(f"    {m1}")
        else:
            print(f"    {m0}  ->  {m1}")

    out_prefix = "metabolite_pseudotime"
    if LINEAGE_NAME is not None:
        out_prefix += f"_{LINEAGE_NAME}"

    clean_names = []
    for m in resolved_metabolites[:4]:
        tmp = sanitize_filename(m, max_len=30)
        clean_names.append(tmp)

    out_prefix += "_" + "_".join(clean_names)
    if len(resolved_metabolites) > 4:
        out_prefix += "_etc"

    print(">>> Plot")
    plot_metabolites(
        plot_df=plot_df,
        metabolites=resolved_metabolites,
        lineage_name=LINEAGE_NAME,
        cat_order=cat_order,
        out_prefix=out_prefix
    )

    print(">>> Done")


if __name__ == "__main__":
    main()
