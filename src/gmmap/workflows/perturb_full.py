#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmMAP-perturb: root-to-terminal fate-bias analysis using metabolite inhibition-like scores.

Designed for AnnData with obs columns such as:
    knn_cell_type, monocle_pseudotime
and obsm:
    X_umap, X_tsne, X_mnn

No CellRank dependency is required. FateBias is estimated from Monocle pseudotime and terminal cell-type anchors.

Core definitions:
    Activity_m   = z(UCell(UP_m genes))
    Inhibition_m = z(UCell(DOWN_m genes))
    Delta_m      = Activity_m - Inhibition_m

For each root/target1/target2 config:
    FateBias = target1-like minus target2-like, scaled from Monocle pseudotime and oriented by terminal anchors.
    rho(inhibition_like_score, FateBias) < 0 predicts target1 -> target2 rerouting.
    rho(inhibition_like_score, FateBias) > 0 predicts target2 -> target1 rerouting.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# Adobe Illustrator / editable-vector font settings
# =========================
# These settings keep text editable in Adobe Illustrator.
# pdf.fonttype=42 embeds TrueType fonts instead of Type 3 vector outlines.
# svg.fonttype="none" keeps SVG text as text rather than converting it to paths.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["savefig.transparent"] = False

# Adobe Illustrator editable text:
# Path effects such as Stroke convert text into vector outlines in many vector editors.
# Therefore UMAP labels are drawn with normal text + editable bbox instead of path effects.
ILLUSTRATOR_EDITABLE_TEXT = True
USE_TEXT_PATH_EFFECTS = False

from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects


# =========================
# Config
# =========================
H5AD_PATH = "data/input.h5ad"
ZMAT_PATH = "data/MAGMA_zstat.csv"

CELLTYPE_COL = "knn_cell_type"
PSEUDOTIME_COL = "monocle_pseudotime"
UMAP_KEY = "X_umap"

PREFIX = "Meta"
OUT_DIR = "Meta_gmMAP_perturb_root_terminal_UP_DOWN_Delta_branchDiff_PDFonly_outputs"

TOP_N_GENES = 1000
MIN_VALID_GENES = 100

N_JOBS_UCELL = -1
N_JOBS_COR = 8

UCELL_CHUNK_SIZE = 500
UCELL_SIGNATURE_BATCH = 128
UCELL_MAX_RANK = None
TIES_METHOD = "average"
MISSING_GENES = "impute"

MIN_CELLS_PER_ANALYSIS = 30
MIN_PCT_DETECTED = 0.01
# Evidence classification thresholds for Activity / Inhibition / Delta support.
SUPPORT_Q_CUTOFF = 0.05
SUPPORT_RHO_MIN = 0.20
DELTA_SUPPORT_Q_CUTOFF = 0.10
DELTA_SUPPORT_RHO_MIN = 0.10


# 如果只分析少数代谢物，在这里填写；留空则分析全部 trait。
# TARGET_TRAITS = ["GCST90200147", "GCST90200378"]
TARGET_TRAITS = []

# Manually force plotting/exporting selected traits in addition to automatically ranked top traits.
# These traits must exist in the Z matrix and pass UP/DOWN signature filtering.
# Example:
# MANUAL_PLOT_TRAITS = ["GCST90200136", "GCST90200541"]
MANUAL_PLOT_TRAITS = ["GCST90200027", "GCST90200834", "GCST90199743", "GCST90200807", "GCST90199750", "GCST90200400", "GCST90199722", "GCST90200430", "GCST90199669", "GCST90200239", "GCST90200012", "GCST90200380", "GCST90200883", "GCST90200355", "GCST90200947", "GCST90200958"]

# Automatic plotting strategy:
#   "branch_difference" ranks metabolites by target1-vs-target2 branch difference.
#   "impact_strength" ranks by the original inhibition-vs-FateBias correlation score.
AUTO_PLOT_RANK_BY = "branch_difference"

# Branch-difference ranking settings.
# terminal_fraction uses the last fraction of pseudotime cells in each branch to estimate terminal differences.
BRANCH_DIFF_TERMINAL_FRACTION = 0.25
BRANCH_DIFF_MIN_ABS_EFFECT = 0.20


TRAJECTORY_CONFIG = {
    "PT_vs_DT": {
        "root": ["NPCd: nephron progenitor cells d"],
        "intermediate": [
            "RVCSBa: renal vesicle/comma-shapedbody a",
            "RVCSB b: renal vesicle/comma-shapedbody b",
        ],
        "target1_name": "PT",
        "target1": [
            "SSBpr: s-shaped body proximal precursor cells",
            "ErPrT: early proximal tubule",
        ],
        "target2_name": "DT",
        "target2": [
            "SSBm/d: s-shaped body medial/distal",
            "DTLH: distal tubule/loop of Henle",
        ],
    },
    "PT_vs_Podo": {
        "root": ["NPCd: nephron progenitor cells d"],
        "intermediate": [
            "RVCSBa: renal vesicle/comma-shapedbody a",
            "RVCSB b: renal vesicle/comma-shapedbody b",
        ],
        "target1_name": "PT",
        "target1": [
            "SSBpr: s-shaped body proximal precursor cells",
            "ErPrT: early proximal tubule",
        ],
        "target2_name": "Podo",
        "target2": [
            "SSBpod: s-shaped body podocyte precursor cells",
            "Pod: podocyte",
        ],
    },
    "DT_vs_Podo": {
        "root": ["NPCd: nephron progenitor cells d"],
        "intermediate": [
            "RVCSBa: renal vesicle/comma-shapedbody a",
            "RVCSB b: renal vesicle/comma-shapedbody b",
        ],
        "target1_name": "DT",
        "target1": [
            "SSBm/d: s-shaped body medial/distal",
            "DTLH: distal tubule/loop of Henle",
        ],
        "target2_name": "Podo",
        "target2": [
            "SSBpod: s-shaped body podocyte precursor cells",
            "Pod: podocyte",
        ],
    },
}

TOP_N_PLOT = 30
TOP_TRAJECTORY_PLOT = 6

# Export plots separately for these evidence classes.
PLOT_SUPPORT_CLASSES = [
    "bidirectional_supported",
    "single_DOWN_supported",
    "DOWN_supported_but_activity_opposite",
]
TOP_TRAJECTORY_PLOT_PER_SUPPORT_CLASS = 6

# Only export PDF figures.
EXPORT_FORMATS = ["pdf"]




# =========================
# Local pyUCell-like implementation
# =========================
class uc:
    @staticmethod
    def _parse_signature(sig_genes):
        pos, neg = [], []
        for g in sig_genes:
            g = str(g)
            if g.endswith("+"):
                pos.append(g[:-1])
            elif g.endswith("-"):
                neg.append(g[:-1])
            else:
                pos.append(g)
        return pos, neg

    @staticmethod
    def _prepare_sig_indices(signatures, genes, missing_genes="impute"):
        gene_to_idx = {str(g): i for i, g in enumerate(genes)}
        sig_indices = {}
        for sig_name, sig_genes in signatures.items():
            pos_genes, neg_genes = uc._parse_signature(sig_genes)
            if missing_genes == "impute":
                pos_idx = [gene_to_idx.get(g, -1) for g in pos_genes]
                neg_idx = [gene_to_idx.get(g, -1) for g in neg_genes]
            elif missing_genes == "skip":
                pos_idx = [gene_to_idx[g] for g in pos_genes if g in gene_to_idx]
                neg_idx = [gene_to_idx[g] for g in neg_genes if g in gene_to_idx]
            else:
                raise ValueError("missing_genes must be 'impute' or 'skip'")
            sig_indices[sig_name] = {"pos": pos_idx, "neg": neg_idx}
        return sig_indices

    @staticmethod
    def get_rankings(data, layer=None, max_rank=1500, ties_method="average"):
        X = data.layers[layer] if isinstance(data, AnnData) and layer else (data.X if isinstance(data, AnnData) else data)
        n_cells, n_genes = X.shape
        data_parts, row_parts, col_parts = [], [], []

        for j in range(n_cells):
            if sparse.issparse(X):
                row = X.getrow(j)
                nz_idx, nz_vals = row.indices, row.data
            else:
                row = np.asarray(X[j]).ravel()
                np.nan_to_num(row, copy=False)
                nz_idx = np.flatnonzero(row)
                nz_vals = row[nz_idx]
            if len(nz_idx) == 0:
                continue
            ranks = rankdata(-nz_vals, method=ties_method).astype(np.int32)
            keep = ranks <= max_rank
            if not np.any(keep):
                continue
            data_parts.append(ranks[keep].astype(np.int32))
            row_parts.append(np.asarray(nz_idx[keep], dtype=np.int32))
            col_parts.append(np.full(np.sum(keep), j, dtype=np.int32))

        if len(data_parts) == 0:
            return sparse.csr_matrix((n_genes, n_cells), dtype=np.int32)

        return sparse.coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(n_genes, n_cells), dtype=np.int32
        ).tocsr()

    @staticmethod
    def _calculate_U(ranks, idx, max_rank=1500):
        idx = np.asarray(idx, dtype=np.int32)
        lgt = len(idx)
        n_cells = ranks.shape[1]
        if lgt == 0:
            return np.zeros(n_cells, dtype=np.float32)

        missing_idx = idx[idx == -1]
        present_idx = idx[idx != -1]
        rank_sum = np.full(n_cells, len(missing_idx) * max_rank, dtype=np.float32)

        if len(present_idx) > 0:
            present_ranks = ranks[present_idx, :]
            present_ranks = present_ranks.toarray() if sparse.issparse(present_ranks) else np.asarray(present_ranks)
            if present_ranks.ndim == 1:
                present_ranks = present_ranks[np.newaxis, :]
            present_ranks = present_ranks.astype(np.float32)
            present_ranks[present_ranks == 0] = max_rank
            rank_sum += present_ranks.sum(axis=0)

        s_min = lgt * (lgt + 1) / 2.0
        s_max = lgt * max_rank
        denom = s_max - s_min
        if denom <= 0:
            return np.zeros(n_cells, dtype=np.float32)
        return np.clip(1.0 - (rank_sum - s_min) / denom, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _score_rank_matrix(ranks, sig_indices, w_neg=1.0, max_rank=1500):
        n_cells = ranks.shape[1]
        sig_names = list(sig_indices.keys())
        scores = np.zeros((n_cells, len(sig_names)), dtype=np.float32)
        for j, sig_name in enumerate(sig_names):
            idx_dict = sig_indices[sig_name]
            pos_score = uc._calculate_U(ranks, idx_dict["pos"], max_rank=max_rank) if len(idx_dict["pos"]) > 0 else np.zeros(n_cells, dtype=np.float32)
            neg_score = uc._calculate_U(ranks, idx_dict["neg"], max_rank=max_rank) if len(idx_dict["neg"]) > 0 else np.zeros(n_cells, dtype=np.float32)
            signed_score = pos_score - w_neg * neg_score
            signed_score[signed_score < 0] = 0.0
            scores[:, j] = signed_score
        return sig_names, scores

    @staticmethod
    def compute_ucell_scores(adata, signatures, layer=None, max_rank=1500, ties_method="average", missing_genes="impute", chunk_size=500, w_neg=1.0, suffix="_UCell", n_jobs=-1):
        if max_rank is None:
            max_rank = min(1500, adata.n_vars)
        genes = np.asarray(adata.var_names).astype(str)
        sig_indices = uc._prepare_sig_indices(signatures, genes, missing_genes=missing_genes)
        sig_names = list(sig_indices.keys())
        chunks_local = [(s, min(s + chunk_size, adata.n_obs)) for s in range(0, adata.n_obs, chunk_size)]

        def process_chunk(start, end):
            X_chunk = adata.layers[layer][start:end, :] if layer else adata.X[start:end, :]
            ranks_chunk = uc.get_rankings(X_chunk, max_rank=max_rank, ties_method=ties_method)
            _, scores_chunk = uc._score_rank_matrix(ranks_chunk, sig_indices, w_neg=w_neg, max_rank=max_rank)
            return start, end, scores_chunk

        results = [process_chunk(s, e) for s, e in chunks_local] if n_jobs == 1 else Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_chunk)(s, e) for s, e in chunks_local)
        scores_all = np.zeros((adata.n_obs, len(sig_names)), dtype=np.float32)
        for start, end, scores_chunk in results:
            scores_all[start:end, :] = scores_chunk
        for j, sig_name in enumerate(sig_names):
            adata.obs[f"{sig_name}{suffix}"] = scores_all[:, j]


# =========================
# Helpers
# =========================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if m.sum() == 0:
        return q
    p0 = p[m]
    n = p0.size
    order = np.argsort(p0)
    ranked = p0[order]
    q0 = ranked * n / np.arange(1, n + 1)
    q0 = np.minimum.accumulate(q0[::-1])[::-1]
    q0 = np.clip(q0, 0, 1)
    out = np.empty_like(p0)
    out[order] = q0
    q[m] = out
    return q


def list_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def auto_choose_max_rank(adata):
    X = adata.X
    detected = np.diff(X.tocsr().indptr) if issparse(X) else np.asarray((X > 0).sum(axis=1)).ravel()
    med = float(np.median(detected))
    return int(round(min(1500, max(800, med))))


def load_Z_mat_robust(z_path, adata):
    Z = pd.read_csv(z_path, index_col=0)
    if Z.shape[1] >= 2:
        first_col = Z.columns[0]
        vals = Z[first_col].astype(str)
        var_names = set(map(str, adata.var_names))
        overlap_first = len(set(vals).intersection(var_names)) / max(1, min(len(vals), len(var_names)))
        overlap_index = len(set(map(str, Z.index)).intersection(var_names)) / max(1, min(len(Z.index), len(var_names)))
        if (overlap_first > 0.2) and (overlap_index < 0.05) and vals.is_unique:
            Z = Z.set_index(first_col)
    Z.index = Z.index.astype(str)
    return Z


def extract_signature_up_down(Z_mat, trait, top_n=1000):
    z = pd.to_numeric(Z_mat[trait], errors="coerce").dropna()
    up = z[z > 0].sort_values(ascending=False).head(top_n)
    dn = z[z < 0].sort_values(ascending=True).head(top_n)
    return up, dn


def build_up_down_signatures(Z_mat, adata, top_n=1000, min_valid_genes=100, target_traits=None):
    gene_universe = set(map(str, adata.var_names))
    if target_traits:
        traits = [t for t in target_traits if t in Z_mat.columns]
        missing = sorted(set(target_traits) - set(traits))
        if missing:
            print(f"[WARN] TARGET_TRAITS not found: {missing[:10]}")
    else:
        traits = list(Z_mat.columns)

    signatures, rows = {}, []
    for trait in traits:
        up_z, dn_z = extract_signature_up_down(Z_mat, trait, top_n=top_n)
        up_genes = [g for g in up_z.index.astype(str) if g in gene_universe]
        dn_genes = [g for g in dn_z.index.astype(str) if g in gene_universe]
        keep_up = len(up_genes) >= min_valid_genes
        keep_dn = len(dn_genes) >= min_valid_genes
        rows.append({
            "trait": trait, "n_up_requested": len(up_z), "n_down_requested": len(dn_z),
            "n_up_in_data": len(up_genes), "n_down_in_data": len(dn_genes),
            "keep_up": keep_up, "keep_down": keep_dn, "keep": keep_up or keep_dn,
        })
        if keep_up:
            signatures[f"{trait}__UP"] = [f"{g}+" for g in up_genes]
        if keep_dn:
            # DOWN genes are scored positively to represent a low-metabolite / inhibition-like program.
            signatures[f"{trait}__DOWN"] = [f"{g}+" for g in dn_genes]
    return signatures, pd.DataFrame(rows)


def compute_signature_score_df(adata, signatures, max_rank):
    score_blocks = []
    sig_names = list(signatures.keys())
    for batch in list_chunks(sig_names, UCELL_SIGNATURE_BATCH):
        sig_batch = {k: signatures[k] for k in batch}
        uc.compute_ucell_scores(
            adata, signatures=sig_batch, max_rank=max_rank, ties_method=TIES_METHOD,
            chunk_size=UCELL_CHUNK_SIZE, missing_genes=MISSING_GENES,
            w_neg=1.0, suffix="_UCell", n_jobs=N_JOBS_UCELL,
        )
        cols = [f"{k}_UCell" for k in batch]
        block = adata.obs[cols].copy()
        block.columns = batch
        score_blocks.append(block)
        adata.obs.drop(columns=cols, inplace=True, errors="ignore")
    if not score_blocks:
        raise ValueError("No UCell score blocks were generated.")
    out = pd.concat(score_blocks, axis=1)
    out.index = adata.obs_names
    return out


def zscore_series(x):
    x = pd.Series(x).astype(float)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def build_activity_inhibition_scores(raw_score_df):
    """
    Build three independent/composite gmMAP-perturb score matrices.

    Definitions:
        Activity_m   = z(UCell(UP_m genes))
        Inhibition_m = z(UCell(DOWN_m genes))
        Delta_m      = Activity_m - Inhibition_m

    Rationale:
        Activity_m and Inhibition_m are no longer forced to be exact opposites.
        Delta_m is retained as a high-vs-low metabolite-state axis.
    """
    traits_up = {c.replace("__UP", "") for c in raw_score_df.columns if c.endswith("__UP")}
    traits_dn = {c.replace("__DOWN", "") for c in raw_score_df.columns if c.endswith("__DOWN")}
    traits = sorted(traits_up.intersection(traits_dn))

    up_df = pd.DataFrame(index=raw_score_df.index)
    down_df = pd.DataFrame(index=raw_score_df.index)
    activity_df = pd.DataFrame(index=raw_score_df.index)
    inhibition_df = pd.DataFrame(index=raw_score_df.index)
    delta_df = pd.DataFrame(index=raw_score_df.index)

    for tr in traits:
        up = raw_score_df[f"{tr}__UP"].astype(float)
        dn = raw_score_df[f"{tr}__DOWN"].astype(float)

        activity = zscore_series(up)
        inhibition = zscore_series(dn)
        delta = activity - inhibition

        up_df[tr] = up
        down_df[tr] = dn
        activity_df[tr] = activity
        inhibition_df[tr] = inhibition
        delta_df[tr] = delta

    activity_df = activity_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    inhibition_df = inhibition_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    delta_df = delta_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return up_df, down_df, activity_df, inhibition_df, delta_df


def scale01(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return out
    xmin, xmax = np.nanmin(x[finite]), np.nanmax(x[finite])
    out[finite] = 0.0 if xmax == xmin else (x[finite] - xmin) / (xmax - xmin)
    return out


def choose_terminal_by_median_pt(meta_sub, config):
    t1_mask = meta_sub[CELLTYPE_COL].isin(config["target1"])
    t2_mask = meta_sub[CELLTYPE_COL].isin(config["target2"])
    if t1_mask.sum() == 0 or t2_mask.sum() == 0:
        raise ValueError("target1 or target2 has zero cells. Check cell-type names in TRAJECTORY_CONFIG.")

    med1 = float(np.nanmedian(meta_sub.loc[t1_mask, PSEUDOTIME_COL].values))
    med2 = float(np.nanmedian(meta_sub.loc[t2_mask, PSEUDOTIME_COL].values))
    pt01 = scale01(meta_sub[PSEUDOTIME_COL].values)

    # FateBias is always defined as target1-like minus target2-like.
    if med1 >= med2:
        fate_bias = 2.0 * pt01 - 1.0
        orientation = f"higher_pseudotime_more_{config['target1_name']}_like"
    else:
        fate_bias = -(2.0 * pt01 - 1.0)
        orientation = f"higher_pseudotime_more_{config['target2_name']}_like_flipped_to_{config['target1_name']}_positive"
    return fate_bias, med1, med2, orientation


def get_config_cells_from_meta(meta_df, config):
    all_types = list(config.get("root", [])) + list(config.get("intermediate", [])) + list(config.get("target1", [])) + list(config.get("target2", []))
    keep = meta_df[CELLTYPE_COL].isin(all_types) & np.isfinite(meta_df[PSEUDOTIME_COL].values)
    return meta_df.index[keep]


def spearman_one_trait(x, y, trait, min_pct_detected=0.01):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    detected = (np.abs(x[ok]) > 1e-12).mean()
    if detected < min_pct_detected or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return None
    rho, p = spearmanr(x[ok], y[ok], nan_policy="omit")
    return {"trait": trait, "rho": float(rho), "pvalue": float(p), "detect_rate": float(detected), "n_cells_used": int(ok.sum())}


def compute_fate_rerouting_for_config(activity_df, inhibition_df, delta_df, meta_df, config_name, config):
    cells = get_config_cells_from_meta(meta_df, config)
    if len(cells) < MIN_CELLS_PER_ANALYSIS:
        print(f"[WARN] {config_name}: too few cells ({len(cells)}). Skip.")
        return pd.DataFrame(), pd.DataFrame()

    meta_sub = meta_df.loc[cells].copy().sort_values(PSEUDOTIME_COL)
    fate_bias, med1, med2, orientation = choose_terminal_by_median_pt(meta_sub, config)
    meta_sub["pt01"] = scale01(meta_sub[PSEUDOTIME_COL].values)
    meta_sub["fate_bias_target1_minus_target2"] = fate_bias

    inhibition_sub = inhibition_df.loc[meta_sub.index]
    activity_sub = activity_df.loc[meta_sub.index]
    delta_sub = delta_df.loc[meta_sub.index]
    traits = list(inhibition_sub.columns)

    inhib_results = Parallel(n_jobs=N_JOBS_COR)(
        delayed(spearman_one_trait)(inhibition_sub[tr].values, meta_sub["fate_bias_target1_minus_target2"].values, tr, MIN_PCT_DETECTED)
        for tr in traits
    )
    inhib_results = [r for r in inhib_results if r is not None]
    if not inhib_results:
        return pd.DataFrame(), meta_sub

    act_results = Parallel(n_jobs=N_JOBS_COR)(
        delayed(spearman_one_trait)(activity_sub[tr].values, meta_sub["fate_bias_target1_minus_target2"].values, tr, MIN_PCT_DETECTED)
        for tr in traits
    )
    act_results = [r for r in act_results if r is not None]

    delta_results = Parallel(n_jobs=N_JOBS_COR)(
        delayed(spearman_one_trait)(delta_sub[tr].values, meta_sub["fate_bias_target1_minus_target2"].values, tr, MIN_PCT_DETECTED)
        for tr in traits
    )
    delta_results = [r for r in delta_results if r is not None]

    res = pd.DataFrame(inhib_results).rename(columns={
        "rho": "rho_inhibition_vs_fatebias", "pvalue": "p_inhibition_vs_fatebias",
        "detect_rate": "detect_rate_inhibition",
    })
    res["q_inhibition_vs_fatebias"] = bh_fdr(res["p_inhibition_vs_fatebias"].values)

    if act_results:
        act = pd.DataFrame(act_results).rename(columns={
            "rho": "rho_activity_vs_fatebias", "pvalue": "p_activity_vs_fatebias",
            "detect_rate": "detect_rate_activity",
        })
        act["q_activity_vs_fatebias"] = bh_fdr(act["p_activity_vs_fatebias"].values)
        res = res.merge(act[["trait", "rho_activity_vs_fatebias", "p_activity_vs_fatebias", "q_activity_vs_fatebias", "detect_rate_activity"]], on="trait", how="left")

    if delta_results:
        delt = pd.DataFrame(delta_results).rename(columns={
            "rho": "rho_delta_vs_fatebias", "pvalue": "p_delta_vs_fatebias",
            "detect_rate": "detect_rate_delta",
        })
        delt["q_delta_vs_fatebias"] = bh_fdr(delt["p_delta_vs_fatebias"].values)
        res = res.merge(delt[["trait", "rho_delta_vs_fatebias", "p_delta_vs_fatebias", "q_delta_vs_fatebias", "detect_rate_delta"]], on="trait", how="left")

    t1, t2 = config["target1_name"], config["target2_name"]
    res["config"] = config_name
    res["target1_name"] = t1
    res["target2_name"] = t2
    res["root_celltypes"] = ";".join(config.get("root", []))
    res["intermediate_celltypes"] = ";".join(config.get("intermediate", []))
    res["target1_celltypes"] = ";".join(config.get("target1", []))
    res["target2_celltypes"] = ";".join(config.get("target2", []))
    res["n_cells_config"] = int(meta_sub.shape[0])
    res["median_pt_target1"] = med1
    res["median_pt_target2"] = med2
    res["fatebias_orientation"] = orientation

    res["predicted_direction"] = np.where(res["rho_inhibition_vs_fatebias"] < 0, f"{t1}_to_{t2}", np.where(res["rho_inhibition_vs_fatebias"] > 0, f"{t2}_to_{t1}", "Neutral"))
    q = res["q_inhibition_vs_fatebias"].astype(float).clip(lower=1e-300)
    res["impact_strength"] = np.abs(res["rho_inhibition_vs_fatebias"].astype(float)) * (-np.log10(q))
    res["impact_score_target1_to_target2"] = (-res["rho_inhibition_vs_fatebias"].astype(float)) * (-np.log10(q))
    res["impact_score_target2_to_target1"] = (res["rho_inhibition_vs_fatebias"].astype(float)) * (-np.log10(q))
    res["evidence_class"] = "weak_or_neutral"
    res.loc[(res["q_inhibition_vs_fatebias"] <= 0.05) & (res["rho_inhibition_vs_fatebias"] < -0.2), "evidence_class"] = f"strong_{t1}_to_{t2}"
    res.loc[(res["q_inhibition_vs_fatebias"] <= 0.05) & (res["rho_inhibition_vs_fatebias"] > 0.2), "evidence_class"] = f"strong_{t2}_to_{t1}"
    return res.sort_values("impact_strength", ascending=False).reset_index(drop=True), meta_sub




def _get_branch_values(meta_sub, score_df, trait, config, branch_name):
    """
    Return branch-specific pseudotime and score values.

    target1 branch = root + intermediate + target1
    target2 branch = root + intermediate + target2
    """
    root_types = list(config.get("root", []))
    inter_types = list(config.get("intermediate", []))

    if branch_name == config["target1_name"]:
        branch_types = root_types + inter_types + list(config.get("target1", []))
    elif branch_name == config["target2_name"]:
        branch_types = root_types + inter_types + list(config.get("target2", []))
    else:
        raise ValueError(f"Unknown branch_name: {branch_name}")

    cells = meta_sub.index[
        meta_sub[CELLTYPE_COL].isin(branch_types) &
        np.isfinite(meta_sub[PSEUDOTIME_COL].values)
    ]

    if len(cells) == 0 or trait not in score_df.columns:
        return None, None

    sub = meta_sub.loc[cells].copy().sort_values(PSEUDOTIME_COL)
    pt01 = scale01(sub[PSEUDOTIME_COL].values)
    values = score_df.loc[sub.index, trait].astype(float).values

    ok = np.isfinite(pt01) & np.isfinite(values)
    return pt01[ok], values[ok]


def _terminal_mean_from_branch(pt01, values, terminal_fraction=0.25):
    """
    Mean score in the terminal pseudotime segment of one branch.
    """
    if pt01 is None or values is None or len(values) < 10:
        return np.nan, 0

    cutoff = np.nanquantile(pt01, 1.0 - terminal_fraction)
    keep = pt01 >= cutoff
    if keep.sum() < 5:
        return np.nan, int(keep.sum())

    return float(np.nanmean(values[keep])), int(keep.sum())


def add_branch_difference_metrics(res, meta_sub, activity_df, inhibition_df, delta_df, config):
    """
    Add target1-vs-target2 branch-difference metrics for Activity, Inhibition and Delta.

    Difference is defined as:
        diff_target2_minus_target1 = terminal_mean(target2 branch) - terminal_mean(target1 branch)

    For target1_to_target2 prediction:
        expected Inhibition difference > 0
        expected Activity and/or Delta difference < 0

    For target2_to_target1 prediction:
        expected Inhibition difference < 0
        expected Activity and/or Delta difference > 0
    """
    if res is None or res.empty:
        return res

    out = res.copy()
    t1 = config["target1_name"]
    t2 = config["target2_name"]

    rows = []
    for tr in out["trait"].astype(str).tolist():
        row = {"trait": tr}

        for score_name, score_df in [
            ("activity", activity_df),
            ("inhibition", inhibition_df),
            ("delta", delta_df),
        ]:
            pt1, y1 = _get_branch_values(meta_sub, score_df, tr, config, t1)
            pt2, y2 = _get_branch_values(meta_sub, score_df, tr, config, t2)

            mean1, n1 = _terminal_mean_from_branch(
                pt1, y1, terminal_fraction=BRANCH_DIFF_TERMINAL_FRACTION
            )
            mean2, n2 = _terminal_mean_from_branch(
                pt2, y2, terminal_fraction=BRANCH_DIFF_TERMINAL_FRACTION
            )

            diff = mean2 - mean1 if np.isfinite(mean1) and np.isfinite(mean2) else np.nan

            row[f"terminal_{score_name}_mean_{t1}"] = mean1
            row[f"terminal_{score_name}_mean_{t2}"] = mean2
            row[f"terminal_{score_name}_n_{t1}"] = n1
            row[f"terminal_{score_name}_n_{t2}"] = n2
            row[f"terminal_{score_name}_diff_{t2}_minus_{t1}"] = diff
            row[f"abs_terminal_{score_name}_diff"] = abs(diff) if np.isfinite(diff) else np.nan

        rows.append(row)

    diff_df = pd.DataFrame(rows)
    out = out.merge(diff_df, on="trait", how="left")

    inhib_diff = pd.to_numeric(out[f"terminal_inhibition_diff_{t2}_minus_{t1}"], errors="coerce")
    activity_diff = pd.to_numeric(out[f"terminal_activity_diff_{t2}_minus_{t1}"], errors="coerce")
    delta_diff = pd.to_numeric(out[f"terminal_delta_diff_{t2}_minus_{t1}"], errors="coerce")

    pred = out["predicted_direction"].astype(str)
    is_t1_to_t2 = pred.eq(f"{t1}_to_{t2}")
    is_t2_to_t1 = pred.eq(f"{t2}_to_{t1}")

    out["branchdiff_inhibition_support"] = (
        (is_t1_to_t2 & (inhib_diff >= BRANCH_DIFF_MIN_ABS_EFFECT)) |
        (is_t2_to_t1 & (inhib_diff <= -BRANCH_DIFF_MIN_ABS_EFFECT))
    )

    out["branchdiff_activity_support"] = (
        (is_t1_to_t2 & (activity_diff <= -BRANCH_DIFF_MIN_ABS_EFFECT)) |
        (is_t2_to_t1 & (activity_diff >= BRANCH_DIFF_MIN_ABS_EFFECT))
    )

    out["branchdiff_delta_support"] = (
        (is_t1_to_t2 & (delta_diff <= -BRANCH_DIFF_MIN_ABS_EFFECT)) |
        (is_t2_to_t1 & (delta_diff >= BRANCH_DIFF_MIN_ABS_EFFECT))
    )

    out["branch_difference_score"] = np.nan_to_num(np.abs(inhib_diff), nan=0.0)
    out.loc[out["branchdiff_activity_support"], "branch_difference_score"] += (
        0.5 * np.nan_to_num(np.abs(activity_diff[out["branchdiff_activity_support"]]), nan=0.0)
    )
    out.loc[out["branchdiff_delta_support"], "branch_difference_score"] += (
        0.5 * np.nan_to_num(np.abs(delta_diff[out["branchdiff_delta_support"]]), nan=0.0)
    )

    opposite_inhibition = (
        (is_t1_to_t2 & (inhib_diff <= -BRANCH_DIFF_MIN_ABS_EFFECT)) |
        (is_t2_to_t1 & (inhib_diff >= BRANCH_DIFF_MIN_ABS_EFFECT))
    )
    out.loc[opposite_inhibition, "branch_difference_score"] *= -1.0
    out["branchdiff_inhibition_opposite_to_prediction"] = opposite_inhibition

    return out


def select_traits_for_plotting(res_df, top_n=6, manual_traits=None, rank_by="branch_difference"):
    """
    Select automatically ranked traits plus manually specified traits.

    Automatic ranking:
      - branch_difference: largest target1-vs-target2 branch score difference
      - impact_strength: original correlation-based impact strength
    """
    if res_df is None or res_df.empty:
        return pd.DataFrame()

    df = res_df.copy()

    if rank_by == "branch_difference" and "branch_difference_score" in df.columns:
        sort_cols = ["branch_difference_score", "impact_strength"]
        ascending = [False, False]
    elif "support_score" in df.columns:
        sort_cols = ["support_score", "impact_strength"]
        ascending = [False, False]
    else:
        sort_cols = ["impact_strength"]
        ascending = [False]

    auto_df = df.sort_values(sort_cols, ascending=ascending).head(top_n).copy()

    if manual_traits is None:
        manual_traits = []

    manual_traits = [str(x) for x in manual_traits if str(x).strip()]
    if len(manual_traits) > 0:
        manual_df = df[df["trait"].astype(str).isin(manual_traits)].copy()
        out = pd.concat([auto_df, manual_df], axis=0, ignore_index=True)
        out = out.drop_duplicates(subset=["trait", "config"], keep="first")
    else:
        out = auto_df

    return out.reset_index(drop=True)


def classify_support_patterns(res):
    """
    Classify candidates into:
      bidirectional_supported:
          DOWN/Inhibition supports predicted rerouting, and Activity or Delta supports the opposite active-state axis.
      single_DOWN_supported:
          Only DOWN/Inhibition supports predicted rerouting; Activity/Delta do not significantly support it.
      DOWN_supported_but_activity_opposite:
          DOWN supports predicted rerouting, but Activity points in the same direction as the inhibited endpoint.
      activity_or_delta_only_no_DOWN:
          Activity or Delta has signal but DOWN/Inhibition does not support the perturbation hypothesis.
      weak_or_unclear:
          no clear support.
    """
    if res is None or res.empty:
        return res

    out = res.copy()

    for col in [
        "rho_activity_vs_fatebias", "q_activity_vs_fatebias",
        "rho_inhibition_vs_fatebias", "q_inhibition_vs_fatebias",
        "rho_delta_vs_fatebias", "q_delta_vs_fatebias",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    t1 = str(out["target1_name"].iloc[0]) if "target1_name" in out.columns else "target1"
    t2 = str(out["target2_name"].iloc[0]) if "target2_name" in out.columns else "target2"

    pred = out["predicted_direction"].astype(str)
    is_t1_to_t2 = pred.eq(f"{t1}_to_{t2}")
    is_t2_to_t1 = pred.eq(f"{t2}_to_{t1}")

    rho_i = pd.to_numeric(out["rho_inhibition_vs_fatebias"], errors="coerce")
    q_i = pd.to_numeric(out["q_inhibition_vs_fatebias"], errors="coerce")
    rho_a = pd.to_numeric(out["rho_activity_vs_fatebias"], errors="coerce")
    q_a = pd.to_numeric(out["q_activity_vs_fatebias"], errors="coerce")
    rho_d = pd.to_numeric(out["rho_delta_vs_fatebias"], errors="coerce")
    q_d = pd.to_numeric(out["q_delta_vs_fatebias"], errors="coerce")

    # FateBias = target1 - target2.
    # For target1_to_target2, inhibition support is rho_inhibition < 0.
    # For target2_to_target1, inhibition support is rho_inhibition > 0.
    inhibition_support = (
        (is_t1_to_t2 & (rho_i <= -SUPPORT_RHO_MIN) & (q_i <= SUPPORT_Q_CUTOFF)) |
        (is_t2_to_t1 & (rho_i >= SUPPORT_RHO_MIN) & (q_i <= SUPPORT_Q_CUTOFF))
    )

    # Delta = Activity - Inhibition. If inhibition pushes target1->target2,
    # Delta/active axis should preferentially remain target1-like: rho_delta > 0.
    # Opposite for target2->target1.
    delta_support = (
        (is_t1_to_t2 & (rho_d >= DELTA_SUPPORT_RHO_MIN) & (q_d <= DELTA_SUPPORT_Q_CUTOFF)) |
        (is_t2_to_t1 & (rho_d <= -DELTA_SUPPORT_RHO_MIN) & (q_d <= DELTA_SUPPORT_Q_CUTOFF))
    )

    # Activity = UP program. Activity support means the active UP program points to the original state.
    activity_support = (
        (is_t1_to_t2 & (rho_a >= DELTA_SUPPORT_RHO_MIN) & (q_a <= DELTA_SUPPORT_Q_CUTOFF)) |
        (is_t2_to_t1 & (rho_a <= -DELTA_SUPPORT_RHO_MIN) & (q_a <= DELTA_SUPPORT_Q_CUTOFF))
    )

    # Caution: activity points toward the inhibited endpoint rather than the original active endpoint.
    activity_opposite = (
        (is_t1_to_t2 & (rho_a <= -DELTA_SUPPORT_RHO_MIN) & (q_a <= DELTA_SUPPORT_Q_CUTOFF)) |
        (is_t2_to_t1 & (rho_a >= DELTA_SUPPORT_RHO_MIN) & (q_a <= DELTA_SUPPORT_Q_CUTOFF))
    )

    out["support_inhibition_DOWN"] = inhibition_support
    out["support_activity_UP"] = activity_support
    out["support_delta"] = delta_support
    out["activity_opposite_to_prediction"] = activity_opposite

    out["support_class"] = "weak_or_unclear"
    out.loc[inhibition_support & (delta_support | activity_support), "support_class"] = "bidirectional_supported"
    out.loc[inhibition_support & (~delta_support) & (~activity_support) & (~activity_opposite), "support_class"] = "single_DOWN_supported"
    out.loc[inhibition_support & activity_opposite, "support_class"] = "DOWN_supported_but_activity_opposite"
    out.loc[(~inhibition_support) & (delta_support | activity_support), "support_class"] = "activity_or_delta_only_no_DOWN"

    out["support_score"] = 0.0
    out.loc[inhibition_support, "support_score"] += np.abs(rho_i[inhibition_support].fillna(0))
    out.loc[delta_support, "support_score"] += 0.5 * np.abs(rho_d[delta_support].fillna(0))
    out.loc[activity_support, "support_score"] += 0.5 * np.abs(rho_a[activity_support].fillna(0))
    out.loc[activity_opposite, "support_score"] -= 0.5 * np.abs(rho_a[activity_opposite].fillna(0))

    class_order = {
        "bidirectional_supported": 0,
        "single_DOWN_supported": 1,
        "DOWN_supported_but_activity_opposite": 2,
        "activity_or_delta_only_no_DOWN": 3,
        "weak_or_unclear": 4,
    }
    out["support_class_order"] = out["support_class"].map(class_order).fillna(9).astype(int)
    out = out.sort_values(
        ["support_class_order", "impact_strength", "support_score"],
        ascending=[True, False, False]
    ).drop(columns=["support_class_order"]).reset_index(drop=True)

    return out



def save_pdf_ai_editable(fig, path, **kwargs):
    """
    Save a PDF with TrueType fonts embedded and editable text where possible.
    For UMAP labels, avoid path effects to prevent text-to-outline conversion in Adobe Illustrator.
    """
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("metadata", {"Creator": "matplotlib; pdf.fonttype=42; editable text"})
    fig.savefig(path, **kwargs)


# =========================
# Plotting
# =========================
def save_top_barplot(res_df, config_name, outdir, top_n=30):
    if res_df is None or res_df.empty:
        return
    df = res_df.head(top_n).iloc[::-1].copy()
    fig_h = max(4.0, 0.22 * df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ax.barh(df["trait"].astype(str), df["impact_strength"].astype(float))
    ax.set_xlabel("Impact strength")
    ax.set_ylabel("")
    ax.set_title(f"{config_name}: top gmMAP-perturb candidates")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{PREFIX}_{config_name}_TopImpact_barplot.pdf"), bbox_inches="tight")
    plt.close(fig)


def _short_celltype_label(x):
    """
    Convert long cell-type names into compact labels for UMAP annotation.
    """
    x = str(x)
    mapping = {
        "NPCd: nephron progenitor cells d": "NPCd",
        "RVCSBa: renal vesicle/comma-shapedbody a": "RVCSBa",
        "RVCSB b: renal vesicle/comma-shapedbody b": "RVCSBb",
        "SSBpr: s-shaped body proximal precursor cells": "SSBpr",
        "ErPrT: early proximal tubule": "ErPrT",
        "SSBm/d: s-shaped body medial/distal": "SSBm/d",
        "DTLH: distal tubule/loop of Henle": "DTLH",
        "SSBpod: s-shaped body podocyte precursor cells": "SSBpod",
        "Pod: podocyte": "Pod",
    }
    if x in mapping:
        return mapping[x]
    if ":" in x:
        return x.split(":")[0].strip()
    return x[:18]


def _celltype_centroid(coords, ct_series, celltypes):
    """
    Median UMAP coordinate for one or several cell types.
    """
    mask = ct_series.isin(celltypes)
    if mask.sum() == 0:
        return None
    return np.nanmedian(coords.loc[mask, ["UMAP1", "UMAP2"]].values, axis=0)


def _draw_curved_arrow(ax, points, color, label=None, linestyle="-", linewidth=2.2, alpha=0.95, rad=0.12):
    """
    Draw a curved multi-segment arrow through a list of UMAP coordinates.
    """
    pts = [np.asarray(p, dtype=float) for p in points if p is not None and np.all(np.isfinite(p))]
    if len(pts) < 2:
        return

    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        arrow = FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>" if i == len(pts) - 2 else "-",
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=14,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            alpha=alpha,
            zorder=8,
        )
        ax.add_patch(arrow)

    if label is not None:
        mid = pts[len(pts) // 2]
        txt = ax.text(
            mid[0], mid[1], label,
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
        )
        if USE_TEXT_PATH_EFFECTS:
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()
            ])


def save_umap_trait_plot(adata, meta_sub, inhibition_df, trait, config_name, config, outdir):
    """
    UMAP plot for one metabolite perturbation candidate.

    Modified visualization:
    1. No cell-type outline circles are drawn.
    2. Selected cell types are annotated by text labels at median UMAP positions.
    3. A blue curved arrow marks the original/default differentiation direction.
    4. A red curved dashed arrow marks the predicted direction under inhibition-like state.
    5. Points are still colored by inhibition-like score.

    Note:
    In the previous version, terminal cells looked gray because target/root cells were overlaid with
    open gray/black circles on top of the inhibition-score scatter. This function removes that overlay.
    """
    if UMAP_KEY not in adata.obsm.keys() or trait not in inhibition_df.columns:
        return

    common = meta_sub.index.intersection(inhibition_df.index)
    if len(common) == 0:
        return

    coords = pd.DataFrame(
        adata.obsm[UMAP_KEY],
        index=adata.obs_names,
        columns=["UMAP1", "UMAP2"]
    ).loc[common]

    score = inhibition_df.loc[common, trait].astype(float)
    ct = adata.obs.loc[common, CELLTYPE_COL].astype(str)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    finite_score = np.isfinite(score.values)
    if finite_score.sum() == 0:
        return

    vmin = np.nanpercentile(score.values[finite_score], 1)
    vmax = np.nanpercentile(score.values[finite_score], 99)

    # Draw all selected trajectory cells colored by inhibition-like score.
    # No outline-circle overlay is used, so terminal cells will not become gray.
    sca = ax.scatter(
        coords["UMAP1"],
        coords["UMAP2"],
        c=score.values,
        s=7,
        linewidths=0,
        alpha=0.88,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )
    cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Inhibition-like score", fontsize=10)

    # Annotate root, intermediate, target1 and target2 cell-type labels.
    all_label_celltypes = (
        list(config.get("root", [])) +
        list(config.get("intermediate", [])) +
        list(config.get("target1", [])) +
        list(config.get("target2", []))
    )

    # Small manual offset to reduce label overlap. You can adjust if needed.
    label_offsets = {
        "NPCd": (-0.15, -0.75),
        "RVCSBa": (-0.35, -0.55),
        "RVCSBb": (-0.35, 0.35),
        "SSBpr": (-0.30, 0.55),
        "ErPrT": (0.25, 0.45),
        "SSBm/d": (0.35, 0.45),
        "DTLH": (0.35, -0.55),
        "SSBpod": (-0.20, 0.55),
        "Pod": (0.25, 0.60),
    }

    for celltype in all_label_celltypes:
        pos = _celltype_centroid(coords, ct, [celltype])
        if pos is None:
            continue
        lab = _short_celltype_label(celltype)
        dx, dy = label_offsets.get(lab, (0.15, 0.25))
        txt = ax.text(
            pos[0] + dx, pos[1] + dy,
            lab,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
        )
        if USE_TEXT_PATH_EFFECTS:
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.8, foreground="white"),
                path_effects.Normal()
            ])

    # Build centroids for arrow paths.
    root_types = list(config.get("root", []))
    inter_types = list(config.get("intermediate", []))
    target1_types = list(config.get("target1", []))
    target2_types = list(config.get("target2", []))

    # Original/default direction:
    # root -> intermediate -> target1. This is the assumed untreated/default trajectory.
    # If you want the original direction to be target2, swap target1/target2 in TRAJECTORY_CONFIG.
    normal_path_celltype_groups = []
    if root_types:
        normal_path_celltype_groups.append(root_types)
    for x in inter_types:
        normal_path_celltype_groups.append([x])
    if target1_types:
        normal_path_celltype_groups.append(target1_types)

    normal_points = [
        _celltype_centroid(coords, ct, group)
        for group in normal_path_celltype_groups
    ]

    _draw_curved_arrow(
        ax,
        normal_points,
        color="#2166ac",
        label=f"normal → {config['target1_name']}",
        linestyle="-",
        linewidth=2.4,
        alpha=0.96,
        rad=0.10,
    )

    # Predicted inhibited direction:
    # Decide from the actual rerouting result direction when available in config is not enough.
    # Here the plot function is called for top traits from one config. We use the standard gmMAP-perturb interpretation:
    #   high inhibition-like state shifts away from target1 if rho < 0; however this function does not receive rho.
    # Therefore we draw the inhibition direction as root/intermediate -> target2, which matches the common
    # target1_to_target2 rerouting visualization. If the result table says target2_to_target1, swap target1/target2 in config
    # or use the optional direction-aware version below.
    inhibition_path_celltype_groups = []
    if root_types:
        inhibition_path_celltype_groups.append(root_types)
    for x in inter_types:
        inhibition_path_celltype_groups.append([x])
    if target2_types:
        inhibition_path_celltype_groups.append(target2_types)

    inhibition_points = [
        _celltype_centroid(coords, ct, group)
        for group in inhibition_path_celltype_groups
    ]

    _draw_curved_arrow(
        ax,
        inhibition_points,
        color="#d73027",
        label=f"inhibition → {config['target2_name']}",
        linestyle="--",
        linewidth=2.4,
        alpha=0.96,
        rad=-0.16,
    )

    # Add compact legend-like explanation inside figure.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        xlim[0] + 0.02 * (xlim[1] - xlim[0]),
        ylim[0] + 0.05 * (ylim[1] - ylim[0]),
        "Blue: original differentiation trajectory\n"
        f"Red dashed: predicted direction under {trait} inhibition-like state",
        fontsize=7.5,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.88),
        zorder=20,
    )

    ax.set_title(f"{config_name} | {trait}", fontsize=12)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    safe_trait = str(trait).replace("/", "_").replace(" ", "_")
    save_pdf_ai_editable(
        fig,
        os.path.join(outdir, f"{PREFIX}_{config_name}_{safe_trait}_UMAP_inhibition.pdf")
    )
    plt.close(fig)



def save_fatebias_curve(meta_sub, inhibition_df, activity_df, trait, config_name, outdir):
    if trait not in inhibition_df.columns:
        return
    common = meta_sub.index.intersection(inhibition_df.index)
    df = pd.DataFrame({
        "pt01": meta_sub.loc[common, "pt01"].astype(float),
        "fate_bias": meta_sub.loc[common, "fate_bias_target1_minus_target2"].astype(float),
        "inhibition": inhibition_df.loc[common, trait].astype(float),
        "activity": activity_df.loc[common, trait].astype(float) if trait in activity_df.columns else np.nan,
    }).dropna(subset=["pt01", "fate_bias", "inhibition"])
    if df.shape[0] < 20:
        return
    n_bins = min(20, max(4, df.shape[0] // 20))
    df["bin"] = pd.qcut(df["pt01"], q=n_bins, duplicates="drop")
    binned = df.groupby("bin", observed=True).agg(pt01=("pt01", "mean"), inhibition=("inhibition", "mean"), activity=("activity", "mean"), fate_bias=("fate_bias", "mean")).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(5.8, 4.2))
    ax1.plot(binned["pt01"], binned["inhibition"], marker="o", markersize=3, label="Inhibition-like score")
    if np.isfinite(binned["activity"]).any():
        ax1.plot(binned["pt01"], binned["activity"], marker="o", markersize=3, label="Activity score")
    ax1.set_xlabel("Scaled Monocle pseudotime")
    ax1.set_ylabel("gmMAP score")
    ax1.legend(frameon=False, fontsize=8, loc="best")
    ax2 = ax1.twinx()
    ax2.plot(binned["pt01"], binned["fate_bias"], linestyle="--", marker="s", markersize=3)
    ax2.set_ylabel("FateBias target1 − target2")
    ax1.set_title(f"{config_name} | {trait}", fontsize=10)
    fig.tight_layout()
    safe_trait = str(trait).replace("/", "_").replace(" ", "_")
    fig.savefig(os.path.join(outdir, f"{PREFIX}_{config_name}_{safe_trait}_pseudotime_fatebias_curve.pdf"), bbox_inches="tight")
    plt.close(fig)




def _branch_cells_for_comparison(meta_sub, config, branch_name):
    """
    Return cells for target1 or target2 branch:
        target1 branch = root + intermediate + target1
        target2 branch = root + intermediate + target2
    """
    root_types = list(config.get("root", []))
    inter_types = list(config.get("intermediate", []))

    if branch_name == config["target1_name"]:
        branch_types = root_types + inter_types + list(config.get("target1", []))
    elif branch_name == config["target2_name"]:
        branch_types = root_types + inter_types + list(config.get("target2", []))
    else:
        raise ValueError(f"Unknown branch_name: {branch_name}")

    cells = meta_sub.index[
        meta_sub[CELLTYPE_COL].isin(branch_types) &
        np.isfinite(meta_sub[PSEUDOTIME_COL].values)
    ]
    return cells


def _binned_mean_curve(x, y, n_bins=35):
    """
    Bin x in [0, 1] and return mean y per bin.
    """
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[0] < 20:
        return pd.DataFrame(columns=["x", "y", "n"])

    n_bins = min(n_bins, max(6, df.shape[0] // 25))
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.digitize(df["x"].values, bins, right=False) - 1
    which = np.clip(which, 0, n_bins - 1)
    df["bin"] = which

    out = (
        df.groupby("bin", observed=True)
        .agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return out


def save_target1_target2_comparison_curves(
    meta_sub,
    inhibition_df,
    activity_df,
    delta_df,
    trait,
    config_name,
    config,
    outdir,
    predicted_direction=None,
):
    """
    Draw target1 and target2 branches separately in the style of the reference figure:
        left panel:  activity score vs branch-specific pseudotime
        right panel: inhibition-like score vs branch-specific pseudotime

    Each panel contains target1-branch cells/mean and target2-branch cells/mean.
    """
    if trait not in inhibition_df.columns or trait not in activity_df.columns or trait not in delta_df.columns:
        return

    target1_name = config["target1_name"]
    target2_name = config["target2_name"]

    branch_defs = [
        (target1_name, "#1f77b4"),
        (target2_name, "#ff7f0e"),
    ]

    if predicted_direction is None:
        predicted_direction = f"{target1_name} → {target2_name}"

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.3), sharex=True)

    for ax, score_label, score_df in [
        (axes[0], "Activity", activity_df),
        (axes[1], "Inhibition", inhibition_df),
        (axes[2], "Delta", delta_df),
    ]:
        for branch_name, color in branch_defs:
            cells = _branch_cells_for_comparison(meta_sub, config, branch_name)
            if len(cells) < 20:
                continue

            sub = meta_sub.loc[cells].copy().sort_values(PSEUDOTIME_COL)
            pt01 = scale01(sub[PSEUDOTIME_COL].values)
            y = score_df.loc[sub.index, trait].astype(float).values

            valid = np.isfinite(pt01) & np.isfinite(y)
            if valid.sum() < 20:
                continue

            ax.scatter(
                pt01[valid],
                y[valid],
                s=12,
                alpha=0.16,
                color=color,
                linewidths=0,
                label=f"{branch_name} cells",
            )

            curve = _binned_mean_curve(pt01[valid], y[valid], n_bins=35)
            if curve.shape[0] > 0:
                ax.plot(
                    curve["x"],
                    curve["y"],
                    color=color,
                    linewidth=2.6,
                    label=f"{branch_name} mean",
                )

        ax.axhline(0, color="0.35", linewidth=0.8, alpha=0.55)
        ax.set_xlabel("Lineage pseudotime (0–1)")
        ax.set_ylabel(f"{score_label} score")
        ax.set_title(f"{score_label}: {target1_name} vs {target2_name}", fontsize=13)
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        f"gmMAP-perturb pseudotime profile: {trait}\n"
        f"Predicted rerouting: {predicted_direction}",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()

    safe_trait = str(trait).replace("/", "_").replace(" ", "_")
    fig.savefig(
        os.path.join(outdir, f"{PREFIX}_{config_name}_{safe_trait}_target1_target2_comparison_curves.pdf"),
        bbox_inches="tight"
    )
    plt.close(fig)


def save_branch_separated_curves(adata, meta_sub, inhibition_df, activity_df, delta_df, trait, config_name, config, outdir):
    """
    Plot target1 and target2 branches separately, closer to the old gmMAP_perturb_pseudotime.py style.

    Instead of mixing root + intermediate + target1 + target2 into one curve, this function builds:
        branch1 = root + intermediate + target1
        branch2 = root + intermediate + target2

    Then each branch gets its own scaled pseudotime and its own mean-binned curve.
    """
    if trait not in inhibition_df.columns or trait not in activity_df.columns:
        return

    root_types = list(config.get("root", []))
    inter_types = list(config.get("intermediate", []))
    target1_types = list(config.get("target1", []))
    target2_types = list(config.get("target2", []))

    branches = {
        config["target1_name"]: root_types + inter_types + target1_types,
        config["target2_name"]: root_types + inter_types + target2_types,
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)

    for ax, (branch_name, branch_celltypes) in zip(axes, branches.items()):
        cells = meta_sub.index[
            meta_sub[CELLTYPE_COL].isin(branch_celltypes) &
            np.isfinite(meta_sub[PSEUDOTIME_COL].values)
        ]

        if len(cells) < 20:
            ax.set_title(f"{branch_name}: too few cells")
            ax.axis("off")
            continue

        sub = meta_sub.loc[cells].copy()
        sub = sub.sort_values(PSEUDOTIME_COL)
        sub["pt_branch_01"] = scale01(sub[PSEUDOTIME_COL].values)

        x = sub["pt_branch_01"].values
        y_inhib = inhibition_df.loc[sub.index, trait].astype(float).values
        y_act = activity_df.loc[sub.index, trait].astype(float).values
        y_delta = delta_df.loc[sub.index, trait].astype(float).values

        n_bins = min(20, max(6, len(sub) // 30))
        sub_plot = pd.DataFrame({
            "pt": x,
            "inhibition": y_inhib,
            "activity": y_act,
            "delta": y_delta,
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if sub_plot.shape[0] < 20:
            ax.set_title(f"{branch_name}: too few valid cells")
            ax.axis("off")
            continue

        sub_plot["bin"] = pd.qcut(sub_plot["pt"], q=n_bins, duplicates="drop")

        binned = (
            sub_plot
            .groupby("bin", observed=True)
            .agg(
                pt=("pt", "mean"),
                inhibition=("inhibition", "mean"),
                activity=("activity", "mean"),
                delta=("delta", "mean"),
                n=("pt", "size"),
            )
            .reset_index(drop=True)
        )

        ax.plot(binned["pt"], binned["inhibition"], marker="o", linewidth=2,
                markersize=4, label="Inhibition-like score")
        ax.plot(binned["pt"], binned["activity"], marker="o", linewidth=2,
                markersize=4, label="Activity score")
        ax.plot(binned["pt"], binned["delta"], marker="o", linewidth=2,
                markersize=4, label="Delta score")

        terminal_mask = sub[CELLTYPE_COL].isin(
            target1_types if branch_name == config["target1_name"] else target2_types
        )
        if terminal_mask.sum() > 0:
            terminal_pt = np.nanmedian(sub.loc[terminal_mask, "pt_branch_01"].values)
            ax.axvline(terminal_pt, linestyle="--", linewidth=1, alpha=0.55)

        ax.axhline(0, linewidth=0.8, alpha=0.35)
        ax.set_title(f"{branch_name} branch")
        ax.set_xlabel("Branch-specific scaled Monocle pseudotime")
        ax.grid(alpha=0.20)

    axes[0].set_ylabel("gmMAP score")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"{config_name} | {trait}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    safe_trait = str(trait).replace("/", "_").replace(" ", "_")
    fig.savefig(os.path.join(outdir, f"{PREFIX}_{config_name}_{safe_trait}_branch_separated_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_top_traits_for_config(adata, res_df, meta_sub, inhibition_df, activity_df, delta_df, config_name, config, outdir, top_n=6):
    if res_df is None or res_df.empty:
        return

    top_rows = select_traits_for_plotting(
        res_df=res_df,
        top_n=top_n,
        manual_traits=MANUAL_PLOT_TRAITS,
        rank_by=AUTO_PLOT_RANK_BY,
    )

    for _, row in top_rows.iterrows():
        tr = str(row["trait"])
        predicted_direction = str(row.get("predicted_direction", f"{config['target1_name']}_to_{config['target2_name']}"))
        predicted_direction_for_title = predicted_direction.replace("_to_", " → ").replace("_", " ")

        save_umap_trait_plot(adata, meta_sub, inhibition_df, tr, config_name, config, outdir)

        # Original mixed root-terminal curve, kept for reference.
        save_fatebias_curve(meta_sub, inhibition_df, activity_df, tr, config_name, outdir)

        # New target1-vs-target2 comparison curve:
        # left panel = activity, right panel = inhibition; each panel contains target1 and target2 curves.
        save_target1_target2_comparison_curves(
            meta_sub=meta_sub,
            inhibition_df=inhibition_df,
            activity_df=activity_df,
            delta_df=delta_df,
            trait=tr,
            config_name=config_name,
            config=config,
            outdir=outdir,
            predicted_direction=predicted_direction_for_title,
        )

        # Older two-branch plot is still exported for reference.
        save_branch_separated_curves(
            adata=adata,
            meta_sub=meta_sub,
            inhibition_df=inhibition_df,
            activity_df=activity_df,
            delta_df=delta_df,
            trait=tr,
            config_name=config_name,
            config=config,
            outdir=outdir,
        )



def save_support_class_barplot(res_df, config_name, support_class, outdir, top_n=30):
    """
    Barplot for one evidence support class.
    """
    if res_df is None or res_df.empty:
        return

    df = res_df.query("support_class == @support_class").copy()
    if df.empty:
        return

    if "support_score" in df.columns:
        df = df.sort_values(["support_score", "impact_strength"], ascending=[False, False])
        score_col = "support_score"
    else:
        df = df.sort_values("impact_strength", ascending=False)
        score_col = "impact_strength"

    df = df.head(top_n).iloc[::-1].copy()

    fig_h = max(4.0, 0.25 * df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.barh(df["trait"].astype(str), df[score_col].astype(float))
    ax.set_xlabel(score_col.replace("_", " "))
    ax.set_ylabel("")
    ax.set_title(f"{config_name}: {support_class}")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()

    safe_class = support_class.replace("/", "_").replace(" ", "_")
    fig.savefig(os.path.join(outdir, f"{PREFIX}_{config_name}_{safe_class}_TopCandidates_barplot.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_support_class_traits_for_config(
    adata,
    res_df,
    meta_sub,
    inhibition_df,
    activity_df,
    delta_df,
    config_name,
    config,
    outdir,
    support_classes=None,
    top_n=6,
):
    """
    Export UMAP and pseudotime plots separately for each evidence class:
      - bidirectional_supported
      - single_DOWN_supported
      - DOWN_supported_but_activity_opposite
    """
    if res_df is None or res_df.empty:
        return

    if support_classes is None:
        support_classes = PLOT_SUPPORT_CLASSES

    base_dir = os.path.join(outdir, "by_support_class", config_name)
    safe_mkdir(base_dir)

    for support_class in support_classes:
        class_df = res_df.query("support_class == @support_class").copy()
        if class_df.empty:
            continue

        if "support_score" in class_df.columns:
            class_df = class_df.sort_values(["support_score", "impact_strength"], ascending=[False, False])
        else:
            class_df = class_df.sort_values("impact_strength", ascending=False)

        class_dir = os.path.join(base_dir, support_class)
        safe_mkdir(class_dir)

        save_support_class_barplot(
            res_df=res_df,
            config_name=config_name,
            support_class=support_class,
            outdir=class_dir,
            top_n=TOP_N_PLOT,
        )

        top_rows = select_traits_for_plotting(
            res_df=class_df,
            top_n=top_n,
            manual_traits=MANUAL_PLOT_TRAITS,
            rank_by=AUTO_PLOT_RANK_BY,
        )

        for _, row in top_rows.iterrows():
            tr = str(row["trait"])
            predicted_direction = str(row.get("predicted_direction", f"{config['target1_name']}_to_{config['target2_name']}"))
            predicted_direction_for_title = predicted_direction.replace("_to_", " → ").replace("_", " ")

            save_umap_trait_plot(
                adata=adata,
                meta_sub=meta_sub,
                inhibition_df=inhibition_df,
                trait=tr,
                config_name=f"{config_name}_{support_class}",
                config=config,
                outdir=class_dir,
            )

            save_target1_target2_comparison_curves(
                meta_sub=meta_sub,
                inhibition_df=inhibition_df,
                activity_df=activity_df,
                delta_df=delta_df,
                trait=tr,
                config_name=f"{config_name}_{support_class}",
                config=config,
                outdir=class_dir,
                predicted_direction=predicted_direction_for_title,
            )

            save_branch_separated_curves(
                adata=adata,
                meta_sub=meta_sub,
                inhibition_df=inhibition_df,
                activity_df=activity_df,
                delta_df=delta_df,
                trait=tr,
                config_name=f"{config_name}_{support_class}",
                config=config,
                outdir=class_dir,
            )


# =========================
# Main
# =========================
def main():
    safe_mkdir(OUT_DIR)
    score_dir = os.path.join(OUT_DIR, "cellTraitMatrices")
    corr_dir = os.path.join(OUT_DIR, "fateBiasCorrelation")
    reroute_dir = os.path.join(OUT_DIR, "perturbRerouting")
    plot_dir = os.path.join(OUT_DIR, "plots")
    for d in [score_dir, corr_dir, reroute_dir, plot_dir]:
        safe_mkdir(d)

    print(">>> Load adata")
    adata = sc.read_h5ad(H5AD_PATH)
    print(adata)

    if CELLTYPE_COL not in adata.obs.columns:
        raise KeyError(f"{CELLTYPE_COL} not found. Available obs columns: {list(adata.obs.columns)}")
    if PSEUDOTIME_COL not in adata.obs.columns:
        raise KeyError(f"{PSEUDOTIME_COL} not found. Available obs columns: {list(adata.obs.columns)}")

    ct_counts = adata.obs[CELLTYPE_COL].astype(str).value_counts().reset_index()
    ct_counts.columns = [CELLTYPE_COL, "n_cells"]
    ct_counts.to_csv(os.path.join(OUT_DIR, f"{PREFIX}_celltype_counts.csv"), index=False)

    print(">>> Load Z matrix")
    Z_mat = load_Z_mat_robust(ZMAT_PATH, adata)

    print(">>> Build UP/DOWN signatures")
    signatures, sig_summary = build_up_down_signatures(Z_mat, adata, TOP_N_GENES, MIN_VALID_GENES, TARGET_TRAITS)
    sig_summary.to_csv(os.path.join(OUT_DIR, f"{PREFIX}_up_down_signature_summary.csv.gz"), index=False, compression="gzip")
    if len(signatures) == 0:
        raise ValueError("No valid signatures remained after filtering.")
    print(f">>> Number of UP/DOWN signatures: {len(signatures)}")

    print(">>> Choose max_rank")
    max_rank = UCELL_MAX_RANK if UCELL_MAX_RANK is not None else auto_choose_max_rank(adata)
    print(f">>> max_rank = {max_rank}")

    print(">>> Compute pyUCell UP/DOWN scores")
    raw_score_df = compute_signature_score_df(adata, signatures, max_rank=max_rank)
    raw_score_df.to_csv(os.path.join(score_dir, f"{PREFIX}_pyUCell_raw_UP_DOWN_scores.csv.gz"), compression="gzip")

    print(">>> Build activity and inhibition-like scores")
    up_df, down_df, activity_df, inhibition_df, delta_df = build_activity_inhibition_scores(raw_score_df)
    up_df.to_csv(os.path.join(score_dir, f"{PREFIX}_pyUCell_UP_scores.csv.gz"), compression="gzip")
    down_df.to_csv(os.path.join(score_dir, f"{PREFIX}_pyUCell_DOWN_scores.csv.gz"), compression="gzip")
    activity_df.to_csv(os.path.join(score_dir, f"{PREFIX}_gmMAP_Activity_ZUP_scores.csv.gz"), compression="gzip")
    inhibition_df.to_csv(os.path.join(score_dir, f"{PREFIX}_gmMAP_Inhibition_ZDOWN_scores.csv.gz"), compression="gzip")
    delta_df.to_csv(os.path.join(score_dir, f"{PREFIX}_gmMAP_Delta_Activity_minus_Inhibition_scores.csv.gz"), compression="gzip")

    meta_df = adata.obs[[CELLTYPE_COL, PSEUDOTIME_COL]].copy()

    print(">>> Run root-terminal fate-bias rerouting analysis")
    all_res = []
    for config_name, config in TRAJECTORY_CONFIG.items():
        print(f"    - {config_name}")
        res, meta_sub = compute_fate_rerouting_for_config(activity_df, inhibition_df, delta_df, meta_df, config_name, config)
        if res is None or res.empty:
            continue

        res = classify_support_patterns(res)
        res = add_branch_difference_metrics(
            res=res,
            meta_sub=meta_sub,
            activity_df=activity_df,
            inhibition_df=inhibition_df,
            delta_df=delta_df,
            config=config,
        )

        res.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_gmMAP_perturb_rerouting_scores.csv.gz"), index=False, compression="gzip")

        bidirectional = res.query("support_class == 'bidirectional_supported'").copy()
        bidirectional.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_bidirectional_supported_candidates.csv.gz"), index=False, compression="gzip")

        single_down = res.query("support_class == 'single_DOWN_supported'").copy()
        single_down.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_single_DOWN_supported_candidates.csv.gz"), index=False, compression="gzip")

        down_activity_opposite = res.query("support_class == 'DOWN_supported_but_activity_opposite'").copy()
        down_activity_opposite.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_DOWN_supported_but_activity_opposite_candidates.csv.gz"), index=False, compression="gzip")

        branchdiff_ranked = res.sort_values(["branch_difference_score", "impact_strength"], ascending=[False, False]).copy()
        branchdiff_ranked.to_csv(
            os.path.join(reroute_dir, f"{PREFIX}_{config_name}_branch_difference_ranked_candidates.csv.gz"),
            index=False,
            compression="gzip"
        )

        if len(MANUAL_PLOT_TRAITS) > 0:
            manual_subset = res[res["trait"].astype(str).isin([str(x) for x in MANUAL_PLOT_TRAITS])].copy()
            manual_subset.to_csv(
                os.path.join(reroute_dir, f"{PREFIX}_{config_name}_manual_selected_traits.csv.gz"),
                index=False,
                compression="gzip"
            )

        support_summary = res.groupby("support_class").size().reset_index(name="n_traits").sort_values("n_traits", ascending=False)
        support_summary.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_support_class_summary.csv"), index=False)

        strong = res.query("q_inhibition_vs_fatebias <= 0.05 and abs(rho_inhibition_vs_fatebias) >= 0.2").copy()
        strong.to_csv(os.path.join(reroute_dir, f"{PREFIX}_{config_name}_strong_rerouting_q0.05_absrho0.2.csv.gz"), index=False, compression="gzip")
        meta_sub.to_csv(os.path.join(corr_dir, f"{PREFIX}_{config_name}_cells_with_fatebias.csv.gz"), compression="gzip")
        save_top_barplot(res, config_name, plot_dir, TOP_N_PLOT)
        plot_top_traits_for_config(adata, res, meta_sub, inhibition_df, activity_df, delta_df, config_name, config, plot_dir, TOP_TRAJECTORY_PLOT)

        # Export class-specific plots for bidirectional, single-DOWN and caution candidates.
        plot_support_class_traits_for_config(
            adata=adata,
            res_df=res,
            meta_sub=meta_sub,
            inhibition_df=inhibition_df,
            activity_df=activity_df,
            delta_df=delta_df,
            config_name=config_name,
            config=config,
            outdir=plot_dir,
            support_classes=PLOT_SUPPORT_CLASSES,
            top_n=TOP_TRAJECTORY_PLOT_PER_SUPPORT_CLASS,
        )

        all_res.append(res)

    if len(all_res) == 0:
        print("No valid gmMAP-perturb results.")
        return

    all_df = pd.concat(all_res, axis=0, ignore_index=True)
    all_df = classify_support_patterns(all_df)
    all_df.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_gmMAP_perturb_rerouting_scores.csv.gz"), index=False, compression="gzip")

    all_bidirectional = all_df.query("support_class == 'bidirectional_supported'").copy()
    all_bidirectional.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_bidirectional_supported_candidates.csv.gz"), index=False, compression="gzip")

    all_single_down = all_df.query("support_class == 'single_DOWN_supported'").copy()
    all_single_down.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_single_DOWN_supported_candidates.csv.gz"), index=False, compression="gzip")

    all_down_activity_opposite = all_df.query("support_class == 'DOWN_supported_but_activity_opposite'").copy()
    all_down_activity_opposite.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_DOWN_supported_but_activity_opposite_candidates.csv.gz"), index=False, compression="gzip")

    all_branchdiff_ranked = all_df.sort_values(["branch_difference_score", "impact_strength"], ascending=[False, False]).copy()
    all_branchdiff_ranked.to_csv(
        os.path.join(reroute_dir, f"{PREFIX}_ALL_branch_difference_ranked_candidates.csv.gz"),
        index=False,
        compression="gzip"
    )

    if len(MANUAL_PLOT_TRAITS) > 0:
        all_manual_subset = all_df[all_df["trait"].astype(str).isin([str(x) for x in MANUAL_PLOT_TRAITS])].copy()
        all_manual_subset.to_csv(
            os.path.join(reroute_dir, f"{PREFIX}_ALL_manual_selected_traits.csv.gz"),
            index=False,
            compression="gzip"
        )

    all_support_summary = all_df.groupby(["config", "support_class"]).size().reset_index(name="n_traits").sort_values(["config", "n_traits"], ascending=[True, False])
    all_support_summary.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_support_class_summary.csv"), index=False)

    strong_all = all_df.query("q_inhibition_vs_fatebias <= 0.05 and abs(rho_inhibition_vs_fatebias) >= 0.2").copy()
    strong_all.to_csv(os.path.join(reroute_dir, f"{PREFIX}_ALL_strong_rerouting_q0.05_absrho0.2.csv.gz"), index=False, compression="gzip")

    print(">>> Done")
    print(f">>> Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
