#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme C (AUROC) Up/Down/Net/Final version (Unified)  +  two requested upgrades

你提出的两个修改点，本脚本都已包含：

(改动1) ΔAUROC（= AUROC_up - AUROC_down；等价于 AUROC_effect_up - AUROC_effect_down）
       的 p/q 计算不再用“单个U统计量的方差公式硬套”，而改为：
       - DeLong（默认，相关AUC差的经典检验；可做两侧）
       - 或 Bootstrap（可选，近似，但可能更慢）
       通过 NET_P_METHOD 控制。

(改动2) 补充“细胞层面 net 分数”的关联性分析（更推荐做“关联性”）：
       score_net(cell) = score_up(cell) - score_down(cell)
       然后对 net 分数做 AUROC/AUROC_effect + p/q（Mann–Whitney U 正态近似）
       输出：
         AUROC_netCell, AUROC_effect_netCell, p_netCell, q_netCell_...

同时也把你之前遇到的两个稳定性问题一起修了（强烈建议保留）：
- Z-score：sd==0 的列/行直接置 0（避免整列 NaN 造成后续崩溃）
- AUCell::exploreThresholds：输入矩阵先 drop 全NA行/列 + fill NA 为中性值 + try/except 保证不会中断流水线

输出（相比你原先脚本，新增了 netCell 相关矩阵/统计列）：
- cellTraitMatrices/{PREFIX}_{method}_{up/down/net}_{stage}.csv.gz   （net 新增）
- SchemeC_AUROC_UpDownNet/
    {PREFIX}_{method}_{stage}_AUROC_up.csv.gz
    {PREFIX}_{method}_{stage}_AUROC_down.csv.gz
    {PREFIX}_{method}_{stage}_AUROC_effect_net_DELTA.csv.gz          （ΔAUROC_effect）
    {PREFIX}_{method}_{stage}_AUROC_netCell.csv.gz                   （cell-level net 的 AUROC）
    {PREFIX}_{method}_{stage}_AUROC_effect_netCell.csv.gz
    {PREFIX}_{method}_{stage}_final_AUROC.csv.gz
    {PREFIX}_{method}_{stage}_final_effect.csv.gz
    {PREFIX}_{method}_{stage}_AUROC_upDownNet_stats_long.csv.gz      （统一 long 表，含 ΔAUC 与 netCell）
    exploreThresholds on final_AUROC（安全防崩）
    TopK summaries（按 Δeffect、final_effect、netCell_effect 排名）
- {PREFIX}_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from typing import Optional
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from joblib import Parallel, delayed
from math import erf, sqrt

try:
    from scipy.stats import rankdata  # tie-aware ranks (average)
except Exception:
    rankdata = None


# =========================
# Config (EDIT ME)
# =========================
H5AD_PATH = "data/input.h5ad"
ZMAT_PATH = "data/MAGMA_zstat.csv"

CELLTYPE_KEY = "cell_ontology_class"
PREFIX = "Meta"
OUT_DIR = "Meta_SchemeC_AUROC_UpDownNet_outputs"

# scoring params
TOP_N_GENES = 1000
MIN_VALID_GENES = 200
N_JOBS = 60

# wAUCell params
TOP_FRAC_WAUCELL = 0.05
TOPK_BATCH_SIZE = 128
WAUCELL_CELL_BATCH_SIZE = 65536

# stage normalization params
COL_BLOCK = 64
DTYPE_OUT = np.float32

# AUROC params
MIN_CELLS_IN_TYPE = 20

# up/down：一般检验 AUROC > 0.5（正向富集）
P_ONE_SIDED_UPDOWN = True

# ΔAUROC (up - down)：推荐 two-sided（可能正也可能负）
DELTA_TWO_SIDED = True

# cell-level net：也推荐 two-sided（净效应可能正负）
NETCELL_TWO_SIDED = True

# stages to run
STAGES_TO_RUN = [
    "S1_raw",
    "S2_geneSetZ",
    "S3_geneSetZ_cellZ",
    "S4_geneSetZ_cellZ_geneSetZ"
]

# methods to run
METHODS_TO_RUN = ["scMRS", "wAUCell"]  # or ["wAUCell"] only

# output options
SCHEMEC_DIRNAME = "SchemeC_AUROC_UpDownNet"
SKIP_EXISTING = True

# final significance source
FINAL_SIG_FROM_DELTA = False  # final_p/final_q 从 ΔAUROC（DeLong/Bootstrap）来

# exploreThresholds params
THRP = 0.01
SMALLEST_POP_PERCENT = 0.25
R_NCORES = 1

# ---------- ΔAUROC p-value method ----------
# "delong" (default) or "bootstrap"
NET_P_METHOD = "delong"

# DeLong/Bootstrap 计算量控制（超大数据强烈建议设置一个上限做分层抽样近似）
RANDOM_SEED = 0
DELONG_MAX_TOTAL_CELLS = 50000     # 每个 celltype 的 DeLong 最多用这么多 cell（分层抽样）
BOOT_N = 200                       # bootstrap 次数（如果启用 bootstrap）
BOOT_MAX_IN = 4000                 # 每次 bootstrap 最大抽样 in
BOOT_MAX_OUT = 4000                # 每次 bootstrap 最大抽样 out

# TopK summaries
TOPK_EXPORT = 20


# =========================================================
# Helpers
# =========================================================
def _chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _bh_fdr_1d(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    m = np.isfinite(p)
    if m.sum() == 0:
        return q
    p0 = p[m]
    n = p0.size
    order = np.argsort(p0)
    ranked = p0[order]
    q0 = ranked * n / (np.arange(1, n + 1))
    q0 = np.minimum.accumulate(q0[::-1])[::-1]
    q0 = np.clip(q0, 0, 1)
    out = np.empty_like(p0)
    out[order] = q0
    q[m] = out
    return q


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _norm_sf(z: float) -> float:
    return 1.0 - _norm_cdf(z)


def save_stage(df: pd.DataFrame, path: str):
    if SKIP_EXISTING and os.path.exists(path):
        return
    df.to_csv(path, compression="gzip")


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if rankdata is not None:
        return rankdata(x, method="average")
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(x, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return r


# =========================================================
# (A) DeLong test for correlated AUCs (two predictors, same labels)
# =========================================================
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks, 1..N, average for ties."""
    x = np.asarray(x, dtype=np.float64)
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = Z.size
    T = np.empty(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        T[i:j] = mid
        i = j
    out = np.empty(N, dtype=np.float64)
    out[J] = T
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """
    preds_sorted: shape (k, N), labels已经按 pos(1)在前 排好
    m: #positives
    returns: aucs (k,), cov (k,k)
    """
    k, N = preds_sorted.shape
    n = N - m
    if m <= 0 or n <= 0:
        return np.full(k, np.nan), np.full((k, k), np.nan)

    tx = np.zeros((k, m), dtype=np.float64)
    ty = np.zeros((k, n), dtype=np.float64)
    tz = np.zeros((k, N), dtype=np.float64)

    for r in range(k):
        tx[r, :] = _compute_midrank(preds_sorted[r, :m])
        ty[r, :] = _compute_midrank(preds_sorted[r, m:])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    # covariance across classifiers
    # np.cov expects rows=variables, columns=observations; bias=True => population covariance
    cov01 = np.cov(v01, bias=True)
    cov10 = np.cov(v10, bias=True)
    S = cov01 / m + cov10 / n
    return aucs, S


def delong_auc_diff_pvalue(
    scores1: np.ndarray,
    scores2: np.ndarray,
    labels01: np.ndarray,
    two_sided: bool = True,
    max_total_cells: int = 50000,
    rng: Optional[np.random.Generator] = None,
):
    """
    DeLong test for AUC difference of two correlated predictors.
    Returns dict: delta_auc, z, p, var, auc1, auc2, n_used, n_pos, n_neg
    为了可运行性，若样本太大，会对 pos/neg 分层无放回抽样到 max_total_cells。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y = np.asarray(labels01, dtype=np.int8)
    s1 = np.asarray(scores1, dtype=np.float64)
    s2 = np.asarray(scores2, dtype=np.float64)

    # 过滤非有限值（两边都要有限）
    finite = np.isfinite(s1) & np.isfinite(s2) & (y >= 0)
    y = y[finite]; s1 = s1[finite]; s2 = s2[finite]

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    m = pos_idx.size
    n = neg_idx.size
    if m == 0 or n == 0:
        return dict(delta_auc=np.nan, z=np.nan, p=np.nan, var=np.nan,
                    auc1=np.nan, auc2=np.nan, n_used=0, n_pos=m, n_neg=n)

    # 分层抽样（无放回）控制规模
    N = m + n
    if max_total_cells is not None and N > max_total_cells:
        frac_pos = m / N
        m_s = int(max(1, round(max_total_cells * frac_pos)))
        n_s = max_total_cells - m_s
        m_s = min(m_s, m)
        n_s = min(n_s, n)
        pos_s = rng.choice(pos_idx, size=m_s, replace=False)
        neg_s = rng.choice(neg_idx, size=n_s, replace=False)
        take = np.concatenate([pos_s, neg_s])
        y = y[take]; s1 = s1[take]; s2 = s2[take]
        # 重新统计
        m = int((y == 1).sum()); n = int((y == 0).sum())

    # labels: pos first
    order = np.argsort(-y, kind="mergesort")
    y_sorted = y[order]
    m = int(y_sorted.sum())
    preds = np.vstack([s1[order], s2[order]])  # (2, N)
    aucs, S = _fast_delong(preds, m)

    if not np.all(np.isfinite(aucs)) or not np.all(np.isfinite(S)):
        return dict(delta_auc=np.nan, z=np.nan, p=np.nan, var=np.nan,
                    auc1=float(aucs[0]) if np.isfinite(aucs[0]) else np.nan,
                    auc2=float(aucs[1]) if np.isfinite(aucs[1]) else np.nan,
                    n_used=int(y.size), n_pos=m, n_neg=int(y.size - m))

    delta = float(aucs[0] - aucs[1])
    var = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    if not np.isfinite(var) or var <= 0:
        return dict(delta_auc=delta, z=np.nan, p=np.nan, var=var,
                    auc1=float(aucs[0]), auc2=float(aucs[1]),
                    n_used=int(y.size), n_pos=m, n_neg=int(y.size - m))

    z = delta / sqrt(var)
    p = 2.0 * _norm_sf(abs(z)) if two_sided else _norm_sf(z)
    return dict(delta_auc=delta, z=float(z), p=float(p), var=var,
                auc1=float(aucs[0]), auc2=float(aucs[1]),
                n_used=int(y.size), n_pos=m, n_neg=int(y.size - m))


# =========================================================
# (B) Bootstrap for AUC difference (optional)
# =========================================================
def _auc_from_scores_binary(scores: np.ndarray, y01: np.ndarray) -> float:
    """AUC via rank-sum with average ranks (ties handled)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y01, dtype=np.int8)
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    if m == 0 or n == 0:
        return np.nan
    r = _rankdata_average(scores)
    sum_r_pos = float(r[y == 1].sum())
    U = sum_r_pos - m * (m + 1) / 2.0
    return float(U / (m * n))


def bootstrap_auc_diff_pvalue(
    scores1: np.ndarray,
    scores2: np.ndarray,
    labels01: np.ndarray,
    n_boot: int = 200,
    two_sided: bool = True,
    max_in: int = 4000,
    max_out: int = 4000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Stratified bootstrap within pos/neg; returns delta, p.
    注意：对每个 trait×celltype 做 bootstrap 很耗时；通常只建议在小规模数据或抽样后使用。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    y = np.asarray(labels01, dtype=np.int8)
    s1 = np.asarray(scores1, dtype=np.float64)
    s2 = np.asarray(scores2, dtype=np.float64)
    finite = np.isfinite(s1) & np.isfinite(s2)
    y = y[finite]; s1 = s1[finite]; s2 = s2[finite]

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if pos.size == 0 or neg.size == 0:
        return dict(delta_auc=np.nan, z=np.nan, p=np.nan, var=np.nan,
                    auc1=np.nan, auc2=np.nan, n_used=0, n_pos=pos.size, n_neg=neg.size)

    # 限制每次抽样规模（无放回先截断，再 bootstrap 有放回）
    if pos.size > max_in:
        pos0 = rng.choice(pos, size=max_in, replace=False)
    else:
        pos0 = pos
    if neg.size > max_out:
        neg0 = rng.choice(neg, size=max_out, replace=False)
    else:
        neg0 = neg

    # 观测 delta
    y0 = np.zeros(pos0.size + neg0.size, dtype=np.int8)
    y0[:pos0.size] = 1
    idx0 = np.concatenate([pos0, neg0])
    auc1 = _auc_from_scores_binary(s1[idx0], y0)
    auc2 = _auc_from_scores_binary(s2[idx0], y0)
    if not np.isfinite(auc1) or not np.isfinite(auc2):
        return dict(delta_auc=np.nan, z=np.nan, p=np.nan, var=np.nan,
                    auc1=auc1, auc2=auc2, n_used=int(idx0.size), n_pos=int(pos0.size), n_neg=int(neg0.size))

    delta_obs = float(auc1 - auc2)

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pb = rng.choice(pos0, size=pos0.size, replace=True)
        nb = rng.choice(neg0, size=neg0.size, replace=True)
        idxb = np.concatenate([pb, nb])
        yb = np.zeros(pos0.size + neg0.size, dtype=np.int8)
        yb[:pos0.size] = 1

        a1 = _auc_from_scores_binary(s1[idxb], yb)
        a2 = _auc_from_scores_binary(s2[idxb], yb)
        deltas[b] = a1 - a2

    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return dict(delta_auc=delta_obs, z=np.nan, p=np.nan, var=np.nan,
                    auc1=float(auc1), auc2=float(auc2),
                    n_used=int(idx0.size), n_pos=int(pos0.size), n_neg=int(neg0.size))

    if two_sided:
        p = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1.0) / (deltas.size + 1.0)
    else:
        p = (np.sum(deltas >= delta_obs) + 1.0) / (deltas.size + 1.0)

    var = float(np.var(deltas, ddof=0)) if deltas.size > 1 else np.nan
    z = delta_obs / sqrt(var) if (np.isfinite(var) and var > 0) else np.nan
    return dict(delta_auc=delta_obs, z=float(z) if np.isfinite(z) else np.nan, p=float(p), var=var,
                auc1=float(auc1), auc2=float(auc2),
                n_used=int(idx0.size), n_pos=int(pos0.size), n_neg=int(neg0.size))


# =========================================================
# 1) gene noise
# =========================================================
def compute_gene_noise(adata):
    X = adata.X
    mean = np.asarray(X.mean(axis=0)).ravel()
    if hasattr(X, "power"):  # sparse
        second_moment = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = second_moment - mean**2
    else:
        var = np.var(X, axis=0)
    var = np.asarray(var, dtype=np.float64)
    var = np.maximum(var, 0.0)
    noise = np.sqrt(var)
    noise[~np.isfinite(noise)] = np.nan
    noise[noise == 0] = np.nan
    return dict(zip(adata.var_names, noise))


# =========================================================
# 2) Up/Down signature from Z_mat
# =========================================================
def extract_signature_up_down(Z_mat, trait, top_n=1000):
    z = pd.to_numeric(Z_mat[trait], errors="coerce").dropna()
    up = z[z > 0].sort_values(ascending=False).head(top_n)
    dn = z[z < 0].sort_values(ascending=True).head(top_n)
    dn = (-dn)  # make down-strength positive
    return up, dn


# =========================================================
# 3) weights: w = z / noise (positive only)
# =========================================================
def compute_weights_from_z(gene_z_series, gene_noise, adata):
    genes = [g for g in gene_z_series.index if g in adata.var_names and g in gene_noise]
    if len(genes) == 0:
        return None, None

    z = pd.to_numeric(gene_z_series.loc[genes], errors="coerce").values.astype(float)
    noise = np.array([gene_noise[g] for g in genes], dtype=float)

    mask = np.isfinite(z) & np.isfinite(noise) & (noise > 0) & (z > 0)
    genes = [g for g, m in zip(genes, mask) if m]
    if len(genes) == 0:
        return None, None

    w = z[mask] / noise[mask]
    return genes, w


# =========================================================
# 4) scMRS (weighted sum / sum|w|)
# =========================================================
def compute_scMRS_norm_by_idx(X, gene_idx, weights):
    denom = np.sum(np.abs(weights))
    if (not np.isfinite(denom)) or denom == 0:
        return None

    X_sub = X[:, gene_idx]
    if issparse(X_sub):
        score = X_sub.dot(weights)
    else:
        score = X_sub @ weights

    score = np.asarray(score).ravel() / denom
    return score


# =========================================================
# 5) Precompute per-cell top-k genes (for wAUCell)
# =========================================================
def precompute_cell_topk_genes(adata, top_frac=0.05, batch_size=256):
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    top_k = int(max(1, np.floor(top_frac * n_genes)))
    topk_idx = np.empty((n_cells, top_k), dtype=np.int32)

    for start in range(0, n_cells, batch_size):
        end = min(n_cells, start + batch_size)
        X = adata.X[start:end, :]
        X = X.toarray() if issparse(X) else np.asarray(X)

        part = np.argpartition(X, -top_k, axis=1)[:, -top_k:]
        rows = np.arange(end - start)[:, None]
        vals = X[rows, part]
        order = np.argsort(vals, axis=1)[:, ::-1]
        topk_idx[start:end, :] = part[rows, order]

    return topk_idx, top_k


# =========================================================
# 6) wAUCell-like
# =========================================================
def compute_wAUCell_from_topk_chunked(
    topk_idx, n_genes, sig_gene_idx, sig_weights, pos_weight, cell_batch_size=65536
):
    w_lookup = np.zeros(n_genes, dtype=np.float32)
    sig_gene_idx = np.asarray(sig_gene_idx, dtype=np.int64)
    w = np.asarray(sig_weights, dtype=np.float32)
    w_lookup[sig_gene_idx] = w

    denom = float(w.sum())
    if (not np.isfinite(denom)) or denom == 0:
        return np.full(topk_idx.shape[0], np.nan, dtype=np.float32)

    n_cells = topk_idx.shape[0]
    out = np.empty(n_cells, dtype=np.float32)

    for start in range(0, n_cells, cell_batch_size):
        end = min(n_cells, start + cell_batch_size)
        idx_chunk = topk_idx[start:end, :]
        W = w_lookup[idx_chunk]
        out[start:end] = (W * pos_weight[None, :]).sum(axis=1) / denom

    return out


# =========================================================
# 7) Worker per trait -> up/down/net score vectors (cell-level)
# =========================================================
def score_trait_worker_ud(
    trait, adata, Z_mat, gene_noise, top_n, min_valid_genes,
    topk_idx, top_k, gene_to_idx, pos_weight, waucell_cell_batch_size
):
    up_z, dn_z = extract_signature_up_down(Z_mat, trait, top_n=top_n)
    up_genes, up_w = compute_weights_from_z(up_z, gene_noise, adata)
    dn_genes, dn_w = compute_weights_from_z(dn_z, gene_noise, adata)

    if (up_genes is None or len(up_genes) < min_valid_genes) and (dn_genes is None or len(dn_genes) < min_valid_genes):
        return None

    X = adata.X
    out = {"trait": trait}

    # UP
    if up_genes is not None and len(up_genes) >= min_valid_genes:
        mask = np.array([g in gene_to_idx for g in up_genes], dtype=bool)
        if mask.sum() >= min_valid_genes:
            up_genes_f = [g for g, m in zip(up_genes, mask) if m]
            up_idx = [gene_to_idx[g] for g in up_genes_f]
            up_w2 = up_w[mask]
            sc_up = compute_scMRS_norm_by_idx(X, up_idx, up_w2)
            out["scMRS_up"] = sc_up if sc_up is not None else np.full(adata.n_obs, np.nan)
            out["wAUCell_up"] = compute_wAUCell_from_topk_chunked(
                topk_idx, adata.n_vars, up_idx, up_w2, pos_weight,
                cell_batch_size=waucell_cell_batch_size
            )
        else:
            out["scMRS_up"] = np.full(adata.n_obs, np.nan)
            out["wAUCell_up"] = np.full(adata.n_obs, np.nan)
    else:
        out["scMRS_up"] = np.full(adata.n_obs, np.nan)
        out["wAUCell_up"] = np.full(adata.n_obs, np.nan)

    # DOWN
    if dn_genes is not None and len(dn_genes) >= min_valid_genes:
        mask = np.array([g in gene_to_idx for g in dn_genes], dtype=bool)
        if mask.sum() >= min_valid_genes:
            dn_genes_f = [g for g, m in zip(dn_genes, mask) if m]
            dn_idx = [gene_to_idx[g] for g in dn_genes_f]
            dn_w2 = dn_w[mask]
            sc_dn = compute_scMRS_norm_by_idx(X, dn_idx, dn_w2)
            out["scMRS_down"] = sc_dn if sc_dn is not None else np.full(adata.n_obs, np.nan)
            out["wAUCell_down"] = compute_wAUCell_from_topk_chunked(
                topk_idx, adata.n_vars, dn_idx, dn_w2, pos_weight,
                cell_batch_size=waucell_cell_batch_size
            )
        else:
            out["scMRS_down"] = np.full(adata.n_obs, np.nan)
            out["wAUCell_down"] = np.full(adata.n_obs, np.nan)
    else:
        out["scMRS_down"] = np.full(adata.n_obs, np.nan)
        out["wAUCell_down"] = np.full(adata.n_obs, np.nan)

    out["scMRS_net"] = out["scMRS_up"] - out["scMRS_down"]
    out["wAUCell_net"] = out["wAUCell_up"] - out["wAUCell_down"]
    return out


# =========================================================
# 8) Compute RAW full wide table
# =========================================================
def score_all_traits_parallel_ud_raw(
    adata,
    Z_mat,
    top_n=1000,
    min_valid_genes=200,
    n_jobs=35,
    top_frac=0.05,
    topk_batch_size=256,
    waucell_cell_batch_size=65536,
    prefix="Meta",
):
    gene_noise = compute_gene_noise(adata)
    topk_idx, top_k = precompute_cell_topk_genes(adata, top_frac=top_frac, batch_size=topk_batch_size)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    pos_weight = (np.arange(top_k, 0, -1, dtype=np.float32) / float(top_k))

    results = Parallel(n_jobs=n_jobs, verbose=10, prefer="threads")(
        delayed(score_trait_worker_ud)(
            trait, adata, Z_mat, gene_noise, top_n, min_valid_genes,
            topk_idx, top_k, gene_to_idx, pos_weight, waucell_cell_batch_size
        )
        for trait in Z_mat.columns
    )

    score_dict = {}
    for res in results:
        if res is None:
            continue
        trait = res["trait"]
        for kind in ["scMRS_up", "scMRS_down", "scMRS_net", "wAUCell_up", "wAUCell_down", "wAUCell_net"]:
            score_dict[f"{prefix}_{kind}_{trait}"] = res[kind]

    return pd.DataFrame(score_dict, index=adata.obs_names)


def extract_kind_matrix(score_df_raw_full: pd.DataFrame, prefix: str, kind: str) -> pd.DataFrame:
    col_prefix = f"{prefix}_{kind}_"
    cols = [c for c in score_df_raw_full.columns if c.startswith(col_prefix)]
    traits = [c[len(col_prefix):] for c in cols]
    out = score_df_raw_full[cols].copy()
    out.columns = traits
    return out


# =========================================================
# 9) Stage normalizations (FIX: sd==0 -> all zeros, not NaN)
# =========================================================
def _col_zscore_inplace(df: pd.DataFrame, col_block=64, dtype_out=np.float32):
    cols = list(df.columns)
    for block in _chunks(cols, col_block):
        M = df[block].to_numpy(dtype=np.float64, copy=True)
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)

        bad = (~np.isfinite(sd)) | (sd == 0)
        sd2 = sd.copy()
        sd2[bad] = 1.0

        M = (M - mu) / sd2
        if bad.any():
            M[:, bad] = 0.0

        df.loc[:, block] = M.astype(dtype_out, copy=False)


def _row_zscore_inplace(df: pd.DataFrame, col_block=64, dtype_out=np.float32):
    n = df.shape[0]
    cols = list(df.columns)

    sumv = np.zeros(n, dtype=np.float64)
    sumsq = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int32)

    for block in _chunks(cols, col_block):
        M = df[block].to_numpy(dtype=np.float64, copy=False)
        m = np.isfinite(M)
        M0 = np.nan_to_num(M, nan=0.0)
        sumv += M0.sum(axis=1)
        sumsq += (M0 * M0).sum(axis=1)
        cnt += m.sum(axis=1).astype(np.int32)

    mu = np.divide(sumv, cnt, out=np.full_like(sumv, np.nan), where=(cnt > 0))
    ex2 = np.divide(sumsq, cnt, out=np.full_like(sumsq, np.nan), where=(cnt > 0))
    var = np.maximum(ex2 - mu * mu, 0.0)
    sd = np.sqrt(var)

    bad = (~np.isfinite(sd)) | (sd == 0)
    sd2 = sd.copy()
    sd2[bad] = 1.0

    for block in _chunks(cols, col_block):
        M = df[block].to_numpy(dtype=np.float64, copy=True)
        M = (M - mu[:, None]) / sd2[:, None]
        if bad.any():
            M[bad, :] = 0.0
        df.loc[:, block] = M.astype(dtype_out, copy=False)


# =========================================================
# 10) Robust Z_mat loader
# =========================================================
def load_Z_mat_robust(z_path: str, adata) -> pd.DataFrame:
    Z = pd.read_csv(z_path, index_col=0)
    if Z.shape[1] >= 2:
        first_col = Z.columns[0]
        vals = Z[first_col].astype(str)
        overlap = len(set(vals).intersection(set(adata.var_names))) / max(1, min(len(vals), len(adata.var_names)))
        idx_overlap = len(set(map(str, Z.index)).intersection(set(adata.var_names))) / max(1, min(len(Z.index), len(adata.var_names)))
        if (overlap > 0.2) and (idx_overlap < 0.05) and vals.is_unique:
            Z = Z.set_index(first_col)
    return Z


# =========================================================
# 11) AUCell exploreThresholds (R if available; else fallback)
# =========================================================
def run_aucell_exploreThresholds_on_matrix(
    auc_like_df: pd.DataFrame,   # rows=items (celltypes), cols=traits
    thrP: float = 0.01,
    smallestPopPercent: float = 0.25,
    nCores: int = 1,
    seed: int = 0,
    neutral_fill: float = 0.5,
):
    # safety: drop all-NA + fill NA neutral
    auc_like_df = auc_like_df.replace([np.inf, -np.inf], np.nan)
    auc_like_df = auc_like_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    auc_like_df = auc_like_df.fillna(neutral_fill)

    items = list(auc_like_df.index)
    traits = list(auc_like_df.columns)
    n_items = len(items)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()

        aucell = importr("AUCell")
        base = importr("base")

        mat = auc_like_df.T  # traits x items
        r_mat = pandas2ri.py2rpy(mat)
        r_mat = base.as_matrix(r_mat)

        ro.r["set.seed"](seed)
        res = aucell.AUCell_exploreThresholds(
            r_mat,
            thrP=thrP,
            nCores=nCores,
            smallestPopPercent=smallestPopPercent,
            plotHist=False,
            assignCells=True,
            verbose=False
        )

        thr_sel = aucell.getThresholdSelected(res)
        asg = aucell.getAssignments(res)

        thr_dict = dict(zip(list(thr_sel.names), list(thr_sel)))

        thr_rows = []
        asg_rows = []
        for tr in traits:
            thrv = float(thr_dict.get(tr, np.nan))
            assigned = []
            if tr in asg.names:
                assigned = list(asg.rx2(tr))
            assigned = [str(x) for x in assigned]

            thr_rows.append({
                "trait": tr,
                "selected_threshold": thrv,
                "method": "AUCell_exploreThresholds",
                "n_assigned": len(assigned),
            })
            asg_rows.append({"trait": tr, "assigned_items": ",".join(assigned)})

        thr_df = pd.DataFrame(thr_rows).set_index("trait")
        assign_df = pd.DataFrame(asg_rows).set_index("trait")
        return thr_df, assign_df

    except Exception:
        # Fallback: right-tail quantile threshold
        q = 1.0 - (thrP / max(1, n_items))
        q = min(max(q, 0.0), 1.0)

        thr_rows = []
        asg_rows = []
        for tr in traits:
            v = auc_like_df[tr].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            thrv = float(np.quantile(v, q)) if v.size > 0 else np.nan

            assigned = (
                auc_like_df.index[(auc_like_df[tr] >= thrv) & np.isfinite(auc_like_df[tr])].tolist()
                if np.isfinite(thrv) else []
            )

            thr_rows.append({
                "trait": tr,
                "selected_threshold": thrv,
                "method": f"PY_quantile(q={q:.6f})",
                "n_assigned": len(assigned),
            })
            asg_rows.append({"trait": tr, "assigned_items": ",".join(map(str, assigned))})

        thr_df = pd.DataFrame(thr_rows).set_index("trait")
        assign_df = pd.DataFrame(asg_rows).set_index("trait")
        return thr_df, assign_df

# =========================================================
# 12) TopK export helper (rank by effect)
# =========================================================
def export_topk_summaries(
    effect_df: pd.DataFrame,         # celltype × trait, EFFECT values
    stats_long: pd.DataFrame,        # reset df with columns trait/celltype
    out_prefix: str,
    out_dir: str,
    topk: int = 20,
):
    os.makedirs(out_dir, exist_ok=True)

    M = effect_df.to_numpy(dtype=float)  # celltype × trait
    celltypes = np.array(effect_df.index, dtype=str)
    traits = np.array(effect_df.columns, dtype=str)

    stats_reset = stats_long.copy()

    # ---------- A) per celltype: topK traits ----------
    rows_pos = []
    rows_neg = []
    for i, ct in enumerate(celltypes):
        v = M[i, :]
        m = np.isfinite(v)
        if m.sum() == 0:
            continue
        vv = v[m]
        tt = traits[m]

        idx_pos = np.argsort(vv)[::-1][:topk]
        for r, jj in enumerate(idx_pos, 1):
            rows_pos.append({"celltype": ct, "trait": tt[jj], "rank": r, "effect_used_for_ranking": float(vv[jj])})

        idx_neg = np.argsort(vv)[:topk]
        for r, jj in enumerate(idx_neg, 1):
            rows_neg.append({"celltype": ct, "trait": tt[jj], "rank": r, "effect_used_for_ranking": float(vv[jj])})

    df_ct_pos = pd.DataFrame(rows_pos).merge(stats_reset, on=["trait", "celltype"], how="left")
    df_ct_neg = pd.DataFrame(rows_neg).merge(stats_reset, on=["trait", "celltype"], how="left")

    df_ct_pos.to_csv(os.path.join(out_dir, f"{out_prefix}_topKtraits_perCelltype_POS.csv.gz"),
                     index=False, compression="gzip")
    df_ct_neg.to_csv(os.path.join(out_dir, f"{out_prefix}_topKtraits_perCelltype_NEG.csv.gz"),
                     index=False, compression="gzip")

    # ---------- B) per trait: topK celltypes ----------
    rows_pos = []
    rows_neg = []
    for j, tr in enumerate(traits):
        v = M[:, j]
        m = np.isfinite(v)
        if m.sum() == 0:
            continue
        vv = v[m]
        cc = celltypes[m]

        idx_pos = np.argsort(vv)[::-1][:topk]
        for r, ii in enumerate(idx_pos, 1):
            rows_pos.append({"trait": tr, "celltype": cc[ii], "rank": r, "effect_used_for_ranking": float(vv[ii])})

        idx_neg = np.argsort(vv)[:topk]
        for r, ii in enumerate(idx_neg, 1):
            rows_neg.append({"trait": tr, "celltype": cc[ii], "rank": r, "effect_used_for_ranking": float(vv[ii])})

    df_tr_pos = pd.DataFrame(rows_pos).merge(stats_reset, on=["trait", "celltype"], how="left")
    df_tr_neg = pd.DataFrame(rows_neg).merge(stats_reset, on=["trait", "celltype"], how="left")

    df_tr_pos.to_csv(os.path.join(out_dir, f"{out_prefix}_topKcelltypes_perTrait_POS.csv.gz"),
                     index=False, compression="gzip")
    df_tr_neg.to_csv(os.path.join(out_dir, f"{out_prefix}_topKcelltypes_perTrait_NEG.csv.gz"),
                     index=False, compression="gzip")


# =========================================================
# 13) Core: AUROC Up/Down + ΔAUROC p (DeLong/Bootstrap) + netCell AUROC
# =========================================================
def compute_relation_auroc_updown_delta_netcells(
    score_up: pd.DataFrame,        # cells × traits
    score_down: pd.DataFrame,      # cells × traits
    celltype_codes: np.ndarray,    # length=cells
    ct_levels: list,
    min_cells_in_type: int = 20,
    one_sided_updown: bool = True,
    delta_two_sided: bool = True,
    netcell_two_sided: bool = True,
    final_sig_from_delta: bool = False,   # 这里不再用 delta 作为 final_p/final_q 来源
    delta_p_method: str = "delong",       # "delong" or "bootstrap"
):
    """
    逻辑修改版：

    1) 先分别计算 AUROC_up / AUROC_down / AUROC_netCell
    2) 用 ef_delta = ef_up - ef_down 判方向：
       - ef_delta > 0  => final_side = "UP"
       - ef_delta < 0  => final_side = "DOWN"
       - ef_delta == 0 => 这里默认归到 "UP"（与你原先 >=0 一致）
    3) final 的取值规则：
       - UP   : final_AUROC  = AUROC_up
                final_effect = AUROC_effect_up
                final_p/q    = p_up / q_up
                final_sig_source = "AUROC_up"
       - DOWN : final_AUROC  = AUROC_down
                final_effect = - AUROC_effect_down
                final_p/q    = p_down / q_down
                final_sig_source = "AUROC_down_negated_effect"

    注意：
    - final_AUROC 保持非负方向判别意义（通常 >= 0.5）
    - final_effect 才是最终 signed trait-celltype 相关性
    """

    traits = list(score_up.columns)
    if traits != list(score_down.columns):
        raise ValueError("score_up/down columns must match (same traits, same order).")

    rng = np.random.default_rng(RANDOM_SEED)

    N = int(score_up.shape[0])
    n_types = len(ct_levels)
    n_ct = np.bincount(celltype_codes, minlength=n_types).astype(int)
    keep = n_ct >= min_cells_in_type
    n_traits = len(traits)

    # ---------- matrices ----------
    au_up = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    ef_up = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    au_dn = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    ef_dn = np.full((n_types, n_traits), np.nan, dtype=np.float32)

    # ef_delta = ef_up - ef_down
    ef_delta = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    p_delta = np.full((n_types, n_traits), np.nan, dtype=np.float64)

    # netCell
    au_netcell = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    ef_netcell = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    p_netcell = np.full((n_types, n_traits), np.nan, dtype=np.float64)

    # final
    au_final = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    ef_final = np.full((n_types, n_traits), np.nan, dtype=np.float32)
    p_final = np.full((n_types, n_traits), np.nan, dtype=np.float64)

    # single-side p
    p_up_mat = np.full((n_types, n_traits), np.nan, dtype=np.float64)
    p_dn_mat = np.full((n_types, n_traits), np.nan, dtype=np.float64)

    recs = []

    if N <= 1:
        m_au_up = pd.DataFrame(au_up, index=ct_levels, columns=traits)
        m_au_dn = pd.DataFrame(au_dn, index=ct_levels, columns=traits)
        m_ef_up = pd.DataFrame(ef_up, index=ct_levels, columns=traits)
        m_ef_dn = pd.DataFrame(ef_dn, index=ct_levels, columns=traits)
        m_ef_del = pd.DataFrame(ef_delta, index=ct_levels, columns=traits)
        m_au_net = pd.DataFrame(au_netcell, index=ct_levels, columns=traits)
        m_ef_net = pd.DataFrame(ef_netcell, index=ct_levels, columns=traits)
        m_au_final = pd.DataFrame(au_final, index=ct_levels, columns=traits)
        m_ef_final = pd.DataFrame(ef_final, index=ct_levels, columns=traits)

        mats = {
            # AUROC matrices
            "AUROC_up": m_au_up,
            "AUROC_down": m_au_dn,
            "AUROC_netCell": m_au_net,
            "final_AUROC": m_au_final,

            # effect matrices: 新旧键名都保留
            "effect_up": m_ef_up,
            "effect_down": m_ef_dn,
            "effect_delta": m_ef_del,
            "effect_netCell": m_ef_net,
            "final_effect": m_ef_final,

            # 新命名也保留
            "AUROC_effect_up": m_ef_up,
            "AUROC_effect_down": m_ef_dn,
            "AUROC_effect_net_DELTA": m_ef_del,
            "AUROC_effect_netCell": m_ef_net,
        } 
        return mats, pd.DataFrame()

    for j, tr in enumerate(traits):
        s_up = score_up[tr].to_numpy(dtype=np.float64)
        s_dn = score_down[tr].to_numpy(dtype=np.float64)

        if (not np.isfinite(s_up).any()) and (not np.isfinite(s_dn).any()):
            continue

        # ranks for AUROC(up/down)
        r_up = _rankdata_average(s_up)
        r_dn = _rankdata_average(s_dn)
        sum_r_up = np.bincount(celltype_codes, weights=r_up, minlength=n_types).astype(np.float64)
        sum_r_dn = np.bincount(celltype_codes, weights=r_dn, minlength=n_types).astype(np.float64)

        # netCell score and rank
        s_net = s_up - s_dn
        r_net = _rankdata_average(s_net)
        sum_r_net = np.bincount(celltype_codes, weights=r_net, minlength=n_types).astype(np.float64)

        # p vectors for BH across celltypes (within this trait)
        p_up_trait = np.full(n_types, np.nan, dtype=np.float64)
        p_dn_trait = np.full(n_types, np.nan, dtype=np.float64)
        p_delta_trait = np.full(n_types, np.nan, dtype=np.float64)
        p_netcell_trait = np.full(n_types, np.nan, dtype=np.float64)
        p_final_trait = np.full(n_types, np.nan, dtype=np.float64)

        tmp_rows = []

        for i in range(n_types):
            if not keep[i]:
                continue

            n1 = int(n_ct[i])
            n0 = int(N - n1)
            if n1 <= 0 or n0 <= 0:
                continue

            denom = float(n1 * n0)
            mean_U = denom / 2.0
            var_U = denom * (N + 1.0) / 12.0   # no tie correction, keep consistent with your original

            # labels for DeLong / bootstrap
            y01 = (celltype_codes == i).astype(np.int8)

            # =====================
            # UP
            # =====================
            Uu = float(sum_r_up[i] - (n1 * (n1 + 1) / 2.0))
            au_u = Uu / denom
            ef_u = au_u - 0.5

            if var_U > 0:
                z_u = (Uu - mean_U) / sqrt(var_U)
                p_u = _norm_sf(z_u) if one_sided_updown else (2.0 * min(_norm_cdf(z_u), _norm_sf(z_u)))
            else:
                z_u = np.nan
                p_u = np.nan

            # =====================
            # DOWN
            # =====================
            Ud = float(sum_r_dn[i] - (n1 * (n1 + 1) / 2.0))
            au_d = Ud / denom
            ef_d = au_d - 0.5

            if var_U > 0:
                z_d = (Ud - mean_U) / sqrt(var_U)
                p_d = _norm_sf(z_d) if one_sided_updown else (2.0 * min(_norm_cdf(z_d), _norm_sf(z_d)))
            else:
                z_d = np.nan
                p_d = np.nan

            # =====================
            # DELTA = effect_up - effect_down
            # 等价于 AUROC_up - AUROC_down
            # =====================
            ef_del = ef_u - ef_d

            if delta_p_method.lower() == "bootstrap":
                dout = bootstrap_auc_diff_pvalue(
                    scores1=s_up,
                    scores2=s_dn,
                    labels01=y01,
                    n_boot=BOOT_N,
                    two_sided=delta_two_sided,
                    max_in=BOOT_MAX_IN,
                    max_out=BOOT_MAX_OUT,
                    rng=rng
                )
            else:
                dout = delong_auc_diff_pvalue(
                    scores1=s_up,
                    scores2=s_dn,
                    labels01=y01,
                    two_sided=delta_two_sided,
                    max_total_cells=DELONG_MAX_TOTAL_CELLS,
                    rng=rng
                )

            z_del = dout["z"]
            p_del = dout["p"]
            del_var = dout["var"]
            del_auc_test = dout["delta_auc"]
            n_used = dout["n_used"]

            # =====================
            # netCell
            # =====================
            Un = float(sum_r_net[i] - (n1 * (n1 + 1) / 2.0))
            au_n = Un / denom
            ef_n = au_n - 0.5

            if var_U > 0:
                z_n = (Un - mean_U) / sqrt(var_U)
                p_n = (2.0 * min(_norm_cdf(z_n), _norm_sf(z_n))) if netcell_two_sided else _norm_sf(z_n)
            else:
                z_n = np.nan
                p_n = np.nan

            # =====================
            # FINAL PICK
            # =====================
            # delta > 0  => UP
            # delta < 0  => DOWN
            # delta == 0 => 仍按 UP 处理（和你原来 ef_del >= 0 一致）
            final_side = "UP" if (np.isfinite(ef_del) and ef_del >= 0) else "DOWN"

            if final_side == "UP":
                au_f = au_u
                ef_f = ef_u
                p_f = p_u
                final_sig_source = "AUROC_up"
            else:
                au_f = au_d
                ef_f = -ef_d      # 这里加负号，输出真正的 signed 相关性
                p_f = p_d
                final_sig_source = "AUROC_down_negated_effect"

            # save matrices
            au_up[i, j] = au_u
            ef_up[i, j] = ef_u
            au_dn[i, j] = au_d
            ef_dn[i, j] = ef_d

            ef_delta[i, j] = ef_del
            p_delta[i, j] = p_del

            au_netcell[i, j] = au_n
            ef_netcell[i, j] = ef_n
            p_netcell[i, j] = p_n

            au_final[i, j] = au_f
            ef_final[i, j] = ef_f
            p_final[i, j] = p_f

            p_up_mat[i, j] = p_u
            p_dn_mat[i, j] = p_d

            p_up_trait[i] = p_u
            p_dn_trait[i] = p_d
            p_delta_trait[i] = p_del
            p_netcell_trait[i] = p_n
            p_final_trait[i] = p_f

            tmp_rows.append({
                "celltype": ct_levels[i],
                "trait": tr,

                "n_in": n1,
                "n_out": n0,

                # up
                "AUROC_up": au_u,
                "AUROC_effect_up": ef_u,
                "z_up": z_u,
                "p_up": p_u,

                # down
                "AUROC_down": au_d,
                "AUROC_effect_down": ef_d,
                "z_down": z_d,
                "p_down": p_d,

                # delta
                "AUROC_effect_delta": ef_del,
                "delta_auc_test_used_by_p": del_auc_test,
                "delta_var": del_var,
                "z_delta": z_del,
                "p_delta": p_del,
                "delta_n_used": n_used,

                # netCell
                "AUROC_netCell": au_n,
                "AUROC_effect_netCell": ef_n,
                "z_netCell": z_n,
                "p_netCell": p_n,

                # final
                "direction": ("POS(up-dominant)" if final_side == "UP" else "NEG(down-dominant)"),
                "final_side": final_side,
                "final_AUROC": au_f,
                "final_effect": ef_f,
                "final_p": p_f,
                "final_sig_source": final_sig_source,

                "_i": i,
                "_j": j,
            })

        if len(tmp_rows) == 0:
            continue

        # ---------- BH across celltypes for this trait ----------
        q_up_trait = _bh_fdr_1d(p_up_trait)
        q_dn_trait = _bh_fdr_1d(p_dn_trait)
        q_delta_trait = _bh_fdr_1d(p_delta_trait)
        q_netcell_trait = _bh_fdr_1d(p_netcell_trait)
        q_final_trait = _bh_fdr_1d(p_final_trait)

        for row in tmp_rows:
            i = row["_i"]

            row["q_up_trait_across_celltypes"] = q_up_trait[i]
            row["q_down_trait_across_celltypes"] = q_dn_trait[i]
            row["q_delta_trait_across_celltypes"] = q_delta_trait[i]
            row["q_netCell_trait_across_celltypes"] = q_netcell_trait[i]
            row["final_q_trait_across_celltypes"] = q_final_trait[i]

            recs.append(row)

    stats_long = pd.DataFrame(recs)

    # ---------- BH across traits within each celltype ----------
    if len(stats_long) > 0:
        stats_long["q_up_celltype_across_traits"] = np.nan
        stats_long["q_down_celltype_across_traits"] = np.nan
        stats_long["q_delta_celltype_across_traits"] = np.nan
        stats_long["q_netCell_celltype_across_traits"] = np.nan
        stats_long["final_q_celltype_across_traits"] = np.nan

        for ct, sub_idx in stats_long.groupby("celltype").groups.items():
            idx = list(sub_idx)

            stats_long.loc[idx, "q_up_celltype_across_traits"] = _bh_fdr_1d(
                stats_long.loc[idx, "p_up"].to_numpy(dtype=float)
            )
            stats_long.loc[idx, "q_down_celltype_across_traits"] = _bh_fdr_1d(
                stats_long.loc[idx, "p_down"].to_numpy(dtype=float)
            )
            stats_long.loc[idx, "q_delta_celltype_across_traits"] = _bh_fdr_1d(
                stats_long.loc[idx, "p_delta"].to_numpy(dtype=float)
            )
            stats_long.loc[idx, "q_netCell_celltype_across_traits"] = _bh_fdr_1d(
                stats_long.loc[idx, "p_netCell"].to_numpy(dtype=float)
            )
            stats_long.loc[idx, "final_q_celltype_across_traits"] = _bh_fdr_1d(
                stats_long.loc[idx, "final_p"].to_numpy(dtype=float)
            )

        stats_long = stats_long.drop(columns=["_i", "_j"], errors="ignore")

    m_au_up = pd.DataFrame(au_up, index=ct_levels, columns=traits)
    m_au_dn = pd.DataFrame(au_dn, index=ct_levels, columns=traits)
    m_ef_up = pd.DataFrame(ef_up, index=ct_levels, columns=traits)
    m_ef_dn = pd.DataFrame(ef_dn, index=ct_levels, columns=traits)
    m_ef_del = pd.DataFrame(ef_delta, index=ct_levels, columns=traits)
    m_au_net = pd.DataFrame(au_netcell, index=ct_levels, columns=traits)
    m_ef_net = pd.DataFrame(ef_netcell, index=ct_levels, columns=traits)
    m_au_final = pd.DataFrame(au_final, index=ct_levels, columns=traits)
    m_ef_final = pd.DataFrame(ef_final, index=ct_levels, columns=traits)

    m_p_up = pd.DataFrame(p_up_mat, index=ct_levels, columns=traits)
    m_p_dn = pd.DataFrame(p_dn_mat, index=ct_levels, columns=traits)
    m_p_del = pd.DataFrame(p_delta, index=ct_levels, columns=traits)
    m_p_net = pd.DataFrame(p_netcell, index=ct_levels, columns=traits)
    m_p_final = pd.DataFrame(p_final, index=ct_levels, columns=traits)

    mats = {
        # 主 AUROC
        "AUROC_up": m_au_up,
        "AUROC_down": m_au_dn,
        "AUROC_netCell": m_au_net,
        "final_AUROC": m_au_final,

        # effect：旧键名
        "effect_up": m_ef_up,
        "effect_down": m_ef_dn,
        "effect_delta": m_ef_del,
        "effect_netCell": m_ef_net,
        "final_effect": m_ef_final,

        # effect：新键名
        "AUROC_effect_up": m_ef_up,
        "AUROC_effect_down": m_ef_dn,
        "AUROC_effect_net_DELTA": m_ef_del,
        "AUROC_effect_netCell": m_ef_net,

        # p
        "p_up": m_p_up,
        "p_down": m_p_dn,
        "p_delta": m_p_del,
        "p_netCell": m_p_net,
        "final_p": m_p_final,
    }
    return mats, stats_long

# =========================================================
# 14) Stage runner: unified output + thresholds + topk
# =========================================================
def process_stage_auroc_all_and_thresholds(
    score_up: pd.DataFrame,
    score_down: pd.DataFrame,
    method: str,
    stage_name: str,
    out_dir: str,
    celltype_codes: np.ndarray,
    ct_levels: list,
):
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_stats_long.csv.gz")

    m_up_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_up.csv.gz")
    m_dn_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_down.csv.gz")
    m_del_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_effect_delta.csv.gz")
    m_netC_auc_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_netCell.csv.gz")
    m_netC_eff_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_AUROC_effect_netCell.csv.gz")
    m_finA_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_final_AUROC.csv.gz")
    m_finE_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_final_effect.csv.gz")

    thr_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_exploreThresholds_thresholds.csv")
    asg_path = os.path.join(out_dir, f"{PREFIX}_{method}_{stage_name}_exploreThresholds_assignments.csv")

    if SKIP_EXISTING and all(os.path.exists(p) for p in [
        stats_path, m_finA_path, m_finE_path, thr_path, asg_path, m_del_path, m_netC_auc_path, m_netC_eff_path
    ]):
        return stats_path

    mats, stats_long = compute_relation_auroc_updown_delta_netcells(
        score_up=score_up,
        score_down=score_down,
        celltype_codes=celltype_codes,
        ct_levels=ct_levels,
        min_cells_in_type=MIN_CELLS_IN_TYPE,
        one_sided_updown=P_ONE_SIDED_UPDOWN,
        delta_two_sided=DELTA_TWO_SIDED,
        netcell_two_sided=NETCELL_TWO_SIDED,
        final_sig_from_delta=FINAL_SIG_FROM_DELTA,
        delta_p_method=NET_P_METHOD,
    )

    # save matrices
    mats["AUROC_up"].to_csv(m_up_path, compression="gzip")
    mats["AUROC_down"].to_csv(m_dn_path, compression="gzip")
    mats["effect_delta"].to_csv(m_del_path, compression="gzip")
    mats["AUROC_netCell"].to_csv(m_netC_auc_path, compression="gzip")
    mats["effect_netCell"].to_csv(m_netC_eff_path, compression="gzip")
    mats["final_AUROC"].to_csv(m_finA_path, compression="gzip")
    mats["final_effect"].to_csv(m_finE_path, compression="gzip")

    # unified long table
    stats_df = stats_long.reset_index()
    stats_df["method"] = method
    stats_df["stage"] = stage_name
    stats_df.to_csv(stats_path, index=False, compression="gzip")

    # TopK summaries
    topk_dir = os.path.join(out_dir, "TopK_Summaries")
    export_topk_summaries(
        effect_df=mats["effect_delta"],
        stats_long=stats_df,
        out_prefix=f"{PREFIX}_{method}_{stage_name}_DELTA",
        out_dir=topk_dir,
        topk=TOPK_EXPORT,
    )
    export_topk_summaries(
        effect_df=mats["effect_netCell"],
        stats_long=stats_df,
        out_prefix=f"{PREFIX}_{method}_{stage_name}_NETCELL",
        out_dir=topk_dir,
        topk=TOPK_EXPORT,
    )
    export_topk_summaries(
        effect_df=mats["final_effect"],
        stats_long=stats_df,
        out_prefix=f"{PREFIX}_{method}_{stage_name}_FINAL",
        out_dir=topk_dir,
        topk=TOPK_EXPORT,
    )

    # exploreThresholds on final_AUROC
    thr_df, assign_df = run_aucell_exploreThresholds_on_matrix(
        auc_like_df=mats["final_AUROC"],
        thrP=THRP,
        smallestPopPercent=SMALLEST_POP_PERCENT,
        nCores=R_NCORES,
        seed=0,
        neutral_fill=0.5
    )
    thr_df.to_csv(thr_path)
    assign_df.to_csv(asg_path)

    return stats_path


# =========================================================
# 15) Stage builder: per method (up/down together; net matrix saved as up-down per stage)
# =========================================================
def build_save_and_run_all_stages_for_method(
    mat_up_raw: pd.DataFrame,
    mat_down_raw: pd.DataFrame,
    method: str,
    mats_dir: str,
    out_dir: str,
    celltype_codes: np.ndarray,
    ct_levels: list,
):
    # S1
    S1_up = mat_up_raw.astype(DTYPE_OUT, copy=False)
    S1_dn = mat_down_raw.astype(DTYPE_OUT, copy=False)
    S1_net = (S1_up - S1_dn).astype(DTYPE_OUT, copy=False)

    if "S1_raw" in STAGES_TO_RUN:
        save_stage(S1_up, os.path.join(mats_dir, f"{PREFIX}_{method}_up_S1_raw.csv.gz"))
        save_stage(S1_dn, os.path.join(mats_dir, f"{PREFIX}_{method}_down_S1_raw.csv.gz"))
        save_stage(S1_net, os.path.join(mats_dir, f"{PREFIX}_{method}_net_S1_raw.csv.gz"))
        yield process_stage_auroc_all_and_thresholds(
            S1_up, S1_dn, method, "S1_raw", out_dir, celltype_codes, ct_levels
        )

    # S2
    S2_up = mat_up_raw.copy()
    S2_dn = mat_down_raw.copy()
    _col_zscore_inplace(S2_up, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    _col_zscore_inplace(S2_dn, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    S2_net = (S2_up - S2_dn).astype(DTYPE_OUT, copy=False)

    if "S2_geneSetZ" in STAGES_TO_RUN:
        save_stage(S2_up, os.path.join(mats_dir, f"{PREFIX}_{method}_up_S2_geneSetZ.csv.gz"))
        save_stage(S2_dn, os.path.join(mats_dir, f"{PREFIX}_{method}_down_S2_geneSetZ.csv.gz"))
        save_stage(S2_net, os.path.join(mats_dir, f"{PREFIX}_{method}_net_S2_geneSetZ.csv.gz"))
        yield process_stage_auroc_all_and_thresholds(
            S2_up, S2_dn, method, "S2_geneSetZ", out_dir, celltype_codes, ct_levels
        )

    # S3
    S3_up = S2_up.copy()
    S3_dn = S2_dn.copy()
    _row_zscore_inplace(S3_up, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    _row_zscore_inplace(S3_dn, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    S3_net = (S3_up - S3_dn).astype(DTYPE_OUT, copy=False)

    if "S3_geneSetZ_cellZ" in STAGES_TO_RUN:
        save_stage(S3_up, os.path.join(mats_dir, f"{PREFIX}_{method}_up_S3_geneSetZ_cellZ.csv.gz"))
        save_stage(S3_dn, os.path.join(mats_dir, f"{PREFIX}_{method}_down_S3_geneSetZ_cellZ.csv.gz"))
        save_stage(S3_net, os.path.join(mats_dir, f"{PREFIX}_{method}_net_S3_geneSetZ_cellZ.csv.gz"))
        yield process_stage_auroc_all_and_thresholds(
            S3_up, S3_dn, method, "S3_geneSetZ_cellZ", out_dir, celltype_codes, ct_levels
        )

    # S4
    S4_up = S3_up.copy()
    S4_dn = S3_dn.copy()
    _col_zscore_inplace(S4_up, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    _col_zscore_inplace(S4_dn, col_block=COL_BLOCK, dtype_out=DTYPE_OUT)
    S4_net = (S4_up - S4_dn).astype(DTYPE_OUT, copy=False)

    if "S4_geneSetZ_cellZ_geneSetZ" in STAGES_TO_RUN:
        save_stage(S4_up, os.path.join(mats_dir, f"{PREFIX}_{method}_up_S4_geneSetZ_cellZ_geneSetZ.csv.gz"))
        save_stage(S4_dn, os.path.join(mats_dir, f"{PREFIX}_{method}_down_S4_geneSetZ_cellZ_geneSetZ.csv.gz"))
        save_stage(S4_net, os.path.join(mats_dir, f"{PREFIX}_{method}_net_S4_geneSetZ_cellZ_geneSetZ.csv.gz"))
        yield process_stage_auroc_all_and_thresholds(
            S4_up, S4_dn, method, "S4_geneSetZ_cellZ_geneSetZ", out_dir, celltype_codes, ct_levels
        )


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mats_dir = os.path.join(OUT_DIR, "cellTraitMatrices")
    schemeC_dir = os.path.join(OUT_DIR, SCHEMEC_DIRNAME)
    os.makedirs(mats_dir, exist_ok=True)
    os.makedirs(schemeC_dir, exist_ok=True)

    print("[1/5] Loading adata...")
    adata = sc.read_h5ad(H5AD_PATH)
    if CELLTYPE_KEY not in adata.obs.columns:
        raise KeyError(f"{CELLTYPE_KEY} not in adata.obs")

    print("[2/5] Normalizing expression (normalize_total + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("[3/5] Loading Z_mat...")
    Z_mat = load_Z_mat_robust(ZMAT_PATH, adata)
    if Z_mat.shape[1] == 0:
        raise ValueError("Z_mat has 0 traits/columns after loading.")
    print(f"      Z_mat genes={Z_mat.shape[0]}, traits={Z_mat.shape[1]}")

    celltype = adata.obs[CELLTYPE_KEY].astype(str)
    ct_levels = sorted(celltype.unique())
    ct_to_code = {ct: i for i, ct in enumerate(ct_levels)}
    celltype_codes = celltype.map(ct_to_code).to_numpy(dtype=np.int32)

    print("[4/5] Computing RAW scMRS/wAUCell scores (parallel)...")
    score_df_raw_full = score_all_traits_parallel_ud_raw(
        adata,
        Z_mat,
        top_n=TOP_N_GENES,
        min_valid_genes=MIN_VALID_GENES,
        n_jobs=N_JOBS,
        top_frac=TOP_FRAC_WAUCELL,
        topk_batch_size=TOPK_BATCH_SIZE,
        waucell_cell_batch_size=WAUCELL_CELL_BATCH_SIZE,
        prefix=PREFIX,
    )
    raw_full_path = os.path.join(OUT_DIR, f"{PREFIX}_ALLKINDS_raw_full.csv.gz")
    if (not SKIP_EXISTING) or (not os.path.exists(raw_full_path)):
        score_df_raw_full.to_csv(raw_full_path, compression="gzip")
    print(f"      saved raw full table -> {raw_full_path}")

    print("[5/5] Building stages + running AUROC Up/Down + Δp(DeLong/Bootstrap) + netCell ...")
    schemeC_all = []

    for method in METHODS_TO_RUN:
        print(f"    - method: {method}")

        mat_up_raw = extract_kind_matrix(score_df_raw_full, prefix=PREFIX, kind=f"{method}_up")
        mat_dn_raw = extract_kind_matrix(score_df_raw_full, prefix=PREFIX, kind=f"{method}_down")

        for stats_path in build_save_and_run_all_stages_for_method(
            mat_up_raw=mat_up_raw,
            mat_down_raw=mat_dn_raw,
            method=method,
            mats_dir=mats_dir,
            out_dir=schemeC_dir,
            celltype_codes=celltype_codes,
            ct_levels=ct_levels,
        ):
            schemeC_all.append(pd.read_csv(stats_path))
            print(f"      saved -> {stats_path}")

        del mat_up_raw, mat_dn_raw

    if len(schemeC_all) > 0:
        df_all = pd.concat(schemeC_all, axis=0, ignore_index=True)
        all_path = os.path.join(OUT_DIR, f"{PREFIX}_SchemeC_AUROC_UpDownNet_ALL_stats_long.csv.gz")
        df_all.to_csv(all_path, index=False, compression="gzip")
        print(f"      saved ALL -> {all_path}")

    print("done")


if __name__ == "__main__":
    main()
