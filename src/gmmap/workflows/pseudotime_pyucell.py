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
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import spearmanr, rankdata
from joblib import Parallel, delayed


# =========================
# Config
# =========================
H5AD_PATH = "data/input.h5ad"
ZMAT_PATH = "data/MAGMA_zstat.csv"

CELLTYPE_COL =  "knn_cell_type"     # 改成你的列名
PSEUDOTIME_COL = "monocle_pseudotime"    # 改成你的拟时序列名
PREFIX = "Meta"
OUT_DIR = "Meta_pyUCell_pseudotime_outputs"

TOP_N_GENES = 1000
MIN_VALID_GENES = 100

N_JOBS_UCELL = -1
N_JOBS_COR = 8

UCELL_CHUNK_SIZE = 500
UCELL_SIGNATURE_BATCH = 128
UCELL_MAX_RANK = None
UCELL_W_NEG = 1.0
TIES_METHOD = "average"

# 更接近官方 pyUCell 默认行为；若想保留你原来逻辑，可改回 "skip"
MISSING_GENES = "impute"

MIN_CELLS_PER_LINEAGE = 30
MIN_PCT_DETECTED = 0.01

DO_KNN_SMOOTH = False
KNN_K = 15
KNN_USE_REP = "X_pca"
KNN_DECAY = 0.1
KNN_GRAPH_KEY = None
KNN_UP_ONLY = False


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
        "RVCSB b: renal vesicle/comma-shapedbody b",
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

            sig_indices[sig_name] = {
                "pos": pos_idx,
                "neg": neg_idx,
            }

        return sig_indices

    @staticmethod
    def get_rankings(data, layer=None, max_rank=1500, ties_method="average"):
        """
        贴近官方 pyUCell:
        - 每个 cell 仅对非零表达基因排名
        - ties 用 scipy.stats.rankdata(..., method=ties_method)
        - 仅保留 rank <= max_rank 的条目
        - 返回 shape=(genes, cells) 的 csr sparse rank matrix
        """
        if isinstance(data, AnnData):
            X = data.layers[layer] if layer else data.X
        else:
            X = data

        n_cells, n_genes = X.shape
        data_parts = []
        row_parts = []
        col_parts = []

        for j in range(n_cells):
            if sparse.issparse(X):
                row = X.getrow(j)
                nz_idx = row.indices
                nz_vals = row.data
            else:
                row = np.asarray(X[j]).ravel()
                np.nan_to_num(row, copy=False)
                nz_idx = np.flatnonzero(row)
                nz_vals = row[nz_idx]

            if len(nz_idx) == 0:
                continue

            # 官方 pyUCell 风格：先 rankdata，再转 int
            ranks = rankdata(-nz_vals, method=ties_method).astype(np.int32)

            keep = ranks <= max_rank
            if not np.any(keep):
                continue

            kept_idx = np.asarray(nz_idx[keep], dtype=np.int32)
            kept_ranks = np.asarray(ranks[keep], dtype=np.int32)

            data_parts.append(kept_ranks)
            row_parts.append(kept_idx)
            col_parts.append(np.full(len(kept_idx), j, dtype=np.int32))

        if len(data_parts) == 0:
            return sparse.csr_matrix((n_genes, n_cells), dtype=np.int32)

        data_all = np.concatenate(data_parts)
        row_all = np.concatenate(row_parts)
        col_all = np.concatenate(col_parts)

        return sparse.coo_matrix(
            (data_all, (row_all, col_all)),
            shape=(n_genes, n_cells),
            dtype=np.int32,
        ).tocsr()

    @staticmethod
    def _calculate_U(ranks, idx, max_rank=1500):
        """
        贴近官方 pyUCell:
        - 缺失基因(-1) 当作 max_rank
        - 稀疏矩阵中未存的位置(=0) 当作 max_rank
        """
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

            if sparse.issparse(present_ranks):
                present_ranks = present_ranks.toarray()
            else:
                present_ranks = np.asarray(present_ranks)

            if present_ranks.ndim == 1:
                present_ranks = present_ranks[np.newaxis, :]

            present_ranks = present_ranks.astype(np.float32)

            # 官方逻辑：没存下来的位置 = 0，按 max_rank 处理
            present_ranks[present_ranks == 0] = max_rank
            rank_sum += present_ranks.sum(axis=0)

        s_min = lgt * (lgt + 1) / 2.0
        s_max = lgt * max_rank
        denom = s_max - s_min

        if denom <= 0:
            return np.zeros(n_cells, dtype=np.float32)

        score = 1.0 - (rank_sum - s_min) / denom
        score = np.clip(score, 0.0, 1.0)
        return score.astype(np.float32)

    @staticmethod
    def _score_rank_matrix(ranks, sig_indices, w_neg=1.0, max_rank=1500):
        n_cells = ranks.shape[1]
        sig_names = list(sig_indices.keys())
        scores = np.zeros((n_cells, len(sig_names)), dtype=np.float32)

        for j, sig_name in enumerate(sig_names):
            idx_dict = sig_indices[sig_name]
            pos_idx = idx_dict["pos"]
            neg_idx = idx_dict["neg"]

            pos_score = (
                uc._calculate_U(ranks, pos_idx, max_rank=max_rank)
                if len(pos_idx) > 0 else np.zeros(n_cells, dtype=np.float32)
            )
            neg_score = (
                uc._calculate_U(ranks, neg_idx, max_rank=max_rank)
                if len(neg_idx) > 0 else np.zeros(n_cells, dtype=np.float32)
            )

            # 官方 signed score 风格：pos - w_neg * neg，再截负值为0
            signed_score = pos_score - (w_neg * neg_score)
            signed_score[signed_score < 0] = 0.0

            scores[:, j] = signed_score

        return sig_names, scores

    @staticmethod
    def compute_ucell_scores(
        adata,
        signatures,
        layer=None,
        max_rank=1500,
        ties_method="average",
        missing_genes="impute",
        chunk_size=500,
        w_neg=1.0,
        suffix="_UCell",
        n_jobs=-1,
    ):
        if max_rank is None:
            max_rank = min(1500, adata.n_vars)

        genes = np.asarray(adata.var_names).astype(str)
        sig_indices = uc._prepare_sig_indices(
            signatures=signatures,
            genes=genes,
            missing_genes=missing_genes,
        )
        sig_names = list(sig_indices.keys())

        if len(sig_names) == 0:
            return None

        chunks = [
            (s, min(s + chunk_size, adata.n_obs))
            for s in range(0, adata.n_obs, chunk_size)
        ]

        def process_chunk(start, end):
            X_chunk = adata.layers[layer][start:end, :] if layer else adata.X[start:end, :]
            ranks_chunk = uc.get_rankings(
                X_chunk,
                max_rank=max_rank,
                ties_method=ties_method,
            )
            _, scores_chunk = uc._score_rank_matrix(
                ranks_chunk,
                sig_indices=sig_indices,
                w_neg=w_neg,
                max_rank=max_rank,
            )
            return start, end, scores_chunk

        if n_jobs == 1:
            results = [process_chunk(start, end) for start, end in chunks]
        else:
            # 用 threading 避免把整个 adata 大量复制到子进程
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(process_chunk)(start, end) for start, end in chunks
            )

        scores_all = np.zeros((adata.n_obs, len(sig_names)), dtype=np.float32)
        for start, end, scores_chunk in results:
            scores_all[start:end, :] = scores_chunk

        for j, sig_name in enumerate(sig_names):
            adata.obs[f"{sig_name}{suffix}"] = scores_all[:, j]

        return None

    @staticmethod
    def smooth_knn_scores(
        adata,
        obs_columns,
        k=10,
        use_rep="X_pca",
        decay=0.1,
        up_only=False,
        graph_key=None,
        suffix="_kNN",
    ):
        if not (0 < decay < 1):
            raise ValueError("decay must be between 0 and 1")

        if graph_key is None:
            sc.pp.neighbors(adata, n_neighbors=k, use_rep=use_rep)
            conn = adata.obsp["connectivities"]
        else:
            if graph_key not in adata.obsp:
                raise KeyError(f"{graph_key} not found in adata.obsp")
            conn = adata.obsp[graph_key]

        if not sparse.issparse(conn):
            conn = sparse.csr_matrix(conn)
        else:
            conn = conn.tocsr()

        n_cells = adata.n_obs

        for col in obs_columns:
            x = adata.obs[col].to_numpy(dtype=np.float32)
            smoothed = np.zeros(n_cells, dtype=np.float32)

            for i in range(n_cells):
                neigh = conn[i].indices
                weights = conn[i].data

                if len(neigh) > 0:
                    order = np.argsort(weights)[::-1]
                    neigh = neigh[order]

                # 把自己放在第一个位置
                all_idx = np.insert(neigh, 0, i)

                # 官方风格的指数衰减加权
                decay_weights = (1.0 - decay) ** np.arange(len(all_idx), dtype=np.float32)
                decay_weights = decay_weights / decay_weights.sum()

                smoothed[i] = np.sum(x[all_idx] * decay_weights)

            if up_only:
                smoothed = np.maximum(smoothed, x)

            adata.obs[f"{col}{suffix}"] = smoothed.astype(np.float32)

        return None


# =========================
# Helpers
# =========================
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
    q0 = ranked * n / (np.arange(1, n + 1))
    q0 = np.minimum.accumulate(q0[::-1])[::-1]
    q0 = np.clip(q0, 0, 1)
    out = np.empty_like(p0)
    out[order] = q0
    q[m] = out
    return q


def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def auto_choose_max_rank(adata):
    X = adata.X
    if issparse(X):
        X = X.tocsr()
        detected = np.diff(X.indptr)
    else:
        detected = np.asarray((X > 0).sum(axis=1)).ravel()
    med = float(np.median(detected))
    return int(round(min(1500, max(800, med))))


def load_Z_mat_robust(z_path, adata):
    Z = pd.read_csv(z_path, index_col=0)

    if Z.shape[1] >= 2:
        first_col = Z.columns[0]
        vals = Z[first_col].astype(str)
        var_names = set(map(str, adata.var_names))

        overlap_first = len(set(vals).intersection(var_names)) / max(
            1, min(len(vals), len(var_names))
        )
        overlap_index = len(set(map(str, Z.index)).intersection(var_names)) / max(
            1, min(len(Z.index), len(var_names))
        )

        if (overlap_first > 0.2) and (overlap_index < 0.05) and vals.is_unique:
            Z = Z.set_index(first_col)

    Z.index = Z.index.astype(str)
    return Z


def extract_signature_up_down(Z_mat, trait, top_n=1000):
    z = pd.to_numeric(Z_mat[trait], errors="coerce").dropna()
    up = z[z > 0].sort_values(ascending=False).head(top_n)
    dn = z[z < 0].sort_values(ascending=True).head(top_n)
    return up, dn


def build_signed_signatures(Z_mat, adata, top_n=1000, min_valid_genes=100):
    gene_universe = set(map(str, adata.var_names))
    signatures = {}
    summary_rows = []

    for trait in Z_mat.columns:
        up_z, dn_z = extract_signature_up_down(Z_mat, trait, top_n=top_n)

        up_genes = [g for g in up_z.index.astype(str) if g in gene_universe]
        dn_genes = [g for g in dn_z.index.astype(str) if g in gene_universe]

        keep = (len(up_genes) >= min_valid_genes) or (len(dn_genes) >= min_valid_genes)

        summary_rows.append({
            "trait": trait,
            "n_up_requested": int(len(up_z)),
            "n_down_requested": int(len(dn_z)),
            "n_up_in_data": int(len(up_genes)),
            "n_down_in_data": int(len(dn_genes)),
            "keep": bool(keep),
        })

        if not keep:
            continue

        signed_genes = [f"{g}+" for g in up_genes] + [f"{g}-" for g in dn_genes]
        signatures[trait] = signed_genes

    summary_df = pd.DataFrame(summary_rows).sort_values("trait")
    return signatures, summary_df


def scale01(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if finite.sum() == 0:
        return out
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax == xmin:
        out[finite] = 0.0
        return out
    out[finite] = (x[finite] - xmin) / (xmax - xmin)
    return out


def correlate_one_trait(scores, pseudotime, trait, min_pct_detected=0.01):
    x = np.asarray(scores, dtype=float)
    y = np.asarray(pseudotime, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None

    detected = (x[ok] > 0).mean()
    if detected < min_pct_detected:
        return None

    if np.nanstd(x[ok]) == 0:
        return None

    rho, p = spearmanr(x[ok], y[ok], nan_policy="omit")
    return {
        "feature": trait,
        "rho": float(rho),
        "pvalue": float(p),
        "detect_rate": float(detected),
        "n_cells_used": int(ok.sum()),
    }


def run_lineage_correlation(score_df, meta_df, lineage_name, lineage_celltypes, min_cells=30, n_jobs=8):
    cells = meta_df.index[
        meta_df[CELLTYPE_COL].isin(lineage_celltypes) &
        np.isfinite(meta_df[PSEUDOTIME_COL].values)
    ]

    if len(cells) < min_cells:
        return pd.DataFrame()

    meta_sub = meta_df.loc[cells].copy()
    meta_sub = meta_sub.sort_values(PSEUDOTIME_COL)
    meta_sub["pt_lineage_01"] = scale01(meta_sub[PSEUDOTIME_COL].values)

    score_sub = score_df.loc[meta_sub.index]
    traits = list(score_sub.columns)

    results = Parallel(n_jobs=n_jobs)(
        delayed(correlate_one_trait)(
            score_sub[tr].values,
            meta_sub["pt_lineage_01"].values,
            tr,
            MIN_PCT_DETECTED
        )
        for tr in traits
    )

    results = [r for r in results if r is not None]
    if len(results) == 0:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    res_df["qvalue"] = bh_fdr(res_df["pvalue"].values)
    res_df["lineage"] = lineage_name
    res_df["n_cells_lineage"] = int(meta_sub.shape[0])

    res_df = res_df.sort_values(["rho", "pvalue"], ascending=[False, True]).reset_index(drop=True)
    return res_df


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    score_dir = os.path.join(OUT_DIR, "cellTraitMatrices")
    corr_dir = os.path.join(OUT_DIR, "pseudotimeCorrelation")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(corr_dir, exist_ok=True)

    print(">>> Load adata")
    adata = sc.read_h5ad(H5AD_PATH)

    if CELLTYPE_COL not in adata.obs.columns:
        raise KeyError(f"{CELLTYPE_COL} not found in adata.obs")
    if PSEUDOTIME_COL not in adata.obs.columns:
        raise KeyError(f"{PSEUDOTIME_COL} not found in adata.obs")

    print(">>> Load Z matrix")
    Z_mat = load_Z_mat_robust(ZMAT_PATH, adata)

    print(">>> Build signed signatures")
    signatures, sig_summary = build_signed_signatures(
        Z_mat=Z_mat,
        adata=adata,
        top_n=TOP_N_GENES,
        min_valid_genes=MIN_VALID_GENES
    )
    sig_summary.to_csv(
        os.path.join(corr_dir, f"{PREFIX}_signed_signature_summary.csv.gz"),
        index=False,
        compression="gzip"
    )

    if len(signatures) == 0:
        raise ValueError("No valid signatures remained after filtering.")

    print(">>> Choose max_rank")
    max_rank = UCELL_MAX_RANK if UCELL_MAX_RANK is not None else auto_choose_max_rank(adata)
    print(f">>> max_rank = {max_rank}")

    print(">>> Compute signed pyUCell-style scores")
    score_blocks = []
    trait_names = list(signatures.keys())

    for batch in chunks(trait_names, UCELL_SIGNATURE_BATCH):
        sig_batch = {k: signatures[k] for k in batch}

        uc.compute_ucell_scores(
            adata,
            signatures=sig_batch,
            max_rank=max_rank,
            ties_method=TIES_METHOD,
            chunk_size=UCELL_CHUNK_SIZE,
            missing_genes=MISSING_GENES,
            w_neg=UCELL_W_NEG,
            suffix="_signedUCell",
            n_jobs=N_JOBS_UCELL,
        )

        cols = [f"{k}_signedUCell" for k in batch]
        block = adata.obs[cols].copy()
        block.columns = batch
        score_blocks.append(block)

        # 清理临时列，避免 obs 过宽
        adata.obs.drop(columns=cols, inplace=True, errors="ignore")

    score_df = pd.concat(score_blocks, axis=1)
    score_df.index = adata.obs_names

    score_df.to_csv(
        os.path.join(score_dir, f"{PREFIX}_pyUCell_signed_scores.csv.gz"),
        compression="gzip"
    )

    if DO_KNN_SMOOTH:
        print(">>> Smooth signed scores with kNN")
        obs_cols = list(score_df.columns)
        adata.obs[obs_cols] = score_df.loc[adata.obs_names, obs_cols]

        uc.smooth_knn_scores(
            adata,
            obs_columns=obs_cols,
            k=KNN_K,
            use_rep=KNN_USE_REP,
            decay=KNN_DECAY,
            up_only=KNN_UP_ONLY,
            graph_key=KNN_GRAPH_KEY,
            suffix="_kNN"
        )

        smooth_cols = [f"{c}_kNN" for c in obs_cols]
        score_df = adata.obs[smooth_cols].copy()
        score_df.columns = obs_cols

        score_df.to_csv(
            os.path.join(score_dir, f"{PREFIX}_pyUCell_signed_scores_kNN.csv.gz"),
            compression="gzip"
        )

    meta_df = adata.obs[[CELLTYPE_COL, PSEUDOTIME_COL]].copy()

    print(">>> Run lineage-wise pseudotime correlations")
    lineage_res = []
    for lin_name, lin_types in LINEAGES.items():
        print(f"    - {lin_name}")
        res = run_lineage_correlation(
            score_df=score_df,
            meta_df=meta_df,
            lineage_name=lin_name,
            lineage_celltypes=lin_types,
            min_cells=MIN_CELLS_PER_LINEAGE,
            n_jobs=N_JOBS_COR
        )
        if res.shape[0] > 0:
            res.to_csv(
                os.path.join(corr_dir, f"{PREFIX}_{lin_name}_signedUCell_pseudotime_correlation.csv.gz"),
                index=False,
                compression="gzip"
            )
            lineage_res.append(res)

    if len(lineage_res) == 0:
        print("No valid lineage results.")
        return

    res_all = pd.concat(lineage_res, axis=0, ignore_index=True)
    res_all.to_csv(
        os.path.join(corr_dir, f"{PREFIX}_ALL_signedUCell_pseudotime_correlation.csv.gz"),
        index=False,
        compression="gzip"
    )

    top_pos = (
        res_all.query("rho > 0 and qvalue < 0.05")
        .sort_values(["lineage", "rho"], ascending=[True, False])
        .groupby("lineage", group_keys=False)
        .head(30)
    )
    top_neg = (
        res_all.query("rho < 0 and qvalue < 0.05")
        .sort_values(["lineage", "rho"], ascending=[True, True])
        .groupby("lineage", group_keys=False)
        .head(30)
    )

    top_pos.to_csv(
        os.path.join(corr_dir, f"{PREFIX}_TopPos_signedUCell_pseudotime.csv.gz"),
        index=False,
        compression="gzip"
    )
    top_neg.to_csv(
        os.path.join(corr_dir, f"{PREFIX}_TopNeg_signedUCell_pseudotime.csv.gz"),
        index=False,
        compression="gzip"
    )

    print(">>> Done")


if __name__ == "__main__":
    main()
