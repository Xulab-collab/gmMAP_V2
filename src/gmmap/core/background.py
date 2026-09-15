"""Matched-gene empirical background for gmMAP (scDRS-inspired, not scDRS).

Rank-based scores use symmetric calibration; no weighted-expression analytic
variance correction or pooled-cell scDRS p-values are claimed.
"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    q = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out


def gene_statistics(x, names, n_mean_bins=20, n_var_bins=20):
    if n_mean_bins < 1 or n_var_bins < 1:
        raise ValueError("Bin counts must be positive")
    if not pd.Index(names).is_unique:
        raise ValueError("Gene names must be unique; resolve duplicated genes first")
    if x.shape[0] < 3:
        raise ValueError("At least three cells are required")
    if sparse.issparse(x):
        x = x.astype(np.float64)
        if not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError("Use finite nonnegative library-normalized log1p expression, not scaled residuals")
        mean = np.asarray(x.mean(axis=0)).ravel()
        var = np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel() - mean**2, 0)
    else:
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("Use finite nonnegative library-normalized log1p expression, not scaled residuals")
        mean, var = x.mean(axis=0), x.var(axis=0)
    stats = pd.DataFrame({"mean": mean, "var": var}, index=pd.Index(names, name="gene"))
    stats["eligible"] = (var > 1e-12) & (mean > 0)
    stats["bin"] = -1
    def quantile_bins(v, n):
        if len(v) < 2 or v.nunique() == 1:
            return pd.Series(0, index=v.index, dtype=int)
        return pd.qcut(v, q=min(n, len(v)), labels=False, duplicates="drop").fillna(0).astype(int)
    eligible = stats.loc[stats.eligible]
    if eligible.empty:
        raise ValueError("No expressed variable genes remain")
    mb = quantile_bins(eligible["mean"], n_mean_bins)
    next_bin = 0
    for _, genes in mb.groupby(mb).groups.items():
        vb = quantile_bins(stats.loc[genes, "var"], n_var_bins)
        for _, members in vb.groupby(vb).groups.items():
            stats.loc[members, "bin"] = next_bin
            next_bin += 1
    return stats


def matched_controls(stats, target_genes, n_ctrl=1000, seed=0, exclude_target=False):
    """Exact set size and joint-bin counts; sampling without replacement per set.

    By default target genes remain eligible, as in scDRS. Exclusion can destroy
    matching in small bins; fail explicitly rather than silently relax bins.
    """
    if n_ctrl < 19:
        raise ValueError("Use at least 19 controls; 1000 is the recommended starting point")
    positions = stats.index.get_indexer(target_genes)
    if len(positions) == 0 or (positions < 0).any() or len(set(positions)) != len(positions):
        raise ValueError("Target genes must be nonempty, unique and present")
    target_bins = stats.iloc[positions]["bin"].to_numpy(int)
    if (target_bins < 0).any():
        raise ValueError("Target genes must be expressed and variable")
    rng = np.random.default_rng(seed)
    ctrl = np.empty((n_ctrl, len(positions)), dtype=np.int64)
    all_bins = stats["bin"].to_numpy(int)
    for b in np.unique(target_bins):
        slots = np.flatnonzero(target_bins == b)
        pool = np.flatnonzero(all_bins == b)
        if exclude_target:
            pool = np.setdiff1d(pool, positions)
        if len(pool) < len(slots):
            raise ValueError(f"Bin {b}: need {len(slots)} genes, only {len(pool)} available; reduce bins")
        for i in range(n_ctrl):
            ctrl[i, slots] = rng.choice(pool, len(slots), replace=False)
    return positions, ctrl


def raw_scores(x, indices, genetic_weights, stats, method="waucell", top_frac=.05,
               variance_alpha=0.0, batch_size=128):
    """Same kernel for target and controls; copy target weights by matched slots.

    Optional noise penalty is recomputed for each selected gene, never using
    the control gene's own GWAS statistic. wAUCell matches core/scoring.py ties.
    """
    if method not in {"waucell", "smrs"} or not 0 < top_frac <= 1:
        raise ValueError("Invalid score method or top_frac")
    if variance_alpha < 0 or batch_size < 1:
        raise ValueError("variance_alpha must be nonnegative and batch_size positive")
    w = np.broadcast_to(np.asarray(genetic_weights, float), indices.shape).copy()
    if not np.isfinite(w).all() or (w < 0).any() or (w.sum(axis=1) <= 0).any():
        raise ValueError("Weights must be finite, nonnegative and have positive sums")
    penalty = (np.sqrt(stats["var"].to_numpy()[indices]) + 1e-8)**variance_alpha
    w /= penalty
    w /= w.sum(axis=1, keepdims=True)
    rows = indices.ravel()
    cols = np.repeat(np.arange(len(indices)), indices.shape[1])
    wm = sparse.csc_matrix((w.ravel(), (rows, cols)), shape=(x.shape[1], len(indices)))
    result = np.empty((x.shape[0], len(indices)), dtype=np.float64)
    k = max(1, int(np.ceil(x.shape[1] * top_frac)))
    for start in range(0, x.shape[0], batch_size):
        block = x[start:start+batch_size]
        if method == "waucell":
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            ranks = rankdata(-block, axis=1, method="average")
            features = np.maximum(k-ranks+1, 0) / k
        else:
            features = block
        val = features @ wm
        result[start:start+batch_size] = val.toarray() if sparse.issparse(val) else np.asarray(val)
    return result


def calibrate(raw):
    """Symmetric set/cell normalization preserves conditional exchangeability.

    Include target and controls symmetrically in all nuisance estimates. Center
    each set over cells, then center/rescale all sets within each cell. Do not
    divide by each set's across-cell SD: that can absorb biological signal.
    MC ranks, not Gaussian tails, give significance. Exchangeability is a
    conditional null assumption; expression-bin matching only approximates it.
    """
    y = np.asarray(raw, dtype=np.float64).copy()
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2 or not np.isfinite(y).all():
        raise ValueError("raw must be a finite cells x (target+controls) matrix")
    y -= y.mean(axis=0, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    scale = y.std(axis=1, keepdims=True)
    np.divide(y, scale, out=y, where=scale > 1e-12)
    y[scale.ravel() <= 1e-12] = 0
    tol = 1e-12
    p = (1 + (y[:, 1:] >= y[:, :1]-tol).sum(axis=1)) / y.shape[1]
    return y, p


def group_tests(corrected, obs, group_key, sample_key=None):
    """One-vs-rest mean contrast, same statistic for target and every control.

    Optional donor aggregation first computes within-donor contrasts. This is
    a gene-set randomization test, not a donor-level treatment-effect test.
    """
    if group_key not in obs or obs[group_key].isna().any():
        raise ValueError("Group key missing or contains missing values")
    if sample_key and (sample_key not in obs or obs[sample_key].isna().any()):
        raise ValueError("Sample key missing or contains missing values")
    groups = obs[group_key].astype(str).to_numpy()
    samples = obs[sample_key].astype(str).to_numpy() if sample_key else np.repeat("all", len(obs))
    rows = []
    for group in sorted(set(groups)):
        contrasts = []
        for sample in sorted(set(samples)):
            a = (groups == group) & (samples == sample)
            b = (groups != group) & (samples == sample)
            if a.any() and b.any():
                contrasts.append(corrected[a].mean(axis=0)-corrected[b].mean(axis=0))
        if not contrasts:
            rows.append({"group": group, "effect": np.nan, "p_mc": np.nan, "n_samples": 0})
            continue
        effects = np.mean(contrasts, axis=0)
        p = (1 + np.sum(effects[1:] >= effects[0]-1e-12)) / len(effects)
        rows.append({"group": group, "effect": effects[0], "p_mc": p,
                     "n_samples": len(contrasts), "n_cells": int(np.sum(groups == group))})
    return pd.DataFrame(rows)


def run_matched_score(adata, z, out, *, layer=None, traits=None, top_n=1000,
                      min_genes=200, n_ctrl=1000, n_mean_bins=20, n_var_bins=20,
                      seed=0, exclude_target=False, method="waucell", top_frac=.05,
                      variance_alpha=0.0, batch_size=128, sample_key=None,
                      group_key=None, save_controls=False, memory_gb=2.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = adata.layers[layer] if layer else adata.X
    if sparse.issparse(x):
        x = x.tocsr()
    if not adata.obs_names.is_unique or not z.index.is_unique or not z.columns.is_unique:
        raise ValueError("Cell, gene and trait identifiers must be unique")
    if top_n < 1 or min_genes < 1 or memory_gb <= 0:
        raise ValueError("Invalid top_n, min_genes or memory limit")
    # Conservative preflight, per trait; do not silently densify the whole expression matrix.
    estimate = 8*(adata.n_obs*(n_ctrl+1)*4 + batch_size*adata.n_vars*3 + top_n*(n_ctrl+1)*6)
    if estimate > memory_gb*1024**3:
        raise MemoryError(f"Estimated workspace {estimate/1024**3:.2f} GiB exceeds memory_gb={memory_gb}; reduce cells/controls or increase limit")
    if n_ctrl < 19 or batch_size < 1 or seed < 0:
        raise ValueError("Use n_ctrl >= 19, positive batch_size, nonnegative seed")
    if group_key and (group_key not in adata.obs or adata.obs[group_key].isna().any()):
        raise ValueError("Group key missing or contains missing values")
    if sample_key and (sample_key not in adata.obs or adata.obs[sample_key].isna().any()):
        raise ValueError("Sample key missing or contains missing values")
    stats = gene_statistics(x, adata.var_names, n_mean_bins, n_var_bins)
    stats.to_csv(out / "gene_matching_stats.csv.gz")
    selected = list(z.columns) if traits is None else traits
    summary, group_rows = [], []
    for trait in selected:
        if trait not in z:
            raise ValueError(f"Unknown trait {trait}")
        # Select AFTER expression overlap, retaining only positive association strength.
        v = pd.to_numeric(z[trait], errors="coerce").reindex(stats.index)
        v = v[stats.eligible & np.isfinite(v) & (v > 0)].sort_values(ascending=False, kind="stable").head(top_n)
        identifier = hashlib.sha256(str(trait).encode()).hexdigest()[:16]
        row = {"trait": trait, "id": identifier, "n_genes": len(v)}
        if len(v) < min_genes:
            row["status"] = "skipped_insufficient_genes"
            summary.append(row)
            continue
        logging.info("Matched background: %s (%d genes, %d controls)", trait, len(v), n_ctrl)
        trait_seed = np.random.SeedSequence([seed, int(identifier[:8], 16)])
        target, controls = matched_controls(stats, v.index, n_ctrl, trait_seed, exclude_target)
        indices = np.vstack([target, controls])
        raw = raw_scores(x, indices, v.to_numpy(), stats, method, top_frac, variance_alpha, batch_size)
        corrected, p = calibrate(raw)
        directory = out / identifier
        directory.mkdir(exist_ok=True)
        result = pd.DataFrame({"raw_score": raw[:, 0], "background_raw_mean": raw[:, 1:].mean(axis=1),
                               "background_excess": raw[:, 0]-raw[:, 1:].mean(axis=1),
                               "calibrated_score": corrected[:, 0], "p_mc": p,
                               "q_bh_within_trait": bh(p)}, index=adata.obs_names)
        result.to_csv(directory / "cell_scores.csv.gz")
        pd.DataFrame({"gene": v.index, "genetic_weight": v.values}).to_csv(directory / "target_genes.csv", index=False)
        # Integer gene indices refer to rows of gene_matching_stats.csv.gz.
        np.savez_compressed(directory / "control_gene_indices.npz", target=target, controls=controls)
        means, variances = stats["mean"].to_numpy(), stats["var"].to_numpy()
        diag = pd.DataFrame({"control": np.arange(n_ctrl),
                             "mean_expression": means[controls].mean(axis=1),
                             "mean_variance": variances[controls].mean(axis=1),
                             "target_overlap_fraction": np.isin(controls, target).mean(axis=1)})
        diag["target_mean_expression"] = means[target].mean()
        diag["target_mean_variance"] = variances[target].mean()
        diag.to_csv(directory / "matching_diagnostics.csv", index=False)
        if save_controls:
            np.savez_compressed(directory / "control_scores.npz", raw=raw[:, 1:], calibrated=corrected[:, 1:])
        if group_key:
            g = group_tests(corrected, adata.obs, group_key, sample_key)
            g["trait"] = trait
            group_rows.append(g)
        row.update(status="ok", min_p=1/(n_ctrl+1),
                   mean_target_overlap=diag.target_overlap_fraction.mean())
        if row["mean_target_overlap"] > .25:
            logging.warning("%s: substantial target/control overlap; inspect bins and power", trait)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "trait_manifest.csv", index=False)
    adata.obs.to_csv(out / "obs_metadata.csv")
    if group_rows:
        g = pd.concat(group_rows, ignore_index=True)
        g["q_bh_all_reported_pairs"] = bh(g.p_mc)
        g.to_csv(out / "group_associations.csv", index=False)
    config = dict(method=method, n_ctrl=n_ctrl, n_mean_bins=n_mean_bins, n_var_bins=n_var_bins,
                  seed=seed, top_n=top_n, min_genes=min_genes, exclude_target=exclude_target,
                  variance_alpha=variance_alpha, top_frac=top_frac, sample_key=sample_key,
                  group_key=group_key, layer=layer, calibration="symmetric_matched_MC_v1",
                  interpretation="association-strength enrichment; no metabolite effect direction",
                  p_resolution=1/(n_ctrl+1), input_requirement="library-normalized nonnegative log1p",
                  scdrs_equivalence=False)
    (out / "background_run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not any(r["status"] == "ok" for r in summary):
        raise ValueError("No traits passed gene count criteria; see trait_manifest.csv")
