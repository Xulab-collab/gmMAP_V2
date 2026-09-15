"""Bulk SMRS exposure validation. No gene-set P is a biological-replicate P."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist, ttest_1samp, ttest_ind
from gmmap.core.background import gene_statistics, matched_controls, raw_scores, bh
from gmmap.io import read_table


def _validate_expression(expr):
    if not expr.index.is_unique or not expr.columns.is_unique:
        raise ValueError('Expression sample and gene IDs must be unique')
    x = expr.to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Use finite nonnegative, per-sample normalized log1p expression')



def preprocess_expression(expr, input_scale='log1p', target_sum=1e6):
    """Per-sample CPM then natural log1p. Never learn normalization across samples.

    Library size uses ALL submitted genes, before GWAS overlap/variance filtering.
    Raw count mode requires integer-valued counts; no gene-length correction.
    """
    _validate_expression(expr)
    if input_scale not in {'counts', 'log1p'}:
        raise ValueError('input_scale must be counts or log1p')
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError('target_sum must be positive and finite')
    x = expr.to_numpy(dtype=float)
    qc = pd.DataFrame(index=expr.index)
    qc['input_scale'] = input_scale
    qc['n_input_genes'] = x.shape[1]
    qc['n_nonzero_genes'] = (x > 0).sum(axis=1)
    if input_scale == 'counts':
        if not np.allclose(x, np.rint(x), rtol=0, atol=1e-6):
            raise ValueError('counts mode requires integer-valued raw counts; do not supply TPM or log expression')
        library = x.sum(axis=1)
        if not np.isfinite(library).all() or (library <= 0).any():
            raise ValueError('Every count sample must have a positive finite total count')
        qc['library_size'] = library
        qc['normalization_factor'] = target_sum/library
        qc['target_sum'] = target_sum
        x = np.log1p((x/library[:, None])*target_sum)
    return pd.DataFrame(x, index=expr.index, columns=expr.columns), qc


def prepare_units(expr, metadata, group_key, treated, control, design, donor_key=None,
                  aggregate=False):
    _validate_expression(expr)
    if not metadata.index.is_unique or set(expr.index) != set(metadata.index):
        raise ValueError('Expression and metadata must contain exactly the same unique sample IDs')
    meta = metadata.loc[expr.index].copy()
    if treated == control or group_key not in meta or meta[group_key].isna().any():
        raise ValueError('Specify distinct treatment/control labels and complete group metadata')
    if set(meta[group_key].astype(str)) != {treated, control}:
        raise ValueError('Metadata must contain exactly the specified two groups')
    if design not in {'paired', 'independent'}:
        raise ValueError('design must be paired or independent')
    if design == 'paired' and not donor_key:
        raise ValueError('Paired design requires donor-key')
    if donor_key and (donor_key not in meta or meta[donor_key].isna().any()):
        raise ValueError('Missing donor IDs')
    meta['donor'] = meta[donor_key].astype(str) if donor_key else meta.index.astype(str)
    meta['label'] = (meta[group_key].astype(str) == treated).astype(int)
    rows, arrays = [], []
    for (donor, label), block in meta.groupby(['donor', 'label'], sort=True):
        if len(block) > 1 and not aggregate:
            raise ValueError('Multiple samples per donor/condition: explicitly use --aggregate-replicates or resolve replicates')
        arrays.append(expr.loc[block.index].to_numpy(float).mean(axis=0))
        rows.append(dict(unit=f'u{len(rows):04d}', donor=donor, label=int(label),
                         n_input_samples=len(block), source_samples=json.dumps(list(block.index))))
    units = pd.DataFrame(rows).set_index('unit')
    sizes = units.groupby('donor').size()
    if design == 'paired':
        if not (sizes == 2).all() or len(sizes) < 4:
            raise ValueError('Paired validation requires >=4 complete donors, each with both conditions')
    elif not (sizes == 1).all() or units.label.value_counts().min() < 3:
        raise ValueError('Independent validation requires distinct donors across groups and >=3 donors per group')
    return pd.DataFrame(arrays, index=units.index, columns=expr.columns), units


def fit_background(expr, z, top_n=1000, min_genes=200, n_ctrl=1000,
                   mean_bins=10, var_bins=10, seed=0, variance_alpha=0.):
    _validate_expression(expr)
    if not z.index.is_unique or min_genes < 1 or top_n < min_genes or seed < 0:
        raise ValueError('Invalid gene identifiers, gene counts, or seed')
    stats = gene_statistics(expr.to_numpy(float), expr.columns, mean_bins, var_bins)
    weights = pd.to_numeric(z, errors='coerce').reindex(expr.columns)
    weights = weights[stats.eligible & np.isfinite(weights) & (weights > 0)]
    weights = weights.sort_values(ascending=False, kind='stable').head(top_n)
    if len(weights) < min_genes:
        raise ValueError('Too few eligible positive MAGMA genes in training data')
    target, controls = matched_controls(stats, weights.index, n_ctrl, seed)
    ix = np.vstack([target, controls])
    # Retain only required genes, so independent prediction needs no unused genes.
    used = np.unique(ix)
    remap = np.full(expr.shape[1], -1, dtype=int); remap[used] = np.arange(len(used))
    model = dict(genes=expr.columns.to_numpy(str)[used], indices=remap[ix],
                 genetic_weights=weights.to_numpy(float), variance=stats['var'].to_numpy()[used],
                 variance_alpha=np.array(variance_alpha), format_version=np.array(1))
    raw = _raw(expr, model)
    model['set_center'] = raw.mean(axis=0)
    diagnostics = dict(n_genes=len(target), target_mean=float(stats['mean'].iloc[target].mean()),
                       target_variance=float(stats['var'].iloc[target].mean()),
                       mean_control_mean=float(stats['mean'].to_numpy()[controls].mean()),
                       mean_control_variance=float(stats['var'].to_numpy()[controls].mean()),
                       overlap=float(np.isin(controls, target).mean()))
    return model, stats, diagnostics


def _raw(expr, model):
    _validate_expression(expr)
    missing = pd.Index(model['genes']).difference(expr.columns)
    if len(missing):
        raise ValueError(f'Missing {len(missing)} model genes; do not fill missing genes with zeros')
    x = expr.loc[:, model['genes']].to_numpy(float)
    stats = pd.DataFrame({'var': model['variance']})
    return raw_scores(x, model['indices'], model['genetic_weights'], stats,
                      method='smrs', variance_alpha=float(model['variance_alpha']))


def transform(expr, model):
    raw = _raw(expr, model)
    residual = raw - model['set_center'][None, :]
    residual -= residual.mean(axis=1, keepdims=True)
    scale = residual.std(axis=1, keepdims=True)
    calibrated = np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 1e-12)
    p = (1 + (calibrated[:, 1:] >= calibrated[:, :1]-1e-12).sum(axis=1))/raw.shape[1]
    return pd.DataFrame({'raw_score': raw[:, 0], 'calibrated_score': calibrated[:, 0],
                         'background_excess': raw[:, 0]-raw[:, 1:].mean(axis=1),
                         'gene_set_p_upper': p}, index=expr.index)


def auc(y, score):
    y = np.asarray(y); score = np.asarray(score)
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    if not n1 or not n0:
        return float('nan')
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def classifier(train_scores, y):
    a, b = train_scores[y == 1], train_scores[y == 0]
    if not len(a) or not len(b):
        raise ValueError('A training fold has only one class')
    difference = a.mean()-b.mean()
    scale = float(np.std(train_scores))
    # No information -> constant decision, not arbitrarily oriented noise.
    direction = float(np.sign(difference)) if scale > 1e-12 else 0.
    return direction, float((a.mean()+b.mean())/2), max(scale, 1e-12)


def fold_cache(expr, units, z, config):
    cache, diagnostics = [], []
    donors = units.donor.to_numpy()
    for donor in sorted(set(donors)):
        test = np.flatnonzero(donors == donor); train = np.flatnonzero(donors != donor)
        model, _, diag = fit_background(expr.iloc[train], z, **config)
        tr, te = transform(expr.iloc[train], model), transform(expr.iloc[test], model)
        cache.append((train, test, tr[['raw_score','calibrated_score']].to_numpy(),
                      te[['raw_score','calibrated_score']].to_numpy()))
        diagnostics.append(dict(held_out_donor=donor, n_train=len(train), n_test=len(test), **diag))
    return cache, pd.DataFrame(diagnostics)


def cross_predict(cache, y):
    out = np.empty((len(y), 2)); directions = np.empty_like(out)
    for train, test, train_scores, test_scores in cache:
        for j in range(2):
            direction, midpoint, scale = classifier(train_scores[:, j], y[train])
            out[test, j] = direction*(test_scores[:, j]-midpoint)/scale
            directions[test, j] = direction
    return out, directions


def permute_labels(y, donors, design, rng):
    result = y.copy()
    if design == 'paired':
        for donor in sorted(set(donors)):
            if rng.integers(2):
                mask = donors == donor; result[mask] = 1-result[mask]
    else:
        result = rng.permutation(y)
    return result


def effect_test(score, y, donors, design):
    a, b = score[y == 1], score[y == 0]
    if design == 'paired':
        delta = np.array([score[(donors == d)&(y == 1)][0]-score[(donors == d)&(y == 0)][0]
                          for d in sorted(set(donors))])
        effect = float(delta.mean()); se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
        df = len(delta)-1
        p = float(ttest_1samp(delta, 0).pvalue) if se > 1e-12 else (1. if abs(effect)<1e-12 else 0.)
    else:
        effect = float(a.mean()-b.mean())
        va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
        se = float(np.sqrt(va+vb))
        df = float((va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))) if se>1e-12 else len(a)+len(b)-2
        p = float(ttest_ind(a,b,equal_var=False).pvalue) if se>1e-12 else (1. if abs(effect)<1e-12 else 0.)
    margin = float(tdist.ppf(.975, df)*se)
    return dict(effect_treated_minus_control=effect, ci95_low=effect-margin,
                ci95_high=effect+margin, parametric_p_two_sided=p)


def analyze(expr, metadata, z, out, *, group_key, treated, control, design,
            donor_key=None, aggregate=False, n_perm=999, seed=0,
            input_scale="log1p", target_sum=1e6, **config):
    if n_perm < 19 or seed < 0:
        raise ValueError('Use >=19 permutations and a nonnegative seed')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty output directory to avoid mixing runs')
    normalized, preprocessing_qc = preprocess_expression(expr, input_scale, target_sum)
    preprocessing_genes = normalized.columns.to_numpy(str)
    expr, units = prepare_units(normalized, metadata, group_key, treated, control, design, donor_key, aggregate)
    config = dict(config, seed=seed)
    y, donors = units.label.to_numpy(), units.donor.to_numpy()
    model, stats, diag = fit_background(expr, z, **config)
    full = transform(expr, model)
    if diag['overlap'] > .25:
        logging.warning('High target/control overlap (%.1f%%); inspect bins and sensitivity',100*diag['overlap'])
    logging.info('Fitting leave-one-donor-out backgrounds (%d donors)', len(set(donors)))
    cache, fold_diag = fold_cache(expr, units, z, config)
    prediction, directions = cross_predict(cache, y)
    names = ['raw_score', 'calibrated_score']
    observed_auc = np.array([auc(y, prediction[:, j]) for j in range(2)])
    observed_effect = np.array([abs(full[name].to_numpy()[y==1].mean()-full[name].to_numpy()[y==0].mean()) for name in names])
    donor_order = sorted(set(donors))
    exact = design == 'paired' and len(donor_order) <= 12 and 2**len(donor_order) <= n_perm
    actual_perm = 2**len(donor_order) if exact else n_perm
    null_auc, null_effect = np.empty((actual_perm, 2)), np.empty((actual_perm, 2))
    rng = np.random.default_rng(seed)
    # Fold backgrounds are label-blind; cached transforms are identical to refitting
    # for every permutation. Training directions/thresholds ARE relearned each time.
    for k in range(actual_perm):
        if exact:
            yp = y.copy()
            for bit, donor in enumerate(donor_order):
                if k & (1 << bit):
                    mask = donors == donor; yp[mask] = 1-yp[mask]
        else:
            yp = permute_labels(y, donors, design, rng)
        pp, _ = cross_predict(cache, yp)
        for j, name in enumerate(names):
            null_auc[k,j] = auc(yp, pp[:,j])
            v = full[name].to_numpy()
            null_effect[k,j] = abs(v[yp==1].mean()-v[yp==0].mean())
    def permutation_p(null, observed):
        count = np.sum(null >= observed-1e-12)
        return float(count/actual_perm) if exact else float((1+count)/(actual_perm+1))
    rows = []
    for j, name in enumerate(names):
        v = full[name].to_numpy()
        pred = prediction[:,j] > 0
        sensitivity = float(pred[y==1].mean()); specificity = float((~pred[y==0]).mean())
        rows.append(dict(score=name, **effect_test(v,y,donors,design),
                         effect_permutation_p=permutation_p(null_effect[:,j],observed_effect[j]),
                         oof_auc=observed_auc[j],
                         auc_permutation_p=permutation_p(null_auc[:,j],observed_auc[j]),
                         oof_sensitivity=sensitivity, oof_specificity=specificity,
                         oof_balanced_accuracy=(sensitivity+specificity)/2))
        if design == 'paired':
            differences = [prediction[(donors==d)&(y==1),j][0]-prediction[(donors==d)&(y==0),j][0] for d in donor_order]
            rows[-1]['oof_within_donor_concordance'] = float(np.mean([1. if d>0 else (.5 if d==0 else 0.) for d in differences]))
        direction, midpoint, scale = classifier(v,y)
        model[f'{name}_classifier'] = np.array([direction,midpoint,scale])
    summary = pd.DataFrame(rows)
    summary['effect_q_two_scores'] = bh(summary.effect_permutation_p)
    summary['auc_q_two_scores'] = bh(summary.auc_permutation_p)
    model['trait'] = np.array(str(z.name)); model['treated_label'] = np.array(treated)
    model['control_label'] = np.array(control)
    model['format_version'] = np.array(2)
    model['input_scale'] = np.array(input_scale)
    model['normalization_target_sum'] = np.array(target_sum)
    model['preprocessing_genes'] = preprocessing_genes
    out.mkdir(parents=True, exist_ok=True)
    preprocessing_qc.to_csv(out/'preprocessing_qc.csv')
    normalized.T.to_csv(out/'normalized_expression_log1p.csv.gz')
    units.join(full).to_csv(out/'sample_scores.csv')
    if design == 'paired':
        differences = pd.DataFrame(index=pd.Index(donor_order,name='donor'))
        for name in names:
            v = full[name].to_numpy()
            differences[name+'_treated_minus_control'] = [v[(donors==d)&(y==1)][0]-v[(donors==d)&(y==0)][0] for d in donor_order]
        differences.to_csv(out/'paired_differences.csv')
    oof = units.copy()
    for j,name in enumerate(names):
        oof[f'{name}_decision'] = prediction[:,j]
        oof[f'{name}_predicted_treated'] = prediction[:,j]>0
        oof[f'{name}_training_direction'] = directions[:,j]
    oof.to_csv(out/'out_of_fold_predictions.csv')
    summary.to_csv(out/'validation_summary.csv', index=False)
    fold_diag.to_csv(out/'fold_diagnostics.csv',index=False)
    stats.to_csv(out/'gene_matching_stats.csv.gz')
    pd.DataFrame({'gene':model['genes'][model['indices'][0]],'weight':model['genetic_weights']}).to_csv(out/'target_genes.csv',index=False)
    np.savez_compressed(out/'fitted_model.npz',**model)
    np.savez_compressed(out/'permutation_null.npz',auc=null_auc,absolute_effect=null_effect)
    config.update(input_scale=input_scale,normalization_target_sum=target_sum,
                  preprocessing='library-size scaling then natural log1p' if input_scale=='counts' else 'already transformed; no additional normalization',
                  design=design,trait=str(z.name),treated=treated,control=control,
                  group_key=group_key,donor_key=donor_key,aggregate_replicates=aggregate,
                  requested_permutations=n_perm,actual_permutations=actual_perm,
                  permutation_mode='exact_paired_swaps' if exact else 'monte_carlo',
                  n_donors=len(set(donors)),n_units=len(units),
                  full_fit_diagnostics=diag,cv='leave-one-donor-out',
                  calibration='training-set centers; within-sample matched-set centering/scaling',
                  covariate_adjustment=False,exact_scdrs=False,
                  permutation_assumption='within-donor treatment exchangeability' if design=='paired' else 'independent donor label exchangeability',
                  ci_note='Parametric intervals are approximate, conditional on cohort-fitted scores; primary P is sample-label permutation')
    (out/'run.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def predict(expr, model_path, input_scale=None):
    with np.load(model_path, allow_pickle=False) as loaded:
        model = {k:loaded[k] for k in loaded.files}
    if int(model['format_version']) not in {1, 2}:
        raise ValueError('Unsupported model format')
    trained_scale = str(model.get('input_scale', np.array('log1p')))
    input_scale = trained_scale if input_scale is None else input_scale
    if input_scale != trained_scale:
        raise ValueError(f'Model expects {trained_scale} input; do not mix training/prediction scales')
    if input_scale == 'counts':
        expected = pd.Index(model['preprocessing_genes'])
        if not expected.is_unique or set(expr.columns) != set(expected):
            raise ValueError('Count prediction requires the same full gene universe as training for library-size normalization')
        expr = expr.loc[:, expected]
    expr, _ = preprocess_expression(expr, input_scale, float(model.get('normalization_target_sum', 1e6)))
    result = transform(expr, model)
    for name in ['raw_score','calibrated_score']:
        direction, midpoint, scale = model[f'{name}_classifier']
        decision = direction*(result[name]-midpoint)/scale
        result[f'{name}_decision'] = decision
        result[f'{name}_predicted_group'] = np.where(decision>0,str(model['treated_label']),str(model['control_label']))
    return result


def main():
    parser = argparse.ArgumentParser(description='Bulk SMRS: matched background, donor-level effects and exposure validation')
    sub = parser.add_subparsers(dest='command',required=True)
    a = sub.add_parser('analyze')
    for flag in ['expression','metadata','magma-z','trait','group-key','treated','control','out']:
        a.add_argument('--'+flag,required=True)
    a.add_argument('--design',choices=['paired','independent'],required=True)
    a.add_argument('--donor-key')
    a.add_argument('--aggregate-replicates',action='store_true')
    input_a = a.add_mutually_exclusive_group(required=True)
    input_a.add_argument('--input-scale',choices=['counts','log1p'])
    input_a.add_argument('--input-is-log1p',action='store_true',help='Compatibility alias for --input-scale log1p')
    a.add_argument('--normalization-target-sum',type=float,default=1e6,help='counts mode: default 1e6 gives CPM before log1p')
    a.add_argument('--top-n-genes',type=int,default=1000)
    a.add_argument('--min-valid-genes',type=int,default=200)
    a.add_argument('--n-ctrl',type=int,default=1000)
    a.add_argument('--mean-bins',type=int,default=10)
    a.add_argument('--var-bins',type=int,default=10)
    a.add_argument('--variance-alpha',type=float,default=0.)
    a.add_argument('--n-perm',type=int,default=999)
    a.add_argument('--seed',type=int,default=0)
    p = sub.add_parser('predict')
    for flag in ['expression','model','out']:
        p.add_argument('--'+flag,required=True)
    input_p = p.add_mutually_exclusive_group(required=True)
    input_p.add_argument('--input-scale',choices=['counts','log1p'])
    input_p.add_argument('--input-is-log1p',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    input_scale = 'log1p' if args.input_is_log1p else args.input_scale
    # Input table is genes x samples, matching conventional bulk expression files.
    expr = read_table(args.expression,index_col=0).T
    expr.index = expr.index.astype(str); expr.columns = expr.columns.astype(str)
    if args.command == 'predict':
        result = predict(expr,args.model,input_scale=input_scale)
        dest = Path(args.out)
        if dest.exists():
            raise ValueError('Prediction output exists; choose a new path')
        dest.parent.mkdir(parents=True,exist_ok=True); result.to_csv(dest)
        return
    metadata = read_table(args.metadata,index_col=0); metadata.index = metadata.index.astype(str)
    z = read_table(args.magma_z,index_col=0); z.index = z.index.astype(str)
    if args.trait not in z:
        raise ValueError('Trait not found in MAGMA matrix')
    result = analyze(expr,metadata,z[args.trait],args.out,group_key=args.group_key,
                     treated=args.treated,control=args.control,design=args.design,
                     donor_key=args.donor_key,aggregate=args.aggregate_replicates,
                     n_perm=args.n_perm,seed=args.seed,input_scale=input_scale,
                     target_sum=args.normalization_target_sum,top_n=args.top_n_genes,
                     min_genes=args.min_valid_genes,n_ctrl=args.n_ctrl,
                     mean_bins=args.mean_bins,var_bins=args.var_bins,variance_alpha=args.variance_alpha)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
