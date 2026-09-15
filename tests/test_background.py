import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from gmmap.core.background import gene_statistics, matched_controls, raw_scores, calibrate, group_tests, bh, run_matched_score
from gmmap.core.scoring import score_waucell, score_smrs


def fixture_data():
    rng=np.random.default_rng(41)
    x=np.log1p(rng.poisson(rng.uniform(.2,4,300),size=(80,300)))
    names=pd.Index([f'g{i}' for i in range(300)])
    return x,names,gene_statistics(x,names,4,3)


def test_matching_exact_size_bins_unique_and_seed():
    x,names,stats=fixture_data()
    target=names[:30]
    a,b=matched_controls(stats,target,99,42)
    _,c=matched_controls(stats,target,99,42)
    assert np.array_equal(b,c)
    for row in b:
        assert len(np.unique(row))==len(a)
        assert np.array_equal(stats.bin.to_numpy()[row],stats.bin.to_numpy()[a])
    assert b.shape==(99,30)


def test_sparse_and_dense_statistics_equal():
    x,names,stats=fixture_data()
    other=gene_statistics(sparse.csr_matrix(x),names,4,3)
    np.testing.assert_allclose(stats[['mean','var']],other[['mean','var']],atol=1e-12)
    assert np.array_equal(stats.bin,other.bin)


@pytest.mark.parametrize('method',['waucell','smrs'])
def test_matches_original_scoring_and_sparse_batches(method):
    x,names,stats=fixture_data()
    # Add ties and zeros already present in count-derived expression.
    target,ctrl=matched_controls(stats,names[:30],19,9)
    weights=np.arange(1,31,dtype=float)
    ix=np.vstack([target,ctrl])
    actual=raw_scores(x,ix,weights,stats,method,.1,batch_size=7)
    other=raw_scores(sparse.csr_matrix(x),ix,weights,stats,method,.1,batch_size=13)
    np.testing.assert_allclose(actual,other,atol=1e-12)
    df=pd.DataFrame(x,columns=names)
    for j in [0,1,7]:
        w=pd.Series(weights,index=names[ix[j]])
        expected=score_waucell(df,w,.1) if method=='waucell' else score_smrs(df,w)
        np.testing.assert_allclose(actual[:,j],expected,atol=1e-12)


def test_calibration_ties_and_permutation_symmetry():
    y,p=calibrate(np.ones((40,100)))
    assert (y==0).all() and (p==1).all()
    raw=np.random.default_rng(5).normal(size=(40,100))
    order=np.random.default_rng(1).permutation(100)
    a,_=calibrate(raw)
    b,_=calibrate(raw[:,order])
    np.testing.assert_allclose(a[:,order],b,atol=1e-12)


def test_exchangeable_null_calibration():
    rng=np.random.default_rng(8)
    rates=[]
    for _ in range(30):
        # Large cell-shared technical background plus gene-set location effects.
        raw=rng.normal(size=(100,200))+rng.normal(0,20,(100,1))+rng.normal(0,20,(1,200))
        _,p=calibrate(raw)
        rates.append(np.mean(p<=.05))
    assert .025 < np.mean(rates) < .075


def test_signal_and_donor_group_test():
    rng=np.random.default_rng(12)
    raw=rng.normal(size=(120,200))
    raw[:60,0]+=5
    y,p=calibrate(raw)
    assert np.mean(p[:60]<=.05)>.8
    obs=pd.DataFrame({'group':['A']*60+['B']*60,'donor':np.tile(np.repeat(['d1','d2','d3'],20),2)})
    g=group_tests(y,obs,'group','donor').set_index('group')
    assert g.loc['A','p_mc']<=.01
    assert g.loc['A','n_samples']==3
    assert g.loc['B','p_mc']>.9


def test_bad_input_and_insufficient_pool():
    x,names,stats=fixture_data()
    with pytest.raises(ValueError): gene_statistics(-x,names)
    with pytest.raises(ValueError): matched_controls(stats,names,19,exclude_target=True)
    with pytest.raises(ValueError): matched_controls(stats,names[:3],3)
    with pytest.raises(ValueError): gene_statistics(x,pd.Index(['same']*300))


def test_gene_selection_overlap_outputs_and_seed_order(tmp_path):
    import anndata as ad
    x,names,stats=fixture_data()
    a=ad.AnnData(sparse.csr_matrix(x),var=pd.DataFrame(index=names),obs=pd.DataFrame(index=[f'c{i}' for i in range(80)]))
    a.obs['type']=['A']*40+['B']*40
    z=pd.DataFrame({'a':np.linspace(3,-1,301),'b':np.linspace(4,-1,301)},index=['missing']+list(names))
    for folder,traits in [('one',['a','b']),('two',['b','a'])]:
        run_matched_score(a,z,tmp_path/folder,traits=traits,top_n=30,min_genes=20,n_ctrl=19,n_mean_bins=4,n_var_bins=3,group_key='type')
    m=pd.read_csv(tmp_path/'one'/'trait_manifest.csv')
    assert list(m.n_genes)==[30,30]
    for ident in m.id:
        p1=pd.read_csv(tmp_path/'one'/ident/'cell_scores.csv.gz')
        p2=pd.read_csv(tmp_path/'two'/ident/'cell_scores.csv.gz')
        pd.testing.assert_frame_equal(p1,p2)
    assert (tmp_path/'one'/'group_associations.csv').exists()


def test_bh():
    np.testing.assert_allclose(bh([.01,.04,.03,np.nan])[:3],[.03,.04,.04])


def test_variance_penalty_transferred_weights():
    x,names,stats=fixture_data()
    t,c=matched_controls(stats,names[:30],19,seed=1)
    ix=np.vstack([t,c]); weights=np.arange(1.,31.)
    y=raw_scores(x,ix,weights,stats,'smrs',variance_alpha=.5)
    for j in [0,1,5]:
        w=weights/(np.sqrt(stats['var'].to_numpy()[ix[j]])+1e-8)**.5
        np.testing.assert_allclose(y[:,j], x[:,ix[j]]@(w/w.sum()),atol=1e-12)


def test_cli_matched_and_legacy(tmp_path):
    import anndata as ad
    from gmmap.cli import build_score_parser, run_score
    x,names,_=fixture_data()
    a=ad.AnnData(sparse.csr_matrix(x),var=pd.DataFrame(index=names),obs=pd.DataFrame(index=[f'c{i}' for i in range(len(x))]))
    a.obs['type']=['A']*40+['B']*40
    a.write_h5ad(tmp_path/'input.h5ad')
    pd.DataFrame({'trait':np.linspace(3,-3,300)},index=names).to_csv(tmp_path/'z.csv')
    base=['--h5ad',str(tmp_path/'input.h5ad'),'--magma-z',str(tmp_path/'z.csv'),'--top-n-genes','30','--min-valid-genes','20','--celltype-key','type']
    parser=build_score_parser()
    run_score(parser.parse_args(base+['--out',str(tmp_path/'legacy')]))
    assert (tmp_path/'legacy'/'gmmap_scores_NET_S3.csv').exists()
    run_score(parser.parse_args(base+['--out',str(tmp_path/'matched'),'--background','matched','--input-is-log1p','--n-ctrl','19','--n-mean-bins','4','--n-var-bins','3']))
    assert (tmp_path/'matched'/'matched_background'/'group_associations.csv').exists()
    with pytest.raises(ValueError,match='input-is-log1p'):
        run_score(parser.parse_args(base+['--out',str(tmp_path/'bad'),'--background','matched']))
