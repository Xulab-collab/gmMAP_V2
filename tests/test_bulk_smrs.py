import numpy as np
import pandas as pd
import pytest
from gmmap.core.bulk_smrs import (prepare_units,fit_background,transform,fold_cache,
    cross_predict,permute_labels,effect_test,auc,analyze,predict,main)
from gmmap.core.scoring import score_smrs


def toy(design='paired',signal=1.0):
    rng=np.random.default_rng(72)
    n=12; g=600
    y=np.tile([0,1],6) if design=='paired' else np.repeat([0,1],6)
    donors=np.repeat([f'd{i}' for i in range(6)],2) if design=='paired' else np.array([f'd{i}' for i in range(n)])
    x=np.maximum(0,rng.normal(3, .8,(n,g))+rng.normal(0,.3,(1,g)))
    x[y==1,:30]+=signal
    expr=pd.DataFrame(x,index=[f's{i}' for i in range(n)],columns=[f'g{i}' for i in range(g)])
    meta=pd.DataFrame({'condition':np.where(y,'treated','control'),'donor':donors},index=expr.index)
    z=pd.Series(np.r_[np.linspace(5,3,30),np.repeat(-1.,g-30)],index=expr.columns,name='test_trait')
    return expr,meta,z


CONFIG=dict(top_n=30,min_genes=20,n_ctrl=39,mean_bins=4,var_bins=3,seed=9)


def units(design='paired',signal=1.):
    x,m,z=toy(design,signal)
    x,u=prepare_units(x,m,'condition','treated','control',design,'donor')
    return x,u,z


def test_raw_matches_smrs_and_frozen_prediction_is_sample_independent():
    x,u,z=units()
    model,_,_=fit_background(x.iloc[:8],z,**CONFIG)
    p=transform(x.iloc[8:],model)
    expected=score_smrs(x.iloc[8:],z[z>0])
    np.testing.assert_allclose(p.raw_score,expected)
    one=transform(x.iloc[[8]],model)
    pd.testing.assert_series_equal(one.iloc[0],p.iloc[0])
    disturbed=x.iloc[8:].copy(); disturbed.iloc[1:]+=100
    pd.testing.assert_series_equal(transform(disturbed,model).iloc[0],p.iloc[0])
    pd.testing.assert_frame_equal(transform(x.iloc[8:][x.columns[::-1]],model),p)
    with pytest.raises(ValueError,match='Missing'):
        transform(x.drop(columns=model['genes'][0]),model)


def test_fold_background_and_prediction_do_not_use_heldout_labels_or_expression():
    x,u,z=units()
    cache,_=fold_cache(x,u,z,CONFIG)
    tr,te,train_values,test_values=cache[0]
    assert set(u.donor.iloc[tr]).isdisjoint(u.donor.iloc[te])
    changed=x.copy(); changed.iloc[te]+=20
    other,_=fold_cache(changed,u,z,CONFIG)
    np.testing.assert_array_equal(other[0][2],train_values)
    y=u.label.to_numpy()
    a,da=cross_predict(cache,y)
    yp=y.copy(); yp[te]=1-yp[te]
    b,db=cross_predict(cache,yp)
    np.testing.assert_array_equal(a[te],b[te])
    np.testing.assert_array_equal(da[te],db[te])


def test_replicates_and_design_validation():
    x,m,z=toy()
    xx=pd.concat([x,x.iloc[[0]].rename(index={'s0':'repeat'})])
    mm=pd.concat([m,m.iloc[[0]].rename(index={'s0':'repeat'})])
    with pytest.raises(ValueError,match='Multiple'):
        prepare_units(xx,mm,'condition','treated','control','paired','donor')
    a,u=prepare_units(xx,mm,'condition','treated','control','paired','donor',True)
    assert len(a)==12 and u.n_input_samples.max()==2
    with pytest.raises(ValueError,match='complete'):
        prepare_units(x.iloc[1:],m.iloc[1:],'condition','treated','control','paired','donor')
    with pytest.raises(ValueError,match='distinct donors'):
        prepare_units(x,m,'condition','treated','control','independent','donor')
    bad=m.copy(); bad.iloc[0,0]='unknown'
    with pytest.raises(ValueError,match='exactly'):
        prepare_units(x,bad,'condition','treated','control','paired','donor')


def test_permutation_preserves_design_and_effect_ci():
    x,u,z=units(); y=u.label.to_numpy(); d=u.donor.to_numpy()
    yp=permute_labels(y,d,'paired',np.random.default_rng(4))
    for donor in set(d): assert yp[d==donor].sum()==1
    result=effect_test(y+np.arange(len(y))*.01,y,d,'paired')
    assert result['effect_treated_minus_control']>1
    assert result['ci95_low']>0
    assert auc(y,y)==1 and auc(y,np.ones(len(y)))==.5


@pytest.mark.parametrize('design',['paired','independent'])
def test_end_to_end_signal_and_saved_predict(tmp_path,design):
    x,m,z=toy(design,signal=1.5)
    cfg={k:v for k,v in CONFIG.items() if k!='seed'}
    summary=analyze(x,m,z,tmp_path/'run',group_key='condition',treated='treated',control='control',
                    design=design,donor_key='donor',n_perm=39,seed=9,**cfg)
    assert summary.loc[0,'oof_auc']>.9
    assert summary.loc[0,'effect_permutation_p']<=.1
    pred=predict(x,tmp_path/'run'/'fitted_model.npz')
    assert len(pred)==len(x) and np.isfinite(pred.raw_score).all()
    assert (tmp_path/'run'/'out_of_fold_predictions.csv').exists()
    with pytest.raises(ValueError,match='empty'):
        analyze(x,m,z,tmp_path/'run',group_key='condition',treated='treated',control='control',
                design=design,donor_key='donor',n_perm=19,**cfg)


def test_constant_data_rejected_no_false_prediction():
    x,u,z=units()
    with pytest.raises(ValueError,match='No expressed variable'):
        fit_background(x*0+1,z,**CONFIG)
    from gmmap.core.bulk_smrs import classifier
    assert classifier(np.ones(len(u)),u.label.to_numpy())[0]==0


def test_cli_analyze_and_predict(tmp_path,monkeypatch):
    x,m,z=toy(signal=1.5)
    x.T.to_csv(tmp_path/'x.csv'); m.to_csv(tmp_path/'m.csv'); z.to_frame().to_csv(tmp_path/'z.csv')
    monkeypatch.setattr('sys.argv',['gmmap-bulk-smrs','analyze','--expression',str(tmp_path/'x.csv'),
        '--metadata',str(tmp_path/'m.csv'),'--magma-z',str(tmp_path/'z.csv'),'--trait','test_trait',
        '--group-key','condition','--treated','treated','--control','control','--design','paired',
        '--donor-key','donor','--input-is-log1p','--top-n-genes','30','--min-valid-genes','20',
        '--n-ctrl','19','--mean-bins','4','--var-bins','3','--n-perm','19','--out',str(tmp_path/'run')])
    main()
    monkeypatch.setattr('sys.argv',['gmmap-bulk-smrs','predict','--expression',str(tmp_path/'x.csv'),
        '--model',str(tmp_path/'run'/'fitted_model.npz'),'--input-is-log1p','--out',str(tmp_path/'pred.csv')])
    main()
    assert len(pd.read_csv(tmp_path/'pred.csv'))==12


def test_null_no_class_separation_and_exact_paired_permutation(tmp_path):
    x,m,z=toy(signal=0.)
    # Same expression in the two conditions of each donor: no exposure information.
    for i in range(0,len(x),2): x.iloc[i+1]=x.iloc[i].to_numpy()
    cfg={k:v for k,v in CONFIG.items() if k!='seed'}
    s=analyze(x,m,z,tmp_path/'null',group_key='condition',treated='treated',control='control',
              design='paired',donor_key='donor',n_perm=99,seed=9,**cfg)
    np.testing.assert_allclose(s.oof_auc,.5)
    np.testing.assert_allclose(s.effect_permutation_p,1.)
    np.testing.assert_allclose(s.auc_permutation_p,1.)
    import json
    run=json.loads((tmp_path/'null'/'run.json').read_text())
    assert run['permutation_mode']=='exact_paired_swaps' and run['actual_permutations']==64
    assert (tmp_path/'null'/'paired_differences.csv').exists()


def test_counts_preprocessing_formula_and_sample_independence():
    from gmmap.core.bulk_smrs import preprocess_expression
    x=pd.DataFrame([[10,30,60],[100,300,600]],index=['a','b'],columns=['g1','g2','g3'])
    a,q=preprocess_expression(x,'counts')
    np.testing.assert_allclose(a.iloc[0],np.log1p([1e5,3e5,6e5]))
    np.testing.assert_allclose(a.iloc[0],a.iloc[1])
    assert list(q.library_size)==[100,1000]
    alone,_=preprocess_expression(x.iloc[[0]],'counts')
    pd.testing.assert_frame_equal(alone,a.iloc[[0]])
    unchanged,_=preprocess_expression(a,'log1p')
    pd.testing.assert_frame_equal(unchanged,a)


@pytest.mark.parametrize('bad',[np.array([[0,0],[1,2]]),np.array([[1.5,2],[2,3]]),np.array([[-1,2],[2,3]]),np.array([[np.nan,2],[2,3]])])
def test_counts_bad_inputs(bad):
    from gmmap.core.bulk_smrs import preprocess_expression
    with pytest.raises(ValueError):
        preprocess_expression(pd.DataFrame(bad),'counts')


def test_counts_end_to_end_equals_manual_and_predict_guard(tmp_path):
    from gmmap.core.bulk_smrs import preprocess_expression
    x,m,z=toy(signal=1.5)
    counts=pd.DataFrame(np.rint(np.expm1(x)),index=x.index,columns=x.columns)
    normalized,_=preprocess_expression(counts,'counts')
    cfg={k:v for k,v in CONFIG.items() if k!='seed'}
    common=dict(group_key='condition',treated='treated',control='control',design='paired',
                donor_key='donor',n_perm=19,seed=9,**cfg)
    a=analyze(counts,m,z,tmp_path/'counts',input_scale='counts',**common)
    b=analyze(normalized,m,z,tmp_path/'manual',**common)
    pd.testing.assert_frame_equal(a,b)
    pred=predict(counts,tmp_path/'counts'/'fitted_model.npz',input_scale='counts')
    other=predict(normalized,tmp_path/'manual'/'fitted_model.npz',input_scale='log1p')
    pd.testing.assert_frame_equal(pred,other)
    pd.testing.assert_frame_equal(predict(counts.iloc[[0]],tmp_path/'counts'/'fitted_model.npz'),pred.iloc[[0]])
    pd.testing.assert_frame_equal(predict(counts[counts.columns[::-1]],tmp_path/'counts'/'fitted_model.npz'),pred)
    with pytest.raises(ValueError,match='same full gene universe'):
        predict(counts.iloc[:,:-1],tmp_path/'counts'/'fitted_model.npz')
    with pytest.raises(ValueError,match='expects counts'):
        predict(normalized,tmp_path/'counts'/'fitted_model.npz',input_scale='log1p')
    assert (tmp_path/'counts'/'preprocessing_qc.csv').exists()
    recovered=pd.read_csv(tmp_path/'counts'/'normalized_expression_log1p.csv.gz',index_col=0).T
    np.testing.assert_allclose(recovered,normalized)


def test_counts_cli(tmp_path,monkeypatch):
    x,m,z=toy()
    counts=np.rint(np.expm1(x))
    counts.T.to_csv(tmp_path/'x.csv'); m.to_csv(tmp_path/'m.csv'); z.to_frame().to_csv(tmp_path/'z.csv')
    monkeypatch.setattr('sys.argv',['gmmap-bulk-smrs','analyze','--expression',str(tmp_path/'x.csv'),
        '--metadata',str(tmp_path/'m.csv'),'--magma-z',str(tmp_path/'z.csv'),'--trait','test_trait',
        '--group-key','condition','--treated','treated','--control','control','--design','paired',
        '--donor-key','donor','--input-scale','counts','--top-n-genes','30','--min-valid-genes','20',
        '--n-ctrl','19','--mean-bins','4','--var-bins','3','--n-perm','19','--out',str(tmp_path/'run')])
    main()
    monkeypatch.setattr('sys.argv',['gmmap-bulk-smrs','predict','--expression',str(tmp_path/'x.csv'),
        '--model',str(tmp_path/'run'/'fitted_model.npz'),'--input-scale','counts','--out',str(tmp_path/'pred.csv')])
    main()
    assert len(pd.read_csv(tmp_path/'pred.csv'))==12
