"""Generate a deliberately strong synthetic paired exposure example and run bulk SMRS.
Not evidence of biological sensitivity. Usage: python demo.py --out demo_run
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gmmap.core.bulk_smrs import analyze

p=argparse.ArgumentParser(); p.add_argument('--out',required=True); args=p.parse_args()
out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20260915)
y=np.tile([0,1],8)
genes=[f'GENE{i:04d}' for i in range(800)]
samples=[f'D{i+1:02d}_{c}' for i in range(8) for c in ['control','treated']]
x=np.maximum(0,rng.normal(3,.7,(16,800))+np.repeat(rng.normal(0,.3,(8,800)),2,axis=0))
x[y==1,:50]+=1.
expr=pd.DataFrame(x,index=samples,columns=genes)
meta=pd.DataFrame({'condition':np.where(y,'treated','control'),'donor':np.repeat([f'D{i+1:02d}' for i in range(8)],2)},index=samples)
z=pd.Series(np.r_[np.linspace(5,3,50),np.repeat(-1.,750)],index=genes,name='synthetic_metabolite')
expr.T.to_csv(out/'expression_log1p.csv'); meta.to_csv(out/'metadata.csv'); z.to_frame().to_csv(out/'magma_z.csv')
summary=analyze(expr,meta,z,out/'results',group_key='condition',treated='treated',control='control',
                design='paired',donor_key='donor',top_n=50,min_genes=30,n_ctrl=99,
                mean_bins=5,var_bins=4,n_perm=999,seed=0)
print(summary.to_string(index=False))
