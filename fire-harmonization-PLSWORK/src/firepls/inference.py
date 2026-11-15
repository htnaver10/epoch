"""Label-free inference from prepared native MODIS inputs with no VIIRS dependency."""
import numpy as np
from .common import path,read,config,sha
from .features import extract,query_indices


def save_npz(filename,**arrays):
    dest=path(filename);dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.partial')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(dest)


def infer(tile,output,method='mlp'):
    cfg=config()
    with np.load(path(tile),allow_pickle=False) as z:
        inputs={k:z[k] for k in ('thermal','reflective_500','reflective_250','geometry','query_grid')}
        grid=inputs['query_grid'];shape=grid.shape[:2]
        if 'inference_valid' in z:valid=z['inference_valid']>0
        else:
            iy,ix,_,_=query_indices(grid,*inputs['thermal'].shape[-2:])
            valid=(np.abs(grid)<=1).all(-1)&(inputs['thermal'][6:,iy,ix].sum(0)>=2)
    x,names=extract(inputs,cfg);flat=x.reshape(-1,len(names))
    if method=='mlp':
        import torch
        from .mlp import load_model,normalize_numpy,predict_tensor
        model,saved=load_model();stats=saved['stats']
        if names!=stats['feature_names']:raise ValueError('Inference feature schema changed')
        logits=predict_tensor(model,torch.from_numpy(normalize_numpy(flat,stats)))
    else:
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(path('models/tree/model.txt')))
        if names!=read('data/features/statistics.json')['feature_names']:raise ValueError('Tree feature schema changed')
        logits=model.predict(flat,raw_score=True,num_threads=8)
    from firefullres.calibration import sigmoid
    arrays={'raw_probability':sigmoid(logits).reshape(shape),'valid':valid}
    if path('results/operating_policy.json').exists():
        from .evaluation import checkpoint_digest
        policies=read('results/operating_policy.json')
        if policies['model_sha256'][method]!=checkpoint_digest(method):raise ValueError('Operating policy does not match inference weights')
        p=policies['policies'][method]
        arrays.update(adjusted_probability=sigmoid(logits+p['logit_offset']).reshape(shape),
            frozen_policy_detection=(logits.astype(np.float64)>=p['raw_logit_cutoff']).reshape(shape)&valid)
    save_npz(output,**arrays);return {'output':output,'method':method,'requires_viirs':False,'query_shape':shape}
