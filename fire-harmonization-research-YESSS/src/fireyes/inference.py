"""MODIS-only inference. Stage 2 never invents additional coarse alerts."""
import numpy as np
from .features import extract,INPUT_KEYS
from .model import predict,normalize_y
from .policy import load_policy,residual_score
from .common import read,save_npz,path

def score_native(inputs,stage,policy):
    x,measured,valid,day,names=extract(inputs,stage)
    score=np.full(valid.shape,-1.,np.float64)
    if valid.any():
        stats=read(f'data/features/{stage}_stats.json')
        if names!=stats['feature_names']:raise ValueError('Inference schema differs from trained model')
        error=normalize_y(measured[valid],stats)-predict(stage,x[valid])
        d=day[valid];values=np.empty(len(error),np.float64)
        for name,mask in [('day',d),('night',~d)]:
            if not mask.any():continue
            p=policy['stages'][stage][name]
            values[mask]=residual_score(error[mask],p['residual_center'],p['residual_scale'],stage)
        score[valid]=values
    return score,valid,day

def localize(coarse_detection,fine_score,fine_valid,fine_cutoff):
    h,w=coarse_detection.shape
    if fine_score.shape!=(2*h,2*w) or fine_valid.shape!=fine_score.shape:raise ValueError('Resolution mismatch')
    s=fine_score.reshape(h,2,w,2).transpose(0,2,1,3).reshape(h,w,4)
    valid=fine_valid.reshape(h,2,w,2).transpose(0,2,1,3).reshape(h,w,4)
    supported=valid.all(-1)&(s.max(-1)>=fine_cutoff)
    clipped=np.clip(s,0,10);ex=np.exp(clipped-clipped.max(-1,keepdims=True));weights=ex/ex.sum(-1,keepdims=True)
    weights=np.where(supported[...,None],weights,.25)
    second=np.sort(s,axis=-1)[...,2]
    keep=np.where(supported[...,None],s>=second[...,None],True)&coarse_detection[...,None]
    fine=keep.reshape(h,w,2,2).transpose(0,2,1,3).reshape(2*h,2*w)
    mass=weights.reshape(h,w,2,2).transpose(0,2,1,3).reshape(2*h,2*w)
    entropy=-(weights*np.log(np.maximum(weights,1e-30))).sum(-1)/np.log(4)
    return fine,mass,coarse_detection&supported,entropy

def infer_arrays(inputs,policy=None):
    p=policy or load_policy();cs,cv,day=score_native(inputs,'coarse',p)
    threshold=np.where(day,p['stages']['coarse']['day']['cutoff'],p['stages']['coarse']['night']['cutoff'])
    detect=cv&(cs>=threshold)
    fs,fv,_=score_native(inputs,'fine',p)
    fine,weights,resolved,entropy=localize(detect,fs,fv,p['stages']['fine']['day']['cutoff'])
    return {'coarse_score':cs,'coarse_valid':cv,'coarse_detection':detect,'fine_score':fs,'fine_valid':fv,
            'fine_detection':fine,'localization_weights':weights,'localization_supported':resolved,
            'localization_entropy':entropy,'coarse_daylight':day}

def infer_file(filename,output):
    with np.load(path(filename),allow_pickle=False) as z:
        if set(z.files)!=INPUT_KEYS:raise ValueError('Input file must contain MODIS-only arrays')
        arrays={k:z[k] for k in INPUT_KEYS}
    result=infer_arrays(arrays);save_npz(output,**result)
    return {'output':output,'coarse_alerts':int(result['coarse_detection'].sum()),
            'localized_alerts':int(result['localization_supported'].sum()),'uses_viirs':False,
            'uses_modis_fire_product':False,'scores_are_probabilities':False}
