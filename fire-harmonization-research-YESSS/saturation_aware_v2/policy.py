"""Channel-aware unlabeled day/night calibration, without pooled nighttime fallback."""
import numpy as np
from common import path,read,write,sha,load_array,config

def artifact_hashes():
    names=['models/coarse/inference.pt','models/fine/inference.pt','data/features/receipt.json','results/source_lock.json']
    return {n:sha(path(n)) for n in names}

def alert_cutoff(score,fraction):
    scores=np.asarray(score,np.float64)
    if not 0<fraction<1 or not len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid alert distribution')
    allowed=int(np.floor(len(scores)*fraction));rank=np.sort(scores)[::-1]
    return float(np.nextafter(rank[min(allowed,len(rank)-1)],np.inf))

def selected_channel(observed):
    observed=np.asarray(observed,bool)
    if observed.shape[-1]!=2:raise ValueError('Expected band21/band22 observation mask')
    return np.where(observed[...,1],1,np.where(observed[...,0],0,-1))

def standardized(error,center,scale):
    return (np.asarray(error,np.float64)-np.asarray(center,np.float64))/np.asarray(scale,np.float64)

def residual_score(error,group,stage,observed):
    z=standardized(error,group['residual_center'],group['residual_scale'])
    if stage=='fine':return np.maximum(0,z[...,0])
    selected=selected_channel(observed)
    score=np.take_along_axis(z,np.maximum(selected,0)[...,None],axis=-1)[...,0]
    return np.where(selected>=0,np.maximum(0,score),-1.)

def load_policy():
    p=read('models/policy.json')
    if p['artifacts']!=artifact_hashes() or p['uses_reference_labels']:raise ValueError('Policy provenance failed')
    return p

def calibrate():
    if path('models/policy.json').exists():return load_policy()
    from model import predict,normalize_y
    policies={}
    for stage in ('coarse','fine'):
        prefix=f'data/features/{stage}_development'
        x=load_array(prefix+'_X.npy');measured=load_array(prefix+'_measured.npy');observed=load_array(prefix+'_target_valid.npy')
        stats=read(f'data/features/{stage}_stats.json');error=normalize_y(measured,stats)-predict(stage,x)
        day=load_array(prefix+'_day.npy');groups={}
        for name,mask in [('day',day>0),('night',day==0)]:
            if stage=='fine' and name=='night':continue
            if int(mask.sum())<config()['minimum_calibration_samples']:raise ValueError(f'Insufficient unlabeled {stage}/{name} calibration samples')
            e=error[mask];tv=observed[mask];centers=[];scales=[];counts=[]
            for j in range(e.shape[1]):
                sample=e[tv[:,j],j]
                if len(sample)<config()['minimum_calibration_samples']:raise ValueError('Insufficient valid per-channel calibration')
                center=float(np.median(sample));scale=float(max(1.4826*np.median(np.abs(sample-center)),.05))
                centers.append(center);scales.append(scale);counts.append(len(sample))
            group={'residual_center':centers,'residual_scale':scales}
            scores=residual_score(e,group,stage,tv)
            fraction=config()['coarse_alert_fraction' if stage=='coarse' else 'localization_alert_fraction']
            cutoff=alert_cutoff(scores,fraction)
            groups[name]={**group,'cutoff':cutoff,'calibration_samples':len(scores),'per_channel_observed_samples':counts,
                          'requested_alert_fraction':fraction,'realized_alert_fraction':float((scores>=cutoff).mean()),
                          'selected_channel_counts':np.bincount(selected_channel(tv),minlength=2).tolist() if stage=='coarse' else None}
        policies[stage]=groups
    p={'model_name':config()['name'],'stages':policies,'artifacts':artifact_hashes(),'fit_split':'unlabeled_whole_swath_development',
       'uses_reference_labels':False,'threshold_uses_modis_fire_flags':False,'nighttime_fallback':False,
       'meaning':'Empirical alert-volume control; anomaly scores are not fire probabilities and alert volume is not false-positive rate',
       'channel_rule':config()['scoring_rule'],'fixed_before_reference_evaluation':True,
       'localization':'Unchanged V1 heuristic: valid 500m daytime children, strong residual, top two including ties; unresolved alerts retain all children.'}
    write('models/policy.json',p);return p
