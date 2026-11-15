"""Unlabeled development-score calibration. Alert fractions are not false-positive rates."""
import numpy as np
from .common import path,read,write,sha,load_array,config

def artifact_hashes():
    names=['models/coarse/inference.pt','models/fine/inference.pt','data/features/receipt.json','results/fit_lock.json']
    return {n:sha(path(n)) for n in names}

def alert_cutoff(score,fraction):
    if not 0<fraction<1 or len(score)==0 or not np.isfinite(score).all():raise ValueError('Invalid alert distribution')
    allowed=int(np.floor(len(score)*fraction));rank=np.sort(np.asarray(score,np.float64))[::-1]
    # Strictly exceed the first disallowed sample, so tied groups cannot inflate the alert count.
    return float(np.nextafter(rank[min(allowed,len(rank)-1)],np.inf))

def residual_score(error,center,scale,stage):
    # Use identical float64 arithmetic for calibration arrays and JSON-reloaded inference.
    # NumPy can otherwise round a nextafter(float64) cutoff back to float32 on comparison.
    z=(np.asarray(error,np.float64)-np.asarray(center,np.float64))/np.asarray(scale,np.float64)
    return np.maximum(0,z.min(-1) if stage=='coarse' else z[...,0])

def load_policy():
    p=read('models/policy.json')
    if p['artifacts']!=artifact_hashes() or p['uses_reference_labels']:raise ValueError('Policy provenance failed')
    return p

def calibrate():
    if path('models/policy.json').exists():return load_policy()
    from .model import predict,normalize_y
    policies={}
    for stage in ('coarse','fine'):
        x=load_array(f'data/features/{stage}_validation_X.npy');measured=load_array(f'data/features/{stage}_validation_measured.npy')
        stats=read(f'data/features/{stage}_stats.json');error=normalize_y(measured,stats)-predict(stage,x)
        day=load_array(f'data/features/{stage}_validation_day.npy');groups={}
        for name,mask in [('day',day>0),('night',day==0)]:
            if stage=='fine' and name=='night':continue
            fallback=int(mask.sum())<1000
            sample=error if fallback else error[mask]
            center=np.median(sample,axis=0);scale=np.maximum(1.4826*np.median(np.abs(sample-center),axis=0),.05)
            scores=residual_score(sample,center,scale,stage)
            fraction=config()['coarse_alert_fraction' if stage=='coarse' else 'localization_alert_fraction']
            cutoff=alert_cutoff(scores,fraction)
            groups[name]={'residual_center':center.tolist(),'residual_scale':scale.tolist(),'cutoff':cutoff,
                          'calibration_samples':len(sample),'requested_alert_fraction':fraction,
                          'realized_alert_fraction':float((scores>=cutoff).mean()),'pooled_fallback':fallback}
        policies[stage]=groups
    p={'stages':policies,'artifacts':artifact_hashes(),'fit_split':'validation_MODIS_measurements_only',
       'uses_reference_labels':False,'threshold_uses_modis_fire_flags':False,
       'meaning':'Empirical unlabeled alert-volume thresholds, not fire probabilities or false-positive guarantees',
       'localization':'Daytime only; all four 500m samples must be valid; require SWIR anomaly above unlabeled 99th percentile; retain top two scores including ties; otherwise unresolved',
       'fixed_before_reference_evaluation':True}
    write('models/policy.json',p);return p
