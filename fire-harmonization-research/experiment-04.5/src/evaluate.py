"""Frozen Experiment 04 evaluation; all writes isolated to experiment-04.5."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from datetime import date,timedelta
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'experiment-04/src'))
from larger_model import (Archive,AREA_WEIGHT,REGIONS,SHAPE,HistogramBoost,
                          feature_codes,unpack,supported,protected_files)
import numpy as np

MODEL_FILES={
    'boosted_14year':RESEARCH/'experiment-04/models/boosted_14year_final.json',
    'boosted_11year':RESEARCH/'experiment-04/models/boosted_11year_benchmark.json',
    'boosted_8year':RESEARCH/'experiment-04/models/boosted_8year_selected.json',
}


def output(name):
    p=(ROOT/name).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Writes must stay in experiment-04.5')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p


def write_json(name,obj):output(name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def write_csv(name,rows):
    with output(name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def history_baselines(archive,static):
    rows=[]
    for year in range(2012,2026):
        m,v=archive.arrays(year);valid=archive.valid_dates(year)
        for week in range(14):
            sl=slice(7*week,min(92,7*week+7));days=int(valid[sl].sum())
            if not days:continue
            for region,geo in REGIONS.items():
                w=AREA_WEIGHT*(geo&~static)
                rows.append({'year':year,'week':week,'region':region,'valid_days':days,
                             'modis':float(np.sum((m[sl][valid[sl]]>0)*w)),
                             'viirs':float(np.sum((v[sl][valid[sl]]>0)*w))})
    result={}
    for region in REGIONS:
        r=[x for x in rows if x['region']==region]
        x=np.array([x['modis']/x['valid_days'] for x in r]);y=np.array([x['viirs']/x['valid_days'] for x in r])
        b,a=np.polyfit(np.log1p(x),np.log1p(y),1)
        result[region]={'ratio':sum(x['viirs'] for x in r)/sum(x['modis'] for x in r),
                        'power_a':float(a),'power_b':float(b)}
    write_csv('results/baseline_training_observations.csv',rows)
    write_json('models/training_only_baselines.json',{'years':list(range(2012,2026)),'power_fit_unit':'mean daily activity in weekly bins','regions':result})
    return result


def binary_scores(prob,y,w):
    p=prob.ravel();t=y.ravel().astype(float);weights=np.broadcast_to(w,y.shape).ravel()
    clip=np.clip(p,1e-12,1-1e-12);yes=p>=.5
    tp=float(np.sum(weights*t*yes));fp=float(np.sum(weights*(1-t)*yes));fn=float(np.sum(weights*t*(~yes)))
    levels,inv=np.unique(p,return_inverse=True)
    pos=np.bincount(inv,weights=weights*t);neg=np.bincount(inv,weights=weights*(1-t))
    tp_curve=np.cumsum(pos[::-1]);fp_curve=np.cumsum(neg[::-1])
    tpr=np.r_[0,tp_curve/pos.sum()];fpr=np.r_[0,fp_curve/neg.sum()]
    auc=float(np.trapezoid(tpr,fpr))
    ap=float(np.sum(pos[::-1]/pos.sum()*tp_curve/(tp_curve+fp_curve)))
    return {'brier':float(np.sum(weights*(p-t)**2)/weights.sum()),
            'logloss':float(-np.sum(weights*(t*np.log(clip)+(1-t)*np.log1p(-clip)))/weights.sum()),
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'roc_auc':auc,'average_precision':ap}


def summarize(rows):
    out=[]
    for scenario,model,region in sorted({(r['scenario'],r['model'],r['region']) for r in rows}):
        g=[r for r in rows if (r['scenario'],r['model'],r['region'])==(scenario,model,region)]
        obs=np.array([r['observed_viirs'] for r in g]);pred=np.array([r['predicted_viirs'] for r in g])
        out.append({'scenario':scenario,'model':model,'region':region,'matched_days':sum(r['valid_days'] for r in g),
                    'calendar_bins':len(g),'weekly_mae':float(np.mean(np.abs(pred-obs))),
                    'weekly_rmse':float(np.sqrt(np.mean((pred-obs)**2))),'weekly_wape':float(np.abs(pred-obs).sum()/obs.sum()),
                    'aggregate_bias':float((pred.sum()-obs.sum())/obs.sum()),
                    'observed_viirs':float(obs.sum()),'predicted_viirs':float(pred.sum())})
    return out


def run():
    main_before=protected_files()
    protected={}
    # Protect all existing research content without reading/writing dependency caches.
    for p in RESEARCH.rglob('*'):
        if p.is_file() and not p.is_symlink() and not p.is_relative_to(ROOT) and not any(x in ('.cache','.venv','__pycache__') for x in p.relative_to(RESEARCH).parts):
            protected[str(p.relative_to(RESEARCH))]=sha(p)
    write_json('results/protected_hashes.json',{'main':main_before,'prior_research':protected})
    write_json('results/protocol_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'PROTOCOL.md',Path(__file__))})
    archive=Archive()
    inputs=[archive.root/'seasons.json',*[archive.root/f'seasons/{y}.json' for y in range(2012,2027)]]
    write_json('results/input_hashes.json',{str(p):sha(p) for p in inputs})
    cfg=json.loads((RESEARCH/'experiment-04/results/config.json').read_text())
    static=np.zeros(SHAPE,bool)
    for y,x in cfg['static_cells']:static[y,x]=True
    model_audit={}
    for name,p in MODEL_FILES.items():
        obj=json.loads(p.read_text());years=obj['metadata']['training']['years']
        assert max(years)<2026
        model_audit[name]={'path':str(p),'sha256':sha(p),'training_years':years,'trees':len(obj['trees']),
                           'test_year':2026,'test_year_outside_training':True}
    write_json('results/frozen_models.json',model_audit)
    m,v=archive.arrays(2026)
    modis_any=m.sum(axis=(1,2))>0;viirs_any=v.sum(axis=(1,2))>0
    candidate=np.arange(92)<61
    gap=np.zeros(92,bool)
    for sensor in ('modis','viirs'):
        for day in archive.meta.get('gaps',{}).get(sensor,{}).get('2026',[]):gap[day]=True
    masks={'paired_active_dates':candidate&~gap&modis_any&viirs_any,'metadata_window':candidate&~gap}
    dates=[]
    for day in range(92):
        dates.append({'date':(date(2026,6,1)+timedelta(days=day)).isoformat(),
                      'modis_raw_detections':int(m[day].sum()),'viirs_raw_detections':int(v[day].sum()),
                      'metadata_gap_flag':bool(gap[day]),'candidate_june_july':bool(candidate[day]),
                      **{s:bool(mask[day]) for s,mask in masks.items()}})
    write_csv('results/date_audit.csv',dates)
    write_json('results/coverage_policy.json',{'candidate_window':['2026-06-01','2026-07-31'],
                'window_dates_inferred_not_request_verified':True,
                'days_by_scenario':{s:int(mask.sum()) for s,mask in masks.items()},
                'primary_population_conditioned_on_both_sensors_globally_active':True,
                'clear_sky_exposure_available':False,'august_scored':False,
                'inference_uses_modis_only_coverage_flags':True})
    # No model score has been generated before masks, sources and protocol are frozen.
    m[:,static]=0;v[:,static]=0
    modis_valid=np.ones(92,bool)
    for day in archive.meta.get('gaps',{}).get('modis',{}).get('2026',[]):modis_valid[day]=False
    codes=feature_codes(m,modis_valid)
    altered=m.copy();altered[61:]+=123
    assert np.array_equal(codes[:61],feature_codes(altered,modis_valid)[:61]),'Future MODIS data leaked'
    eligible=~static;keys,inv=np.unique(codes[:61,eligible],return_inverse=True)
    features=unpack(keys)
    predictions={'raw_modis':(m[:61]>0).astype(float)}
    for name,path in MODEL_FILES.items():
        model=HistogramBoost.load(path);p=model.predict(features)
        values=np.zeros((61,*SHAPE));values[:,eligible]=p[inv].reshape(61,-1)
        predictions[name]=values
        print('Scored frozen model:',name,flush=True)
    del codes,altered
    baseline=history_baselines(archive,static)
    arrays={'modis':m[:61]>0,'viirs':v[:61]>0,'static':static,
            **{s:mask[:61] for s,mask in masks.items()},**predictions}
    np.savez_compressed(output('results/prediction_grids_2026.npz'),**arrays)
    weekly=[];daily=[];binary=[];support=[]
    for scenario,mask_full in masks.items():
        mask=mask_full[:61]
        for region,geo in REGIONS.items():
            w=AREA_WEIGHT*(geo&eligible)
            for day in np.flatnonzero(mask):
                daily.append({'scenario':scenario,'date':dates[day]['date'],'region':region,
                              'observed_viirs':float(np.sum((v[day]>0)*w)),
                              **{name:float(np.sum(value[day]*w)) for name,value in predictions.items()}})
            for week in range(9):
                daymask=mask.copy();daymask[:week*7]=False;daymask[min(61,week*7+7):]=False
                days=int(daymask.sum())
                if not days:continue
                observed=float(np.sum((v[:61][daymask]>0)*w));raw=float(np.sum((m[:61][daymask]>0)*w))
                totals={name:float(np.sum(values[daymask]*w)) for name,values in predictions.items()}
                b=baseline[region]
                totals['constant_ratio_14year']=raw*b['ratio']
                totals['power_14year_rate']=max(0.,math.expm1(b['power_a']+b['power_b']*math.log1p(raw/days)))*days
                for name,total in totals.items():
                    weekly.append({'scenario':scenario,'year':2026,'week':week,'region':region,'valid_days':days,
                                   'observed_viirs':observed,'observed_modis':raw,'model':name,'predicted_viirs':total})
            zero_support=~supported(features)
            support_mask=np.zeros((61,*SHAPE),bool);support_mask[:,eligible]=zero_support[inv].reshape(61,-1)
            ref=float(np.sum((v[:61][mask]>0)*w));unsupported=float(np.sum((v[:61][mask]>0)*support_mask[mask]*w))
            support.append({'scenario':scenario,'region':region,'reference_activity':ref,'outside_modis_context':unsupported,'unsupported_share':unsupported/ref if ref else None})
        # Full-grid probability metrics: no fitting or calibration on this year.
        w=AREA_WEIGHT[eligible][None,:]
        y=(v[:61][mask][:,eligible]>0)
        for name,prob in predictions.items():
            binary.append({'scenario':scenario,'model':name,'eligible_cell_days':int(mask.sum()*eligible.sum()),
                           **binary_scores(prob[mask][:,eligible],y,w)})
    write_csv('results/weekly_predictions.csv',weekly);write_csv('results/daily_predictions.csv',daily)
    write_csv('results/metrics.csv',summarize(weekly));write_json('results/metrics.json',summarize(weekly))
    write_csv('results/binary_metrics.csv',binary);write_csv('results/support_diagnostics.csv',support)
    assert all(sha(Path(a['path']))==a['sha256'] for a in model_audit.values())
    assert protected_files()==main_before
    assert all(sha(RESEARCH/name)==h for name,h in protected.items())
    write_json('results/isolation.json',{'main_files_unchanged':len(main_before),'prior_research_files_unchanged':len(protected),
                'boosted_models_retrained':False,'new_outputs_only_in_experiment_04_5':True})
    print('Completed frozen 2026 holdout evaluation.',flush=True)


if __name__=='__main__':run()
