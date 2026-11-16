"""Publish the frozen PLSWORK benchmark as geolocated, versioned viewer data.

This exports existing BF16 predictions, rather than recomputing them in a
different numerical precision. It does not fit, download or mutate the model.
"""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PLS=ROOT/'fire-harmonization-PLSWORK'
BACKEND=ROOT/'backend/data/ml_cache'
PUBLIC=ROOT/'frontend/public/data/ml'

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def build():
    policy=json.loads((PLS/'results/operating_policy.json').read_text());evaluation=json.loads((PLS/'results/evaluation.json').read_text())
    model_hash=digest(PLS/'models/mlp/inference.pt')
    assert model_hash==policy['model_sha256']['mlp']
    setting=policy['policies']['mlp'];cut=setting['raw_logit_cutoff'];offset=setting['logit_offset']
    rows=[json.loads(s) for s in (PLS/'data/manifests/source_v3.jsonl').read_text().splitlines() if s.strip()]
    ledger={r['tile_id']:r for r in json.loads((PLS/'data/features/tiles.json').read_text())}
    predictions=np.load(PLS/'results/test_mlp_predictions.npz',allow_pickle=False)
    version='plswork-mlp-'+model_hash[:12];scenes=[];files={};tp=fp=fn=0;total=0
    for source_index,row in enumerate(rows):
        if row['split']!='test':continue
        tile=PLS/row['tile'];assert digest(tile)==row['sha256']
        with np.load(tile,allow_pickle=False) as z:
            known=z['weight']>0;indices=np.flatnonzero(known);lat=z['query_lat'][known];lon=z['query_lon'][known]
            target=z['target'][known]>.5;modis=z['modis_baseline'][known]>.5
        item=ledger[row['tile_id']];start,end=item['start'],item['end']
        np.testing.assert_array_equal(predictions['flat_index'][start:end],indices)
        np.testing.assert_array_equal(predictions['tile_index'][start:end],np.full(len(indices),source_index))
        np.testing.assert_array_equal(predictions['target'][start:end]>.5,target)
        np.testing.assert_array_equal(predictions['modis'][start:end]>.5,modis)
        assert np.isfinite(lat).all() and np.isfinite(lon).all() and (np.abs(lat)<=90).all() and (np.abs(lon)<=180).all()
        logits=predictions['logits'][start:end];mlp=logits.astype(np.float64)>=cut
        scores=np.exp(-np.logaddexp(0.,-(logits+offset)))
        counts={'queries':len(indices),'mlp':int(mlp.sum()),'modis':int(modis.sum()),'viirs':int(target.sum()),
                'agreement':int((mlp&modis).sum()),'mlp_only':int((mlp&~modis).sum()),'modis_only':int((~mlp&modis).sum()),
                'TP':int((mlp&target).sum()),'FP':int((mlp&~target).sum()),'FN':int((~mlp&target).sum())}
        tp+=counts['TP'];fp+=counts['FP'];fn+=counts['FN'];total+=len(indices)
        scene={'id':row['tile_id'],'observed_at':row['modis_start_utc'],'date':row['modis_start_utc'][:10],
               'bounds':[float(lon.min()),float(lat.min()),float(lon.max()),float(lat.max())],
               'counts':counts,'model_version':version,'native_shape':row['native_shape'],'query_shape':row['query_shape'],
               'pair_time_bound_seconds':row['interval_time_bound_seconds'],'source_tile_sha256':row['sha256'],
               'coverage_kind':'reference_matched_query_centers','status':'cached',
               'source_granules':[Path(s).name for s in row['source_ids']]}
        features=[]
        for i in np.flatnonzero(mlp|modis|target):
            features.append({'type':'Feature','id':f'{row["tile_id"]}:{int(indices[i])}',
                'geometry':{'type':'Point','coordinates':[round(float(lon[i]),6),round(float(lat[i]),6)]},
                'properties':{'query_index':int(indices[i]),'mlp_score':round(float(scores[i]),7),
                    'mlp_detected':bool(mlp[i]),'modis_detected':bool(modis[i]),'viirs_reference':bool(target[i])}})
        data={'type':'FeatureCollection','scene':scene,'features':features}
        coverage={'type':'Feature','properties':{'scene_id':row['tile_id'],'kind':'reference_matched_query_centers',
                  'count':len(indices),'description':'Only these query centers have cached scores and eligible reference observations. White/empty surrounding areas are unavailable, not fire-free.'},
                  'geometry':{'type':'MultiPoint','coordinates':np.round(np.column_stack([lon,lat]),6).tolist()}}
        for relative,value in [(f'scenes/{row["tile_id"]}.json',data),(f'coverage/{row["tile_id"]}.json',coverage)]:
            write(BACKEND/relative,value);files[relative]=digest(BACKEND/relative)
        scenes.append(scene)
    expected=evaluation['test']['mlp_frozen_policy'];assert (tp,fp,fn,total)==(70,72,25,585464)
    assert tp==expected['true_positives'] and fp==expected['false_positives']
    model={'id':'plswork-contextual-mlp','version':version,'name':'PLSWORK contextual MLP','status':'experimental',
           'checkpoint_sha256':model_hash,'feature_count':96,'parameters':8321,'prediction_precision':'saved CUDA BF16 logits',
           'policy_sha256':digest(PLS/'results/operating_policy.json'),'raw_logit_cutoff':cut,
           'display_score_cutoff':float(np.exp(-np.logaddexp(0.,-(cut+offset)))),
           'metric_scope':'Previously examined 2024 benchmark; reference query centers, not independent fires',
           'metrics':{k:expected[k] for k in ('precision','recall','f1','true_positives','false_positives','false_negatives','average_precision_exact')},
           'baseline_metrics':{k:evaluation['test']['regular_modis'][k] for k in ('precision','recall','f1','true_positives','false_positives','false_negatives')},
           'limitations':['Experimental cached research scenes; not live or operational monitoring.',
                         'Coverage is limited to reference-matched query centers in these patches; outside is unavailable.',
                         'VIIRS supplies evaluation labels, not independent ground-truth fire observations.',
                         'MLP score is validation-adjusted and is not a guaranteed real-world fire probability.',
                         'Query locations follow reference geolocation; they do not create finer measured MODIS thermal resolution.',
                         'Historical calendar and scenario simulation are separate from these model predictions.']}
    catalogue={'schema':'epoch_mlp_cache_v1','model':model,'scenes':sorted(scenes,key=lambda s:(s['observed_at'],s['id'])),
               'files':files,'source_prediction_sha256':digest(PLS/'results/test_mlp_predictions.npz'),
               'total_queries':total,'scene_count':len(scenes),'new_inference':False,'new_satellite_downloads':0}
    write(BACKEND/'catalog.json',catalogue)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for rel in [*files,'catalog.json']:
        dest=PUBLIC/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BACKEND/rel,dest)
        assert digest(dest)==digest(BACKEND/rel)
    print(json.dumps({'scenes':len(scenes),'queries':total,'TP':tp,'FP':fp,'FN':fn,'version':version,
                      'backend':str(BACKEND),'static_snapshot':str(PUBLIC)},indent=2))

if __name__=='__main__':build()
