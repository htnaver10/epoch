"""Freeze validation calibration/threshold, then report untouched test performance."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core,engine
from firefullres.runtime import resolve_config,configure_model,move
from firefullres.calibration import sigmoid,fit_intercept,matched_false_alarm_cutoff,metrics


def readable_report(report,name):
    directory=f'results/{name}';rows=[]
    for split in ('validation','test'):
        for model,values in report[split].items():rows.append({'split':split,'model':model,**values})
    with core.path(f'{directory}/metrics.csv',True).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Expanded real-data round: '+name,'',
        'The operating threshold and intercept adjustment were fitted on development validation, then frozen before test predictions.',
        'This is an all-land VIIRS-reference experiment. It is not a forest-only or operational wildfire model.','']
    for split in ('validation','test'):
        lines+=['## '+split.capitalize(),'',
            '| Method | Precision | Recall | False positives | False negatives | Brier | Score-sum activity |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for label,key in [('MODIS','modis'),('Raw model at 0.5','raw_model_at_0_5'),
            ('Validation-adjusted model, frozen operating point','validation_adjusted_model_at_frozen_threshold'),
            ('Always-clear reference','always_clear_reference')]:
            value=report[split][key]
            lines.append(f'| {label} | {100*value["precision"]:.3f}% | {100*value["recall"]:.3f}% | '
                f'{value["false_positives"]:,.0f} | {value["false_negatives"]:,.0f} | {value["brier"]:.6f} | '
                f'{value["score_sum_active_query_equivalents"]:,.1f} |')
        lines.append('')
    lines+=['## Interpretation','',
        'Matching validation activity totals is a consequence of the fitted intercept, not independent calibration evidence.',
        'The threshold permits at most the MODIS validation false-positive exposure; test false alarms can reveal a failure of that policy to transfer.',
        'An always-clear baseline can have low Brier loss in rare-fire data while detecting no fires.',
        'Compare models on the same new test exposures. Pooled counts are not directly comparable with the earlier one-scene pilot.',
        'Read evaluation.json for exact tied-score ranking metrics, activity bias, checkpoint hashes and coverage limitations.','',
        'No automatic usefulness or deployment approval is made. More independent events and uncertainty analysis are needed before a broad claim.','']
    core.path(f'{directory}/EVALUATION.md',True).write_text('\n'.join(lines),encoding='utf-8')


def collect(manifest,checkpoint,split):
    rows=[r for r in core.validate_tiles(manifest,False,False,False) if r['split']==split]
    if not rows:raise ValueError('Missing evaluation split')
    device=engine.device_for('cuda');model,saved=engine.load_checkpoint(checkpoint,device)
    config=resolve_config(saved['config']);model=configure_model(model,config,device,compile_model=False);model.eval()
    values={k:[] for k in ('logits','target','weight','modis','tile_index')}
    tiles=[]
    with torch.inference_mode():
        for raw in engine.loader(rows,saved['stats'],config):
            batch=move(raw,device,config,labels=False)
            with engine.amp_context(device,config['precision']):out=model(**batch['inputs'])
            logits=out['logits'].float().cpu().numpy()[:,0]
            for i,row in enumerate(raw['record']):
                h,w=raw['query_shapes'][i];weight=raw['weight'][i,0,:h,:w].numpy();known=weight>0
                with np.load(core.path(row['tile']),allow_pickle=False) as z:baseline=z['modis_baseline']
                index=len(tiles);tiles.append({'tile_id':row['tile_id'],'event_group':row['event_group'],'region_id':row['region_id'],
                    'acquisition_utc':row.get('modis_start_utc'),'source_ids':row['source_ids']})
                values['logits'].append(logits[i,:h,:w][known]);values['target'].append(raw['target'][i,0,:h,:w].numpy()[known])
                values['weight'].append(weight[known]);values['modis'].append(baseline[known])
                values['tile_index'].append(np.full(int(known.sum()),index,np.int32))
    return {k:np.concatenate(v) for k,v in values.items()},tiles


def evaluate(config_path):
    config=core.read_json(config_path);name=config['name'];folder=f'results/{name}'
    manifest=f'data/manifests/{name}.jsonl';checkpoint=f'models/{name}/best.pt'
    calibration_path=f'{folder}/calibration.json';report_path=f'{folder}/evaluation.json'
    checkpoint_digest=core.sha256(checkpoint);manifest_digest=core.sha256(manifest)
    if core.path(report_path).exists():
        existing=core.read_json(report_path)
        if existing['checkpoint_sha256']!=checkpoint_digest or existing['manifest_sha256']!=manifest_digest:
            raise ValueError('Sealed evaluation belongs to a different checkpoint/dataset; preserve it and plan a fresh experiment')
        readable_report(existing,name)
        return existing
    print('Collecting development validation predictions; test predictions are not used for fitting.',flush=True)
    validation,val_tiles=collect(manifest,checkpoint,'validation')
    core.write_npz(f'{folder}/validation_predictions.npz',**validation)
    if core.path(calibration_path).exists():
        calibration=core.read_json(calibration_path)
        if calibration['checkpoint_sha256']!=checkpoint_digest or calibration['manifest_sha256']!=manifest_digest:
            raise ValueError('Frozen calibration belongs to another checkpoint/dataset')
        offset=calibration['logit_offset'];cut=calibration['raw_logit_cutoff']
    else:
        offset=fit_intercept(validation['logits'],validation['target'],validation['weight'])
        cut,budget=matched_false_alarm_cutoff(validation['logits'],validation['target'],validation['weight'],validation['modis'])
        calibration={'checkpoint':checkpoint,'checkpoint_sha256':checkpoint_digest,'manifest_sha256':manifest_digest,
            'fit_split':'validation','test_used_for_fit':False,'logit_offset':offset,'raw_logit_cutoff':cut,
            'display_calibrated_probability_threshold':float(sigmoid(cut+offset)),
            'validation_false_positive_budget_from_modis':budget,
            'method':'validation-only intercept adjustment and conservative cutoff at MODIS validation false-positive exposure',
            'caution':'Matching activity totals on calibration data is by construction, not proof of generalization.'}
        core.write_json(calibration_path,calibration)
    def score(values):
        z,y,w,b=(values[k] for k in ('logits','target','weight','modis'))
        return {'modis':metrics(b,y,w,b>=.5),
            'always_clear_reference':metrics(np.zeros_like(z),y,w,np.zeros(z.shape,bool)),
            'raw_model_at_0_5':metrics(sigmoid(z),y,w,z>=0,ranking_score=z),
            'validation_adjusted_model_at_frozen_threshold':metrics(sigmoid(z+offset),y,w,np.asarray(z,np.float64)>=cut,ranking_score=z)}
    validation_scores=score(validation)
    print('Calibration and operating threshold frozen. Evaluating the reserved test scenes.',flush=True)
    test,test_tiles=collect(manifest,checkpoint,'test');core.write_npz(f'{folder}/test_predictions.npz',**test)
    report={'checkpoint':checkpoint,'checkpoint_sha256':checkpoint_digest,'manifest_sha256':manifest_digest,
        'calibration':calibration,'validation':validation_scores,'test':score(test),
        'validation_tiles':val_tiles,'test_tiles':test_tiles,
        'scope':config['scope'],'model_usefulness_established':False,
        'limitations':['This expansion is not the full 100-event feasibility target.',
            'Cohorts are not verified wildfire incidents; reference pixels within events are correlated.',
            'Test labels were read for integrity/coverage only, not used for checkpoint, intercept or threshold selection.',
            'Offset/threshold apply in this evaluation; generic run.py inference still emits raw model scores.',
            'No operational wildfire, forest-only or absolute subpixel location claim.']}
    readable_report(report,name);core.write_json(report_path,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/substantial_round_v2.json')
    args=p.parse_args();result=evaluate(args.config)
    print(json.dumps({'test':result['test'],'report':f'results/{core.read_json(args.config)["name"]}/evaluation.json'},indent=2))
