"""Freeze validation policies, then evaluate on the reused benchmark exactly once."""
import csv
import json
import subprocess
import sys

import numpy as np

from firefullres.calibration import sigmoid,fit_intercept,matched_false_alarm_cutoff,metrics,exact_ranking
from .common import ROOT,path,read,write,sha,config,validate_parent
from .prepare import load_split


def checkpoint_digest(method):return sha('models/mlp/inference.pt' if method=='mlp' else 'models/tree/model.txt')


def predict_method(method,split):
    if method=='mlp':
        from .mlp import predict
        return predict(split)
    cfg=config();python=(ROOT/cfg['tree_python']).resolve()
    result=subprocess.run([str(python),str(ROOT/'run.py'),'predict-tree','--split',split],check=True)
    return np.load(path(f'results/{split}_tree_logits.npy'),allow_pickle=False)


def write_predictions(method,split,z):
    from .inference import save_npz
    raw=load_split(split)
    if z.shape!=raw['y'].shape or not np.isfinite(z).all():raise ValueError('Invalid model predictions')
    save_npz(f'results/{split}_{method}_predictions.npz',logits=z.astype(np.float64),
        target=np.asarray(raw['y']),weight=np.asarray(raw['w']),modis=np.asarray(raw['modis']),
        tile_index=np.asarray(raw['tile_index']),flat_index=np.asarray(raw['flat_index']))


def freeze_policies():
    output='results/operating_policy.json';models={method:checkpoint_digest(method) for method in ('mlp','tree')}
    if path(output).exists():
        frozen=read(output)
        if frozen['model_sha256']!=models or frozen['features_sha256']!=sha('data/features/receipt.json'):
            raise ValueError('Frozen operating policy belongs to another model/dataset')
        return frozen
    raw=load_split('validation');policies={}
    for method in ('mlp','tree'):
        z=predict_method(method,'validation');write_predictions(method,'validation',z)
        cut,budget=matched_false_alarm_cutoff(z,raw['y'],raw['w'],raw['modis'])
        offset=fit_intercept(z,raw['y'],raw['w'])
        rank=exact_ranking(z,raw['y'],raw['w'])
        policies[method]={'raw_logit_cutoff':cut,'logit_offset':offset,'false_positive_budget':budget,
            'validation_average_precision':rank['average_precision_exact']}
    selected=max(policies,key=lambda key:policies[key]['validation_average_precision'])
    result={'fit_split':'validation','test_used_for_fit':False,'benchmark_previously_examined':True,
        'features_sha256':sha('data/features/receipt.json'),'model_sha256':models,'policies':policies,
        'selected_model':selected,'selection_rule':'highest exact validation average precision',
        'policy':'conservative raw-logit cutoff at MODIS validation false-positive exposure'}
    write(output,result);print('Validation policies and model selection frozen before benchmark prediction.',flush=True);return result


def reference_predictions(split):
    import torch
    from firefullres import engine
    from firefullres.runtime import move
    checkpoint='models/reference/supervised_modis.pt'
    if sha(checkpoint)!=config()['source_baseline_sha256']:raise ValueError('Copied supervised baseline changed')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model,saved=engine.load_checkpoint(checkpoint,device)
    cfg=saved['config'];model.eval();parent=validate_parent();raw=load_split(split);chunks=[]
    with torch.inference_mode():
        for row in parent:
            if row['split']!=split:continue
            batch=engine.Tiles([row],saved['stats'])[0];batch=engine.collate([batch]);batch=move(batch,device,cfg,labels=False)
            with engine.amp_context(device,cfg['precision']):out=model(**batch['inputs'])['logits'][0,0].float().cpu().numpy()
            with np.load(path(row['tile']),allow_pickle=False) as z:chunks.append(out[z['weight']>0])
    result=np.concatenate(chunks)
    if len(result)!=len(raw['y']):raise ValueError('Reference and new model exposure differ')
    return result


def render(report):
    lines=['# Contextual MLP and tree: reused VIIRS-reference benchmark','',
        '**This benchmark and development cohort were examined in earlier experiments; it is not a new blind test.**','',
        f'Selected using validation average precision: **{report["operating_policy"]["selected_model"]}**.',
        'Features use MODIS radiances, annular background contrast, reflective measurements and geometry. VIIRS labels and MODIS detections are not feature inputs.','']
    rows=[]
    for split in ('validation','test'):
        lines += ['## '+split.capitalize(),'','| Method | Precision | Recall | TP | FP | FN | F1 | AP |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for method,value in report[split].items():
            lines.append(f'| {method} | {100*value["precision"]:.2f}% | {100*value["recall"]:.2f}% | {value["true_positives"]:,.0f} | '
                f'{value["false_positives"]:,.0f} | {value["false_negatives"]:,.0f} | {value["f1"]:.4f} | {value["average_precision_exact"]:.4f} |')
            rows.append({'split':split,'method':method,**value})
        lines.append('')
    lines += ['## Interpretation limits','',
        'The operating policies and model selection were fitted on validation before these benchmark predictions. No post-test threshold tuning was performed.',
        'Training has three regional cohorts; validation and test each have one. Query labels are correlated and do not represent independent wildfire incidents.',
        'VIIRS provides reference observations, not verified ground-truth wildfire. Not matching a VIIRS detection does not alone establish that a MODIS alert is false.',
        'Raw and validation-adjusted probabilities may remain miscalibrated on other regions. The target is active-fire reference detection, not forecasting or burned area.',
        'A finer query grid does not create new thermal spatial resolution or guarantee detection of fires absent from MODIS measurements.',
        'All imagery was physically copied from the previous research folder. No new satellite data were downloaded.','']
    path('results/EVALUATION.md').write_text('\n'.join(lines),encoding='utf-8')
    with path('results/metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate():
    policies=freeze_policies();output='results/evaluation.json'
    if path(output).exists():
        report=read(output)
        if report['operating_policy']!=policies:raise ValueError('Sealed evaluation policy changed')
        render(report);return report
    report={'scope':config()['scope'],'operating_policy':policies,'new_blind_test':False,'model_usefulness_established':False}
    for split in ('validation','test'):
        raw=load_split(split);y,w,b=(np.asarray(raw[k]) for k in ('y','w','modis'))
        scores={'regular_modis':metrics(b,y,w,b>=.5)}
        baseline=reference_predictions(split);scores['previous_cnn_raw_0.5']=metrics(sigmoid(baseline),y,w,baseline>=0,ranking_score=baseline)
        for method in ('mlp','tree'):
            if split=='validation':
                with np.load(path(f'results/{split}_{method}_predictions.npz')) as z:logits=z['logits']
            else:logits=predict_method(method,split);write_predictions(method,split,logits)
            policy=policies['policies'][method];prob=sigmoid(logits)
            scores[method+'_raw_0.5']=metrics(prob,y,w,logits>=0,ranking_score=logits)
            scores[method+'_frozen_policy']=metrics(sigmoid(logits+policy['logit_offset']),y,w,
                logits.astype(np.float64)>=policy['raw_logit_cutoff'],ranking_score=logits)
        report[split]=scores
    report['feature_receipt']=read('data/features/receipt.json');report['mlp_training']=read('models/mlp/completed.json')
    report['tree_training']=read('models/tree/completed.json');write(output,report);render(report)
    print(json.dumps({'selected':policies['selected_model'],'test':report['test']},indent=2),flush=True);return report
