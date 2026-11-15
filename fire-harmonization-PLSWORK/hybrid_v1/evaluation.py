"""Freeze both scales' operating points on validation, then report the reused benchmark."""
import json
import numpy as np
from common import ROOT,path,config,read,write,sha,load,save_array
from preprocessing import aggregate_cells
from firepls.prepare import load_split
from firefullres.calibration import sigmoid,fit_intercept,matched_false_alarm_cutoff,metrics

def digests():
    return {name:sha(path(name)) for name in ('models/base_mlp.pt','models/residual.txt','data/receipt.json','results/software_lock.json')}

def operating_point(z,y,w,b):
    cut,budget = matched_false_alarm_cutoff(z,y,w,b)
    return {'cutoff':cut,'logit_offset':fit_intercept(z,y,w),'validation_false_positive_budget':budget}

def calibrate():
    if path('results/operating_policy.json').exists(): return verify_policy()
    raw = load_split('validation'); z = load('results/validation_logits.npy')
    query = operating_point(z,raw['y'],raw['w'],raw['modis'])
    cz,cy,cw,cb = aggregate_cells(z,raw['y'],raw['modis'],load('data/validation_cell.npy'))
    cell = operating_point(cz,cy,cw,cb)
    # Baseline MLP cell operating point is fitted on the same validation support for fair scale comparison.
    bz,by,bw,bb = aggregate_cells(load('data/validation_prior.npy'),raw['y'],raw['modis'],load('data/validation_cell.npy'))
    mlp_cell = operating_point(bz,by,bw,bb)
    result = {'query':query,'native_cell':cell,'base_mlp_cell':mlp_cell,'artifact_sha256':digests(),
              'fit_split':'validation','test_used_for_fit':False,'new_blind_test':False,
              'selection':config()['policy'],'native_cell_selection':config()['native_cell_policy']}
    write('results/operating_policy.json',result)
    print('Query and native-cell policies frozen on validation before test prediction.',flush=True)
    return result

def verify_policy():
    p = read('results/operating_policy.json')
    if p['artifact_sha256'] != digests(): raise ValueError('Frozen policy artifacts changed')
    if p['fit_split'] != 'validation' or p['test_used_for_fit']: raise ValueError('Invalid policy provenance')
    return p

def scored(z,y,w,p):
    return metrics(sigmoid(z+p['logit_offset']),y,w,z.astype(np.float64)>=p['cutoff'],ranking_score=z)

def evaluate():
    from model import predict
    p = verify_policy()
    if path('results/evaluation.json').exists():
        saved = read('results/evaluation.json')
        if saved['operating_policy'] != p: raise ValueError('Sealed evaluation policy mismatch')
        return saved
    old = json.loads((ROOT/'results/evaluation.json').read_text())
    report = {'scope':config()['scope'],'operating_policy':p,'new_blind_test':False,
              'training':read('models/completed.json'),'preprocessing':read('data/receipt.json')}
    for split in ('validation','test'):
        raw = load_split(split); y,w,b = (raw[k] for k in ('y','w','modis'))
        z = predict(split); save_array(f'results/{split}_logits.npy',z)
        query = {'VIIRS_reference_self_comparison':metrics(y,y,w,y>.5),
                 'standard_MODIS':metrics(b,y,w,b>.5),
                 'previous_MLP_frozen_policy':old[split]['mlp_frozen_policy'],
                 'hybrid_raw_0.5':metrics(sigmoid(z),y,w,z>=0,ranking_score=z),
                 'hybrid_frozen_policy':scored(z,y,w,p['query'])}
        cz,cy,cw,cb = aggregate_cells(z,y,b,load(f'data/{split}_cell.npy'))
        bz,_,_,_ = aggregate_cells(load(f'data/{split}_prior.npy'),y,b,load(f'data/{split}_cell.npy'))
        cell = {'VIIRS_reference_self_comparison':metrics(cy,cy,cw,cy>.5),
                'standard_MODIS':metrics(cb,cy,cw,cb>.5),
                'base_MLP_cell_policy':scored(bz,cy,cw,p['base_mlp_cell']),
                'hybrid_cell_policy':scored(cz,cy,cw,p['native_cell'])}
        # Compare actual changes relative to MODIS, preserving same evaluation exposure.
        pred=z>=p['query']['cutoff']; base=b>.5; pos=y>.5
        changes={name:int(mask.sum()) for name,mask in {
            'additional_reference_positives':pred&~base&pos,
            'lost_reference_positives':~pred&base&pos,
            'removed_reference_false_positives':~pred&base&~pos,
            'added_reference_false_positives':pred&~base&~pos}.items()}
        report[split] = {'query':query,'native_cell':cell,'changes_vs_modis':changes}
    write('results/evaluation.json',report); return report
