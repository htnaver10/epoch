"""Small feature MLP, train-only scaling and standard standalone export."""
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from firefullres.calibration import exact_ranking,sigmoid
from .common import path,read,write,sha,config
from .prepare import load_split


class ContextMLP(nn.Module):
    def __init__(self,features,hidden=(64,32),dropout=.1):
        super().__init__();layers=[];before=features
        for width in hidden:
            layers += [nn.Linear(before,width),nn.GELU(),nn.Dropout(dropout)];before=width
        layers.append(nn.Linear(before,1));self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x).squeeze(-1)
    def initialize_prior(self,p):
        if not 0<p<1:raise ValueError('Require positive and negative training labels')
        with torch.no_grad():self.layers[-1].bias.fill_(math.log(p/(1-p)))


def atomic_save(filename,state):
    destination=path(filename);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name(destination.name+'.partial');torch.save(state,tmp);tmp.replace(destination)


def normalize_numpy(x,stats):
    return np.clip((x-np.array(stats['mean'],np.float32))/np.array(stats['scale'],np.float32),
        -stats['normalization_clip'],stats['normalization_clip']).astype(np.float32)


def device_data(split,stats,device):
    raw=load_split(split);result={key:torch.tensor(raw[key],device=device) for key in ('X','y','w')}
    mean=torch.tensor(stats['mean'],device=device,dtype=torch.float32);scale=torch.tensor(stats['scale'],device=device,dtype=torch.float32)
    result['X'].sub_(mean).div_(scale).clamp_(-stats['normalization_clip'],stats['normalization_clip'])
    return result,raw


def predict_tensor(model,x,batch_size=65536):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.device.type=='cuda'):
                z=model(x[start:start+batch_size])
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def load_model(checkpoint='models/mlp/inference.pt',device='cpu'):
    saved=torch.load(path(checkpoint),map_location=device,weights_only=True)
    if saved.get('kind')!='contextual_mlp_v1' or saved.get('requires_viirs_at_inference') is not False:
        raise ValueError('Unsupported feature-MLP export')
    model=ContextMLP(len(saved['stats']['feature_names']),**saved['architecture']).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval();return model,saved


def predict(split):
    model,saved=load_model(device='cuda' if torch.cuda.is_available() else 'cpu')
    device=next(model.parameters()).device;data,_=device_data(split,saved['stats'],device)
    z=predict_tensor(model,data['X']);del data
    return z


def train():
    cfg=config();options=cfg['mlp'];stats=read('data/features/statistics.json');receipt=sha('data/features/receipt.json')
    done='models/mlp/completed.json'
    if path(done).exists():
        state=read(done)
        if state['receipt_sha256']!=receipt or state['inference_sha256']!=sha('models/mlp/inference.pt'):
            raise ValueError('Completed MLP artifacts/data changed')
        return state
    if options['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA required by this experiment')
    torch.set_num_threads(4);random.seed(options['seed']);torch.manual_seed(options['seed'])
    device=torch.device(options['device']);architecture={'hidden':options['hidden'],'dropout':options['dropout']}
    model=ContextMLP(len(stats['mean']),**architecture).to(device);model.initialize_prior(stats['prevalence'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    start=0;best=-1.;stale=0;history=[]
    if path('models/mlp/last.pt').exists():
        saved=torch.load(path('models/mlp/last.pt'),map_location=device,weights_only=True)
        if saved['options']!=options or saved['stats']!=stats or saved['receipt_sha256']!=receipt:
            raise ValueError('MLP resume configuration/data mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if saved['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        start=saved['epoch']+1;best=saved['best_ap'];stale=saved['stale'];history=saved['history']
    train_data,raw_train=device_data('train',stats,device);validation,raw_validation=device_data('validation',stats,device)
    total=float(train_data['w'].sum());batch_size=options['batch_size'];n=len(train_data['y'])
    print(f'MLP: {sum(p.numel() for p in model.parameters()):,} parameters; {n:,} training queries on {device}',flush=True)
    for epoch in range(start,options['epochs']):
        started=time.perf_counter();model.train();order=torch.randperm(n,device=device);loss_total=0.
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for begin in range(0,n,batch_size):
            index=order[begin:begin+batch_size];x=train_data['X'][index];y=train_data['y'][index];weight=train_data['w'][index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=model(x)
            losses=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
            weighted=(losses*weight).sum();loss=weighted/weight.sum()
            if not torch.isfinite(loss):raise ValueError('Nonfinite MLP loss; optimizer update refused')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),options['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();loss_total+=float(weighted.detach())
        z=predict_tensor(model,validation['X'],batch_size)
        ranking=exact_ranking(z,raw_validation['y'],raw_validation['w']);ap=ranking['average_precision_exact']
        q=sigmoid(z);y=raw_validation['y'];w=raw_validation['w']
        logloss=float(np.sum(w*(np.logaddexp(0,z.astype(np.float64))-y*z))/np.sum(w))
        improved=ap>best
        if improved:best=ap;stale=0
        else:stale+=1
        row={'epoch':epoch,'training_queries':n,'training_logloss':loss_total/total,'validation_average_precision':ap,
            'validation_logloss':logloss,'validation_roc_auc':ranking['roc_auc_exact'],'seconds':time.perf_counter()-started,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None}
        history.append(row)
        state={'kind':'contextual_mlp_v1','requires_viirs_at_inference':False,'model':model.state_dict(),
            'architecture':architecture,'options':options,'stats':stats,'receipt_sha256':receipt,'epoch':epoch,
            'best_ap':best,'stale':stale,'history':history,'optimizer':optimizer.state_dict(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        if improved:atomic_save('models/mlp/best.pt',state)
        atomic_save('models/mlp/last.pt',state);write('results/mlp_history.json',history);print(json.dumps(row),flush=True)
        if stale>=options['patience']:break
    selected=torch.load(path('models/mlp/best.pt'),map_location='cpu',weights_only=True)
    export={key:selected[key] for key in ('kind','requires_viirs_at_inference','model','architecture','stats','receipt_sha256','epoch')}
    atomic_save('models/mlp/inference.pt',export);load_model()
    result={'epochs_completed':len(history),'selected_epoch':selected['epoch']+1,'best_validation_average_precision':best,
        'receipt_sha256':receipt,'inference_sha256':sha('models/mlp/inference.pt'),'standalone_export_reload_passed':True,
        'early_stopping':len(history)<options['epochs']}
    write(done,result);return result
