from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .core import (ROOT,INPUT_KEYS,path,relative,read_json,read_jsonl,write_json,write_npz,
                   sha256,validate_tiles,validate_arrays,normalize)
from .network import NativeFireNet
from .metrics import BinaryMetrics


class Tiles(Dataset):
    def __init__(self,rows,stats):self.rows=rows;self.stats=stats;self.verified=set()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        if r['tile'] not in self.verified:
            if sha256(r['tile'])!=r['sha256']:raise ValueError('Tile changed since manifest freeze')
            self.verified.add(r['tile'])
        with np.load(path(r['tile']),allow_pickle=False) as z:
            validate_arrays(z)
            inputs={k:torch.from_numpy(normalize(k,z[k],self.stats) if k!='query_grid' else z[k].copy()) for k in INPUT_KEYS}
            data={k:torch.from_numpy(z[k].copy()).float() for k in ['target','weight','native_target','native_weight']}
        return {'inputs':inputs,**data,'record':r}


def collate(batch):
    h=max(b['target'].shape[0] for b in batch);w=max(b['target'].shape[1] for b in batch)
    inputs={}
    for key in INPUT_KEYS:
        if key=='query_grid':
            out=torch.full((len(batch),h,w,2),2.,dtype=torch.float32)
            for i,b in enumerate(batch):
                y,x=b['target'].shape;out[i,:y,:x]=b['inputs'][key]
            inputs[key]=out
        else:inputs[key]=torch.stack([b['inputs'][key] for b in batch])
    result={'inputs':inputs,'record':[b['record'] for b in batch]}
    for key in ('target','weight'):
        out=torch.zeros((len(batch),1,h,w),dtype=torch.float32)
        for i,b in enumerate(batch):
            y,x=b[key].shape;out[i,0,:y,:x]=b[key]
        result[key]=out
    for key in ('native_target','native_weight'):result[key]=torch.stack([b[key] for b in batch])[:,None]
    return result


def device_for(value):
    if value=='auto':value='cuda' if torch.cuda.is_available() else 'cpu'
    if value=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; configure RTX PyTorch or explicitly use --device cpu for a smoke test')
    return torch.device(value)


def amp_context(device,precision):
    if precision=='float32':return contextlib.nullcontext()
    if device.type!='cuda':raise ValueError('Mixed precision is restricted to CUDA; use float32 for the CPU smoke test')
    if precision=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 unavailable')
    return torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bfloat16' else torch.float16)


def loader(rows,stats,config,shuffle=False,epoch=0):
    workers=int(config.get('workers',0))
    gen=torch.Generator().manual_seed(int(config['seed'])+epoch)
    kwargs={'num_workers':workers,'pin_memory':config.get('pin_memory',False),'collate_fn':collate,
            'batch_size':int(config['microbatch']),'shuffle':shuffle,'generator':gen}
    if workers:kwargs['prefetch_factor']=int(config.get('prefetch_factor',2))
    return DataLoader(Tiles(rows,stats),**kwargs)


def move(batch,device):
    return {**batch,'inputs':{k:v.to(device,non_blocking=True) for k,v in batch['inputs'].items()},
            **{k:batch[k].to(device,non_blocking=True) for k in ['target','weight','native_target','native_weight']}}


def loss_sums(output,batch):
    # Each objective has its own exposure denominator, accumulated across the effective batch.
    fine=F.binary_cross_entropy_with_logits(output['logits'].float(),batch['target'],reduction='none')
    native=F.binary_cross_entropy_with_logits(output['native_logits'].float(),batch['native_target'],reduction='none')
    return (fine*batch['weight']).sum(),(native*batch['native_weight']).sum()


def save_checkpoint(filename,model,optimizer,scaler,config,stats,manifest_digest,epoch,best):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
           'config':config,'stats':stats,'manifest_sha256':manifest_digest,'epoch':epoch,'best_validation_logloss':best,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'python_rng':random.getstate(),'torch_version':str(torch.__version__),'checkpoint_scope':'epoch_boundary'}
    p=path(filename,True);tmp=p.with_name(p.name+'.partial');torch.save(state,tmp);tmp.replace(p)


def load_checkpoint(filename,device):
    saved=torch.load(path(filename),map_location=device,weights_only=True)
    model=NativeFireNet(**saved['config']['model']).to(device)
    model.load_state_dict(saved['model'])
    return model,saved


def score(model,rows,stats,config,device):
    metric=BinaryMetrics(threshold=config.get('threshold',.5));model.eval()
    with torch.no_grad():
        for raw in loader(rows,stats,config):
            b=move(raw,device)
            with amp_context(device,config['precision']):out=model(**b['inputs'])
            p=torch.sigmoid(out['logits'].float()).cpu().numpy()
            metric.add(p,b['target'].cpu().numpy(),b['weight'].cpu().numpy())
    return metric.result()


def train(manifest,statistics_file,config_file,run_name,allow_synthetic=False,resume=False,device_override=None,epochs_override=None):
    if not run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_name):raise ValueError('Use an alphanumeric run name')
    config=read_json(config_file);stats=read_json(statistics_file)
    if device_override:config['device']=device_override
    if epochs_override is not None:config['epochs']=epochs_override
    if int(config['epochs'])<1:raise ValueError('epochs must be positive')
    rows=validate_tiles(manifest,allow_synthetic,False,False)
    if stats['manifest_sha256']!=sha256(manifest):raise ValueError('Statistics belong to another tile manifest')
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='validation']
    if not train_rows or not val_rows:raise ValueError('Both train and validation partitions are required')
    if int(config['accumulation_steps'])<1 or int(config['microbatch'])<1:raise ValueError('Invalid batch configuration')
    folder=path(f'models/{run_name}')
    if folder.exists() and not resume:raise ValueError('Run already exists; use --resume or a new run name')
    if resume and not (folder/'last.pt').exists():raise ValueError('No last.pt checkpoint to resume')
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=device_for(config['device'])
    if device.type=='cpu' and config['precision']!='float32':raise ValueError('Use the smoke_cpu.json config for CPU tests')
    model=NativeFireNet(**config['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['precision']=='float16')
    start=0;best=float('inf')
    if resume:
        model,saved=load_checkpoint(folder/'last.pt',device)
        if saved['manifest_sha256']!=sha256(manifest) or saved['stats']!=stats:raise ValueError('Resume data/statistics mismatch')
        original={k:v for k,v in saved['config'].items() if k not in ['epochs','device']}
        current={k:v for k,v in config.items() if k not in ['epochs','device']}
        if original!=current:raise ValueError('Resume configuration changed; start a new run')
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start=saved['epoch']+1;best=saved['best_validation_logloss']
    write_json(f'models/{run_name}/run.json',{'config':config,'manifest':relative(manifest),'manifest_sha256':sha256(manifest),
               'statistics':relative(statistics_file),'parameters':sum(p.numel() for p in model.parameters()),
               'device':str(device),'dataset_kinds':sorted({r['dataset_kind'] for r in rows}),
               'input_keys':list(INPUT_KEYS),'not_claimed':'ground-truth fire probability, exact subpixel location or tested RTX memory fit'})
    for epoch in range(start,int(config['epochs'])):
        model.train();iterator=iter(loader(train_rows,stats,config,True,epoch));processed=0;started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        while True:
            window=[]
            for _ in range(int(config['accumulation_steps'])):
                try:window.append(next(iterator))
                except StopIteration:break
            if not window:break
            fine_weight=sum(float(b['weight'].sum()) for b in window)
            native_weight=sum(float(b['native_weight'].sum()) for b in window)
            if fine_weight<=0:continue
            optimizer.zero_grad(set_to_none=True)
            for raw in window:
                batch=move(raw,device)
                with amp_context(device,config['precision']):out=model(**batch['inputs'])
                with torch.autocast(device_type=device.type,enabled=False):
                    sf,sn=loss_sums(out,batch)
                    loss=sf/fine_weight+config['native_loss_weight']*sn/max(native_weight,1.)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward();processed+=len(raw['record'])
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        validation=score(model,val_rows,stats,config,device)
        improved=validation['logloss']<best
        if improved:best=validation['logloss']
        save_checkpoint(folder/'last.pt',model,optimizer,scaler,config,stats,sha256(manifest),epoch,best)
        if improved:save_checkpoint(folder/'best.pt',model,optimizer,scaler,config,stats,sha256(manifest),epoch,best)
        record={'epoch':epoch,'processed_tiles':processed,'training_seconds':elapsed,'validation':validation,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None}
        if device.type=='cuda':record['reserved_fraction_of_device']=record['peak_reserved_bytes']/torch.cuda.get_device_properties(device).total_memory
        with path(f'logs/{run_name}/epochs.jsonl',True).open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    return {'run':run_name,'best_validation_logloss':best,'checkpoint':relative(folder/'best.pt')}


def predict(manifest,checkpoint,output_dir,split,device_name='auto',allow_synthetic=False,evaluate=False):
    outdir=path(output_dir)
    if outdir.exists():raise ValueError('Prediction output exists; use a new output directory')
    rows=validate_tiles(manifest,allow_synthetic,False,False);rows=[r for r in rows if r['split']==split]
    if not rows:raise ValueError(f'No {split} tiles')
    device=device_for(device_name);model,saved=load_checkpoint(checkpoint,device);model.eval()
    trained_synthetic=any(k!='nasa_native' for k in saved['stats'].get('dataset_kinds',[]))
    if trained_synthetic and not allow_synthetic:raise ValueError('Synthetic-trained checkpoints require --allow-synthetic')
    if evaluate:
        groups=set(saved['stats']['training_groups']);sources=set(saved['stats']['training_sources'])
        tiles=set(saved['stats']['training_tile_ids'])
        if any(r['event_group'] in groups or r['tile_id'] in tiles or sources.intersection(r['source_ids']) for r in rows):
            raise ValueError('Evaluation data overlap checkpoint training groups/sources/tiles')
    config={**saved['config'],'microbatch':1,'workers':0,'pin_memory':False}
    if device.type=='cpu':config['precision']='float32'
    if evaluate and split=='train':raise ValueError('Evaluation must use validation/test, not training')
    metric=BinaryMetrics(threshold=config.get('threshold',.5));baseline=BinaryMetrics(threshold=config.get('threshold',.5))
    outdir.mkdir(parents=True);ledger=[]
    with torch.no_grad():
        for raw in loader(rows,saved['stats'],config):
            row=raw['record'][0];b=move(raw,device)
            with amp_context(device,config['precision']):out=model(**b['inputs'])
            p=torch.sigmoid(out['logits'].float())[0,0].cpu().numpy()
            with np.load(path(row['tile'])) as z:
                valid=z['weight'] if evaluate else z['inference_valid']
                if evaluate:
                    metric.add(p,z['target'],valid);baseline.add(z['modis_baseline'],z['target'],valid)
                dest=outdir/(row['tile_id']+'.npz')
                write_npz(dest,probability=p,valid=valid,latitude=z['query_lat'],longitude=z['query_lon'])
                # Candidate query centers; no claim of independent incidents or calibrated ground accuracy.
                candidate=(p>=config.get('threshold',.5))&(valid>0)
                supported_scores=np.where(valid>0,p,-np.inf)
                for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    other=np.full_like(p,-np.inf)
                    ys=slice(max(0,dr),p.shape[0]+min(0,dr));xs=slice(max(0,dc),p.shape[1]+min(0,dc))
                    other[ys,xs]=supported_scores[slice(max(0,-dr),p.shape[0]+min(0,-dr)),slice(max(0,-dc),p.shape[1]+min(0,-dc))]
                    candidate &= p>other  # strict peaks suppress equal-probability plateaus
                features=[{'type':'Feature','geometry':{'type':'Point','coordinates':[float(z['query_lon'][y,x]),float(z['query_lat'][y,x])]},
                           'properties':{'probability_reference_activity':float(p[y,x]),'basis':'model_estimate','tile_id':row['tile_id'],
                                         'source_ids':row['source_ids'],'acquisition_utc':row.get('modis_start_utc'),
                                         'forest_context_only':row.get('forest_only',False),'uncertainty_calibrated':False}}
                          for y,x in np.argwhere(candidate)]
                write_json(outdir/(row['tile_id']+'.geojson'),{'type':'FeatureCollection','features':features})
            ledger.append({'tile_id':row['tile_id'],'prediction':relative(dest),'sha256':sha256(dest)})
    report={'checkpoint':relative(checkpoint),'checkpoint_sha256':sha256(checkpoint),'manifest_sha256':sha256(manifest),
            'split':split,'tiles':len(rows),'synthetic_smoke_only':trained_synthetic or any(r['dataset_kind']!='nasa_native' for r in rows),
            'predictions':ledger,'ground_location_accuracy_validated':False}
    if evaluate:report.update(model_metrics=metric.result(),modis_baseline_metrics=baseline.result())
    write_json(outdir/'report.json',report)
    return report


def profile(manifest,statistics_file,config_file,output,steps=10,allow_synthetic=False):
    config=read_json(config_file);device=device_for('cuda');stats=read_json(statistics_file)
    rows=validate_tiles(manifest,allow_synthetic,False,False);rows=[r for r in rows if r['split']=='train']
    if not rows:raise ValueError('No training tiles')
    model=NativeFireNet(**config['model']).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['precision']=='float16');times=[];count=0
    def batches():
        while True:yield from loader(rows,stats,config)
    it=iter(batches())
    for step in range(steps+3):
        raw=next(it);torch.cuda.synchronize();started=time.perf_counter();b=move(raw,device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device,config['precision']):out=model(**b['inputs'])
        sf,sn=loss_sums(out,b)
        loss=sf/b['weight'].sum().clamp_min(1)+config['native_loss_weight']*sn/b['native_weight'].sum().clamp_min(1)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        if step==2:torch.cuda.reset_peak_memory_stats()
        if step>=3:times.append(time.perf_counter()-started);count+=len(raw['record'])
    result={'gpu':torch.cuda.get_device_name(),'model_parameters':sum(p.numel() for p in model.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'tiles_per_second_excluding_loader_wait':count/sum(times),'measured_steps':steps,
            'method':'three warmups; full forward/backward/optimizer every microbatch; excludes loader wait and accumulated-window amortization',
            'synthetic':any(r['dataset_kind']!='nasa_native' for r in rows),'config':config}
    write_json(output,result);return result
