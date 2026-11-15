from __future__ import annotations

import contextlib
import json
import random
import time
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .core import (INPUT_KEYS,path,relative,read_json,write_json,write_npz,sha256,validate_tiles)
from .network import NativeFireNet
from .metrics import BinaryMetrics
from .loading import TileStore, WindowSampler
from .runtime import (resolve_config, configure_model, optimizer_for, move,
                      device_batches, region, runtime_info)


class Tiles(Dataset):
    def __init__(self,rows,stats,cache_mib=0):self.store=TileStore(rows,stats,cache_mib)
    def __len__(self):return len(self.store)
    def __getitem__(self,index):
        value=self.store[index]
        return {**value,'inputs':{k:torch.from_numpy(v) for k,v in value['inputs'].items()},
                **{k:torch.from_numpy(value[k]) for k in ['target','weight','native_target','native_weight']}}


def collate(batch,bucket_size=0):
    h=max(b['target'].shape[0] for b in batch);w=max(b['target'].shape[1] for b in batch)
    if bucket_size:
        h=((h+bucket_size-1)//bucket_size)*bucket_size
        w=((w+bucket_size-1)//bucket_size)*bucket_size
    inputs={}
    for key in INPUT_KEYS:
        if key=='query_grid':
            out=torch.full((len(batch),h,w,2),2.,dtype=torch.float32)
            for i,b in enumerate(batch):
                y,x=b['target'].shape;out[i,:y,:x]=b['inputs'][key]
            inputs[key]=out
        else:inputs[key]=torch.stack([b['inputs'][key] for b in batch])
    result={'inputs':inputs,'record':[b['record'] for b in batch],
            'query_shapes':[tuple(b['target'].shape) for b in batch]}
    for key in ('target','weight'):
        out=torch.zeros((len(batch),1,h,w),dtype=torch.float32)
        for i,b in enumerate(batch):
            y,x=b[key].shape;out[i,0,:y,:x]=b[key]
        result[key]=out
    for key in ('native_target','native_weight'):result[key]=torch.stack([b[key] for b in batch])[:,None]
    return result


class WindowCollate:
    def __init__(self,microbatch,bucket_size):self.microbatch=microbatch;self.bucket_size=bucket_size
    def __call__(self,items):
        batches=[collate(items[i:i+self.microbatch],self.bucket_size) for i in range(0,len(items),self.microbatch)]
        return {'batches':batches,'tiles':len(items),
                'fine_weight':sum(float(b['weight'].sum(dtype=torch.float64)) for b in batches),
                'native_weight':sum(float(b['native_weight'].sum(dtype=torch.float64)) for b in batches)}


def seed_worker(_):
    torch.set_num_threads(1)
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


def device_for(value):
    if value=='auto':value='cuda' if torch.cuda.is_available() else 'cpu'
    if value=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; configure RTX PyTorch or explicitly use --device cpu for a smoke test')
    return torch.device(value)


def amp_context(device,precision):
    if precision=='float32':return contextlib.nullcontext()
    if device.type!='cuda':raise ValueError('Mixed precision is restricted to CUDA; use float32 for the CPU smoke test')
    if precision=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 unavailable')
    return torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bfloat16' else torch.float16)


def loader(rows,stats,config,shuffle=False,epoch=0,accumulated=False):
    config=resolve_config(config)
    workers=int(config.get('workers',0))
    microbatch=int(config['microbatch']);bucket=int(config['query_bucket_size'])
    sampler=WindowSampler(rows,microbatch*(int(config['accumulation_steps']) if accumulated else 1),
                          bucket,shuffle,config['seed'])
    sampler.set_epoch(epoch)
    kwargs={'num_workers':workers,'pin_memory':config.get('pin_memory',False),
            'collate_fn':WindowCollate(microbatch,bucket) if accumulated else partial(collate,bucket_size=bucket),
            'batch_sampler':sampler,'generator':torch.Generator().manual_seed(int(config['seed'])),
            'worker_init_fn':seed_worker}
    if workers:
        kwargs.update(prefetch_factor=int(config.get('prefetch_factor',2)),
                      persistent_workers=bool(config['persistent_workers']),
                      multiprocessing_context=config['worker_start_method'])
    return DataLoader(Tiles(rows,stats,config['tile_cache_mib']),**kwargs)


def loss_sums(output,batch):
    # Each objective has its own exposure denominator, accumulated across the effective batch.
    fine=F.binary_cross_entropy_with_logits(output['logits'].float(),batch['target'],reduction='none')
    native=F.binary_cross_entropy_with_logits(output['native_logits'].float(),batch['native_target'],reduction='none')
    return (fine*batch['weight']).sum(),(native*batch['native_weight']).sum()


def train_window(model,window,optimizer,scaler,config,device,trace=False):
    """Shared train/profile path; one exposure-normalized optimizer update."""
    fine_weight=window['fine_weight'];native_weight=window['native_weight']
    if fine_weight<=0:return 0
    optimizer.zero_grad(set_to_none=True)
    finite=torch.ones((),dtype=torch.bool,device=device)
    for batch in device_batches(window['batches'],device,config):
        with region('forward_and_loss',trace):
            with amp_context(device,config['precision']):out=model(**batch['inputs'])
            with torch.autocast(device_type=device.type,enabled=False):
                sf,sn=loss_sums(out,batch)
                loss=sf/fine_weight+config['native_loss_weight']*sn/max(native_weight,1.)
            finite=finite & torch.isfinite(loss.detach())
        with region('backward',trace):scaler.scale(loss).backward()
    with region('optimizer',trace):
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=False)
        # One host decision per update, retaining both loss and gradient guards.
        if not bool(finite & torch.isfinite(norm)):
            raise ValueError('Nonfinite training loss or gradients; optimizer update refused')
        scaler.step(optimizer);scaler.update()
    return window['tiles']


def memory_usage(device):
    if device.type!='cuda':return {'peak_allocated_bytes':None,'peak_reserved_bytes':None}
    allocated=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
    return {'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
            'reserved_fraction_of_device':reserved/torch.cuda.get_device_properties(device).total_memory}


def save_checkpoint(filename,model,optimizer,scaler,config,stats,manifest_digest,epoch,best):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
           'config':config,'stats':stats,'manifest_sha256':manifest_digest,'epoch':epoch,'best_validation_logloss':best,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'python_rng':random.getstate(),'torch_version':str(torch.__version__),'checkpoint_scope':'epoch_boundary',
           'training_format':2}
    p=path(filename,True);tmp=p.with_name(p.name+'.partial');torch.save(state,tmp);tmp.replace(p)


def load_checkpoint(filename,device):
    saved=torch.load(path(filename),map_location=device,weights_only=True)
    model=NativeFireNet(**saved['config']['model']).to(device)
    model.load_state_dict(saved['model'])
    return model,saved


def score(model,rows,stats,config,device,data_loader=None):
    metric=BinaryMetrics(threshold=config.get('threshold',.5));model.eval()
    with torch.inference_mode():
        for raw in data_loader if data_loader is not None else loader(rows,stats,config):
            b=move(raw,device,config,labels=False)
            with amp_context(device,config['precision']):out=model(**b['inputs'])
            p=torch.sigmoid(out['logits'].float()).cpu().numpy()
            metric.add(p,raw['target'].numpy(),raw['weight'].numpy())
    return metric.result()


def train(manifest,statistics_file,config_file,run_name,allow_synthetic=False,resume=False,device_override=None,epochs_override=None):
    if not run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_name):raise ValueError('Use an alphanumeric run name')
    config=resolve_config(read_json(config_file));stats=read_json(statistics_file)
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
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=device_for(config['device'])
    if device.type=='cpu' and config['precision']!='float32':raise ValueError('Use the smoke_cpu.json config for CPU tests')
    model=NativeFireNet(**config['model']).to(device)
    if config['initialize_output_prior'] and not resume:
        priors=stats.get('training_target_prevalence')
        if not priors:raise ValueError('Prior initialization requires training-only target prevalences in statistics')
        model.initialize_output_priors(priors['fine'],priors['native'])
    executor=configure_model(model,config,device)
    optimizer=optimizer_for(model,config,device)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['precision']=='float16')
    start=0;best=float('inf')
    if resume:
        model,saved=load_checkpoint(folder/'last.pt',device)
        if saved.get('training_format')!=2:raise ValueError('Legacy training order cannot be resumed exactly; start a new run')
        if saved['manifest_sha256']!=sha256(manifest) or saved['stats']!=stats:raise ValueError('Resume data/statistics mismatch')
        original={k:v for k,v in resolve_config(saved['config']).items() if k not in ['epochs','device']}
        current={k:v for k,v in config.items() if k not in ['epochs','device']}
        if original!=current:raise ValueError('Resume configuration changed; start a new run')
        executor=configure_model(model,config,device)
        optimizer=optimizer_for(model,config,device)
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start=saved['epoch']+1;best=saved['best_validation_logloss']
    train_loader=loader(train_rows,stats,config,True,accumulated=True)
    val_loader=loader(val_rows,stats,config)
    folder.mkdir(parents=True,exist_ok=True)
    write_json(f'models/{run_name}/run.json',{'config':config,'manifest':relative(manifest),'manifest_sha256':sha256(manifest),
               'statistics':relative(statistics_file),'parameters':sum(p.numel() for p in model.parameters()),
               'device':str(device),'runtime':runtime_info(config,device),'training_format':2,
               'dataset_kinds':sorted({r['dataset_kind'] for r in rows}),
               'input_keys':list(INPUT_KEYS),'not_claimed':'ground-truth fire probability, exact subpixel location or tested RTX memory fit'})
    for epoch in range(start,int(config['epochs'])):
        epoch_started=time.perf_counter();executor.train();train_loader.batch_sampler.set_epoch(epoch)
        started=time.perf_counter();iterator=iter(train_loader);processed=0;updates=0;loader_wait=0.
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        while True:
            waiting=time.perf_counter()
            try:window=next(iterator)
            except StopIteration:break
            loader_wait+=time.perf_counter()-waiting
            count=train_window(executor,window,optimizer,scaler,config,device)
            processed+=count;updates+=bool(count)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        memory=memory_usage(device);validation_started=time.perf_counter()
        validation=score(executor,val_rows,stats,config,device,val_loader)
        validation_seconds=time.perf_counter()-validation_started
        improved=validation['logloss']<best
        if improved:best=validation['logloss']
        saving=time.perf_counter()
        save_checkpoint(folder/'last.pt',model,optimizer,scaler,config,stats,sha256(manifest),epoch,best)
        if improved:save_checkpoint(folder/'best.pt',model,optimizer,scaler,config,stats,sha256(manifest),epoch,best)
        record={'epoch':epoch,'processed_tiles':processed,'optimizer_updates':updates,'training_seconds':elapsed,
                'tiles_per_second_including_loader':processed/elapsed,'loader_wait_seconds':loader_wait,
                'validation_seconds':validation_seconds,'checkpoint_seconds':time.perf_counter()-saving,
                'epoch_seconds':time.perf_counter()-epoch_started,'validation':validation,**memory}
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
    config=resolve_config(saved['config'])
    if device.type=='cpu':config['precision']='float32'
    executor=configure_model(model,config,device,compile_model=False);executor.eval()
    if evaluate and split=='train':raise ValueError('Evaluation must use validation/test, not training')
    metric=BinaryMetrics(threshold=config.get('threshold',.5));baseline=BinaryMetrics(threshold=config.get('threshold',.5))
    outdir.mkdir(parents=True);ledger=[]
    with torch.inference_mode():
        for raw in loader(rows,saved['stats'],config):
            b=move(raw,device,config,labels=False)
            with amp_context(device,config['precision']):out=executor(**b['inputs'])
            probabilities=torch.sigmoid(out['logits'].float())[:,0].cpu().numpy()
            for i,row in enumerate(raw['record']):
                h,w=raw['query_shapes'][i];p=probabilities[i,:h,:w]
                with np.load(path(row['tile']),allow_pickle=False) as source:
                    z={key:source[key] for key in ('weight','inference_valid','target','modis_baseline','query_lat','query_lon')}
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


def profile(manifest,statistics_file,config_file,output,steps=10,allow_synthetic=False,
            warmup_steps=3,trace=None,worst_case=False,device_override=None):
    if steps<1 or warmup_steps<1:raise ValueError('Profile steps and warmup_steps must be positive')
    if trace and (path(trace)==path(output) or path(trace).exists()):raise ValueError('Use a fresh separate trace path')
    setup=time.perf_counter()
    config=resolve_config(read_json(config_file));stats=read_json(statistics_file)
    if stats['manifest_sha256']!=sha256(manifest):raise ValueError('Statistics belong to another tile manifest')
    device=device_for(device_override or config['device'])
    if device.type=='cpu' and config['precision']!='float32':raise ValueError('CPU profile requires float32 config')
    rows=validate_tiles(manifest,allow_synthetic,False,False);rows=[r for r in rows if r['split']=='train']
    if not rows:raise ValueError('No training tiles')
    if worst_case:
        from .loading import tile_shapes
        rows.sort(key=lambda r:np.prod(tile_shapes(r)[1]),reverse=True)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=NativeFireNet(**config['model']).to(device);executor=configure_model(model,config,device)
    if config['initialize_output_prior']:
        priors=stats.get('training_target_prevalence')
        if not priors:raise ValueError('Prior initialization requires training-only target prevalences in statistics')
        model.initialize_output_priors(priors['fine'],priors['native'])
    optimizer=optimizer_for(model,config,device)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['precision']=='float16')
    data_loader=loader(rows,stats,config,shuffle=not worst_case,accumulated=True)
    def windows():
        epoch=0
        while True:
            data_loader.batch_sampler.set_epoch(epoch)
            yield from data_loader
            epoch+=1
    it=iter(windows());setup_seconds=time.perf_counter()-setup
    warmup=time.perf_counter()
    for _ in range(warmup_steps):train_window(executor,next(it),optimizer,scaler,config,device)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    warmup_seconds=time.perf_counter()-warmup
    # Revisit the leading (largest, when requested) windows in measurement too.
    it=iter(windows())
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    context=torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) if trace else contextlib.nullcontext()
    count=updates=real_slots=padded_slots=0;loader_wait=0.;device_events=[];microbatches=0
    observed_queries=[];observed_native=[];window_tiles=[]
    with context as captured:
        started=time.perf_counter()
        for _ in range(steps):
            waiting=time.perf_counter()
            with region('loader_wait',bool(trace)):window=next(it)
            loader_wait+=time.perf_counter()-waiting
            if device.type=='cuda':
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            processed=train_window(executor,window,optimizer,scaler,config,device,bool(trace))
            if device.type=='cuda':end.record();device_events.append((begin,end))
            count+=processed;updates+=bool(processed);window_tiles.append(processed)
            for b in window['batches']:
                microbatches+=1;observed_queries.extend(b['query_shapes'])
                observed_native.append(tuple(b['inputs']['thermal'].shape[-2:]))
                real_slots+=sum(h*w for h,w in b['query_shapes']);padded_slots+=b['target'].numel()
            if captured is not None:captured.step()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;memory=memory_usage(device)
    if trace:captured.export_chrome_trace(str(path(trace,True)))
    if not count:raise ValueError('Profile has no valid training exposure')
    result={'runtime':runtime_info(config,device),'model_parameters':sum(p.numel() for p in model.parameters()),**memory,
            'tiles_per_second_including_loader':count/elapsed,'training_seconds':elapsed,
            'loader_wait_seconds':loader_wait,'loader_wait_fraction':loader_wait/elapsed,
            'measured_updates':updates,'measured_microbatches':microbatches,'processed_tiles':count,
            'tiles_per_update_min':min(window_tiles),'tiles_per_update_max':max(window_tiles),
            'query_padding_fraction':1-real_slots/padded_slots,'setup_seconds':setup_seconds,
            'warmup_steps':warmup_steps,'warmup_seconds':warmup_seconds,
            'max_observed_query_shape':[max(s[i] for s in observed_queries) for i in (0,1)],
            'max_observed_native_shape':[max(s[i] for s in observed_native) for i in (0,1)],
            'sample_order':'largest_query_bucket_first' if worst_case else 'train_seed_epoch_shuffle',
            'device_window_timeline_ms':[a.elapsed_time(b) for a,b in device_events],
            'trace':relative(trace) if trace else None,
            'method':'same accumulated update as train; includes loader wait, copies, guards and optimizer; excludes validation/checkpoint; trace adds overhead',
            'not_measured':'SM occupancy, Tensor Core utilization, all-dataset worst-case memory fit, or fire accuracy',
            'manifest_sha256':sha256(manifest),'synthetic':any(r['dataset_kind']!='nasa_native' for r in rows),'config':config}
    write_json(output,result);return result
