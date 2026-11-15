"""Train/export the unchanged MODIS network with immutable teacher soft targets."""
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import core,engine
from ..network import NativeFireNet
from ..loading import WindowSampler
from ..runtime import resolve_config,configure_model,optimizer_for,device_batches,runtime_info
from .models import bernoulli_kl_sum
from .teacher import validate_soft_targets
from .history import reconcile_history


class DistilledTiles(engine.Tiles):
    def __init__(self,rows,stats,soft,cache_mib=0):
        super().__init__(rows,stats,cache_mib)
        self.soft={r['tile_id']:r for r in soft['rows']};self.verified_soft={}
    def __getitem__(self,index):
        value=super().__getitem__(index);row=value['record']
        if row['split']!='train':raise ValueError('Distillation loader must never read held-out tiles')
        soft=self.soft[row['tile_id']];path=core.path(soft['path']);stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
        if self.verified_soft.get(soft['path'])!=signature:
            if core.sha256(path)!=soft['sha256']:raise ValueError('Frozen teacher logits checksum changed')
            self.verified_soft[soft['path']]=signature
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'teacher_logits','distill_weight'}:raise ValueError('Invalid teacher cache arrays')
            logits=z['teacher_logits'];weight=z['distill_weight']
        if logits.shape!=tuple(value['target'].shape) or weight.shape!=logits.shape:
            raise ValueError('Teacher logits do not align with the student query grid')
        if logits.dtype!=np.float32 or weight.dtype!=np.float32 or not np.isfinite(logits).all() or not np.isfinite(weight).all():
            raise ValueError('Teacher logits/weights must be finite float32')
        if (weight<0).any() or np.any(weight>value['weight'].numpy()):
            raise ValueError('Distillation extends outside known training query support')
        if not np.isclose(weight.sum(dtype=np.float64),soft['known_weight']):raise ValueError('Teacher cache exposure changed')
        return {**value,'teacher_logits':torch.from_numpy(logits),'distill_weight':torch.from_numpy(weight)}


class DistillCollate(engine.WindowCollate):
    def __call__(self,items):
        result=super().__call__(items)
        for start,batch in zip(range(0,len(items),self.microbatch),result['batches']):
            group=items[start:start+self.microbatch];h,w=batch['target'].shape[-2:]
            for key in ('teacher_logits','distill_weight'):
                out=torch.zeros((len(group),1,h,w),dtype=torch.float32)
                for i,item in enumerate(group):
                    a,b=item[key].shape;out[i,0,:a,:b]=item[key]
                batch[key]=out
        result['distill_weight']=sum(float(batch['distill_weight'].sum(dtype=torch.float64)) for batch in result['batches'])
        return result


def distilled_loader(rows,stats,soft,config):
    sampler=WindowSampler(rows,config['microbatch']*config['accumulation_steps'],config['query_bucket_size'],True,config['seed'])
    options={'batch_sampler':sampler,'num_workers':config['workers'],'pin_memory':config['pin_memory'],
        'collate_fn':DistillCollate(config['microbatch'],config['query_bucket_size']),
        'worker_init_fn':engine.seed_worker,'generator':torch.Generator().manual_seed(config['seed'])}
    if config['workers']:
        options.update(prefetch_factor=config['prefetch_factor'],persistent_workers=config['persistent_workers'],
            multiprocessing_context=config['worker_start_method'])
    return DataLoader(DistilledTiles(rows,stats,soft,config['tile_cache_mib']),**options)


def student_window(model,window,optimizer,scaler,config,device,temperature,coefficient):
    if window['fine_weight']<=0:return {'tiles':0,'hard_sum':0.,'soft_sum':0.,'native_sum':0.}
    if not np.isfinite(coefficient) or coefficient<0:raise ValueError('Distillation coefficient must be nonnegative')
    optimizer.zero_grad(set_to_none=True);finite=torch.ones((),device=device,dtype=torch.bool)
    totals=torch.zeros(3,device=device)
    for batch in device_batches(window['batches'],device,config):
        with engine.amp_context(device,config['precision']):out=model(**batch['inputs'])
        hard,native=engine.loss_sums(out,batch)
        teacher_logits=batch['teacher_logits'].to(device,non_blocking=True)
        distill_weight=batch['distill_weight'].to(device,non_blocking=True)
        soft=bernoulli_kl_sum(out['logits'],teacher_logits,distill_weight,temperature)
        # Keep full hard-label supervision. Each term uses its own actual exposure.
        loss=hard/window['fine_weight']+config['native_loss_weight']*native/max(window['native_weight'],1.)
        if window['distill_weight']>0:loss=loss+coefficient*soft/window['distill_weight']
        finite &= torch.isfinite(loss.detach());scaler.scale(loss).backward()
        totals+=torch.stack([hard.detach(),soft.detach(),native.detach()])
    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=False)
    if not bool(finite & torch.isfinite(norm)):raise ValueError('Nonfinite distilled student loss/gradient; update refused')
    scaler.step(optimizer);scaler.update();values=totals.cpu().tolist()
    return dict(tiles=window['tiles'],hard_sum=values[0],soft_sum=values[1],native_sum=values[2])


def save_student(file,model,optimizer,scaler,config,stats,parent_digest,epoch,best,provenance,record=None):
    state={'kind':'modis_distilled_student_v1','training_format':3,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'config':config,'stats':stats,
        'manifest_sha256':parent_digest,'epoch':epoch,'best_validation_logloss':best,
        'distillation':provenance,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'inference_input_keys':list(core.INPUT_KEYS),'requires_viirs_at_inference':False,'epoch_record':record}
    destination=core.path(file,True);temporary=destination.with_name(destination.name+'.partial')
    torch.save(state,temporary);temporary.replace(destination)


def train_student(experiment,cache_path,run_name,resume=False,allow_synthetic=False):
    config=resolve_config(core.read_json(experiment['student_config']))
    manifest=experiment['parent_manifest'];stats=core.read_json(experiment['student_statistics']);soft=core.read_json(cache_path)
    rows=core.validate_tiles(manifest,allow_synthetic,False,False);digest=core.sha256(manifest)
    if digest!=experiment['parent_manifest_sha256'] or stats['manifest_sha256']!=digest or soft['parent_manifest_sha256']!=digest:
        raise ValueError('Distilled student manifest/statistics/cache mismatch')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    validate_soft_targets(soft,training)
    if not validation:raise ValueError('Student checkpoint selection requires held-out development validation')
    provenance={'teacher_checkpoint':soft['teacher_checkpoint'],'teacher_checkpoint_sha256':soft['teacher_checkpoint_sha256'],
        'soft_targets':cache_path,'soft_targets_sha256':core.sha256(cache_path),
        'paired_manifest_sha256':soft['paired_manifest_sha256'],'temperature':experiment['temperature'],
        'coefficient':experiment['distillation_weight'],'teacher_frozen':True,'soft_targets_partition':'train',
        'formula':'hard_BCE/known + native_weight*native_BCE/native_known + lambda*T^2*Bernoulli_KL/teacher_known'}
    if core.sha256(soft['teacher_checkpoint'])!=soft['teacher_checkpoint_sha256']:raise ValueError('Frozen teacher checkpoint changed')
    folder=core.path(f'models/{run_name}');last=folder/'last.pt'
    if folder.exists() and not resume:raise ValueError('Student run exists; resume or choose a new version')
    if resume and not last.exists():raise ValueError('No distilled student checkpoint to resume')
    random.seed(config['seed']);torch.manual_seed(config['seed']);device=engine.device_for(config['device'])
    model=NativeFireNet(**config['model']).to(device)
    if config['initialize_output_prior']:
        prior=stats['training_target_prevalence'];model.initialize_output_priors(prior['fine'],prior['native'])
    optimizer=optimizer_for(model,config,device)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['precision']=='float16')
    start=0;best=float('inf')
    if resume:
        model,saved=engine.load_checkpoint(last,device)
        if saved.get('kind')!='modis_distilled_student_v1' or saved['config']!=config or saved['stats']!=stats or saved['distillation']!=provenance or saved['manifest_sha256']!=digest:
            raise ValueError('Distilled student resume data/configuration changed')
        optimizer=optimizer_for(model,config,device);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start=saved['epoch']+1;best=saved['best_validation_logloss']
        reconcile_history(run_name,saved)
    executor=configure_model(model,config,device)
    train_loader=distilled_loader(training,stats,soft,config);val_loader=engine.loader(validation,stats,config)
    folder.mkdir(parents=True,exist_ok=True)
    core.write_json(f'models/{run_name}/run.json',{'config':config,'manifest':manifest,'manifest_sha256':digest,
        'statistics':experiment['student_statistics'],'parameters':sum(p.numel() for p in model.parameters()),
        'runtime':runtime_info(config,device),'distillation':provenance,'input_keys':list(core.INPUT_KEYS),
        'requires_viirs_at_inference':False,'initialized_from_baseline_checkpoint':False})
    for epoch in range(start,config['epochs']):
        begin=time.perf_counter();executor.train();train_loader.batch_sampler.set_epoch(epoch);processed=updates=0
        sums=np.zeros(3);weights=np.zeros(3)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for window in train_loader:
            result=student_window(executor,window,optimizer,scaler,config,device,experiment['temperature'],experiment['distillation_weight'])
            processed+=result['tiles'];updates+=bool(result['tiles'])
            sums+=np.array([result['hard_sum'],result['soft_sum'],result['native_sum']])
            weights+=np.array([window['fine_weight'],window['distill_weight'],window['native_weight']])
        validation_score=engine.score(executor,validation,stats,config,device,val_loader)
        improved=validation_score['logloss']<best
        if improved:best=validation_score['logloss']
        record={'epoch':epoch,'processed_tiles':processed,'optimizer_updates':updates,
            'training_hard_logloss':float(sums[0]/max(weights[0],1)),
            'training_temperature_scaled_kl':float(sums[1]/max(weights[1],1)),
            'validation':validation_score,'epoch_seconds':time.perf_counter()-begin,**engine.memory_usage(device)}
        if improved:save_student(folder/'best.pt',model,optimizer,scaler,config,stats,digest,epoch,best,provenance,record)
        save_student(last,model,optimizer,scaler,config,stats,digest,epoch,best,provenance,record)
        with core.path(f'logs/{run_name}/epochs.jsonl',True).open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    selected=folder/'best.pt';saved=torch.load(selected,map_location='cpu',weights_only=True)
    export={k:saved[k] for k in ('kind','model','config','stats','manifest_sha256','epoch','best_validation_logloss','inference_input_keys','requires_viirs_at_inference')}
    export['distillation_teacher_sha256']=provenance['teacher_checkpoint_sha256']
    exported=folder/'student_inference.pt';temporary=exported.with_name(exported.name+'.partial');torch.save(export,temporary);temporary.replace(exported)
    # Existing MODIS-only loading works without constructing or opening a teacher.
    reloaded,_=engine.load_checkpoint(exported,torch.device('cpu'))
    if set(reloaded.state_dict())!=set(model.state_dict()):raise ValueError('Student export architecture changed')
    return core.relative(selected)
