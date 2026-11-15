"""Exposure-normalized teacher fitting, exact resume and train-only soft targets."""
from collections import OrderedDict
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from .. import core,engine
from ..loading import WindowSampler
from ..metrics import BinaryMetrics
from ..runtime import resolve_config,configure_model,optimizer_for,runtime_info
from .data import validate_pairs,normalize_teacher,validate_teacher_arrays
from .models import VIIRSTeacher,frozen_teacher
from .history import reconcile_history


class TeacherTiles(Dataset):
    def __init__(self,rows,stats,cache_mib=0):
        self.rows=rows;self.stats=stats;self.limit=int(cache_mib*1024**2)
        self.verified={};self.cache=OrderedDict();self.cache_bytes=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        for file,checksum in ((row['tile'],row['sha256']),(row['teacher_tile'],row['teacher_sha256'])):
            stat=core.path(file).stat();signature=(stat.st_size,stat.st_mtime_ns)
            if self.verified.get(file)!=signature:
                if core.sha256(file)!=checksum:raise ValueError('Teacher input or parent label checksum changed')
                self.verified[file]=signature
        if index in self.cache:
            value,size=self.cache.pop(index);self.cache[index]=(value,size);return value
        with np.load(core.path(row['teacher_tile']),allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files};validate_teacher_arrays(arrays,tuple(row['query_shape']))
            image=normalize_teacher(arrays['radiance'],arrays['valid'],self.stats)
            support=arrays['valid'].any(0)
        with np.load(core.path(row['tile']),allow_pickle=False) as z:
            target=z['target'].astype(np.float32);weight=(z['weight']*support).astype(np.float32)
        value={'image':torch.from_numpy(image)[None], 'target':torch.from_numpy(target)[None,None],
            'weight':torch.from_numpy(weight)[None,None],'record':row}
        size=image.nbytes+target.nbytes+weight.nbytes
        if self.limit and size<=self.limit:
            while self.cache_bytes+size>self.limit:
                _,(_,old_size)=self.cache.popitem(last=False);self.cache_bytes-=old_size
            self.cache[index]=(value,size);self.cache_bytes+=size
        return value


def list_collate(items):return items


def teacher_loader(rows,stats,config,training=False):
    size=int(config['microbatch'])*int(config['accumulation_steps']) if training else 1
    sampler=WindowSampler(rows,size,config['query_bucket_size'],training,config['seed'])
    options={'batch_sampler':sampler,'num_workers':config['workers'],'pin_memory':config['pin_memory'],
        'collate_fn':list_collate,'worker_init_fn':engine.seed_worker,
        'generator':torch.Generator().manual_seed(config['seed'])}
    if config['workers']:
        options.update(prefetch_factor=config['prefetch_factor'],persistent_workers=config['persistent_workers'],
            multiprocessing_context=config['worker_start_method'])
    return DataLoader(TeacherTiles(rows,stats,config['tile_cache_mib']),**options)


def move_image(value,device,config):
    return value.to(device,non_blocking=True,memory_format=torch.channels_last if config['memory_format']=='channels_last' else torch.contiguous_format)


def score_teacher(model,loader,config,device):
    model.eval();metric=BinaryMetrics(threshold=config['threshold'])
    with torch.inference_mode():
        for items in loader:
            for item in items:
                with engine.amp_context(device,config['precision']):logits=model(move_image(item['image'],device,config))
                metric.add(torch.sigmoid(logits.float()).cpu().numpy(),item['target'].numpy(),item['weight'].numpy())
    return metric.result()


def save_teacher(path,model,optimizer,scaler,config,stats,digest,epoch,best,record=None):
    saved={'kind':'viirs_thermal_teacher_v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scaler':scaler.state_dict(),'config':config,'stats':stats,'paired_manifest_sha256':digest,
        'epoch':epoch,'best_validation_logloss':best,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate(),
        'input_channels':['I04_radiance','I05_radiance','I04_valid','I05_valid'],
        'target_not_in_inputs':True,'epoch_record':record}
    destination=core.path(path,True);temporary=destination.with_name(destination.name+'.partial')
    torch.save(saved,temporary);temporary.replace(destination)


def load_teacher(checkpoint,device):
    saved=torch.load(core.path(checkpoint),map_location=device,weights_only=True)
    if saved.get('kind')!='viirs_thermal_teacher_v1':raise ValueError('Not a VIIRS thermal teacher checkpoint')
    model=VIIRSTeacher(**saved['config']['model']).to(device);model.load_state_dict(saved['model'],strict=True)
    return model,saved


def qualify_teacher(model,saved,rows,device,experiment_config):
    validation=[row for row in rows if row['split']=='validation']
    if not validation:raise ValueError('Teacher quality check requires development validation')
    score=score_teacher(model,teacher_loader(validation,saved['stats'],saved['config']),saved['config'],device)
    prevalence=score['positive_query_weight']/max(score['valid_query_weight'],1)
    ap=score.get('average_precision')
    passed=(ap is not None and np.isfinite(ap) and ap>max(.01,5*prevalence) and score['brier']<prevalence)
    report={'passed':bool(passed),'validation':score,'always_clear_brier':prevalence,
        'minimum_average_precision_exclusive':max(.01,5*prevalence),'fit_or_check_partition':'validation',
        'test_used':False,'note':'Quality gate checks label prediction, not verified wildfire truth.'}
    if not passed:raise ValueError('Selected VIIRS teacher did not beat the validation quality guard; student distillation refused')
    return report


def train_teacher(manifest,stats_path,config_path,run_name,resume=False,allow_synthetic=False):
    config=resolve_config(core.read_json(config_path));stats=core.read_json(stats_path)
    rows=core.read_jsonl(manifest)
    if not allow_synthetic and any(r['dataset_kind']!='nasa_native' for r in rows):raise ValueError('Synthetic teacher data require explicit test authorization')
    digest=core.sha256(manifest)
    if stats['paired_manifest_sha256']!=digest or stats['fit_partition']!='train':raise ValueError('Teacher statistics mismatch')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    if not training or not validation:raise ValueError('Teacher needs train and validation partitions')
    if config['microbatch']!=1:raise ValueError('Teacher microbatch is one unpadded image; use accumulation for larger batches')
    folder=core.path(f'models/{run_name}');last=folder/'last.pt'
    if folder.exists() and not resume:raise ValueError('Teacher run exists; resume or choose a new version')
    if resume and not last.exists():raise ValueError('No teacher checkpoint to resume')
    random.seed(config['seed']);torch.manual_seed(config['seed']);device=engine.device_for(config['device'])
    model=VIIRSTeacher(**config['model']).to(device);model.initialize_prior(stats['positive_prevalence'])
    optimizer=optimizer_for(model,config,device)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['precision']=='float16')
    start=0;best=float('inf')
    if resume:
        model,saved=load_teacher(last,device)
        if saved['config']!=config or saved['stats']!=stats or saved['paired_manifest_sha256']!=digest:
            raise ValueError('Teacher resume configuration/data changed')
        optimizer=optimizer_for(model,config,device);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['torch_rng'].cpu());random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        start=saved['epoch']+1;best=saved['best_validation_logloss']
        reconcile_history(run_name,saved)
    model=configure_model(model,config,device,compile_model=False)
    train_loader=teacher_loader(training,stats,config,True);val_loader=teacher_loader(validation,stats,config)
    folder.mkdir(parents=True,exist_ok=True)
    core.write_json(f'models/{run_name}/run.json',{'kind':'viirs_thermal_teacher_v1','config':config,
        'paired_manifest_sha256':digest,'parameters':sum(p.numel() for p in model.parameters()),
        'runtime':runtime_info(config,device),'no_fire_mask_input':True})
    for epoch in range(start,config['epochs']):
        begin=time.perf_counter();model.train();train_loader.batch_sampler.set_epoch(epoch);processed=updates=0
        total_loss=total_weight=0.
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for items in train_loader:
            denominator=sum(float(item['weight'].sum(dtype=torch.float64)) for item in items)
            if denominator<=0:continue
            optimizer.zero_grad(set_to_none=True);finite=torch.ones((),device=device,dtype=torch.bool)
            for item in items:
                image=move_image(item['image'],device,config);target=item['target'].to(device,non_blocking=True);weight=item['weight'].to(device,non_blocking=True)
                with engine.amp_context(device,config['precision']):logits=model(image)
                loss=(F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')*weight).sum()/denominator
                finite &= torch.isfinite(loss.detach());scaler.scale(loss).backward()
                total_loss+=float(loss.detach())*denominator;total_weight+=float(weight.sum())
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=False)
            if not bool(finite & torch.isfinite(norm)):raise ValueError('Nonfinite teacher loss/gradient; optimizer update refused')
            scaler.step(optimizer);scaler.update();processed+=len(items);updates+=1
        validation_score=score_teacher(model,val_loader,config,device)
        improved=validation_score['logloss']<best
        if improved:best=validation_score['logloss']
        record={'epoch':epoch,'processed_tiles':processed,'optimizer_updates':updates,
            'training_logloss':total_loss/max(total_weight,1),'validation':validation_score,
            'epoch_seconds':time.perf_counter()-begin,**engine.memory_usage(device)}
        if improved:save_teacher(folder/'best.pt',model,optimizer,scaler,config,stats,digest,epoch,best,record)
        save_teacher(last,model,optimizer,scaler,config,stats,digest,epoch,best,record)
        with core.path(f'logs/{run_name}/epochs.jsonl',True).open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    return core.relative(folder/'best.pt')


def cache_teacher_targets(manifest,checkpoint,experiment_config,allow_synthetic=False):
    name=experiment_config['name'];meta=f'data/manifests/{name}_soft_targets.json'
    rows=validate_pairs(manifest,experiment_config,allow_synthetic);training=[r for r in rows if r['split']=='train']
    checkpoint_digest=core.sha256(checkpoint);manifest_digest=core.sha256(manifest)
    if core.path(meta).exists():
        result=core.read_json(meta)
        if result['teacher_checkpoint_sha256']!=checkpoint_digest or result['paired_manifest_sha256']!=manifest_digest:
            raise ValueError('Frozen soft targets belong to another teacher/dataset')
        quality=core.read_json(result['quality_report'])
        if not quality['passed'] or quality['teacher_checkpoint_sha256']!=checkpoint_digest:
            raise ValueError('Teacher quality receipt does not match the frozen checkpoint')
        validate_soft_targets(result,training);return meta
    device=engine.device_for(core.read_json(experiment_config['teacher_config'])['device'])
    model,saved=load_teacher(checkpoint,device)
    if saved['paired_manifest_sha256']!=manifest_digest:raise ValueError('Teacher checkpoint and paired manifest differ')
    config=saved['config'];model=configure_model(model,config,device,compile_model=False)
    quality=qualify_teacher(model,saved,rows,device,experiment_config)
    quality.update(teacher_checkpoint_sha256=checkpoint_digest,paired_manifest_sha256=manifest_digest)
    quality_path=f'results/{name}/teacher_quality.json';core.write_json(quality_path,quality)
    model=frozen_teacher(model)
    loader=teacher_loader(training,saved['stats'],config);result=[]
    with torch.inference_mode():
        for items in loader:
            for item in items:
                with engine.amp_context(device,config['precision']):logits=model(move_image(item['image'],device,config))
                logits=logits[0,0].float().cpu().numpy();weight=item['weight'][0,0].numpy();row=item['record']
                if not np.isfinite(logits).all():raise ValueError('Nonfinite teacher soft targets')
                destination=f'data/teacher_cache/{name}/{row["tile_id"]}.npz'
                core.write_npz(destination,teacher_logits=logits,distill_weight=weight)
                result.append({'tile_id':row['tile_id'],'split':'train','student_sha256':row['sha256'],
                    'teacher_tile_sha256':row['teacher_sha256'],'path':destination,'sha256':core.sha256(destination),
                    'known_weight':float(weight.sum(dtype=np.float64))})
    frozen={'schema':'frozen_viirs_soft_targets_v1','teacher_checkpoint':core.relative(checkpoint),
        'teacher_checkpoint_sha256':checkpoint_digest,'paired_manifest_sha256':manifest_digest,
        'parent_manifest_sha256':experiment_config['parent_manifest_sha256'],
        'training_only':True,'temperature_applied_at_loss':True,'quality_report':quality_path,'rows':result}
    validate_soft_targets(frozen,training);core.write_json(meta,frozen)
    print(f'Cached frozen teacher logits for {len(result)} training tiles; no validation/test targets cached.',flush=True)
    return meta


def validate_soft_targets(meta,training):
    if meta.get('schema')!='frozen_viirs_soft_targets_v1' or meta.get('training_only') is not True:
        raise ValueError('Soft targets must be a frozen training-only cache')
    expected={r['tile_id']:r for r in training};seen=set()
    for row in meta['rows']:
        key=row['tile_id']
        if row['split']!='train' or key not in expected or key in seen:raise ValueError('Held-out, duplicate or foreign soft target')
        seen.add(key)
        if row['student_sha256']!=expected[key]['sha256']:raise ValueError('Soft target refers to another student tile')
        if 'teacher_sha256' in expected[key] and row['teacher_tile_sha256']!=expected[key]['teacher_sha256']:
            raise ValueError('Soft target refers to another VIIRS image crop')
    if seen!=set(expected):raise ValueError('Soft targets do not cover exactly the training tiles')
