"""CUDA-stage entry point; native preparation stays in the separate GDAL env."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.nn import functional as F
from firefullres import core,engine
from firefullres.runtime import resolve_config,configure_model,optimizer_for
from firefullres.distill.models import VIIRSTeacher
from firefullres.distill.teacher import TeacherTiles,move_image,train_teacher,cache_teacher_targets
from firefullres.distill.student import train_student


def profile_teacher(config):
    name=config['name'];cfg=resolve_config(core.read_json(config['teacher_config']))
    stats=core.read_json(f'data/manifests/{name}_teacher_stats.json')
    rows=[r for r in core.read_jsonl(f'data/manifests/{name}_pairs.jsonl') if r['split']=='train']
    row=max(rows,key=lambda r:r['query_shape'][0]*r['query_shape'][1])
    item=TeacherTiles([row],stats)[0];device=engine.device_for(cfg['device'])
    model=VIIRSTeacher(**cfg['model']).to(device);model.initialize_prior(stats['positive_prevalence'])
    model=configure_model(model,cfg,device,compile_model=False);optimizer=optimizer_for(model,cfg,device)
    image=move_image(item['image'],device,cfg);target=item['target'].to(device);weight=item['weight'].to(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with engine.amp_context(device,cfg['precision']):logits=model(image)
        loss=(F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        if not torch.isfinite(loss):raise ValueError('Nonfinite actual-data teacher profile loss')
        optimizer.step()
    if device.type=='cuda':torch.cuda.synchronize()
    result={'passed':True,'query_shape':row['query_shape'],'parameters':sum(p.numel() for p in model.parameters()),
        'seconds':time.perf_counter()-begin,'actual_data':True,'updates':2,**engine.memory_usage(device)}
    core.write_json(f'results/{name}/teacher_gpu_profile.json',result);print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('profile','teacher','cache','student','evaluate'))
    parser.add_argument('--config',default='configs/teacher_student_v1.json')
    args=parser.parse_args();config=core.read_json(args.config);name=config['name']
    pairs=f'data/manifests/{name}_pairs.jsonl';stats=f'data/manifests/{name}_teacher_stats.json'
    if args.stage=='profile':profile_teacher(config)
    elif args.stage=='teacher':
        run=name+'_teacher';train_teacher(pairs,stats,config['teacher_config'],run,core.path(f'models/{run}/last.pt').exists())
    elif args.stage=='cache':cache_teacher_targets(pairs,f'models/{name}_teacher/best.pt',config)
    elif args.stage=='student':
        run=name+'_student';train_student(config,f'data/manifests/{name}_soft_targets.json',run,core.path(f'models/{run}/last.pt').exists())
    else:
        from firefullres.distill.evaluation import evaluate
        evaluate(config)


if __name__=='__main__':main()
