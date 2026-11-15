"""Self-supervised robust regression of withheld MODIS measurements."""
import gc
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import config,read,write,path,sha,load_array

class BackgroundNet(nn.Module):
    def __init__(self,n_inputs,n_outputs,hidden=(64,32)):
        super().__init__();layers=[];last=n_inputs
        for width in hidden:layers.extend([nn.Linear(last,width),nn.GELU()]);last=width
        layers.append(nn.Linear(last,n_outputs));self.network=nn.Sequential(*layers)
    def forward(self,x):return self.network(x)

def masked_loss(prediction,target,observed):
    # Unavailable/saturated values contribute neither a gradient nor a denominator.
    losses=F.smooth_l1_loss(prediction,target,reduction='none',beta=1.)
    return (losses*observed).sum()/observed.sum().clamp_min(1.)

def atomic_save(name,value):
    dest=path(name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial')
    torch.save(value,tmp);tmp.replace(dest)

def normalize_x(x,stats):
    return np.clip((x-np.asarray(stats['input_mean'],np.float32))/np.asarray(stats['input_scale'],np.float32),-10,10).astype(np.float32)

def normalize_y(y,stats):
    return (y-np.asarray(stats['target_center'],np.float32))/np.asarray(stats['target_scale'],np.float32)

def load_model(stage,device='cpu'):
    saved=torch.load(path(f'models/{stage}/inference.pt'),map_location=device,weights_only=True)
    if saved['stage']!=stage or saved['uses_fire_labels'] or saved['model_name']!=config()['name']:raise ValueError('Wrong model provenance')
    net=BackgroundNet(**saved['architecture']).to(device);net.load_state_dict(saved['model'],strict=True);net.eval()
    return net,saved

def predict(stage,x,device='cpu'):
    torch.set_num_threads(4);net,saved=load_model(stage,device);result=[]
    with torch.inference_mode():
        for start in range(0,len(x),65536):
            a=torch.from_numpy(normalize_x(x[start:start+65536],saved['stats'])).to(device)
            result.append(net(a).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,saved['architecture']['n_outputs']),np.float32)

def train(stage):
    cfg=config()['training'];stats=read(f'data/features/{stage}_stats.json');done=f'models/{stage}/completed.json'
    if path(done).exists():
        old=read(done)
        if old['export_sha256']!=sha(path(f'models/{stage}/inference.pt')):raise ValueError('Trained export changed')
        return old
    torch.set_num_threads(4);torch.manual_seed(config()['seed']+(stage=='fine'))
    device=torch.device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('This run requires the configured CUDA environment')
    architecture={'n_inputs':len(stats['input_mean']),'n_outputs':len(stats['target_center']),'hidden':cfg['hidden']}
    net=BackgroundNet(**architecture).to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start=0;best=float('inf');stale=0;history=[]
    signature={'stats':stats,'config':cfg,'feature_receipt_sha256':sha(path('data/features/receipt.json'))}
    checkpoint=f'models/{stage}/last.pt'
    if path(checkpoint).exists():
        saved=torch.load(path(checkpoint),map_location=device,weights_only=True)
        if saved['signature']!=signature:raise ValueError('Resume inputs changed')
        net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch']+1;best=saved['best'];stale=saved['stale'];history=saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    data={}
    for split in ('train','development'):
        x=load_array(f'data/features/{stage}_{split}_X.npy');y=load_array(f'data/features/{stage}_{split}_measured.npy')
        tv=load_array(f'data/features/{stage}_{split}_target_valid.npy')
        norm=np.where(tv,normalize_y(y,stats),0).astype(np.float32)
        data[split]=(torch.from_numpy(normalize_x(x,stats)).to(device),torch.from_numpy(norm).to(device),torch.tensor(tv,dtype=torch.float32,device=device))
    tx,ty,tw=data['train'];vx,vy,vw=data['development'];batch=cfg['batch_size']
    print(f'{stage}: {len(tx):,} unlabeled training samples; {len(vx):,} development samples; {sum(p.numel() for p in net.parameters()):,} parameters',flush=True)
    started=time.perf_counter()
    for epoch in range(start,cfg['epochs']):
        if stale>=cfg['patience']:break
        net.train();order=torch.randperm(len(tx),device=device);total=0.;tick=time.perf_counter()
        for begin in range(0,len(tx),batch):
            index=order[begin:begin+batch];optimizer.zero_grad(set_to_none=True)
            loss=masked_loss(net(tx[index]),ty[index],tw[index])
            if not torch.isfinite(loss):raise ValueError('Nonfinite reconstruction loss')
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*float(tw[index].sum())
        net.eval();validation=0.
        with torch.inference_mode():
            for begin in range(0,len(vx),batch):
                a=vx[begin:begin+batch];weights=vw[begin:begin+batch]
                validation+=float(masked_loss(net(a),vy[begin:begin+batch],weights))*float(weights.sum())
        validation/=float(vw.sum());improved=validation<best
        if improved:best=validation;stale=0
        else:stale+=1
        row={'epoch':epoch+1,'training_reconstruction_loss':total/float(tw.sum()),'validation_reconstruction_loss':validation,
             'seconds':time.perf_counter()-tick,'training_samples':len(tx)}
        history.append(row)
        state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,
               'history':history,'signature':signature,'architecture':architecture,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        if improved:atomic_save(f'models/{stage}/best.pt',state)
        atomic_save(checkpoint,state);write(f'results/{stage}_history.json',history);print(row,flush=True)
    selected=torch.load(path(f'models/{stage}/best.pt'),map_location='cpu',weights_only=True)
    export={'model_name':config()['name'],'stage':stage,'model':selected['model'],'architecture':architecture,'stats':stats,
            'selected_epoch':selected['epoch']+1,'uses_fire_labels':False,'target':'withheld MODIS measurement',
            'feature_receipt_sha256':signature['feature_receipt_sha256']}
    atomic_save(f'models/{stage}/inference.pt',export);load_model(stage)
    result={'model_name':config()['name'],'stage':stage,'epochs_completed':len(history),'selected_epoch':export['selected_epoch'],
            'selection_metric':'unlabeled validation SmoothL1 reconstruction loss','best_loss':best,
            'training_samples':len(tx),'validation_samples':len(vx),'parameters':sum(p.numel() for p in net.parameters()),
            'seconds_this_invocation':time.perf_counter()-started,'export_sha256':sha(path(f'models/{stage}/inference.pt')),
            'uses_viirs_labels':False,'uses_modis_fire_labels':False,'torch_version':str(torch.__version__)}
    write(done,result);del data,tx,ty,tw,vx,vy,vw,net,optimizer;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result
