"""Synthetic implementation fixtures. Explicit opt-in required for training."""
import numpy as np
from .core import SCHEMA,path,sha256,write_npz,write_jsonl,statistics
from .geometry import reference_targets


def make_demo(output='data/manifests/demo.jsonl',size=32,query_size=64,train_tiles=4):
    if size<16 or size%8 or query_size<8 or train_tiles<1:
        raise ValueError('Demo size must be a multiple of 8 >=16; query size >=8; train tiles >=1')
    if path(output).exists():raise ValueError('Demo exists; use a new manifest name')
    rows=[]
    for split,count in [('train',train_tiles),('validation',2),('test',2)]:
        for i in range(count):
            seed=1000+100*len(rows)+i;rng=np.random.default_rng(seed)
            y,x=np.indices((size,size));cy,cx=rng.uniform(10,22,2)*size/32
            def image(n,channels):
                yy,xx=np.indices((n,n));heat=np.exp(-((yy*size/n-cy)**2+(xx*size/n-cx)**2)/(8*(size/32)**2))
                value=np.stack([rng.normal(.4,.03,(n,n))+(j+1)*heat for j in range(channels)]).astype(np.float32)
                return np.concatenate([value,np.ones_like(value)])
            qr,qc=np.meshgrid(np.linspace(4*size/32,27*size/32,query_size),np.linspace(4*size/32,27*size/32,query_size),indexing='ij')
            grid=np.stack([2*(qc+.5)/size-1,2*(qr+.5)/size-1],-1).astype(np.float32)
            heat=np.exp(-((qr-cy)**2+(qc-cx)**2)/(8*(size/32)**2))
            classes=np.where(heat>.55,8,5).astype(np.uint8)
            classes[:3,:]=4
            target,weight,nt,nw=reference_targets(classes,np.zeros(classes.shape,np.uint32),grid,np.ones(classes.shape,bool),(size,size),4)
            geometry=np.zeros((14,size,size),np.float32);geometry[[1,3,5,7,8,9,10,11,12,13]]=1
            arrays={'thermal':image(size,6),'reflective_500':image(2*size,5),'reflective_250':image(4*size,2),
                    'geometry':geometry,'query_grid':grid,'target':target,'weight':weight,'native_target':nt,'native_weight':nw,
                    'query_lat':(50+.01*qr).astype(np.float64),'query_lon':(-120+.02*qc).astype(np.float64),
                    'inference_valid':np.ones_like(target),'modis_baseline':(heat>.8).astype(np.float32)}
            tile=f'data/tiles/{path(output).stem}/{split}_{i}.npz';write_npz(tile,**arrays)
            rows.append({'schema':SCHEMA,'dataset_kind':'synthetic_smoke_only','tile_id':f'demo_{split}_{i}',
                         'tile':tile,'sha256':sha256(tile),'split':split,'event_group':f'demo_event_{split}_{i}',
                         'region_id':f'demo_region_{split}','source_ids':[f'synthetic_{split}_{i}'],'forest_only':False})
            rows[-1].update(native_shape=[size,size],query_shape=list(target.shape))
    write_jsonl(output,rows)
    return {'manifest':output,'tiles':len(rows),'warning':'Synthetic software smoke test only. No NASA observations or fire-accuracy evidence.'}
