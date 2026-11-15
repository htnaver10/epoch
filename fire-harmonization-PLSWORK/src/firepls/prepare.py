"""Build full-exposure local feature matrices and training-only normalizers."""
import numpy as np
from .common import path,read,write,sha,config,validate_parent
from .features import extract


def files_for(split):return {key:f'data/features/{split}_{key}.npy' for key in ('X','y','w','modis','tile_index','flat_index')}
def load_split(split,mmap=True):
    return {k:np.load(path(v),mmap_mode='r' if mmap else None,allow_pickle=False) for k,v in files_for(split).items()}


def prepare():
    cfg=config();parent=validate_parent();receipt='data/features/receipt.json'
    if path(receipt).exists():
        frozen=read(receipt)
        if frozen['config_sha256']!=sha('configs/experiment.json') or frozen['parent_sha256']!=sha('data/manifests/source_v3.jsonl'):
            raise ValueError('Feature preparation configuration/data changed')
        for filename,digest in frozen['files'].items():
            if sha(filename)!=digest:raise ValueError('Frozen feature matrix or normalizer changed')
        return frozen
    path('data/features').mkdir(parents=True,exist_ok=True)
    counts={split:int(sum(r['valid_queries'] for r in parent if r['split']==split)) for split in ('train','validation','test')}
    with np.load(path(parent[0]['tile']),allow_pickle=False) as z:
        _,names=extract({k:z[k] for k in ('thermal','reflective_500','reflective_250','geometry','query_grid')},cfg)
    matrices={};positions={split:0 for split in counts};tile_ledger=[]
    for split,n in counts.items():
        if n<=0:raise ValueError('Every partition requires known exposure')
        matrices[split]={}
        for key,file in files_for(split).items():
            dtype=np.float32 if key in ('X','y','w','modis') else np.int32
            shape=(n,len(names)) if key=='X' else (n,)
            matrices[split][key]=np.lib.format.open_memmap(path(file+'.partial'),mode='w+',dtype=dtype,shape=shape)
    total=np.zeros(len(names));squares=total.copy();exposure=0.;positive=0.
    split_positive={split:0. for split in counts}
    for index,row in enumerate(parent):
        if sha(row['tile'])!=row['sha256']:raise ValueError('Copied parent tile checksum changed')
        with np.load(path(row['tile']),allow_pickle=False) as z:
            x,actual=extract({k:z[k] for k in ('thermal','reflective_500','reflective_250','geometry','query_grid')},cfg)
            if actual!=names:raise ValueError('Feature schema changed')
            mask=z['weight']>0;flat=np.flatnonzero(mask).astype(np.int32);values=x[mask]
            y=z['target'][mask];weight=z['weight'][mask];baseline=z['modis_baseline'][mask]
            if not np.isin(weight,[1.]).all():raise ValueError('Parent exposure count assumes unit weights at known queries')
        split=row['split'];start=positions[split];end=start+len(y);positions[split]=end
        for key,values_ in [('X',values),('y',y),('w',weight),('modis',baseline),('flat_index',flat),('tile_index',np.full(len(y),index,np.int32))]:
            matrices[split][key][start:end]=values_
        tile_ledger.append({'tile_id':row['tile_id'],'split':split,'start':start,'end':end,'query_shape':row['query_shape'],
            'event_group':row['event_group'],'source_ids':row['source_ids'],'row_index':index})
        split_positive[split]+=float(np.sum(y*weight,dtype=np.float64))
        if split=='train':
            a=values.astype(np.float64);b=weight.astype(np.float64)[:,None]
            total+=(a*b).sum(0);squares+=(a*a*b).sum(0);exposure+=float(weight.sum());positive+=float((y*weight).sum())
        if (index+1)%100==0:print(f'Context features {index+1}/{len(parent)} tiles',flush=True)
    if positions!=counts:raise ValueError('Feature matrix exposure does not reconcile with parent')
    mean=total/exposure;scale=np.sqrt(np.maximum(squares/exposure-mean**2,1e-8))
    stats={'fit_partition':'train','feature_names':names,'mean':mean.tolist(),'scale':scale.tolist(),
        'prevalence':positive/exposure,'known_exposure':exposure,'normalization_clip':10.0,
        'parent_sha256':sha('data/manifests/source_v3.jsonl')}
    all_files=[]
    for split in counts:
        for key,file in files_for(split).items():
            matrices[split][key].flush();matrices[split][key]._mmap.close()
            path(file+'.partial').replace(path(file));all_files.append(file)
    write('data/features/statistics.json',stats);write('data/features/tiles.json',tile_ledger)
    all_files+=['data/features/statistics.json','data/features/tiles.json']
    frozen={'parent_sha256':sha('data/manifests/source_v3.jsonl'),'config_sha256':sha('configs/experiment.json'),
        'features':names,'feature_count':len(names),'counts':counts,'positive_counts':split_positive,
        'files':{file:sha(file) for file in all_files},'no_data_downloads':True,'all_known_queries_retained':True}
    write(receipt,frozen);return frozen
