"""Materialize and normalize self-supervised measurements, with no reference file access."""
import hashlib
import collections
import numpy as np
from common import read,read_shared,write,sha,path,shared_path,save_array,load_array,config
from features import INPUT_KEYS,extract,interior

def load_inputs(row):
    with np.load(shared_path(row['input']),allow_pickle=False) as z:
        if set(z.files)!=INPUT_KEYS:raise ValueError('MODIS training view contains forbidden arrays')
        return {k:z[k] for k in INPUT_KEYS}

def check_no_overlap(rows):
    seen={}
    for r in rows:
        swath=r['modis_source'];split=r['split'];y,x,h,w=r['source_window']
        if swath in seen and seen[swath]['split']!=split:raise ValueError('Swath overlaps dataset partitions')
        bucket=seen.setdefault(swath,{'split':split,'windows':[]})['windows']
        # Only halo-excluded interiors are fitted. Reject duplicated interior exposure.
        from common import config
        halo=config()['halo'];rect=(y+halo,x+halo,y+h-halo,x+w-halo)
        for other in bucket:
            if max(rect[0],other[0])<min(rect[2],other[2]) and max(rect[1],other[1])<min(rect[3],other[3]):
                raise ValueError('Overlapping native training interiors')
        bucket.append(rect)

def partition():
    dest='data/partition.json'
    if path(dest).exists():return read(dest)
    rows=read_shared('data/modis_manifest.json');check_no_overlap(rows)
    by_swath={};strata=collections.defaultdict(list)
    for row in rows:
        if row['split']!='train' or row['modis_source'] in by_swath:continue
        # No reference or fire-mask access; classify the swath's solar geometry only.
        inputs=load_inputs(row)
        day=float(np.median(inputs['geometry'][5]))>=config()['minimum_daylight_cosine']
        key=(row['event_group'],'day' if day else 'night')
        by_swath[row['modis_source']]=key;strata[key].append(row['modis_source'])
    held=[]
    for key,sources in sorted(strata.items()):
        if len(sources)<2:continue
        order=sorted(sources,key=lambda s:hashlib.sha256(f'{config()["seed"]}:{s}'.encode()).hexdigest())
        n=max(1,int(len(sources)*config()['development_holdout_fraction']))
        held.extend(order[:n])
    result=[]
    for row in rows:
        fit_split='test' if row['split']=='test' else ('development' if row['split']=='validation' or row['modis_source'] in held else 'train')
        result.append({**row,'fit_split':fit_split})
    splits={s:{r['modis_source'] for r in result if r['fit_split']==s} for s in ('train','development','test')}
    if any(splits[a]&splits[b] for a,b in [('train','development'),('train','test'),('development','test')]):raise ValueError('Fit swath overlap')
    write(dest,{'rows':result,'held_out_training_swaths':held,
                'swath_counts':{s:len(v) for s,v in splits.items()},'uses_reference_labels':False})
    return read(dest)

def prepare():
    dest='data/features/receipt.json'
    if path(dest).exists():
        saved=read(dest)
        for f,d in saved['files'].items():
            if sha(path(f))!=d:raise ValueError('Self-supervised features changed')
        return saved
    rows=partition()['rows']
    chunks={(stage,split):{'X':[],'measured':[],'day':[],'target_valid':[]} for stage in ('coarse','fine') for split in ('train','development')}
    feature_names={}
    for i,row in enumerate(rows):
        if row['fit_split']=='test':continue
        inputs=load_inputs(row)
        for stage in ('coarse','fine'):
            x,y,mask,day,names,tv=extract(inputs,stage);feature_names[stage]=names
            # Fit only disjoint original interiors; prediction separately has full context support.
            scale=1 if stage=='coarse' else 2
            mask &= interior(mask.shape,scale*config()['halo'])
            bucket=chunks[(stage,row['fit_split'])]
            bucket['X'].append(x[mask]);bucket['measured'].append(y[mask]);bucket['day'].append(day[mask]);bucket['target_valid'].append(tv[mask])
        if (i+1)%100==0:print(f'Blind-spot features: {i+1} source tiles processed',flush=True)
    files=[];counts={}
    for stage in ('coarse','fine'):
        counts[stage]={}
        for split in ('train','development'):
            for key in ('X','measured','day','target_valid'):
                bucket=chunks[(stage,split)];value=np.concatenate(bucket[key]);bucket[key].clear()
                filename=f'data/features/{stage}_{split}_{key}.npy';save_array(filename,value);files.append(filename)
                if key=='X':counts[stage][split]=len(value)
                del value
        x=load_array(f'data/features/{stage}_train_X.npy');y=load_array(f'data/features/{stage}_train_measured.npy')
        # Stats are fitted solely on unlabeled training measurements; no test or VIIRS values.
        mean=np.zeros(x.shape[1],np.float64);sq=mean.copy()
        for start in range(0,len(x),65536):
            a=np.asarray(x[start:start+65536],np.float64);mean+=a.sum(0);sq+=(a*a).sum(0)
        mean/=len(x);scale=np.sqrt(np.maximum(sq/len(x)-mean*mean,1e-8))
        tv=load_array(f'data/features/{stage}_train_target_valid.npy')
        median=[];yscale=[]
        for j in range(y.shape[1]):
            observed=y[tv[:,j],j]
            if len(observed)<1000:raise ValueError('Insufficient observed training targets')
            median.append(float(np.median(observed)))
            yscale.append(float(max((np.quantile(observed,.75)-np.quantile(observed,.25))/1.349,.01)))
        stats={'input_mean':mean.tolist(),'input_scale':scale.tolist(),'target_center':median,'target_scale':yscale,
               'feature_names':feature_names[stage],'fit_split':'train','target':'withheld MODIS measurement; not fire label'}
        filename=f'data/features/{stage}_stats.json';write(filename,stats);files.append(filename)
    strata={}
    for stage in ('coarse','fine'):
        for split in ('train','development'):
            day=load_array(f'data/features/{stage}_{split}_day.npy');tv=load_array(f'data/features/{stage}_{split}_target_valid.npy')
            strata[f'{stage}_{split}']={'day':int(day.sum()),'night':int((~day).sum()),'observed_targets':tv.sum(0).tolist()}
    files+=['data/partition.json']
    result={'model_name':config()['name'],'counts':counts,'strata':strata,'feature_names':feature_names,'files':{f:sha(path(f)) for f in files},
            'test_inputs_used':False,'reference_arrays_used':False,'modis_fire_flags_used':False,
            'upstream_limitation':'Inherited tile availability comes from the prior paired-scene corpus; upstream selection is not an uncurated global MODIS sample.'}
    write(dest,result);return result
