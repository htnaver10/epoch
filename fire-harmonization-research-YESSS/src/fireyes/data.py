"""Materialize and normalize self-supervised measurements, with no reference file access."""
import numpy as np
from .common import read,write,sha,path,save_array,load_array
from .features import INPUT_KEYS,extract

def load_inputs(row):
    with np.load(path(row['input']),allow_pickle=False) as z:
        if set(z.files)!=INPUT_KEYS:raise ValueError('MODIS training view contains forbidden arrays')
        return {k:z[k] for k in INPUT_KEYS}

def check_no_overlap(rows):
    seen={}
    for r in rows:
        swath=r['modis_source'];split=r['split'];y,x,h,w=r['source_window']
        if swath in seen and seen[swath]['split']!=split:raise ValueError('Swath overlaps dataset partitions')
        bucket=seen.setdefault(swath,{'split':split,'windows':[]})['windows']
        # Only halo-excluded interiors are fitted. Reject duplicated interior exposure.
        from .common import config
        halo=config()['halo'];rect=(y+halo,x+halo,y+h-halo,x+w-halo)
        for other in bucket:
            if max(rect[0],other[0])<min(rect[2],other[2]) and max(rect[1],other[1])<min(rect[3],other[3]):
                raise ValueError('Overlapping native training interiors')
        bucket.append(rect)

def prepare():
    dest='data/features/receipt.json'
    if path(dest).exists():
        saved=read(dest)
        for f,d in saved['files'].items():
            if sha(path(f))!=d:raise ValueError('Self-supervised features changed')
        return saved
    rows=read('data/modis_manifest.json');check_no_overlap(rows)
    chunks={(stage,split):{'X':[],'measured':[],'day':[]} for stage in ('coarse','fine') for split in ('train','validation')}
    feature_names={}
    for i,row in enumerate(rows):
        if row['split']=='test':continue
        inputs=load_inputs(row)
        for stage in ('coarse','fine'):
            x,y,mask,day,names=extract(inputs,stage);feature_names[stage]=names
            bucket=chunks[(stage,row['split'])]
            bucket['X'].append(x[mask]);bucket['measured'].append(y[mask]);bucket['day'].append(day[mask])
        if (i+1)%100==0:print(f'Blind-spot features: {i+1} source tiles processed',flush=True)
    files=[];counts={}
    for stage in ('coarse','fine'):
        counts[stage]={}
        for split in ('train','validation'):
            for key in ('X','measured','day'):
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
        median=np.median(y,axis=0);yscale=np.maximum((np.quantile(y,.75,axis=0)-np.quantile(y,.25,axis=0))/1.349,.01)
        stats={'input_mean':mean.tolist(),'input_scale':scale.tolist(),'target_center':median.tolist(),'target_scale':yscale.tolist(),
               'feature_names':feature_names[stage],'fit_split':'train','target':'withheld MODIS measurement; not fire label'}
        filename=f'data/features/{stage}_stats.json';write(filename,stats);files.append(filename)
    result={'counts':counts,'feature_names':feature_names,'files':{f:sha(path(f)) for f in files},
            'test_inputs_used':False,'reference_arrays_used':False,'modis_fire_flags_used':False,
            'upstream_limitation':'Inherited tile availability comes from the prior paired-scene corpus; upstream selection is not an uncurated global MODIS sample.'}
    write(dest,result);return result
