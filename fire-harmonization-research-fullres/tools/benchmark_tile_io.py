"""Compare the former repeated-NPZ loading path with the new single-read loader.

Warm-cache, single-process NumPy microbenchmark only; not GPU/NASA accuracy proof.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from firefullres.core import (INPUT_KEYS,REQUIRED_KEYS,path,read_json,sha256,
                             validate_tiles,validate_arrays,normalize,write_json)
from firefullres.loading import TileStore


def before(arrays,stats,row):
    validate_arrays(arrays)
    return {'inputs':{key:normalize(key,arrays[key],stats) if key!='query_grid' else arrays[key].copy() for key in INPUT_KEYS},
            **{key:arrays[key].copy().astype(np.float32) for key in ('target','weight','native_target','native_weight')},
            'record':row}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--stats',required=True)
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--output',default='results/tile_io_benchmark.json')
    parser.add_argument('--allow-synthetic',action='store_true');args=parser.parse_args()
    if args.iterations<5:parser.error('Use at least five iterations')
    rows=validate_tiles(args.manifest,args.allow_synthetic,False,False);stats=read_json(args.stats)
    if stats['manifest_sha256']!=sha256(args.manifest):raise ValueError('Statistics belong to another manifest')
    if not 0<=args.index<len(rows):parser.error('index outside manifest')
    row=rows[args.index];store=TileStore([row],stats,cache_mib=0)
    class Counted:
        def __init__(self,z):self.z=z;self.count=collections.Counter()
        def __contains__(self,k):return k in self.z.files
        def __getitem__(self,k):self.count[k]+=1;return self.z[k]
    with np.load(path(row['tile']),allow_pickle=False) as z:
        counted=Counted(z);old=before(counted,stats,row);old_reads=sum(counted.count.values())
    new=store[0]
    for key in INPUT_KEYS:np.testing.assert_array_equal(old['inputs'][key],new['inputs'][key])
    for key in ('target','weight','native_target','native_weight'):np.testing.assert_array_equal(old[key],new[key])
    def old_load():
        with np.load(path(row['tile']),allow_pickle=False) as z:return before(z,stats,row)
    timings={'before':[],'after':[]}
    for _ in range(3):old_load();store[0]
    for i in range(args.iterations):
        order=[('before',old_load),('after',lambda:store[0])]
        if i%2:order.reverse()
        for name,fn in order:
            start=time.perf_counter();fn();timings[name].append(time.perf_counter()-start)
    medians={name:statistics.median(values) for name,values in timings.items()}
    result={'scope':'warm-cache NumPy loading/normalization of one tile; excludes hashing warmup, Torch, workers, CUDA and NASA-reader validation',
            'manifest_sha256':sha256(args.manifest),'tile_id':row['tile_id'],'dataset_kind':row['dataset_kind'],
            'native_shape':list(new['inputs']['thermal'].shape[-2:]),'query_shape':list(new['target'].shape),
            'iterations':args.iterations,'arrays_bitwise_equal':True,'before_member_reads':old_reads,
            'after_member_reads':len(REQUIRED_KEYS),'median_seconds':medians,
            'local_loader_speed_ratio':medians['before']/medians['after'],
            'not_claimed':'end-to-end training speedup, RTX utilization, real NASA I/O throughput or fire accuracy'}
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
