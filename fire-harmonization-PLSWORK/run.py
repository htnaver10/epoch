"""Local-only full pipeline; uses existing Python environments, no data downloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='all',choices=('all','copy','prepare','train-mlp','train-tree','predict-tree','calibrate','evaluate','infer'))
    p.add_argument('--split',choices=('train','validation','test'),default='validation')
    p.add_argument('--tile');p.add_argument('--output');p.add_argument('--method',choices=('mlp','tree'),default='mlp')
    args=p.parse_args(argv)
    if args.stage in ('all','copy'):
        from bootstrap_copy import copy_existing
        copy_existing()
        if args.stage=='copy':return
    from firepls.common import config,path,write,freeze_sources
    cfg=config()
    if args.stage in ('all','prepare'):
        from firepls.prepare import prepare
        prepare()
        if args.stage=='prepare':return
    if args.stage in ('all','train-mlp','train-tree','calibrate','evaluate'):freeze_sources()
    env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    if args.stage in ('all','train-mlp'):
        from firepls.mlp import train
        train()
        if args.stage=='train-mlp':return
    if args.stage=='all':
        python=(ROOT/cfg['tree_python']).resolve()
        subprocess.run([str(python),str(ROOT/'run.py'),'train-tree'],check=True,env=env)
    if args.stage=='train-tree':
        from firepls.tree import train
        print(json.dumps(train(),indent=2));return
    if args.stage=='predict-tree':
        import numpy as np
        from firepls.tree import predict
        np.save(path(f'results/{args.split}_tree_logits.npy'),predict(args.split),allow_pickle=False);return
    if args.stage=='calibrate':
        from firepls.evaluation import freeze_policies
        freeze_policies();return
    if args.stage in ('all','evaluate'):
        from firepls.evaluation import evaluate
        evaluate();return
    if args.stage=='infer':
        if not args.tile or not args.output:p.error('infer requires --tile and --output')
        if args.method=='tree':
            import importlib.util
            if importlib.util.find_spec('lightgbm') is None:
                subprocess.run([str((ROOT/cfg['tree_python']).resolve()),str(ROOT/'run.py'),'infer','--tile',args.tile,'--output',args.output,'--method','tree'],check=True,env=env);return
        from firepls.inference import infer
        print(json.dumps(infer(args.tile,args.output,args.method)))


if __name__=='__main__':main()
