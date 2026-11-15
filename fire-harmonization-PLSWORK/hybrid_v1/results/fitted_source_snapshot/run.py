"""One-command local hybrid pipeline using the two existing Python environments."""
import argparse
import json
import subprocess
import sys
from common import ROOT,HERE,path,config,read,write,lock

def worker(stage,*args):
    parent=json.loads((ROOT/'configs/experiment.json').read_text())
    key='torch_python' if stage in ('prior','infer-prior') else 'tree_python'
    subprocess.run([str((ROOT/parent[key]).resolve()),str(HERE/'run.py'),stage,*args],check=True,cwd=ROOT)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['all','prepare','prior','train','calibrate','evaluate','report','verify','infer','infer-prior','infer-finish'])
    parser.add_argument('--split',choices=['train','validation','test']);parser.add_argument('--tile');parser.add_argument('--output')
    args=parser.parse_args();lock()
    if args.stage=='all':
        for stage in ('prepare','train','calibrate','evaluate','report','verify'):
            if stage=='train':
                for split in ('train','validation'):worker('prior','--split',split)
            if stage=='evaluate':worker('prior','--split','test')
            write('results/run_status.json',{'status':'running','stage':stage,'new_data_downloads':False})
            worker(stage)
        write('results/run_status.json',{'status':'complete','new_data_downloads':False,
                                       'verification':read('results/completion_verification.json')})
        return
    if args.stage=='prepare':
        from preprocessing import prepare
        result=prepare()
    elif args.stage=='prior':
        if args.split=='test' and not path('results/operating_policy.json').exists():
            raise ValueError('Freeze operating policy before test prediction')
        if args.split is None:raise ValueError('--split required')
        from model import cache_prior
        result=cache_prior(args.split)
    elif args.stage=='train':
        from model import train
        result=train()
    elif args.stage=='calibrate':
        from evaluation import calibrate
        result=calibrate()
    elif args.stage=='evaluate':
        from evaluation import evaluate
        result=evaluate()
        print(json.dumps(result['test'],indent=2),flush=True);return
    elif args.stage=='report':
        from report import render
        result=render()
    elif args.stage=='verify':
        from verification import verify
        result=verify()
    elif args.stage in ('infer','infer-prior'):
        if not args.tile or not args.output:raise ValueError('--tile and --output required')
        if args.stage=='infer':
            worker('infer-prior','--tile',args.tile,'--output',args.output);worker('infer-finish');return
        from model import infer_prior
        result=infer_prior(args.tile,args.output)
    else:
        from model import infer_finish
        result=infer_finish()
    if args.stage not in ('prepare',):print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
