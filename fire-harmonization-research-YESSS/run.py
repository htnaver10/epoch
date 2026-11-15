"""Complete local unsupervised experiment. No network or reference-label fitting."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['all','copy','prepare','train','calibrate','evaluate','report','verify','infer'])
    parser.add_argument('--tile');parser.add_argument('--output');args=parser.parse_args()
    from fireyes.common import freeze_fit,write,read
    if args.stage in ('all','copy'):
        from copy_local import copy
        write('results/run_status.json',{'status':'running','stage':'copy'});copy()
        if args.stage=='copy':return
    freeze_fit()
    if args.stage in ('all','prepare'):
        from fireyes.data import prepare
        write('results/run_status.json',{'status':'running','stage':'prepare'});prepare()
        if args.stage=='prepare':return
    if args.stage in ('all','train'):
        from fireyes.model import train
        for stage in ('coarse','fine'):
            write('results/run_status.json',{'status':'running','stage':'train_'+stage});train(stage)
        if args.stage=='train':return
    if args.stage in ('all','calibrate'):
        from fireyes.policy import calibrate
        write('results/run_status.json',{'status':'running','stage':'calibrate'});calibrate()
        if args.stage=='calibrate':return
    if args.stage in ('all','evaluate'):
        from fireyes.evaluation import evaluate
        write('results/run_status.json',{'status':'running','stage':'evaluate'});result=evaluate()
        print(json.dumps(result['test']['query'],indent=2),flush=True)
        if args.stage=='evaluate':return
    if args.stage in ('all','report'):
        from report import render
        render()
        if args.stage=='report':return
    if args.stage in ('all','verify'):
        from verify import verify
        result=verify();write('results/run_status.json',{'status':'complete','verification':result});print(json.dumps(result,indent=2),flush=True)
    if args.stage=='infer':
        if not args.tile or not args.output:parser.error('--tile and --output required')
        from fireyes.inference import infer_file
        print(json.dumps(infer_file(args.tile,args.output),indent=2))

if __name__=='__main__':main()
