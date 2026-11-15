"""Corrected pipeline with outputs confined to saturation_aware_v2."""
import argparse
import json
from common import freeze_fit,write

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['all','prepare','train','calibrate','evaluate','report','verify','infer'])
    parser.add_argument('--tile');parser.add_argument('--output');args=parser.parse_args();freeze_fit()
    if args.stage in ('all','prepare'):
        from data import prepare
        write('results/run_status.json',{'status':'running','stage':'prepare'});prepare()
        if args.stage=='prepare':return
    if args.stage in ('all','train'):
        from model import train
        for s in ('coarse','fine'):
            write('results/run_status.json',{'status':'running','stage':'train_'+s});train(s)
        if args.stage=='train':return
    if args.stage in ('all','calibrate'):
        from policy import calibrate
        write('results/run_status.json',{'status':'running','stage':'calibrate'});calibrate()
        if args.stage=='calibrate':return
    if args.stage in ('all','evaluate'):
        from evaluation import evaluate
        write('results/run_status.json',{'status':'running','stage':'evaluate'});r=evaluate()
        print(json.dumps({'coverage':r['test']['coverage'],'query':r['test']['query']},indent=2),flush=True)
        if args.stage=='evaluate':return
    if args.stage in ('all','report'):
        from report import render
        render()
        if args.stage=='report':return
    if args.stage in ('all','verify'):
        from verify import verify
        r=verify();write('results/run_status.json',{'status':'complete','verification':r});print(json.dumps(r,indent=2))
    if args.stage=='infer':
        if not args.tile or not args.output:parser.error('--tile and --output required')
        from inference import infer_file
        print(json.dumps(infer_file(args.tile,args.output),indent=2))

if __name__=='__main__':main()
