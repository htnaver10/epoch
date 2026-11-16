"""Run the second expansion's isolated size/data controls and larger model."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import settings

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('all', 'features', 'profile', 'train', 'evaluate', 'fit', 'infer', 'verify', 'compare'))
    parser.add_argument('--run', choices=tuple(settings.read('config.json')['runs']), default=settings.DEFAULT_RUN)
    parser.add_argument('--tile'); parser.add_argument('--output')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.stage in ('all', 'features'):
        from dataset import prepare_features
        print(json.dumps(prepare_features(), indent=2), flush=True)
    runs = list(settings.read('config.json')['runs']) if args.stage == 'all' else [args.run]
    for run_id in runs:
        if args.stage in ('all', 'profile', 'fit'):
            from training import profile
            profile(run_id)
        if args.stage in ('all', 'train', 'fit'):
            from training import train
            print(json.dumps(train(run_id), indent=2), flush=True)
        if args.stage in ('all', 'evaluate', 'fit'):
            from training import evaluate
            evaluate(run_id)
        if args.stage in ('all', 'verify', 'fit'):
            from verification import verify
            print(json.dumps(verify(run_id), indent=2), flush=True)
    if args.stage == 'infer':
        if not args.tile or not args.output: parser.error('infer requires --tile and --output relative to wide_diverse_v2')
        from inference import infer
        print(json.dumps(infer(args.tile, args.output, args.device, args.run), indent=2))
    if args.stage in ('all', 'compare'):
        from comparison import compare
        print(json.dumps(compare(), indent=2), flush=True)

if __name__ == '__main__': main()
