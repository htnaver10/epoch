"""Train and use the isolated 50-million-parameter fire experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('all', 'features', 'profile', 'train', 'evaluate', 'infer', 'verify'))
    parser.add_argument('--tile')
    parser.add_argument('--output')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.stage in ('all', 'features'):
        from dataset import prepare_features
        print(json.dumps(prepare_features(), indent=2))
    if args.stage in ('all', 'profile'):
        from training import profile
        profile()
    if args.stage in ('all', 'train'):
        from training import train
        print(json.dumps(train(), indent=2))
    if args.stage in ('all', 'evaluate'):
        from training import evaluate
        evaluate()
    if args.stage == 'infer':
        if not args.tile or not args.output:
            parser.error('infer requires --tile and --output, relative to wide_diverse_v1')
        from inference import infer
        print(json.dumps(infer(args.tile, args.output, args.device), indent=2))
    if args.stage in ('all', 'verify'):
        from verification import verify
        print(json.dumps(verify(), indent=2))


if __name__ == '__main__':
    main()
