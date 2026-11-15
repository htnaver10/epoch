"""Prompt locally for Earthdata credentials and download the checked real pilot.

Uses the bounded 2023-07-01 western Canada inventory. Produces downloaded-source
and proposed-pair manifests, not reviewed training splits or a trained model.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import getpass
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres.acquisition import acquire,propose_pairs

CREDENTIAL_KEYS=('EARTHDATA_USERNAME','EARTHDATA_PASSWORD','EARTHDATA_TOKEN')


@contextmanager
def local_credentials(allow_prompt=True,launcher='download-real-data.cmd'):
    previous={name:os.environ.get(name) for name in CREDENTIAL_KEYS}
    configured=bool(previous['EARTHDATA_TOKEN'] or
        previous['EARTHDATA_USERNAME'] and previous['EARTHDATA_PASSWORD'])
    try:
        if not configured:
            if not allow_prompt or not sys.stdin.isatty():
                raise ValueError(f'Open {launcher} in a local terminal to enter your Earthdata login.')
            username=input('NASA Earthdata username: ').strip()
            password=getpass.getpass('NASA Earthdata password (hidden): ')
            if not username or not password:raise ValueError('Both username and password are required.')
            os.environ['EARTHDATA_USERNAME']=username
            os.environ['EARTHDATA_PASSWORD']=password
        yield
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only',action='store_true',help='Check public metadata without credentials or science-file downloads')
    parser.add_argument('--no-prompt',action='store_true',help='Require credentials already in this process environment')
    parser.add_argument('--start',default='2023-07-01T20:00:00Z',help='Acquisition range start in UTC')
    parser.add_argument('--end',default='2023-07-01T23:59:59Z',help='Acquisition range end in UTC')
    parser.add_argument('--bbox',type=float,nargs=4,default=[-125,50,-110,65],metavar=('WEST','SOUTH','EAST','NORTH'))
    parser.add_argument('--tag',default='pilot_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'),
        help='Fresh output name using letters, numbers, underscores or hyphens')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.tag):parser.error('Use an alphanumeric output tag with underscores/hyphens')
    inventory=f'data/manifests/{args.tag}_'+('inventory' if args.inventory_only else 'downloads')+'.jsonl'
    bundles=f'data/manifests/{args.tag}_proposed_bundles.jsonl'
    print(f'Real NASA pilot: {args.start} to {args.end}, bbox {args.bbox}, all seven MODIS/VIIRS products.',flush=True)
    if not args.inventory_only:
        print('Enter your Earthdata login here. Your password is hidden and is not saved to a file.',flush=True)
    try:
        def fetch():
            return acquire(args.start,args.end,args.bbox,
                inventory,download=not args.inventory_only,max_per_product=50)
        if args.inventory_only:
            result=fetch()
            print(f'Metadata check completed: {result["granules"]} granules. No science files downloaded.')
        else:
            with local_credentials(not args.no_prompt):result=fetch()
            pairs=propose_pairs(inventory,bundles)
            print(f'Download completed: {result["granules"]} source granules.')
            print(f'Proposed pairs: {pairs["proposed_pairs"]}; manifest: {bundles}')
            print('Next: inspect the real readers/alignment and review event groups and splits.')
            print('The pilot is for real-file QA; independent training/validation/test events still need collection.')
        print(f'Source manifest: {inventory}')
        return 0
    except (KeyboardInterrupt,EOFError):
        print('\nStopped. Downloaded files remain available for a later attempt.',file=sys.stderr)
        return 130
    except Exception as error:
        # Remote errors may contain signed URLs; keep the wrapper's error free
        # of credentials and leave provider diagnostics in the local terminal.
        print(f'Pilot setup failed ({type(error).__name__}). Check your Earthdata login, provider access and connection.',file=sys.stderr)
        if isinstance(error,ValueError):
            print('Use a fresh output tag. For login entry, open download-real-data.cmd in a local terminal.',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
