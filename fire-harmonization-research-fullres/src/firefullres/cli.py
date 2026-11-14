import argparse
import json
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description='Isolated native MODIS -> VIIRS-reference image-model pipeline. All paths are relative to fire-harmonization-research-fullres.')
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('acquire',help='Bounded NASA CMR inventory, optionally authenticated download')
    a.add_argument('--start',required=True);a.add_argument('--end',required=True);a.add_argument('--bbox',nargs=4,type=float,required=True)
    a.add_argument('--output',required=True);a.add_argument('--download',action='store_true');a.add_argument('--products',nargs='+')
    a.add_argument('--max-per-product',type=int,default=50)
    a=sub.add_parser('propose-pairs',help='Propose close-time bundles; reviewed event/region/split fields remain required')
    a.add_argument('--inventory',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('prepare',help='Read native NASA HDF4/HDF5 bundles into georeferenced tile files')
    a.add_argument('--bundles',required=True);a.add_argument('--output',required=True)
    a.add_argument('--tile-size',type=int,default=96);a.add_argument('--stride',type=int,default=64)
    a.add_argument('--halo',type=int,default=16);a.add_argument('--max-query-side',type=int,default=384)
    a.add_argument('--all-land',action='store_true',help='Explicit thermal-activity baseline without a forest-only claim')
    a.add_argument('--inference',action='store_true',help='MODIS-only bundle, metric query grid, no VIIRS labels')
    a=sub.add_parser('make-demo',help='Synthetic software fixtures; not NASA data')
    a.add_argument('--output',default='data/manifests/demo.jsonl')
    a=sub.add_parser('preview',help='Inspect native radiance/reference alignment before training')
    a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a.add_argument('--index',type=int,default=0);a.add_argument('--allow-synthetic',action='store_true')
    for name in ('audit','stats'):
        a=sub.add_parser(name);a.add_argument('--manifest',required=True);a.add_argument('--allow-synthetic',action='store_true')
        a.add_argument('--output',required=True)
    a=sub.add_parser('train',help='Train/resume the native multiresolution CNN')
    for key in ('manifest','stats','config','run'):a.add_argument('--'+key,required=True)
    a.add_argument('--resume',action='store_true');a.add_argument('--device',choices=['auto','cpu','cuda']);a.add_argument('--epochs',type=int)
    a.add_argument('--allow-synthetic',action='store_true')
    for name in ('evaluate','infer'):
        a=sub.add_parser(name);a.add_argument('--manifest',required=True);a.add_argument('--checkpoint',required=True)
        a.add_argument('--output',required=True);a.add_argument('--split',choices=['validation','test'] if name=='evaluate' else ['train','validation','test','inference'],default='test' if name=='evaluate' else 'inference')
        a.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a.add_argument('--allow-synthetic',action='store_true')
    a=sub.add_parser('profile',help='End-to-end accumulated training profile, optionally with CPU/CUDA trace')
    for key in ('manifest','stats','config','output'):a.add_argument('--'+key,required=True)
    a.add_argument('--steps',type=int,default=10);a.add_argument('--allow-synthetic',action='store_true')
    a.add_argument('--warmup-steps',type=int,default=3);a.add_argument('--trace')
    a.add_argument('--worst-case',action='store_true',help='Start with largest query buckets; not an exhaustive fit guarantee')
    a.add_argument('--device',choices=['auto','cpu','cuda'])
    args=parser.parse_args(argv)
    try:
        command=args.command
        if command=='acquire':
            from .acquisition import acquire
            result=acquire(args.start,args.end,args.bbox,args.output,args.download,args.products,args.max_per_product)
        elif command=='propose-pairs':
            from .acquisition import propose_pairs
            result=propose_pairs(args.inventory,args.output)
        elif command=='prepare':
            from .prepare import prepare
            result=prepare(args.bundles,args.output,args.tile_size,args.stride,args.halo,args.max_query_side,not args.all_land,args.inference)
        elif command=='make-demo':
            from .demo import make_demo
            result=make_demo(args.output)
        elif command=='preview':
            from .preview import preview
            result=preview(args.manifest,args.output,args.index,args.allow_synthetic)
        elif command in ('audit','stats'):
            from .core import validate_tiles,statistics,write_json,sha256
            if command=='stats':result=statistics(args.manifest,args.output,args.allow_synthetic)
            else:
                rows=validate_tiles(args.manifest,args.allow_synthetic)
                result={'passed':True,'manifest_sha256':sha256(args.manifest),'tiles':len(rows),
                        'dataset_kinds':sorted({r['dataset_kind'] for r in rows}),'scope':'tile integrity and leakage audit; not scientific sensor validation'}
                write_json(args.output,result)
        else:
            from . import engine
            if command=='train':result=engine.train(args.manifest,args.stats,args.config,args.run,args.allow_synthetic,args.resume,args.device,args.epochs)
            elif command=='profile':result=engine.profile(args.manifest,args.stats,args.config,args.output,args.steps,args.allow_synthetic,
                args.warmup_steps,args.trace,args.worst_case,args.device)
            else:result=engine.predict(args.manifest,args.checkpoint,args.output,args.split,args.device,args.allow_synthetic,command=='evaluate')
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,ImportError,OSError,KeyError) as error:
        if isinstance(error,ModuleNotFoundError) and error.name=='torch':
            message='PyTorch is not installed. Follow RUNNING.md to create the CUDA/CPU full-resolution environment.'
        else:message=str(error)
        parser.exit(2,f'{command}: {message}\n')
