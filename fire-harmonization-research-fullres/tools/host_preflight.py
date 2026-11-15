"""Inspect this training host without downloads or credential disclosure.

Exit 0 means the selected software stage is available, not scientific readiness.
Use --stage readers under the separate GDAL data environment.
"""
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres.core import path,write_json,read_jsonl


def physical_memory():
    if os.name!='nt':return None
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                  *[(name,ctypes.c_ulonglong) for name in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):return None
    return {'total_bytes':memory.total,'available_bytes':memory.available}


def inspect(stage):
    disk=shutil.disk_usage(ROOT)
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
            'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'logical_cpus':os.cpu_count(),'physical_memory':physical_memory(),
            'disk':dict(zip(('total_bytes','used_bytes','free_bytes'),disk)),
            'dependencies':{},'software_errors':[],
            'earthdata_environment_configured':bool(os.environ.get('EARTHDATA_TOKEN') or
                (os.environ.get('EARTHDATA_USERNAME') and os.environ.get('EARTHDATA_PASSWORD'))),
            'credential_note':'Presence only; credentials were not read from files or authenticated.'}
    for name in ('numpy','torch','h5py','pyproj','earthaccess','matplotlib'):
        if name=='torch' and stage=='readers':continue
        try:
            module=importlib.import_module(name)
            report['dependencies'][name]={'available':True,'version':str(getattr(module,'__version__','unknown'))}
            if name=='torch':
                available=module.cuda.is_available()
                report['cuda']={'available':available,'wheel_cuda':module.version.cuda,
                    'architectures':module.cuda.get_arch_list()}
                if available:
                    report['cuda'].update(gpu=module.cuda.get_device_name(0),
                        capability=list(module.cuda.get_device_capability(0)),
                        bf16=module.cuda.is_bf16_supported(),
                        free_bytes=module.cuda.mem_get_info()[0],total_bytes=module.cuda.get_device_properties(0).total_memory)
                if not available:report['software_errors'].append('CUDA unavailable')
                elif not report['cuda']['bf16']:report['software_errors'].append('Configured BF16 unavailable')
        except (ImportError,OSError,RuntimeError) as error:
            report['dependencies'][name]={'available':False,'reason':str(error)}
            if name=='numpy' or name=='torch' and stage=='training' or stage=='readers':
                report['software_errors'].append(name+' unavailable')
    try:
        from firefullres.native import gdal_module
        gdal=gdal_module()
        report['gdal']={'available':True,'version':gdal.VersionInfo('--version'),'HDF4':True,'HDF4Image':True}
    except (ImportError,OSError,RuntimeError) as error:
        report['gdal']={'available':False,'reason':str(error)}
        if stage=='readers':report['software_errors'].append('GDAL HDF4 readers unavailable')
    report['raw_native_files']=sum(1 for p in (ROOT/'data/raw').rglob('*') if p.suffix.lower() in ('.hdf','.nc'))
    manifests=[]
    for manifest in (ROOT/'data/manifests').glob('*.jsonl'):
        try:
            rows=read_jsonl(manifest)
            manifests.append({'manifest':manifest.relative_to(ROOT).as_posix(),'rows':len(rows),
                'dataset_kinds':sorted({r.get('dataset_kind','unknown') for r in rows}),
                'missing_tiles':sum(not path(r['tile']).is_file() for r in rows if 'tile' in r)})
        except (ValueError,OSError,KeyError) as error:
            manifests.append({'manifest':manifest.name,'error':str(error)})
    report['manifests']=manifests
    report['nasa_tile_manifests_present']=any(m.get('rows',0)>0 and m.get('dataset_kinds')==['nasa_native']
        and m.get('missing_tiles')==0 for m in manifests)
    report['scientific_readiness_assessed']=False
    report['scientific_note']='Manifest presence is not an audit or real-file calibration/alignment approval.'
    report['software_stage_ready']=not report['software_errors']
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],
            capture_output=True,text=True,timeout=10,check=True)
        report['nvidia_smi']=query.stdout.strip()
    except (OSError,subprocess.SubprocessError):pass
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('training','readers'),default='training')
    parser.add_argument('--output',default='results/windows_host_preflight.json')
    args=parser.parse_args();result=inspect(args.stage);write_json(args.output,result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['software_stage_ready'] else 1)
