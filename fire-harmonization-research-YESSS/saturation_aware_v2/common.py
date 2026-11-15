"""Version-confined outputs and read-only access to the shared local observation copy."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parent
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'.cache/torch_inductor')
os.environ['TRITON_CACHE_DIR']=str(ROOT/'.cache/triton')

def path(name):
    p=(ROOT/name).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path escapes SaturationAwareBackgroundV2')
    return p

def shared_path(name):
    p=(SHARED/name).resolve()
    if not p.is_relative_to(SHARED) or p.is_relative_to(ROOT):raise ValueError('Invalid shared input path')
    return p

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(name):return json.loads(path(name).read_text(encoding='utf-8'))
def read_shared(name):return json.loads(shared_path(name).read_text(encoding='utf-8'))
def config():return read('config.json')

def write(name,value):
    dest=path(name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(dest)

def save_array(name,value):
    dest=path(name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial')
    with tmp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
    tmp.replace(dest)

def save_npz(name,**values):
    dest=path(name);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**values)
    tmp.replace(dest)

def load_array(name):return np.load(path(name),mmap_mode='r',allow_pickle=False)

def prior_lock():
    files=['config.json','models/policy.json','results/fit_lock.json','results/evaluation.json']
    files += [f'models/{stage}/{name}.pt' for stage in ('coarse','fine') for name in ('best','last','inference')]
    files += [str(p.relative_to(SHARED)).replace('\\','/') for p in (SHARED/'src/fireyes').glob('*.py')]
    state={n:sha(shared_path(n)) for n in sorted(files)}
    if path('results/prior_artifact_lock.json').exists():
        if read('results/prior_artifact_lock.json')!=state:raise ValueError('Original experiment changed')
    else:write('results/prior_artifact_lock.json',state)
    return state

def freeze_fit():
    prior_lock()
    value={'files':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},'config_sha256':sha(path('config.json')),
           'shared_modis_manifest_sha256':sha(shared_path('data/modis_manifest.json')),
           'shared_copy_receipt_sha256':sha(shared_path('results/copy_receipt.json')),
           'prior_artifact_lock_sha256':sha(path('results/prior_artifact_lock.json'))}
    dest='results/source_lock.json'
    if path(dest).exists() and read(dest)!=value:raise ValueError('Frozen V2 source/data changed; use another version')
    if not path(dest).exists():write(dest,value)
    return value
