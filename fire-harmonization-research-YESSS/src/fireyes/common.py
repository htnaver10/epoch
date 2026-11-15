import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def path(name):
    p=(ROOT/name).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Path escapes YESSS experiment')
    return p

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(name):return json.loads(path(name).read_text(encoding='utf-8'))
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

def freeze_fit():
    # Evaluation, reporting and verification are deliberately outside the fitting lock.
    names=['config.json']+[f'src/fireyes/{n}.py' for n in ('common','features','data','model','policy','inference')]
    value={'files':{n:sha(path(n)) for n in names},'input_manifest_sha256':sha(path('data/modis_manifest.json')),
           'copy_receipt_sha256':sha(path('results/copy_receipt.json'))}
    dest='results/fit_lock.json'
    if path(dest).exists() and read(dest)!=value:raise ValueError('Frozen fitting inputs/code changed; create a new run version')
    if not path(dest).exists():write(dest,value)
    return value
