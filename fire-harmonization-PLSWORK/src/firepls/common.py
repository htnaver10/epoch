import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Artifact path escapes fire-harmonization-PLSWORK')
    return p
def read(value):return json.loads(path(value).read_text())
def rows(value):return [json.loads(line) for line in path(value).read_text().splitlines() if line.strip()]
def sha(value):
    with path(value).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(value,obj):
    p=path(value);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def config():return read('configs/experiment.json')
def validate_parent():
    from firefullres.core import validate_tiles
    cfg=config();manifest='data/manifests/source_v3.jsonl'
    if sha(manifest)!=cfg['source_manifest_sha256']:raise ValueError('Local parent manifest changed')
    return validate_tiles(manifest,False,False,False)


def freeze_sources():
    sources=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tools').glob('*.py'))+[ROOT/'run.py']
    lock={'sources':{p.relative_to(ROOT).as_posix():sha(p.relative_to(ROOT)) for p in sources},
        'config_sha256':sha('configs/experiment.json'),
        'copy_receipt_sha256':sha('results/local_copy_receipt.json'),
        'feature_receipt_sha256':sha('data/features/receipt.json')}
    output='results/software_lock.json'
    if path(output).exists() and read(output)!=lock:raise ValueError('Frozen experiment code/config/data changed; create a new version')
    if not path(output).exists():write(output,lock)
    return lock
