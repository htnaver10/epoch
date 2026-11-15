"""Isolated, checksum-bound hybrid experiment; parent experiment is read-only."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))

def path(name):
    p = (HERE / name).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError('Hybrid artifact path escapes experiment')
    return p

def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(name):
    return json.loads(path(name).read_text())

def write(name, obj):
    p = path(name); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)

def config():
    return read('config.json')

def save_array(name, a):
    import numpy as np
    p = path(name); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.partial')
    with tmp.open('wb') as stream:
        np.save(stream, a, allow_pickle=False)
    tmp.replace(p)

def lock():
    from firepls.common import freeze_sources
    freeze_sources()
    cfg = config()
    if sha(ROOT / cfg['base_model']) != cfg['base_model_sha256']:
        raise ValueError('Parent MLP changed')
    if sha(ROOT / 'data/features/receipt.json') != cfg['feature_receipt_sha256']:
        raise ValueError('Parent feature receipt changed')
    files = sorted(HERE.glob('*.py')) + [HERE / 'config.json']
    frozen = {'files': {p.name: sha(p) for p in files},
              'parent_software_lock': sha(ROOT / 'results/software_lock.json')}
    dest = path('results/software_lock.json')
    if dest.exists() and read('results/software_lock.json') != frozen:
        original = read('results/software_lock.json')
        # A recorded post-fit verifier repair is allowed; fitted mathematics and artifacts stay sealed.
        repair_path = path('results/post_training_repair.json')
        if not repair_path.exists():
            raise ValueError('Frozen hybrid source changed; use a new experiment version')
        repair = read('results/post_training_repair.json')
        changed = {name:{'before':original['files'].get(name),'after':digest}
                   for name,digest in frozen['files'].items() if original['files'].get(name)!=digest}
        if (set(frozen['files'])!=set(original['files']) or
            frozen['parent_software_lock']!=original['parent_software_lock'] or
            not set(changed).issubset({'common.py','verification.py'}) or
            repair['original_software_lock_sha256']!=sha(dest) or repair['changes']!=changed):
            raise ValueError('Unapproved modification to fitted hybrid sources')
        for name,digest in original['files'].items():
            if sha(path('results/fitted_source_snapshot')/name)!=digest:
                raise ValueError('Original fitted source snapshot changed')
    if not dest.exists():
        write('results/software_lock.json', frozen)
    return frozen

def load(name):
    import numpy as np
    return np.load(path(name), mmap_mode='r', allow_pickle=False)
