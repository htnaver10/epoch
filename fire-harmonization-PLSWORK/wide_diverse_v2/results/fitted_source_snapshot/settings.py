"""Isolated second expansion and controlled size/data experiments."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
V1 = PARENT / 'wide_diverse_v1'
WORKSPACE = PARENT.parent
DEFAULT_RUN = 'large200_d2'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT / '.cache/torchinductor')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / '.cache/triton')
sys.path.insert(0, str(ROOT / 'vendor'))

def path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Output path escapes wide_diverse_v2')
    return result

def read(value):
    return json.loads(path(value).read_text(encoding='utf-8'))

def rows(filename):
    return [json.loads(line) for line in Path(filename).read_text(encoding='utf-8').splitlines() if line.strip()]

def sha(filename):
    with Path(filename).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(value, data):
    target = path(value); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)

def feature_config():
    return read('config.json')['features']

def run_config(run_id=DEFAULT_RUN):
    cfg = read('config.json')
    if run_id not in cfg['runs']:
        raise ValueError('Unknown controlled run')
    return cfg, cfg['runs'][run_id]

def artifact(run_id, name, kind='results'):
    run_config(run_id)
    return f'{kind}/{run_id}/{name}'

def freeze_sources(run_id=DEFAULT_RUN):
    import torch
    cfg, run = run_config(run_id)
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'vendor').rglob('*.py')) + [path('config.json'), path('expansion_config.json')]
    current = {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}, 'run_id': run_id, 'run': run,
        'parent_manifest_sha256': sha(PARENT / 'data/manifests/source_v3.jsonl'),
        'parent_feature_receipt_sha256': sha(PARENT / 'data/features/receipt.json'),
        'parent_model_sha256': sha(PARENT / 'models/mlp/inference.pt'),
        'v1_manifest_sha256': sha(V1 / 'data/manifests/extra_train.jsonl'),
        'v1_feature_receipt_sha256': sha(V1 / 'data/feature_receipt.json'),
        'v1_model_sha256': sha(V1 / 'models/inference.pt'),
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    if run['dataset'] == 'd2':
        current.update(data_receipt_sha256=sha(path('data/feature_receipt.json')),
            expansion_receipt_sha256=sha(path('results/expansion_receipt.json')))
    output = artifact(run_id, 'software_lock.json')
    if path(output).exists():
        if read(output) != current:
            raise ValueError('Frozen run source/configuration/data changed')
    else:
        write(output, current)
    return current
