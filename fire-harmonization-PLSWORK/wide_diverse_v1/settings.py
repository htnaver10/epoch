"""Experiment paths and atomic artifacts; parent experiments are read-only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
WORKSPACE = PARENT.parent
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT / '.cache/torchinductor')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / '.cache/triton')
sys.path.insert(0, str(ROOT / 'vendor'))


def path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Output path escapes wide_diverse_v1')
    return result


def read(value):
    return json.loads(path(value).read_text(encoding='utf-8'))


def rows(filename):
    return [json.loads(line) for line in Path(filename).read_text().splitlines() if line.strip()]


def sha(filename):
    with Path(filename).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(value, data):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)


def feature_config():
    return read('config.json')['features']


def configure_native_root():
    from firefullres import core
    core.ROOT = ROOT
    return core


def freeze_sources():
    import platform
    import torch
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'vendor').rglob('*.py')) + [path('config.json')]
    if path('expansion_config.json').exists():
        files.append(path('expansion_config.json'))
    current = {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
               'parent_manifest_sha256': sha(PARENT / 'data/manifests/source_v3.jsonl'),
               'parent_feature_receipt_sha256': sha(PARENT / 'data/features/receipt.json'),
               'parent_model_sha256': sha(PARENT / 'models/mlp/inference.pt'),
               'data_receipt_sha256': sha(path('data/feature_receipt.json')),
               'expansion_receipt_sha256': sha(path('results/expansion_receipt.json')),
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    if path('results/software_lock.json').exists():
        if read('results/software_lock.json') != current:
            raise ValueError('Frozen source/data changed; preserve this run and create a new version')
    else:
        write('results/software_lock.json', current)
    return current
