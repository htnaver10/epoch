from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'modis_native_viirs_queries_v1'
THERMAL_BANDS = [20, 21, 22, 23, 31, 32]
INPUT_KEYS = ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')
REQUIRED_KEYS = (*INPUT_KEYS, 'target', 'weight', 'native_target', 'native_weight', 'query_lat', 'query_lon')


def path(value, mkdir=False):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('All full-resolution paths must be inside fire-harmonization-research-fullres')
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def relative(value):
    return str(path(value).relative_to(ROOT))


def sha256(value):
    h = hashlib.sha256()
    with path(value).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(value):
    return json.loads(path(value).read_text())


def write_json(value, obj):
    p = path(value, True)
    temporary = p.with_name(p.name + '.partial')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, p)


def read_jsonl(value):
    return [json.loads(line) for line in path(value).read_text().splitlines() if line.strip()]


def write_jsonl(value, rows):
    p = path(value, True)
    temporary = p.with_name(p.name + '.partial')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    os.replace(temporary, p)


def write_npz(value, **arrays):
    p = path(value, True)
    temporary = p.with_name(p.name + '.partial')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, p)


def utc(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Acquisition times must contain an explicit UTC offset')
    return d.astimezone(timezone.utc)


def time_bound_seconds(bundle):
    a, b = utc(bundle['modis_start_utc']), utc(bundle['modis_end_utc'])
    c, d = utc(bundle['viirs_start_utc']), utc(bundle['viirs_end_utc'])
    if b <= a or d <= c:
        raise ValueError('Invalid acquisition interval')
    return max(abs((a-d).total_seconds()), abs((b-c).total_seconds()))


def validate_tiles(manifest, allow_synthetic=False, check_hashes=True, inspect_arrays=True):
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError('Empty tile manifest')
    seen, groups, sources = set(), {}, {}
    for r in rows:
        if r.get('schema') != SCHEMA or r.get('split') not in ('train', 'validation', 'test', 'inference'):
            raise ValueError('Unsupported tile schema or split')
        if r.get('dataset_kind') != 'nasa_native' and not allow_synthetic:
            raise ValueError('Synthetic data require --allow-synthetic; never report smoke results as NASA performance')
        if not r.get('event_group') or not r.get('region_id'):
            raise ValueError('Reviewed event_group and region_id are required')
        if r['tile_id'] in seen:
            raise ValueError('Duplicate tile_id')
        seen.add(r['tile_id'])
        for key, mapping in ((r['event_group'], groups), *[(s, sources) for s in r['source_ids']]):
            if key in mapping and mapping[key] != r['split']:
                raise ValueError(f'Train/validation/test leakage through group or source: {key}')
            mapping[key] = r['split']
        p = path(r['tile'])
        if check_hashes and sha256(p) != r['sha256']:
            raise ValueError(f'Tile checksum mismatch: {p.name}')
        if inspect_arrays:
            with np.load(p, allow_pickle=False) as z:
                validate_arrays(z)
    return rows


def validate_arrays(z):
    if any(k not in z for k in REQUIRED_KEYS):
        raise ValueError('Missing tile arrays')
    h, w = z['thermal'].shape[-2:]
    for key, channels, shape in [('thermal', 12, (h,w)), ('reflective_500', 10, (2*h,2*w)),
                                 ('reflective_250', 4, (4*h,4*w)), ('geometry', 14, (h,w))]:
        if z[key].shape != (channels, *shape) or not np.isfinite(z[key]).all():
            raise ValueError(f'Invalid native input array: {key}')
        if z[key].dtype != np.float32:
            raise ValueError(f'Native model arrays must be float32: {key}')
    q = z['target'].shape
    if len(q) != 2 or z['query_grid'].shape != (*q,2):
        raise ValueError('Invalid query dimensions')
    if z['query_grid'].dtype != np.float32 or not np.isfinite(z['query_grid']).all():
        raise ValueError('Query grid must be finite float32, never BF16/FP16')
    for key in ('target','weight','query_lat','query_lon'):
        if z[key].shape != q or not np.isfinite(z[key]).all():
            raise ValueError(f'Invalid target/query array: {key}')
    if z['query_lat'].dtype != np.float64 or z['query_lon'].dtype != np.float64:
        raise ValueError('Stored geographic coordinates must be float64')
    for key in ('target','native_target'):
        if np.any((z[key]<0)|(z[key]>1)):
            raise ValueError('Reference targets must be in [0,1]')
    for key in ('weight','native_weight'):
        if np.any(z[key]<0) or not np.isfinite(z[key]).all():
            raise ValueError('Invalid exposure weights')
    if z['native_target'].shape != (h,w) or z['native_weight'].shape != (h,w):
        raise ValueError('Invalid native auxiliary target dimensions')


def statistics(manifest, output, allow_synthetic=False):
    rows = validate_tiles(manifest, allow_synthetic)
    selected = [r for r in rows if r['split'] == 'train']
    if not selected:
        raise ValueError('No training partition; never fit normalizers on validation/test')
    sums = {k: np.zeros(n) for k,n in [('thermal',6),('reflective_500',5),('reflective_250',2)]}
    squares = {k: v.copy() for k,v in sums.items()}
    counts = {k: v.copy() for k,v in sums.items()}
    for row in selected:
        with np.load(path(row['tile'])) as z:
            for key in sums:
                n = len(sums[key]); values, valid = z[key][:n], z[key][n:]
                values = transform_values(key, values).astype(np.float64)
                sums[key] += (values*valid).sum((1,2))
                squares[key] += (values**2*valid).sum((1,2))
                counts[key] += valid.sum((1,2))
    result = {'schema':SCHEMA, 'manifest_sha256':sha256(manifest), 'training_tiles':len(selected),
              'fit_partition':'train', 'channels':{}, 'thermal_transform':'signed_log1p_radiance',
              'training_groups':sorted({r['event_group'] for r in selected}),
              'training_sources':sorted({s for r in selected for s in r['source_ids']}),
              'training_tile_ids':[r['tile_id'] for r in selected],
              'dataset_kinds':sorted({r['dataset_kind'] for r in selected})}
    for key in sums:
        n = counts[key]; mean = sums[key]/np.maximum(n,1)
        scale = np.sqrt(np.maximum(squares[key]/np.maximum(n,1)-mean**2, 1e-8))
        mean[n==0]=0; scale[n==0]=1
        result['channels'][key]={'mean':mean.tolist(),'scale':scale.tolist(),'valid_pixels':n.tolist()}
    write_json(output, result)
    return result


def transform_values(key, values):
    return np.sign(values)*np.log1p(np.abs(values)) if key == 'thermal' else values


def normalize(key, array, stats):
    if key not in stats['channels']:
        return array.astype(np.float32)
    info = stats['channels'][key]; n = len(info['mean'])
    values = transform_values(key,array[:n])
    valid = array[n:]
    values = (values-np.array(info['mean'])[:,None,None])/np.array(info['scale'])[:,None,None]
    return np.concatenate([values*valid,valid]).astype(np.float32)
