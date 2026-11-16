"""Original feature-cache references plus genuinely new native observation windows."""
from __future__ import annotations

from collections import Counter
import numpy as np

from feature_extraction import extract
from settings import ROOT, PARENT, path, read, rows, sha, write, feature_config


def source_identity(source):
    # Different reprocessing filenames from the same acquisition still overlap.
    from pathlib import Path
    name = Path(source).name
    parts = name.split('.')
    if len(parts) >= 3 and parts[1].startswith('A') and len(parts[1]) == 8 and parts[2].isdigit():
        return '.'.join(parts[:3])
    return name


def check_extra_rows(extra, parent):
    heldout = {source_identity(s) for row in parent if row['split'] != 'train' for s in row['source_ids']}
    heldout_groups = {row['event_group'] for row in parent if row['split'] != 'train'}
    original_windows = {(next(source_identity(s) for s in r['source_ids'] if source_identity(s).startswith('MYD021KM.')),
                         tuple(r['source_window'])) for r in parent}
    seen = set()
    for row in extra:
        if row['split'] != 'train' or row['dataset_kind'] != 'nasa_native':
            raise ValueError('Additional observations must be real native training-only data')
        sources = {source_identity(s) for s in row['source_ids']}
        if sources & heldout or row['event_group'] in heldout_groups:
            raise ValueError('Additional training data overlap a held-out source/cohort')
        identity = (next(s for s in sources if s.startswith('MYD021KM.')), tuple(row['source_window']))
        if identity in original_windows or identity in seen:
            raise ValueError('Duplicated native observation window')
        seen.add(identity)
        if not 0 < row['interval_time_bound_seconds'] <= 900:
            raise ValueError('Unacceptable cross-sensor observation interval')
    return True


def prepare_features():
    parent = rows(PARENT / 'data/manifests/source_v3.jsonl')
    extra = rows(path('data/manifests/extra_train.jsonl'))
    check_extra_rows(extra, parent)
    receipt_path = 'data/feature_receipt.json'
    inputs = {'parent_receipt': sha(PARENT / 'data/features/receipt.json'),
              'extra_manifest': sha(path('data/manifests/extra_train.jsonl')),
              'expansion_receipt': sha(path('results/expansion_receipt.json')),
              'config': sha(path('config.json'))}
    parent_receipt = __import__('json').loads((PARENT / 'data/features/receipt.json').read_text())
    for filename, digest in parent_receipt['files'].items():
        if sha(PARENT / filename) != digest:
            raise ValueError('Original frozen feature cache changed')
    if path(receipt_path).exists():
        receipt = read(receipt_path)
        if receipt['inputs'] != inputs:
            raise ValueError('Feature data/configuration changed')
        for filename, digest in receipt['files'].items():
            if sha(path(filename)) != digest:
                raise ValueError('New feature cache changed')
        return receipt
    parent_stats = __import__('json').loads((PARENT / 'data/features/statistics.json').read_text())
    names = parent_stats['feature_names']
    n = int(sum(row['valid_queries'] for row in extra))
    if not extra or n <= 0 or sum(row['positive_queries'] for row in extra) <= 0:
        raise ValueError('Expansion must add genuine positive reference observations')
    path('data/features').mkdir(parents=True, exist_ok=True)
    arrays = {}
    files = {}
    for key in ('X', 'y', 'w', 'group'):
        file = f'data/features/extra_{key}.npy'
        shape = (n, len(names)) if key == 'X' else (n,)
        arrays[key] = np.lib.format.open_memmap(path(file + '.partial'), mode='w+', shape=shape,
                                               dtype=np.int32 if key == 'group' else np.float32)
        files[key] = file
    ledger = __import__('json').loads((PARENT / 'data/features/tiles.json').read_text())
    groups = sorted({r['event_group'] for r in ledger if r['split'] == 'train'} | {r['event_group'] for r in extra})
    group_lookup = {group: i for i, group in enumerate(groups)}
    position = 0
    for i, row in enumerate(extra):
        tile = path(row['tile'])
        if sha(tile) != row['sha256']:
            raise ValueError('Additional source tile checksum changed')
        with np.load(tile, allow_pickle=False) as z:
            x, actual_names = extract({k: z[k] for k in ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')}, feature_config())
            mask = z['weight'] > 0
            if actual_names != names or not np.isin(z['weight'][mask], [1.]).all():
                raise ValueError('Feature schema/exposure differs from parent')
            count = int(mask.sum())
            if count != int(row['valid_queries']):
                raise ValueError('Expansion query counts disagree')
            end = position + count
            arrays['X'][position:end] = x[mask]
            arrays['y'][position:end] = z['target'][mask]
            arrays['w'][position:end] = z['weight'][mask]
            arrays['group'][position:end] = group_lookup[row['event_group']]
        position = end
        if (i + 1) % 50 == 0:
            print(f'Additional features: {i + 1}/{len(extra)} tiles', flush=True)
    if position != n:
        raise ValueError('Additional feature exposure does not reconcile')
    for key, array in arrays.items():
        array.flush()
        array._mmap.close()
        path(files[key] + '.partial').replace(path(files[key]))
    parent_x = np.load(PARENT / 'data/features/train_X.npy', mmap_mode='r', allow_pickle=False)
    extra_x = np.load(path(files['X']), mmap_mode='r', allow_pickle=False)
    total = np.zeros(len(names), np.float64)
    squares = total.copy()
    exposure = 0
    for source in (parent_x, extra_x):
        for start in range(0, len(source), 65536):
            chunk = np.asarray(source[start:start + 65536], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite training features')
            total += chunk.sum(0)
            squares += (chunk * chunk).sum(0)
            exposure += len(chunk)
    mean = total / exposure
    scale = np.sqrt(np.maximum(squares / exposure - mean * mean, 1e-8))
    stats = {'fit_partition': 'train_only_parent_and_expansion', 'feature_names': names,
             'mean': mean.tolist(), 'scale': scale.tolist(), 'normalization_clip': 10.0,
             'known_exposure': exposure}
    write('data/statistics.json', stats)
    parent_groups = np.full(len(parent_x), -1, dtype=np.int32)
    for row in ledger:
        if row['split'] == 'train':
            parent_groups[row['start']:row['end']] = group_lookup[row['event_group']]
    if (parent_groups < 0).any():
        raise ValueError('Parent training cohort ledger is incomplete')
    np.save(path('data/features/parent_group.npy'), parent_groups, allow_pickle=False)
    file_list = list(files.values()) + ['data/statistics.json', 'data/features/parent_group.npy']
    positive = sum(r['positive_queries'] for r in extra)
    receipt = {'inputs': inputs, 'groups': groups, 'new_tiles': len(extra), 'new_queries': n,
               'new_positive_queries': positive, 'parent_training_queries': len(parent_x),
               'training_only_normalization': True, 'source_overlap_with_heldout': False,
               'files': {file: sha(path(file)) for file in file_list}}
    write(receipt_path, receipt)
    return receipt


class TrainingData:
    def __init__(self):
        self.receipt = read('data/feature_receipt.json')
        self.stats = read('data/statistics.json')
        self.shards = []
        for root, prefix in ((PARENT / 'data/features', 'train'), (path('data/features'), 'extra')):
            self.shards.append({k: np.load(root / f'{prefix}_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('X', 'y')})
        self.boundary = len(self.shards[0]['y'])
        self.y = np.concatenate([s['y'] for s in self.shards])
        self.group = np.concatenate([np.load(path('data/features/parent_group.npy'), allow_pickle=False),
                                     np.load(path('data/features/extra_group.npy'), allow_pickle=False)])
        self.pools = []
        for index, name in enumerate(self.receipt['groups']):
            membership = self.group == index
            positives = np.flatnonzero(membership & (self.y == 1))
            negatives = np.flatnonzero(membership & (self.y == 0))
            if not len(positives) or not len(negatives):
                raise ValueError(f'Cohort {name} needs both classes for stratified sampling')
            self.pools.append((positives, negatives))
        self.mean = np.array(self.stats['mean'], np.float32)
        self.scale = np.array(self.stats['scale'], np.float32)
        self.prevalence = float(np.mean([len(p) / (len(p) + len(n)) for p, n in self.pools]))

    def draw(self, rng, size, positive_fraction):
        if not 0 < positive_fraction < 1:
            raise ValueError('Invalid sampling fraction')
        group_ids = rng.integers(len(self.pools), size=size)
        selected_positive = rng.random(size) < positive_fraction
        indices = np.empty(size, np.int64)
        correction = np.empty(size, np.float32)
        for group_id, (pos, neg) in enumerate(self.pools):
            prevalence = len(pos) / (len(pos) + len(neg))
            for positive, pool, probability, target_probability in (
                    (True, pos, positive_fraction, prevalence),
                    (False, neg, 1 - positive_fraction, 1 - prevalence)):
                selected = (group_ids == group_id) & (selected_positive == positive)
                indices[selected] = pool[rng.integers(len(pool), size=int(selected.sum()))]
                correction[selected] = target_probability / probability
        x = np.empty((size, len(self.mean)), np.float32)
        for shard_id, shard in enumerate(self.shards):
            selected = indices < self.boundary if shard_id == 0 else indices >= self.boundary
            local = indices[selected] - (0 if shard_id == 0 else self.boundary)
            x[selected] = shard['X'][local]
        x = np.clip((x - self.mean) / self.scale, -10, 10)
        return x, self.y[indices].astype(np.float32), correction, group_ids


def heldout(split):
    if split not in ('validation', 'test'):
        raise ValueError('Heldout loader only permits validation/test')
    return {key: np.load(PARENT / f'data/features/{split}_{key}.npy', mmap_mode='r', allow_pickle=False)
            for key in ('X', 'y', 'w', 'modis', 'tile_index', 'flat_index')}
