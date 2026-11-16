"""Read-only old shards plus a disjoint second set of real observations."""
import json
import numpy as np
from feature_extraction import extract
from settings import PARENT, V1, path, read, rows, sha, write, feature_config

def source_identity(source):
    from pathlib import Path
    name = Path(source).name; parts = name.split('.')
    return '.'.join(parts[:3]) if len(parts) >= 3 and parts[1].startswith('A') and len(parts[1]) == 8 and parts[2].isdigit() else name

def check_extra_rows(extra, parent):
    heldout = {source_identity(s) for r in parent if r['split'] != 'train' for s in r['source_ids']}
    heldout_groups = {r['event_group'] for r in parent if r['split'] != 'train'}
    seen = {(next(source_identity(s) for s in r['source_ids'] if source_identity(s).startswith('MYD021KM.')), tuple(r['source_window'])) for r in parent}
    for row in extra:
        sources = {source_identity(s) for s in row['source_ids']}
        if row['split'] != 'train' or row['dataset_kind'] != 'nasa_native':
            raise ValueError('Additional observations must be real native training-only data')
        if sources & heldout or row['event_group'] in heldout_groups:
            raise ValueError('Additional data overlap a held-out source/cohort')
        identity = (next(s for s in sources if s.startswith('MYD021KM.')), tuple(row['source_window']))
        if identity in seen:
            raise ValueError('Duplicated native observation window')
        seen.add(identity)
        if not 0 < row['interval_time_bound_seconds'] <= 900:
            raise ValueError('Invalid cross-sensor timing')
    return True

def verify_previous_features():
    for root, filename in [(PARENT, 'data/features/receipt.json'), (V1, 'data/feature_receipt.json')]:
        receipt = json.loads((root / filename).read_text(encoding='utf-8'))
        for name, expected in receipt['files'].items():
            if sha(root / name) != expected:
                raise ValueError('Previous frozen feature artifact changed')

def verify_training_features(regime):
    verify_previous_features()
    if regime == 'd2':
        for name, expected in read('data/feature_receipt.json')['files'].items():
            if sha(path(name)) != expected:
                raise ValueError('Second expansion feature artifact changed')

def prepare_features():
    prior = rows(PARENT / 'data/manifests/source_v3.jsonl') + rows(V1 / 'data/manifests/extra_train.jsonl')
    extra = rows(path('data/manifests/extra_train.jsonl')); check_extra_rows(extra, prior)
    inputs = {'parent_receipt': sha(PARENT / 'data/features/receipt.json'), 'v1_receipt': sha(V1 / 'data/feature_receipt.json'),
        'extra_manifest': sha(path('data/manifests/extra_train.jsonl')), 'expansion_receipt': sha(path('results/expansion_receipt.json')), 'config': sha(path('config.json'))}
    verify_previous_features()
    if path('data/feature_receipt.json').exists():
        receipt = read('data/feature_receipt.json')
        if receipt['inputs'] != inputs: raise ValueError('Feature input/configuration changed')
        for filename, expected in receipt['files'].items():
            if sha(path(filename)) != expected: raise ValueError('Second expansion feature cache changed')
        return receipt
    v1_receipt = json.loads((V1 / 'data/feature_receipt.json').read_text())
    names = json.loads((V1 / 'data/statistics.json').read_text())['feature_names']
    groups = sorted(set(v1_receipt['groups']) | {r['event_group'] for r in extra}); group_lookup = {g: i for i, g in enumerate(groups)}
    n = int(sum(r['valid_queries'] for r in extra))
    if not extra or n <= 0 or sum(r['positive_queries'] for r in extra) <= 0:
        raise ValueError('Second expansion must add genuine positive reference observations')
    path('data/features').mkdir(parents=True, exist_ok=True); files, arrays = {}, {}
    for key in ('X', 'y', 'w', 'group'):
        filename = f'data/features/extra_{key}.npy'; files[key] = filename
        arrays[key] = np.lib.format.open_memmap(path(filename + '.partial'), mode='w+',
            shape=(n, len(names)) if key == 'X' else (n,), dtype=np.int32 if key == 'group' else np.float32)
    position = 0
    for i, row in enumerate(extra):
        if sha(path(row['tile'])) != row['sha256']: raise ValueError('Second expansion native tile changed')
        with np.load(path(row['tile']), allow_pickle=False) as z:
            x, actual_names = extract({k: z[k] for k in ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')}, feature_config())
            mask = z['weight'] > 0
            if actual_names != names or not np.isin(z['weight'][mask], [1.]).all(): raise ValueError('Feature schema/known-query weights changed')
            end = position + int(mask.sum())
            if end - position != int(row['valid_queries']): raise ValueError('Manifest query count differs')
            for key, value in [('X', x[mask]), ('y', z['target'][mask]), ('w', z['weight'][mask]), ('group', group_lookup[row['event_group']])]:
                arrays[key][position:end] = value
        position = end
        if (i + 1) % 50 == 0: print(f'Second expansion features: {i + 1}/{len(extra)} tiles', flush=True)
    if position != n: raise ValueError('Second expansion exposure does not reconcile')
    for key, array in arrays.items():
        array.flush(); array._mmap.close(); path(files[key] + '.partial').replace(path(files[key]))
    shards = [np.load(root / name, mmap_mode='r', allow_pickle=False) for root, name in
        [(PARENT, 'data/features/train_X.npy'), (V1, 'data/features/extra_X.npy'), (path('data/features'), 'extra_X.npy')]]
    total = np.zeros(len(names), np.float64); squares = total.copy(); exposure = 0
    for source in shards:
        for start in range(0, len(source), 65536):
            x = np.asarray(source[start:start + 65536], dtype=np.float64)
            if not np.isfinite(x).all(): raise ValueError('Nonfinite training features')
            total += x.sum(0); squares += (x*x).sum(0); exposure += len(x)
    mean = total / exposure
    write('data/statistics.json', {'fit_partition': 'train_only_original_and_both_expansions', 'feature_names': names,
        'mean': mean.tolist(), 'scale': np.sqrt(np.maximum(squares/exposure - mean*mean, 1e-8)).tolist(), 'normalization_clip': 10., 'known_exposure': exposure})
    old_mapping = np.array([group_lookup[g] for g in v1_receipt['groups']], np.int32)
    for name, input_file in [('parent_group', 'data/features/parent_group.npy'), ('v1_group', 'data/features/extra_group.npy')]:
        np.save(path(f'data/features/{name}.npy'), old_mapping[np.load(V1 / input_file, allow_pickle=False)], allow_pickle=False)
    file_list = list(files.values()) + ['data/statistics.json', 'data/features/parent_group.npy', 'data/features/v1_group.npy']
    positive = int(sum(r['positive_queries'] for r in extra))
    receipt = {'inputs': inputs, 'groups': groups, 'new_tiles': len(extra), 'new_queries': n, 'new_positive_queries': positive,
        'prior_training_queries': len(shards[0]) + len(shards[1]), 'combined_training_queries': exposure,
        'combined_positive_queries': 8132 + positive, 'training_only_normalization': True, 'source_overlap_with_heldout': False,
        'files': {name: sha(path(name)) for name in file_list}}
    write('data/feature_receipt.json', receipt); return receipt

class TrainingData:
    def __init__(self, regime='d2'):
        if regime not in ('d1', 'd2'): raise ValueError('Unknown training data regime')
        self.regime = regime; root = V1 if regime == 'd1' else path('.')
        self.receipt = json.loads((root / 'data/feature_receipt.json').read_text()); self.stats = json.loads((root / 'data/statistics.json').read_text())
        locations = [(PARENT / 'data/features', 'train'), (V1 / 'data/features', 'extra')]
        if regime == 'd2': locations.append((path('data/features'), 'extra'))
        self.shards = [{k: np.load(folder / f'{prefix}_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('X', 'y')} for folder, prefix in locations]
        self.offsets = np.r_[0, np.cumsum([len(s['y']) for s in self.shards])]; self.y = np.concatenate([s['y'] for s in self.shards])
        group_files = [(V1, 'data/features/parent_group.npy'), (V1, 'data/features/extra_group.npy')] if regime == 'd1' else [
            (path('.'), 'data/features/parent_group.npy'), (path('.'), 'data/features/v1_group.npy'), (path('.'), 'data/features/extra_group.npy')]
        self.group = np.concatenate([np.load(folder / name, allow_pickle=False) for folder, name in group_files]); self.pools = []
        for index, name in enumerate(self.receipt['groups']):
            member = self.group == index; pos = np.flatnonzero(member & (self.y == 1)); neg = np.flatnonzero(member & (self.y == 0))
            if not len(pos) or not len(neg): raise ValueError(f'Cohort {name} requires both label classes')
            self.pools.append((pos, neg))
        self.mean = np.array(self.stats['mean'], np.float32); self.scale = np.array(self.stats['scale'], np.float32)
        self.prevalence = float(np.mean([len(p)/(len(p)+len(n)) for p, n in self.pools]))

    def draw(self, rng, size, positive_fraction):
        if not 0 < positive_fraction < 1: raise ValueError('Invalid sampling fraction')
        group_ids = rng.integers(len(self.pools), size=size); selected_positive = rng.random(size) < positive_fraction
        indices = np.empty(size, np.int64); correction = np.empty(size, np.float32)
        for group_id, (pos, neg) in enumerate(self.pools):
            prevalence = len(pos)/(len(pos)+len(neg))
            for positive, pool, probability, target_probability in [(True, pos, positive_fraction, prevalence), (False, neg, 1-positive_fraction, 1-prevalence)]:
                selected = (group_ids == group_id) & (selected_positive == positive)
                indices[selected] = pool[rng.integers(len(pool), size=int(selected.sum()))]; correction[selected] = target_probability/probability
        x = np.empty((size, len(self.mean)), np.float32)
        for i, shard in enumerate(self.shards):
            selected = (indices >= self.offsets[i]) & (indices < self.offsets[i+1]); x[selected] = shard['X'][indices[selected]-self.offsets[i]]
        return np.clip((x-self.mean)/self.scale, -10, 10), self.y[indices].astype(np.float32), correction, group_ids

def heldout(split):
    if split not in ('validation', 'test'): raise ValueError('Only original held-out cohorts may be evaluated')
    return {k: np.load(PARENT / f'data/features/{split}_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('X', 'y', 'w', 'modis', 'tile_index', 'flat_index')}
