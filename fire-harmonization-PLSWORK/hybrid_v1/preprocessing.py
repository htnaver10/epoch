"""Match reflectance support to the MODIS thermal grid without using VIIRS values."""
import numpy as np
from common import ROOT, path, read, write, config, sha, load, save_array
from firepls.features import extract, query_indices
from firepls.common import config as parent_config, validate_parent
from firepls.prepare import load_split, prepare as verify_parent_features

INPUTS = ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')

def pool_valid(values, valid, scale):
    """Valid-only aligned block mean; zero support is explicit, never treated as clear."""
    h, w = values.shape
    if h % scale or w % scale or valid.shape != values.shape:
        raise ValueError('Invalid aligned reflectance dimensions')
    good = (valid > 0) & np.isfinite(values)
    a = np.where(good, values, 0).reshape(h//scale, scale, w//scale, scale)
    n = good.reshape(h//scale, scale, w//scale, scale).sum((1, 3))
    total = a.sum((1, 3), dtype=np.float64)
    mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    return mean, n / float(scale*scale)

def features(inputs, modis):
    """Strict MODIS-only inputs; no VIIRS, reference mask or target accepted."""
    if set(inputs) != set(INPUTS):
        raise ValueError('Hybrid features accept only MODIS imagery and query geometry')
    base, names = extract(inputs, parent_config())
    shape = inputs['query_grid'].shape[:2]
    if modis.shape != shape or not np.isin(modis, [0, 1]).all():
        raise ValueError('MODIS detection input must be a binary query-aligned grid')
    h, w = inputs['thermal'].shape[-2:]
    iy, ix, _, _ = query_indices(inputs['query_grid'], h, w)
    arrays = [base]; names = list(names)
    for key, scale, n, bands in [('reflective_500', 2, 5, (3,4,5,6,7)),
                                  ('reflective_250', 4, 2, (1,2))]:
        arr = inputs[key]
        for i, band in enumerate(bands):
            mean, support = pool_valid(arr[i], arr[n+i], scale)
            m, f = mean[iy, ix], support[iy, ix]
            fine = base[..., names.index(f'reflectance_{band}')]
            fine_valid = base[..., names.index(f'reflectance_valid_{band}')] > 0
            delta = np.where(fine_valid & (f > 0), fine - m, 0)
            arrays += [np.stack([m, delta, f], -1)]
            names += [f'cell_mean_reflectance_{band}', f'cell_contrast_reflectance_{band}',
                      f'cell_valid_fraction_{band}']
    arrays += [modis[..., None]]; names += ['operational_modis_detection']
    x = np.concatenate(arrays, -1).astype(np.float32)
    if x.shape[-1] != 118 or not np.isfinite(x).all():
        raise ValueError('Invalid hybrid features')
    return x, names, iy*w + ix

def prepare():
    receipt = verify_parent_features()
    if path('data/receipt.json').exists():
        saved = read('data/receipt.json')
        for name, digest in saved['files'].items():
            if sha(path(name)) != digest: raise ValueError('Hybrid data changed: ' + name)
        return saved
    parent = validate_parent()
    ledger = __import__('json').loads((ROOT/'data/features/tiles.json').read_text())
    sources = {split: set() for split in ('train','validation','test')}
    for row in parent:
        sources[row['split']].update(row['source_ids'])
        if row['interval_time_bound_seconds'] > 900: raise ValueError('Temporal matching failed')
        if row['max_inverse_residual_m'] > 20.001: raise ValueError('Geolocation consistency failed')
    if any(sources[a] & sources[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Source overlap across splits')
    path('data').mkdir(parents=True, exist_ok=True)
    arrays = {}; raw = {}; ids = {}; observed = {}; next_id = {}
    for split, count in receipt['counts'].items():
        raw[split] = load_split(split)
        arrays[split] = np.lib.format.open_memmap(path(f'data/{split}_X.npy.partial'), mode='w+',
                                                  dtype=np.float32, shape=(count,118))
        ids[split] = np.empty(count, np.int32); observed[split] = {}; next_id[split] = 0
    for index, (row, item) in enumerate(zip(parent, ledger, strict=True)):
        if row['tile_id'] != item['tile_id']: raise ValueError('Tile order mismatch')
        split = row['split']; begin, end = item['start'], item['end']
        with np.load(ROOT/row['tile'], allow_pickle=False) as z:
            x, names, local_ids = features({k:z[k] for k in INPUTS}, z['modis_baseline'])
            flat = raw[split]['flat_index'][begin:end]
            values = x.reshape(-1,118)[flat]
            np.testing.assert_array_equal(values[:,:96], raw[split]['X'][begin:end])
            np.testing.assert_array_equal(values[:,-1], raw[split]['modis'][begin:end])
            cell = local_ids.ravel()[flat]
        arrays[split][begin:end] = values
        unique, inverse = np.unique(cell, return_inverse=True)
        mapped = []
        # Deduplicate native cells by source swath and absolute scan row/column across tiles.
        r0,c0,_,width = row['source_window']
        for c in unique:
            key = (row['source_ids'][0], r0 + int(c)//width, c0 + int(c)%width)
            if key not in observed[split]:
                observed[split][key] = next_id[split]; next_id[split] += 1
            mapped.append(observed[split][key])
        ids[split][begin:end] = np.asarray(mapped, np.int32)[inverse]
        if (index+1)%100 == 0: print(f'Hybrid alignment: {index+1}/{len(parent)} tiles', flush=True)
    files = []; audit = {}
    for split in arrays:
        arrays[split].flush(); arrays[split]._mmap.close()
        dest = f'data/{split}_X.npy'; path(dest+'.partial').replace(path(dest)); files.append(dest)
        dest = f'data/{split}_cell.npy'; save_array(dest, ids[split]); files.append(dest)
        counts = np.bincount(ids[split]); positive = np.bincount(ids[split], weights=raw[split]['y'])
        audit[split] = {'known_queries':len(ids[split]), 'positive_queries':int(raw[split]['y'].sum()),
                        'observed_native_cells':len(counts), 'cells_with_reference_fire':int((positive>0).sum()),
                        'mixed_label_cells':int(((positive>0)&(positive<counts)).sum()),
                        'median_known_queries_per_cell':float(np.median(counts)),
                        'max_known_queries_per_cell':int(counts.max())}
    result = {'feature_names':names, 'feature_count':len(names), 'files':{f:sha(path(f)) for f in files},
              'parent_feature_receipt':config()['feature_receipt_sha256'], 'split_audit':audit,
              'max_pair_interval_seconds':max(r['interval_time_bound_seconds'] for r in parent),
              'max_inverse_consistency_residual_m':max(r['max_inverse_residual_m'] for r in parent),
              'source_overlap_across_splits':False, 'new_data_downloads':False,
              'cell_support':'nearest native MODIS cell by inverse geolocation, not exact footprint overlap',
              'reference':'unmodified VIIRS nominal/high fire vs observed-land labels; unknown is excluded'}
    write('data/receipt.json', result); return result

def aggregate_cells(score, y, baseline, cell):
    """Any-reference-fire and max model score on exactly the same observed cell support."""
    n = int(cell.max())+1
    z = np.full(n, -np.inf); target = np.zeros(n); modis = np.zeros(n)
    np.maximum.at(z, cell, score); np.maximum.at(target, cell, y); np.maximum.at(modis, cell, baseline)
    if not np.isfinite(z).all(): raise ValueError('Empty native cell in aggregation')
    return z, target, np.ones(n), modis
