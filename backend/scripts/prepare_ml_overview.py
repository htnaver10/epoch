"""Aggregate the existing frozen scene cache for a small, backend-free globe.

No downloads, inference or training. Display cells summarize sampled queries;
they do not claim that the whole cell or the whole country was observed.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'frontend/public/data/ml-presentation'
OUTPUT = ROOT / 'frontend/public/data/ml-overview.json'
GRID = 0.25
COLUMNS = ['lat', 'lon', 'queries', 'mlp', 'modis', 'viirs', 'mlp_only', 'modis_only', 'score_max', 'scene_samples']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build():
    catalogue = json.loads((SOURCE / 'catalog.json').read_text(encoding='utf-8'))
    if not catalogue['complete']:
        raise ValueError('Source cache is incomplete')
    groups = defaultdict(dict)
    regions = {}
    sums = dict.fromkeys(['queries', 'mlp', 'modis', 'viirs'], 0)
    for number, scene in enumerate(catalogue['scenes'], 1):
        relative = f'scenes/{scene["id"]}/queries.npz'
        path = SOURCE / relative
        expected = catalogue['files'][relative]
        if isinstance(expected, dict):
            expected = expected['sha256']
        if sha(path) != expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
        with np.load(path, allow_pickle=False) as z:
            if str(z['model_version']) != catalogue['model']['version']:
                raise ValueError('Mixed model versions')
            lat, lon = z['lat'], z['lon']
            if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
                raise ValueError('Invalid source geolocation')
            # Use a common global grid so adjacent patches aggregate identically.
            y = np.floor((lat + 90) / GRID).astype(np.int64)
            x = np.floor((lon + 180) / GRID).astype(np.int64)
            ids, inverse = np.unique(y * 1440 + x, return_inverse=True)
            mlp, modis, viirs = z['mlp'], z['modis'], z['viirs']
            values = [np.bincount(inverse, minlength=len(ids))]
            values += [np.bincount(inverse, weights=a, minlength=len(ids)).astype(np.int64)
                       for a in (mlp, modis, viirs, mlp & ~modis, modis & ~mlp)]
            score = np.zeros(len(ids), dtype=np.float64)
            np.maximum.at(score, inverse, z['score'])
            for key, count in [('queries', len(lat)), ('mlp', int(mlp.sum())), ('modis', int(modis.sum())), ('viirs', int(viirs.sum()))]:
                if count != scene['counts'][key]:
                    raise ValueError(f'Scene counts differ: {scene["id"]}, {key}')
                sums[key] += count
        group = groups[(scene['date'], scene['split'], scene['group'])]
        for i, cell_id in enumerate(ids):
            key = int(cell_id)
            cell = group.setdefault(key, [int(cell_id // 1440) * GRID - 90 + GRID / 2,
                                          int(cell_id % 1440) * GRID - 180 + GRID / 2,
                                          0, 0, 0, 0, 0, 0, 0., 0])
            for j, value in enumerate(values):
                cell[j + 2] += int(value[i])
            cell[8] = max(cell[8], round(float(score[i]), 7))
            cell[9] += 1
        west, south, east, north = scene['bounds']
        bounds = regions.setdefault(scene['group'], [west, south, east, north])
        bounds[:] = [min(bounds[0], west), min(bounds[1], south), max(bounds[2], east), max(bounds[3], north)]
        if number % 200 == 0:
            print(f'Aggregated {number}/{len(catalogue["scenes"])} patches', flush=True)
    if sums['queries'] != catalogue['total_queries']:
        raise ValueError('Query total differs from frozen presentation')
    for key in ('mlp', 'modis', 'viirs'):
        if sums[key] != sum(s['counts'][key] for s in catalogue['scenes']):
            raise ValueError('Alert totals differ from frozen presentation')
    records = [{'date': date, 'split': split, 'group': group, 'cells': [cells[k] for k in sorted(cells)]}
               for (date, split, group), cells in sorted(groups.items())]
    unique_cells = {key for group in groups.values() for key in group}
    value = {'schema': 'epoch_ml_overview_v1', 'model': catalogue['model'],
             'source_catalogue_sha256': sha(SOURCE / 'catalog.json'),
             'grid_degrees': GRID, 'columns': COLUMNS, 'scene_count': catalogue['scene_count'],
             'source_totals': sums, 'display_cell_count': len(unique_cells),
             'dates': sorted({r['date'] for r in records}), 'regions': regions, 'records': records,
             'scope': 'Historical sampled patches in Canada and surrounding regions, not nationwide continuous coverage.',
             'count_semantics': 'Query samples from overlapping patches may repeat. These are not independent fires.',
             'new_inference': False, 'new_satellite_downloads': 0}
    OUTPUT.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    receipt = {'passed': True, 'model_version': value['model']['version'], 'source_scenes_verified': value['scene_count'],
               'source_totals': sums, 'display_cells': len(unique_cells), 'bytes': OUTPUT.stat().st_size,
               'sha256': sha(OUTPUT), 'new_inference': False, 'new_satellite_downloads': 0}
    destination = ROOT / 'backend/data/ml_overview'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    build()
