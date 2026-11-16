"""Independently reconcile every cached score, exported alert and rendered asset."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from prepare_ml_presentation import ROOT, PUBLIC, WORK, read, sha, write


def validate():
    catalogue = read(PUBLIC / 'catalog.json')
    inputs = read(WORK / 'inputs.json')
    scenes = catalogue['scenes']
    if not catalogue['complete'] or len(scenes) != catalogue['scene_count']:
        raise ValueError('Incomplete catalogue')
    if {s['id'] for s in scenes} != {r['tile_id'] for r in inputs['rows']}:
        raise ValueError('Prepared snapshot is not fully represented')
    totals, renders = {}, 0
    cutoff = catalogue['model']['raw_logit_cutoff']
    offset = catalogue['model']['logit_offset']
    for relative, expected in catalogue['files'].items():
        path = (PUBLIC / relative).resolve()
        if not path.is_relative_to(PUBLIC.resolve()) or sha(path) != expected:
            raise ValueError('Changed or unsafe asset: ' + relative)
    for index, scene in enumerate(scenes):
        directory = PUBLIC / 'scenes' / scene['id']
        with np.load(directory / 'queries.npz', allow_pickle=False) as values:
            if str(values['model_version']) != catalogue['model']['version']:
                raise ValueError('Mixed model versions')
            logits, scores = values['raw_logit'], values['score']
            mlp, modis, viirs = values['mlp'], values['modis'], values['viirs']
            for key in ('query_index', 'lat', 'lon', 'score', 'mlp', 'modis', 'viirs'):
                if len(values[key]) != scene['counts']['queries']:
                    raise ValueError('Incomplete query output')
            if not np.isfinite(logits).all() or not np.isfinite(scores).all():
                raise ValueError('Nonfinite scores')
            np.testing.assert_array_equal(mlp, logits.astype(np.float64) >= cutoff)
            np.testing.assert_allclose(scores, np.exp(-np.logaddexp(0., -(logits.astype(np.float64) + offset))), atol=3e-8, rtol=1e-7)
            actual = {'queries': len(logits), 'mlp': int(mlp.sum()), 'modis': int(modis.sum()), 'viirs': int(viirs.sum()),
                      'TP': int((mlp & viirs).sum()), 'FP': int((mlp & ~viirs).sum()), 'FN': int((~mlp & viirs).sum()),
                      'mlp_only': int((mlp & ~modis).sum()), 'modis_only': int((modis & ~mlp).sum())}
            if scene['counts'] != actual:
                raise ValueError('Incorrect catalogue counts')
            data = read(directory / 'predictions.geojson')
            if data['scene'] != scene:
                raise ValueError('GeoJSON metadata mismatch')
            selected = np.flatnonzero(mlp | modis | viirs)
            if len(data['features']) != len(selected):
                raise ValueError('Incomplete GeoJSON alert union')
            for feature, i in zip(data['features'], selected):
                p = feature['properties']
                if p['query_index'] != int(values['query_index'][i]) or p['mlp_detected'] != bool(mlp[i]) or p['modis_detected'] != bool(modis[i]) or p['viirs_reference'] != bool(viirs[i]):
                    raise ValueError('Exported flag/index mismatch')
                np.testing.assert_allclose(feature['geometry']['coordinates'], [values['lon'][i], values['lat'][i]], atol=0, rtol=0)
                np.testing.assert_allclose(p['mlp_score'], scores[i], atol=3e-8, rtol=1e-7)
        for layer in ('compare', 'mlp', 'modis', 'viirs', 'difference'):
            with Image.open(directory / f'{layer}.png') as image:
                expected_size = (1800, 1300) if layer == 'compare' else (900, 650)
                if image.size != expected_size:
                    raise ValueError('Incorrect rendered dimensions')
                image.verify()
                renders += 1
        totals.setdefault(scene['split'], Counter()).update(actual)
        if index % 200 == 0:
            print(f'Verified {index + 1}/{len(scenes)} scenes', flush=True)
    for split, counts in totals.items():
        if dict(counts) != catalogue['splits'][split]['counts']:
            raise ValueError('Incorrect split totals')
    benchmark = catalogue['benchmark']['wide_frozen_policy']
    for key, metric_key in [('TP', 'true_positives'), ('FP', 'false_positives'), ('FN', 'false_negatives')]:
        if totals['test'][key] != benchmark[metric_key]:
            raise ValueError('Sealed benchmark mismatch')
    report = {'passed': True, 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'model_version': catalogue['model']['version'], 'scenes': len(scenes),
              'queries': sum(c['queries'] for c in totals.values()), 'rendered_images': renders,
              'hashed_assets': len(catalogue['files']), 'benchmark': {k: totals['test'][k] for k in ('TP', 'FP', 'FN')},
              'checks': ['All snapshot scenes included', 'All asset hashes match', 'Every eligible query score retained',
                         'Raw-logit threshold and adjusted-score arithmetic', 'GeoJSON coordinates/flags/scores reconcile',
                         'Every rendered image decodes at full dimensions', 'All split totals reconcile', 'Frozen benchmark matches exactly']}
    write(ROOT / 'data-visualization/ml-presentation/verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    validate()
