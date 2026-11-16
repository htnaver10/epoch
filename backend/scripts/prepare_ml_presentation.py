"""Freeze all locally prepared scenes, infer the completed wide MLP, and render them.

No training or satellite downloads. Completed held-out CUDA BF16 predictions are
reused exactly; all other eligible queries are inferred with the frozen model.
An immutable input snapshot makes resumable builds independent of ongoing work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'fire-harmonization-PLSWORK'
WIDE = PARENT / 'wide_diverse_v1'
WORK = ROOT / 'backend/data/ml_presentation'
PUBLIC = ROOT / 'frontend/public/data/ml-presentation'
sys.path.insert(0, str(WIDE))
sys.path.insert(0, str(WIDE / 'vendor'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot(refresh=False):
    path = WORK / 'inputs.json'
    if path.exists() and not refresh:
        return read(path)
    sources = [(PARENT, 'data/manifests/source_v3.jsonl', 'original'),
               (WIDE, 'data/manifests/extra_train.jsonl', 'expansion')]
    second = PARENT / 'wide_diverse_v2'
    for name in ('extra_train.jsonl', 'extra_train.partial.jsonl'):
        if (second / 'data/manifests' / name).exists():
            sources.append((second, 'data/manifests/' + name, 'additional'))
            break
    rows, manifests, seen = [], [], set()
    for base, relative, cohort in sources:
        data = (base / relative).read_bytes()
        # A final unterminated record may still be being written by preparation.
        lines = data.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines.pop()
        frozen = b''.join(lines)
        manifest = {'path': (base / relative).relative_to(ROOT).as_posix(),
                    'sha256': hashlib.sha256(frozen).hexdigest(), 'records': len(lines),
                    'partial_snapshot': '.partial.' in relative}
        manifests.append(manifest)
        position = 0
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['tile_id'] in seen:
                raise ValueError('Duplicate prepared tile: ' + row['tile_id'])
            seen.add(row['tile_id'])
            count = int(row['valid_queries'])
            rows.append({**row, 'root': base.relative_to(ROOT).as_posix(), 'cohort': cohort,
                         'presentation_split': 'additional' if cohort == 'additional' else row['split'],
                         'feature_start': position, 'feature_end': position + count})
            position += count
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'manifests': manifests,
              'checkpoint_sha256': sha(WIDE / 'models/inference.pt'),
              'policy_sha256': sha(WIDE / 'results/operating_policy.json'), 'rows': rows}
    write(path, result)
    return result


def font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


BG, PANEL, SOFT = '#090f18', '#101c29', '#94a9bc'
COLORS = {'mlp': '#55ccff', 'modis': '#ffb224', 'viirs': '#bd9bff', 'agreement': '#e7f1fa'}
LABELS = {'mlp': '50.6M MLP', 'modis': 'Original MODIS', 'viirs': 'VIIRS reference', 'difference': 'MLP / MODIS difference'}
FONTS = {size: font(size) for size in (13, 16, 20, 26, 32)}


def render(scene, arrays, destination):
    """Render every eligible center in a local geographic frame, without a network basemap."""
    width, height = 900, 650
    west, south, east, north = scene['bounds']
    cosine = max(.05, np.cos(np.deg2rad((south + north) / 2)))
    dx, dy = max((east - west) * cosine, .001), max(north - south, .001)
    scale = min(770 / dx, 430 / dy)
    left, top = (width - dx * scale) / 2, 115 + (430 - dy * scale) / 2
    x = left + (arrays['lon'] - west) * cosine * scale
    y = top + (north - arrays['lat']) * scale
    background = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(background)
    draw.rectangle((left, top, left + dx * scale, top + dy * scale), fill=PANEL, outline='#385065')
    # Drawing onto the actual query positions preserves gaps; the bounding box
    # and geographic grid do not imply observed coverage outside these marks.
    pixels = np.asarray(background).copy()
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    pixels[yi, xi] = (53, 74, 92)
    background = Image.fromarray(pixels)
    draw = ImageDraw.Draw(background)
    for fraction in (0, .5, 1):
        gx, gy = left + dx * scale * fraction, top + dy * scale * fraction
        draw.line((gx, top, gx, top + dy * scale), fill='#24394a')
        draw.line((left, gy, left + dx * scale, gy), fill='#24394a')
        draw.text((gx, top + dy * scale + 8), f'{west + (east-west)*fraction:.2f}°', font=FONTS[13], fill=SOFT, anchor='mt')
        draw.text((left - 8, gy), f'{north - (north-south)*fraction:.2f}°', font=FONTS[13], fill=SOFT, anchor='rm')
    draw.text((34, 605), 'Gray = reference-eligible query centers. Other locations are unavailable.', font=FONTS[16], fill=SOFT)
    panels = []
    for layer in LABELS:
        panel = background.copy()
        draw = ImageDraw.Draw(panel)
        draw.text((34, 22), LABELS[layer], font=FONTS[26], fill='#edf5fc')
        count = scene['counts'][layer] if layer != 'difference' else scene['counts']['mlp_only'] + scene['counts']['modis_only']
        draw.text((34, 62), f'{count:,} positive marks  |  {scene["date"]}  |  {scene["split"].upper()}', font=FONTS[16], fill=SOFT)
        mask = arrays[layer] if layer != 'difference' else arrays['mlp'] != arrays['modis']
        for index in np.flatnonzero(mask):
            color = COLORS[layer] if layer != 'difference' else COLORS['mlp' if arrays['mlp'][index] else 'modis']
            draw.ellipse((x[index]-3, y[index]-3, x[index]+3, y[index]+3), fill=color, outline='#e6f2fc')
        if layer == 'difference':
            draw.text((34, 567), 'Cyan: MLP only     Amber: MODIS only', font=FONTS[16], fill=SOFT)
        panel.save(destination / f'{layer}.png', optimize=False)
        panels.append(panel)
    comparison = Image.new('RGB', (1800, 1300), BG)
    for i, panel in enumerate(panels):
        comparison.paste(panel, ((i % 2) * width, (i // 2) * height))
    comparison.save(destination / 'compare.png', optimize=False)


def metric(counts):
    tp, fp, fn = (counts[k] for k in ('TP', 'FP', 'FN'))
    return {'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            'TP': tp, 'FP': fp, 'FN': fn}


def build(refresh=False):
    import torch
    from training import load_model, predict
    from feature_extraction import extract
    from settings import feature_config

    frozen = snapshot(refresh)
    if frozen['checkpoint_sha256'] != sha(WIDE / 'models/inference.pt') or frozen['policy_sha256'] != sha(WIDE / 'results/operating_policy.json'):
        raise ValueError('Presentation snapshot belongs to different weights or policy')
    policy = read(WIDE / 'results/operating_policy.json')
    if policy['model_sha256'] != frozen['checkpoint_sha256']:
        raise ValueError('Policy/model mismatch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model, saved = load_model(device)
    version = 'wide-mlp-' + frozen['checkpoint_sha256'][:12]
    ledger = {r['tile_id']: r for r in read(PARENT / 'data/features/tiles.json')}
    feature_files = {split: np.load(PARENT / f'data/features/{split}_X.npy', mmap_mode='r', allow_pickle=False)
                     for split in ('train', 'validation', 'test')}
    feature_files['expansion'] = np.load(WIDE / 'data/features/extra_X.npy', mmap_mode='r', allow_pickle=False)
    # Verify the exact files that feed inference, rather than trusting their names.
    receipts = [(PARENT, read(PARENT / 'data/features/receipt.json')),
                (WIDE, read(WIDE / 'data/feature_receipt.json'))]
    for base, receipt in receipts:
        for relative, expected in receipt['files'].items():
            if relative.endswith('_X.npy') or relative.endswith('tiles.json'):
                if sha(base / relative) != expected:
                    raise ValueError('Frozen feature cache changed: ' + relative)
    heldout = {split: np.load(WIDE / f'results/{split}_predictions.npz', allow_pickle=False)
               for split in ('validation', 'test')}
    evaluation = read(WIDE / 'results/evaluation.json')
    scenes, files, totals = [], {}, {}
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(frozen['rows']):
        scene_id = row['tile_id']
        directory = PUBLIC / 'scenes' / scene_id
        receipt_path = WORK / 'receipts' / f'{scene_id}.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['version'] != version or receipt['source_sha256'] != row['sha256']:
                raise ValueError('Invalid resume receipt')
            for rel, digest in receipt['files'].items():
                if sha(PUBLIC / rel) != digest:
                    raise ValueError('Presentation asset changed: ' + rel)
            scene = receipt['scene']
            files.update(receipt['files'])
        else:
            tile = ROOT / row['root'] / row['tile']
            if sha(tile) != row['sha256']:
                raise ValueError('Source tile changed: ' + str(tile))
            with np.load(tile, allow_pickle=False) as z:
                known = z['weight'] > 0
                indices = np.flatnonzero(known)
                lat, lon = z['query_lat'][known], z['query_lon'][known]
                viirs, modis = z['target'][known] > .5, z['modis_baseline'][known] > .5
                if row['cohort'] == 'original':
                    item = ledger[scene_id]
                    start, end = item['start'], item['end']
                    features = feature_files[row['split']][start:end]
                elif row['cohort'] == 'expansion':
                    start, end = row['feature_start'], row['feature_end']
                    features = feature_files['expansion'][start:end]
                else:
                    feature_grid, names = extract({key: z[key] for key in ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')}, feature_config())
                    if names != saved['stats']['feature_names']:
                        raise ValueError('Feature schema mismatch')
                    features = feature_grid[known]
                if row['cohort'] == 'original' and row['split'] in heldout:
                    prediction = heldout[row['split']]
                    np.testing.assert_array_equal(prediction['flat_index'][start:end], indices)
                    np.testing.assert_array_equal(prediction['target'][start:end] > .5, viirs)
                    logits = prediction['logits'][start:end]
                    precision = 'saved CUDA BF16 benchmark logits'
                else:
                    logits = predict(model, features, saved['stats'], 2048)
                    precision = 'CUDA BF16' if device == 'cuda' else 'CPU float32'
            if len(indices) != int(row['valid_queries']) or len(logits) != len(indices):
                raise ValueError('Query counts do not reconcile')
            if not all(np.isfinite(a).all() for a in (lat, lon, logits)) or (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
                raise ValueError('Invalid inference output/geolocation')
            mlp = logits.astype(np.float64) >= policy['raw_logit_cutoff']
            scores = np.exp(-np.logaddexp(0., -(logits.astype(np.float64) + policy['logit_offset'])))
            counts = {'queries': len(indices), 'mlp': int(mlp.sum()), 'modis': int(modis.sum()), 'viirs': int(viirs.sum()),
                      'TP': int((mlp & viirs).sum()), 'FP': int((mlp & ~viirs).sum()), 'FN': int((~mlp & viirs).sum()),
                      'mlp_only': int((mlp & ~modis).sum()), 'modis_only': int((modis & ~mlp).sum())}
            scene = {'id': scene_id, 'date': row['modis_start_utc'][:10], 'observed_at': row['modis_start_utc'],
                     'split': row['presentation_split'], 'cohort': row['cohort'], 'group': row['event_group'],
                     'bounds': [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())],
                     'counts': counts, 'model_version': version, 'source_sha256': row['sha256'],
                     'query_shape': row['query_shape'], 'precision': precision,
                     'pair_time_bound_seconds': row['interval_time_bound_seconds'],
                     'source_granules': [Path(s).name for s in row['source_ids']]}
            directory.mkdir(parents=True, exist_ok=True)
            # All query scores are retained, including background and nonalerts.
            with (directory / 'queries.npz.partial').open('wb') as stream:
                np.savez_compressed(stream, query_index=indices, lat=lat, lon=lon, raw_logit=logits,
                                    score=scores.astype(np.float32), mlp=mlp, modis=modis, viirs=viirs,
                                    model_version=np.array(version), raw_logit_cutoff=np.array(policy['raw_logit_cutoff']))
            (directory / 'queries.npz.partial').replace(directory / 'queries.npz')
            features = [{'type': 'Feature', 'id': f'{scene_id}:{int(indices[i])}',
                         'geometry': {'type': 'Point', 'coordinates': [float(lon[i]), float(lat[i])]},
                         'properties': {'query_index': int(indices[i]), 'mlp_score': float(scores[i]),
                                        'mlp_detected': bool(mlp[i]), 'modis_detected': bool(modis[i]), 'viirs_reference': bool(viirs[i])}}
                        for i in np.flatnonzero(mlp | modis | viirs)]
            write(directory / 'predictions.geojson', {'type': 'FeatureCollection', 'scene': scene, 'model_version': version,
                  'coverage_kind': 'reference_matched_query_centers', 'features': features,
                  'note': 'Alert union only. queries.npz retains all eligible scores. Outside sampled coverage is unavailable.'})
            render(scene, {'lat': lat, 'lon': lon, 'mlp': mlp, 'modis': modis, 'viirs': viirs}, directory)
            write(directory / 'scene.json', scene)
            scene_files = {p.relative_to(PUBLIC).as_posix(): sha(p) for p in directory.iterdir() if p.is_file()}
            write(receipt_path, {'version': version, 'source_sha256': row['sha256'], 'scene': scene, 'files': scene_files})
            files.update(scene_files)
        scenes.append(scene)
        split = scene['split']
        totals.setdefault(split, Counter()).update(scene['counts'])
        if number % 20 == 0 or number + 1 == len(frozen['rows']):
            progress = {'status': 'building', 'scenes_complete': number + 1, 'scene_count': len(frozen['rows']),
                        'queries_complete': sum(s['counts']['queries'] for s in scenes), 'model_version': version}
            write(WORK / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    # The test totals must exactly reproduce the sealed evaluation, not a new
    # threshold or scores rounded for display.
    expected = evaluation['test']['wide_frozen_policy']
    for field, original in [('TP', 'true_positives'), ('FP', 'false_positives'), ('FN', 'false_negatives')]:
        if totals['test'][field] != expected[original]:
            raise ValueError('Rendered benchmark differs from frozen evaluation')
    catalogue = {'schema': 'epoch_ml_presentation_v1', 'model': {'name': 'Wide diverse MLP', 'version': version,
                 'parameters': 50567169, 'feature_count': 96, 'checkpoint_sha256': frozen['checkpoint_sha256'],
                 'policy_sha256': frozen['policy_sha256'], 'selected_epoch': saved['epoch'] + 1,
                 'raw_logit_cutoff': policy['raw_logit_cutoff'], 'logit_offset': policy['logit_offset']},
                 'created_utc': frozen['created_utc'], 'manifests': frozen['manifests'],
                 'scene_count': len(scenes), 'total_queries': sum(s['counts']['queries'] for s in scenes),
                 'render_count': 5 * len(scenes), 'scenes': sorted(scenes, key=lambda s: (s['date'], s['id'])),
                 'splits': {split: {'scenes': sum(s['split'] == split for s in scenes), 'counts': dict(counts), 'metrics': metric(counts)} for split, counts in totals.items()},
                 'benchmark': evaluation['test'], 'training_history': read(WIDE / 'results/training_history.json'),
                 'limitations': ['Research results; not live monitoring or an operational fire product.',
                     'VIIRS supplies reference labels, not independent field truth. Query centers are not independent wildfire incidents.',
                     'Training and validation scenes are not test evidence. The 84-patch benchmark has been examined previously.',
                     'Additional windows share training acquisitions; they are not a new independent test.',
                     'Only reference-eligible query centers are included. Outside coverage is unavailable, not fire-free.',
                     'Scores are validation-adjusted and are not guaranteed real-world probabilities. MODIS measured resolution is unchanged.'],
                 'files': files, 'complete': True, 'new_training': False, 'new_satellite_downloads': 0}
    write(PUBLIC / 'catalog.json', catalogue)
    export_index(catalogue)
    write(WORK / 'progress.json', {'status': 'complete', 'scenes_complete': len(scenes), 'queries_complete': catalogue['total_queries'], 'renders': catalogue['render_count'], 'benchmark_reconciled': True})
    print(json.dumps(read(WORK / 'progress.json')), flush=True)


def export_index(catalogue):
    fields = ['id', 'date', 'split', 'cohort', 'group', 'queries', 'mlp', 'modis', 'viirs', 'TP', 'FP', 'FN']
    with (PUBLIC / 'scene-summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene in catalogue['scenes']:
            writer.writerow({**{k: scene[k] for k in fields[:5]}, **{k: scene['counts'][k] for k in fields[5:]}})
    cards = []
    for scene in sorted(catalogue['scenes'], key=lambda s: (s['split'] != 'test', -s['counts']['viirs'])):
        scene_id = scene['id']
        cards.append(f'<article><h2>{html.escape(scene["date"])} · {html.escape(scene["split"].upper())} · {scene_id}</h2>'
                     f'<p>{scene["counts"]["queries"]:,} eligible queries · {scene["counts"]["mlp"]:,} MLP alerts · {scene["counts"]["viirs"]:,} reference positives</p>'
                     f'<img loading="lazy" width="1800" height="1300" src="scenes/{scene_id}/compare.png" alt="MLP, MODIS, VIIRS and difference comparison">'
                     f'<p><a href="scenes/{scene_id}/compare.png">Full-size image</a> · <a href="scenes/{scene_id}/predictions.geojson">GeoJSON</a> · <a href="scenes/{scene_id}/queries.npz">All scores (NPZ)</a></p></article>')
    (PUBLIC / 'gallery.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>EPOCH · All rendered ML scenes</title><style>body{background:#090f18;color:#edf5fc;font:16px system-ui;margin:40px auto;max-width:1300px;padding:0 24px}a{color:#55ccff}h1{font-size:42px}p{color:#a8bdce}article{margin:50px 0;break-inside:avoid}img{width:100%;height:auto} @media print{body{background:white;color:black}article{break-before:page}a{color:black}}</style>'
        f'<h1>EPOCH / Complete scene atlas</h1><p>50,567,169-parameter frozen MLP · {catalogue["scene_count"]:,} scenes · {catalogue["total_queries"]:,} eligible queries. All five views per scene are pre-rendered.</p>'
        '<p>Research only. Training/validation and additional windows are not independent test evidence. VIIRS provides reference labels. Empty areas outside gray coverage are unavailable.</p>'
        '<p><a href="../../../?mode=ml">Open presentation</a> · <a href="scene-summary.csv">Scene summary CSV</a> · <a href="catalog.json">Provenance and catalogue</a></p>'
        + ''.join(cards) + '</html>', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-snapshot', action='store_true', help='Include newly prepared tiles in a new input snapshot')
    args = parser.parse_args()
    build(args.refresh_snapshot)
