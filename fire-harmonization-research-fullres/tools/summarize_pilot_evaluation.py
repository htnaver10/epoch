"""Audit frozen pilot predictions and reproduce the legacy before/after report.

Uses existing native evaluation output only. Does not train or tune thresholds.
Run with the full-resolution .venv Python from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from firefullres import core

COLORS = ['#2878B5', '#D55E00', '#263238']
NAMES = ['Before: MODIS', 'After: pilot CNN', 'VIIRS reference']


def csv_file(filename, rows):
    with core.path(filename, True).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_metrics(p, y, w, threshold):
    """Direct point losses and exact, tie-aware ranking; separate from engine helper."""
    p, y, w = [np.asarray(a, dtype=np.float64) for a in (p, y, w)]
    pred = p >= threshold
    tp, fp = float(w[y.astype(bool) & pred].sum()), float(w[(y == 0) & pred].sum())
    fn, tn = float(w[y.astype(bool) & ~pred].sum()), float(w[(y == 0) & ~pred].sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    order = np.argsort(-p, kind='stable')
    scores, target, weights = p[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    positive = np.add.reduceat(target * weights, starts)
    negative = np.add.reduceat((1 - target) * weights, starts)
    cumulative_positive = np.cumsum(positive)
    cumulative_all = np.cumsum(positive + negative)
    ap = float(np.sum(positive / positive.sum() * cumulative_positive / cumulative_all)) if positive.sum() else None
    auc = float(np.sum(positive * (negative.sum() - np.cumsum(negative) + .5 * negative)) /
                (positive.sum() * negative.sum())) if positive.sum() * negative.sum() else None
    q = np.clip(p, 1e-12, 1 - 1e-12)
    expected = float(np.dot(w, p))
    reference = float(np.dot(w, y))
    return {'valid_query_weight': float(w.sum()), 'positive_query_weight': reference,
            'expected_active_queries': expected, 'predicted_active_queries': tp + fp,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'brier': float(np.dot(w, (p - y) ** 2) / w.sum()),
            'logloss': float(-np.dot(w, y * np.log(q) + (1 - y) * np.log1p(-q)) / w.sum()),
            'roc_auc_exact': auc, 'average_precision_exact': ap,
            'expected_activity_bias_pct': 100 * (expected / reference - 1) if reference else None,
            'threshold': threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', default='results/real_training_pilot_v1_evaluation')
    args = parser.parse_args()
    evaluation = core.path(args.evaluation)
    report = core.read_json(evaluation / 'report.json')
    run = core.read_json(core.path(report['checkpoint']).parent / 'run.json')
    manifest = run['manifest']
    assert core.sha256(manifest) == report['manifest_sha256'] == run['manifest_sha256']
    assert core.sha256(report['checkpoint']) == report['checkpoint_sha256']
    rows = core.validate_tiles(manifest)
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == report['split']]
    assert report['split'] == 'validation' and not report['synthetic_smoke_only']
    ledger = {r['tile_id']: r for r in report['predictions']}
    assert len(ledger) == len(report['predictions']) == len(validation) == report['tiles']
    assert set(ledger) == {r['tile_id'] for r in validation}
    threshold = report['model_metrics']['threshold']
    parts = {key: [] for key in ('p', 'y', 'b', 'w', 'lat', 'lon')}
    tile_rows = []
    for row in validation:
        prediction = ledger[row['tile_id']]
        assert core.sha256(prediction['prediction']) == prediction['sha256']
        with np.load(core.path(row['tile']), allow_pickle=False) as z, \
                np.load(core.path(prediction['prediction']), allow_pickle=False) as out:
            assert np.array_equal(z['weight'], out['valid'])
            assert np.array_equal(z['query_lat'], out['latitude'])
            assert np.array_equal(z['query_lon'], out['longitude'])
            assert z['target'].shape == out['probability'].shape
            keep = z['weight'] > 0
            p, y, b, w = [np.asarray(a[keep], dtype=np.float64) for a in
                          (out['probability'], z['target'], z['modis_baseline'], z['weight'])]
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            assert w.sum() == row['valid_queries'] and np.dot(w, y) == row['positive_queries']
            for key, a in zip(('p', 'y', 'b', 'w', 'lat', 'lon'),
                              (p, y, b, w, z['query_lat'][keep], z['query_lon'][keep])):
                parts[key].append(a)
            tile_rows.append({'tile_id': row['tile_id'], 'region_id': row['region_id'],
                              'acquisition_utc': row['modis_start_utc'],
                              'valid_queries': float(w.sum()), 'observed_viirs': float(np.dot(w, y)),
                              'before_modis': float(np.dot(w, b)), 'after_expected': float(np.dot(w, p)),
                              'after_thresholded': float(w[p >= threshold].sum())})
    data = {key: np.concatenate(value) for key, value in parts.items()}
    p, y, b, w = [data[key] for key in ('p', 'y', 'b', 'w')]
    assert np.all(w == 1), 'Report query counts require binary exposure weights'
    # One MODIS scene has two adjacent VIIRS companions. Repeated exposed
    # coordinates across either companion would duplicate reference support.
    modis_scenes = {s for r in validation for s in r['source_ids'] if Path(s).name.startswith('MYD021KM.')}
    reference_scenes = {s for r in validation for s in r['source_ids'] if Path(s).name.startswith('VNP14IMG.')}
    assert len(modis_scenes) == 1
    unique = len(np.unique(np.column_stack((data['lat'], data['lon'])), axis=0))
    assert unique == len(y), 'Duplicate exposed VIIRS query centers'
    metrics = {'before_modis': independent_metrics(b, y, w, threshold),
               'after_cnn': independent_metrics(p, y, w, threshold),
               'always_zero': independent_metrics(np.zeros_like(p), y, w, threshold)}
    checked = ('valid_query_weight', 'positive_query_weight', 'precision', 'recall', 'f1', 'brier', 'logloss')
    for name, source in [('after_cnn', 'model_metrics'), ('before_modis', 'modis_baseline_metrics')]:
        for key in checked:
            assert np.isclose(metrics[name][key], report[source][key], rtol=1e-10, atol=1e-12), (name, key)
    dates = sorted({r['modis_start_utc'] for r in validation})
    training_dates = sorted({r['modis_start_utc'] for r in train})
    assert len(dates) == len(training_dates) == 1
    assert sum(r['observed_viirs'] for r in tile_rows) == metrics['after_cnn']['positive_query_weight']
    summary = {'scope': 'one real NASA all-land spatial validation scene; no independent test',
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'checkpoint': report['checkpoint'], 'checkpoint_sha256': report['checkpoint_sha256'],
               'manifest': manifest, 'manifest_sha256': report['manifest_sha256'],
               'training_tiles': len(train), 'validation_tiles': len(validation),
               'training_positive_queries': sum(r['positive_queries'] for r in train),
               'training_acquisitions_utc': training_dates, 'validation_acquisitions_utc': dates,
               'validation_positive_tiles': sum(r['observed_viirs'] > 0 for r in tile_rows),
               'validation_modis_granules': len(modis_scenes), 'validation_viirs_granules': len(reference_scenes),
               'unique_exposed_query_centers': unique, 'metrics': metrics,
               'ranking_method': 'exact tied-score groups; engine report uses 4096-bin approximation',
               'validation_is_earlier_than_training': dates[0] < training_dates[0],
               'ground_location_accuracy_validated': False, 'independent_test_available': False}
    core.write_json(evaluation / 'visual_evaluation.json', summary)
    csv_file(evaluation / 'visual_evaluation_by_tile.csv', tile_rows)
    csv_file(evaluation / 'metrics.csv', [{'series': name, **value} for name, value in metrics.items()])

    figure_dir = core.path('visualizations/' + evaluation.name)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    figures = []

    def save(fig, name):
        for suffix in ('png', 'svg'):
            fig.savefig(figure_dir / f'{name}.{suffix}', dpi=170, bbox_inches='tight')
        figures.append(name)
        plt.close(fig)

    before, after = metrics['before_modis'], metrics['after_cnn']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    values = [[before['expected_active_queries'], after['expected_active_queries'], y.sum()],
              [before['precision'] * 100, after['precision'] * 100],
              [before['recall'] * 100, after['recall'] * 100]]
    for ax, numbers, title in zip(axes, values,
                                ['Activity on the same query centers', 'Precision at threshold 0.5 (%)', 'Recall at threshold 0.5 (%)']):
        bars = ax.bar(np.arange(len(numbers)), numbers, color=COLORS[:len(numbers)])
        ax.set_xticks(np.arange(len(numbers)), NAMES[:len(numbers)], rotation=18, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(numbers) * 1.22)
        ax.bar_label(bars, labels=[f'{v:,.1f}' if len(numbers) == 3 else f'{v:.2f}%' for v in numbers], padding=4)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Active-query equivalent; CNN = sum of scores')
    fig.suptitle('Pilot CNN overestimates activity on the Quebec validation scene', fontweight='bold')
    save(fig, '01_before_after_summary')

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout='constrained')
    for ax, key, title in zip(axes, ['true_positive', 'false_positive', 'false_negative'],
                              ['Matched VIIRS fire labels', 'False positives vs VIIRS', 'Missed VIIRS fire labels']):
        vals = [before[key], after[key]]
        bars = ax.bar([0, 1], vals, color=COLORS[:2])
        ax.bar_label(bars, labels=[f'{n:,.0f}' for n in vals], padding=4)
        ax.set_xticks([0, 1], NAMES[:2], rotation=15, ha='right')
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Exposed VIIRS query centers')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('At threshold 0.5, extra CNN positives are mostly false positives', fontweight='bold')
    save(fig, '02_detection_errors')

    # Common geographic bins and support: no unobserved land is filled with zero.
    step = .1
    xs = np.arange(np.floor(data['lon'].min() / step) * step,
                   np.ceil(data['lon'].max() / step) * step + step * 1.01, step)
    ys = np.arange(np.floor(data['lat'].min() / step) * step,
                   np.ceil(data['lat'].max() / step) * step + step * 1.01, step)
    exposure = np.histogram2d(data['lat'], data['lon'], bins=(ys, xs), weights=w)[0]
    grids = [np.histogram2d(data['lat'], data['lon'], bins=(ys, xs), weights=w * a)[0] for a in (b, p, y)]
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 5), layout='constrained')
    for ax, grid, title in zip(axes, grids, NAMES):
        fraction = np.divide(grid, exposure, out=np.full_like(grid, np.nan), where=exposure > 0)
        mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(fraction), cmap='magma', norm=PowerNorm(.4, vmin=0, vmax=1), rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_aspect(1 / np.cos(np.deg2rad(data['lat'].mean())))
    fig.colorbar(mesh, ax=axes, label='Active fraction of covered queries; shared scale', shrink=.8)
    fig.suptitle('Shared 0.1-degree view of the June 5, 2023 validation scene', fontweight='bold')
    save(fig, '03_spatial_before_after')
    map_rows = []
    for iy, ix in np.argwhere(exposure > 0):
        map_rows.append({'latitude': (ys[iy] + ys[iy + 1]) / 2, 'longitude': (xs[ix] + xs[ix + 1]) / 2,
                         'valid_queries': exposure[iy, ix], 'before_modis': grids[0][iy, ix],
                         'after_expected': grids[1][iy, ix], 'observed_viirs': grids[2][iy, ix]})
    csv_file(evaluation / 'spatial_figure_data.csv', map_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    ids = np.arange(len(tile_rows))
    for key, name, color in zip(['before_modis', 'after_expected', 'observed_viirs'], NAMES, COLORS):
        counts = np.array([r[key] for r in tile_rows])
        axes[0].plot(ids, counts / np.array([r['valid_queries'] for r in tile_rows]) * 100,
                     label=name, color=color, linewidth=1.6, marker='.' if key == 'observed_viirs' else None)
    axes[0].set(xlabel='Validation tile index (manifest order)', ylabel='Active fraction of covered queries (%)',
                title='Activity by tile; this axis is not time')
    axes[0].legend()
    for key, name, color in zip(['before_modis', 'after_expected'], NAMES[:2], COLORS[:2]):
        errors = [abs(r[key] - r['observed_viirs']) for r in tile_rows]
        axes[1].plot(ids, errors, label=name, color=color)
    axes[1].set(xlabel='Validation tile index (manifest order)', ylabel='Absolute activity error (query equivalent)',
                title='CNN activity errors extend across the scene')
    axes[1].legend()
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    save(fig, '04_tile_activity_errors')

    calibration = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        keep = (p >= lo) & (p < hi if hi < 1 else p <= 1)
        if not keep.any():
            continue
        calibration.append({'bin_lower': lo, 'bin_upper': min(hi, 1), 'valid_queries': int(keep.sum()),
                            'mean_score': float(p[keep].mean()), 'observed_fire_rate': float(y[keep].mean())})
    csv_file(evaluation / 'calibration_figure_data.csv', calibration)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    bins = np.linspace(0, 1, 31)
    for label, selected, color in [('VIIRS non-fire', y == 0, '#2878B5'), ('VIIRS fire', y == 1, '#D55E00')]:
        density, _ = np.histogram(p[selected], bins=bins)
        axes[0].stairs(density / selected.sum() * 100, bins, label=label, color=color, linewidth=2)
    axes[0].axvline(threshold, color='#555555', linestyle='--', label='Saved threshold 0.5')
    axes[0].set(xlabel='CNN score', ylabel='Share of each reference class (%)', title='Score distributions overlap')
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], color='#777777', linestyle='--', label='Perfect calibration')
    axes[1].plot([r['mean_score'] for r in calibration], [r['observed_fire_rate'] for r in calibration],
                 'o-', color=COLORS[1], label='Pilot CNN')
    axes[1].set(xlabel='Mean CNN score within bin', ylabel='Recorded VIIRS fire fraction',
                title='Scores are far above recorded fire frequency', xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    save(fig, '05_probability_diagnostics')

    positive_tiles = summary['validation_positive_tiles']
    reference = before['positive_query_weight']
    images = '\n\n'.join(f'![{name.replace("_", " ")}](../../visualizations/{evaluation.name}/{name}.png)\n\n'
                           f'[Editable SVG](../../visualizations/{evaluation.name}/{name}.svg)' for name in figures)
    rows_text = []
    for label, key, pattern in [('Precision', 'precision', '.2%'), ('Recall', 'recall', '.2%'), ('F1', 'f1', '.4f'),
                                ('Brier loss (lower is better)', 'brier', '.6f'), ('Log loss (lower is better)', 'logloss', '.6f'),
                                ('ROC AUC (exact)', 'roc_auc_exact', '.4f'), ('Average precision (exact)', 'average_precision_exact', '.4f')]:
        rows_text.append(f'| {label} | {before[key]:{pattern}} | {after[key]:{pattern}} |')
    text = f'''# Before and after native-resolution harmonization: real_training_pilot_v1

**This one-epoch pilot performs substantially worse than the observed MODIS baseline on its Quebec validation scene. Keep it experimental.** [1]

The comparison follows the earlier [coarse-grid evaluation](../../../fire-harmonization-research/results/EVALUATION.md): observed MODIS before, frozen model estimates after, and observed VIIRS as reference. The native target is a recorded VIIRS query fire label, so its numbers are not directly comparable to the older weighted cell-day metrics.

## What the results support

- On **{len(y):,} covered query centers**, observed VIIRS contains **{reference:,.0f} fire labels**. MODIS marks **{before['predicted_active_queries']:,.0f}** centers active; the CNN marks **{after['predicted_active_queries']:,.0f}** at the saved **0.5** threshold. [1]
- CNN false positives rise from **{before['false_positive']:,.0f} to {after['false_positive']:,.0f}**, while matched VIIRS positives fall from **{before['true_positive']:,.0f} to {after['true_positive']:,.0f}**. Precision falls from **{before['precision']:.2%} to {after['precision']:.2%}**; recall falls from **{before['recall']:.2%} to {after['recall']:.2%}**. [1]
- Summing the CNN scores yields **{after['expected_active_queries']:,.1f} active-query equivalents**, versus **{reference:,.0f}** recorded VIIRS positives: **{after['expected_activity_bias_pct']:,.1f}%** bias. These scores are not validated probabilities or counts of independent fires. The corresponding MODIS bias is **{before['expected_activity_bias_pct']:+.1f}%**. [1]
- CNN Brier loss is **{after['brier']:.6f}**, versus **{before['brier']:.6f}** for MODIS and **{metrics['always_zero']['brier']:.6f}** for an always-zero reference baseline. The validation prevalence is only **{reference / len(y):.4%}**; the zero baseline's small loss does not make it useful for finding fires. [1]

## Why the pilot should remain experimental

Training completed **one epoch** using **{len(train)} western-Canada tiles** from **{training_dates[0][:10]}**, with only **{summary['training_positive_queries']:,.0f} positive query labels**. Validation uses **{len(validation)} Quebec tiles** from **{dates[0][:10]}**, with positives in **{positive_tiles} tiles**. It covers one MODIS acquisition paired with **{len(reference_scenes)} adjacent VIIRS granules**; those tiles are not independent events. [2]

Validation groups, source files, tile IDs and buffered geographic footprints are separated from training. However, this same validation set was used to choose the saved best checkpoint; it is **not an untouched test set**. Its date is also earlier than the training date, so this is a spatial scene comparison, not forward temporal validation. No aggregate uncertainty interval or independent-event generalization claim is supported. [2]

Sparse training positives and the short run are possible contributors to failure, but this evaluation does not establish a cause. The slight ROC AUC advantage does not offset the much worse precision, average precision and activity totals. Do not tune a threshold on this scene and then describe its result as independent testing.

## Matched detection and score metrics

All comparisons use the same covered query centers and the saved 0.5 threshold. Ranking metrics below use exact tied-score groups; the native evaluator's [raw report](report.json) uses an approximate 4096-bin histogram. Hard MODIS flags are treated as scores 0/1, so their log loss is sensitive to the shared clipping convention. [1]

| Metric | Before: MODIS | After: pilot CNN |
|---|---:|---:|
{chr(10).join(rows_text)}

## Before/after figures

The maps share one scale and show active fractions among covered query centers in 0.1-degree display bins. Blank bins have no eligible exposure. The display bins do not change the native evaluation target. The tile index is spatial manifest order, not a timeline. All chart values are available in the accompanying CSV files. [1]

{images}

## Interpretation limits and next step

Labels are VIIRS nominal/high-confidence fire classes 8/9 versus observed land class 5, with the pipeline's QA and geometric exclusions. Cloud, water, low-confidence fire and unknown classes are excluded rather than interpreted as non-fire. The MODIS baseline is the native fire flag sampled at those VIIRS query centers. Geometric interpolation residuals do not validate absolute ground location. [3]

This is an **all-land thermal-activity pilot**, not a forest-only wildfire model. Reference disagreement can include sensor resolution, label policy and acquisition-time differences (up to 15 minutes allowed by pairing). Neither source is independent wildfire ground truth. Active queries are not burned area, unique incidents or reconstructed subpixel fire locations.

The older summer totals, weekly profiles, burning calendars, multi-region comparisons and power-law benchmark cannot be reproduced from this one-scene validation set with the same meaning. The earlier power law was fitted to coarse weekly occupied-cell totals; applying it directly to native binary query labels would change the target without validation.

Retain the MODIS baseline. Before further claims, add multiple geographically and temporally separated fire-rich training scenes plus a separately reserved test set. Use development validation for checkpoint selection and any calibration; compare precision, recall, activity bias and timing against MODIS on the untouched test set. A longer run alone would not establish improvement.

## Evidence and reproducibility

1. [Frozen evaluation output](report.json), [independently recomputed metrics](visual_evaluation.json), [metric table](metrics.csv), [per-tile values](visual_evaluation_by_tile.csv), [map values](spatial_figure_data.csv), and [calibration values](calibration_figure_data.csv).
2. [Training record](../../logs/real_training_pilot_v1/epochs.jsonl), [run configuration](../../models/real_training_pilot_v1/run.json), [split audit](../real_training_pilot_v1_audit.json), and [execution verification](../real_training_pilot_v1_execution.json).
3. [Native label and geometric eligibility code](../../src/firefullres/geometry.py) and [MODIS baseline preparation](../../src/firefullres/prepare.py).

Checkpoint: `models/real_training_pilot_v1/last.pt`; SHA-256 `{report['checkpoint_sha256']}`. Predictions and input tiles were checked against their saved hashes; counts, exposure masks, coordinates and direct losses were independently reconciled. All **{unique:,}** exposed coordinates are unique within the validation scene. The existing checkpoint and training data are unchanged.

From the repository root, reproduce with:

```powershell
& 'fire-harmonization-research-fullres/.venv/Scripts/python.exe' 'fire-harmonization-research-fullres/run.py' evaluate --manifest data/manifests/real_training_pilot_v1.jsonl --checkpoint models/real_training_pilot_v1/last.pt --split validation --device cuda --output results/real_training_pilot_v1_evaluation_NEW
& 'fire-harmonization-research-fullres/.venv/Scripts/python.exe' 'fire-harmonization-research-fullres/tools/summarize_pilot_evaluation.py' --evaluation results/real_training_pilot_v1_evaluation_NEW
```

Use a fresh evaluation output directory; the evaluator refuses to overwrite existing predictions. Plotting only needs the saved predictions and does not fit a model or alter the threshold.
'''
    (evaluation / 'EVALUATION.md').write_text(text, encoding='utf-8')
    links = [evaluation / name for name in ('visual_evaluation.json', 'metrics.csv', 'visual_evaluation_by_tile.csv')]
    assert all(link.exists() for link in links)
    provenance = {'passed': True, 'independent_metric_keys_checked': list(checked),
                  'tile_hashes_checked': len(rows), 'prediction_hashes_checked': len(ledger),
                  'unique_exposed_query_centers': unique, 'inputs': {
                      report['checkpoint']: core.sha256(report['checkpoint']), manifest: core.sha256(manifest),
                      core.relative(evaluation / 'report.json'): core.sha256(evaluation / 'report.json'),
                      core.relative(__file__): core.sha256(__file__)},
                  'outputs': {core.relative(f): core.sha256(f) for f in
                              [evaluation / 'EVALUATION.md', evaluation / 'visual_evaluation.json',
                               *sorted(evaluation.glob('*.csv')), *sorted(figure_dir.glob('*.png')), *sorted(figure_dir.glob('*.svg'))]}}
    core.write_json(evaluation / 'visualization_manifest.json', provenance)
    print(json.dumps({'report': core.relative(evaluation / 'EVALUATION.md'), 'metrics': metrics,
                      'validation_positive_tiles': positive_tiles, 'checks_passed': True}, indent=2))


if __name__ == '__main__':
    main()
