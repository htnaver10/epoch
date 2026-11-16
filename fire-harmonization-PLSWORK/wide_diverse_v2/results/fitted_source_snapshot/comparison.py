"""A saved, single-seed 2x2 comparison of network size and available data."""
import json
import numpy as np
from settings import PARENT, V1, path, read, write, run_config, artifact, sha

def counts(y, pred):
    tp = int(np.count_nonzero(pred & y)); fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y)); tn = int(np.count_nonzero(~pred & ~y))
    precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': precision, 'recall': recall,
        'F1': 2*tp/max(2*tp+fp+fn, 1)}

def compare():
    cfg = read('config.json'); results = []
    with np.load(V1 / 'results/test_predictions.npz', allow_pickle=False) as z:
        baseline = {k: z[k] for k in z.files}
    y = baseline['target'] > .5
    old = json.loads((V1 / 'results/evaluation.json').read_text())
    old_policy = json.loads((V1 / 'results/operating_policy.json').read_text())
    first = counts(y, baseline['logits'].astype(np.float64) >= old_policy['raw_logit_cutoff'])
    first.update(run_id='reference50_d1', parameters=50567169, dataset='d1',
        validation_AP=old['validation']['wide_frozen_policy']['average_precision_exact'],
        AP=old['test']['wide_frozen_policy']['average_precision_exact'])
    results.append(first)
    for run_id, run in cfg['runs'].items():
        report = read(artifact(run_id, 'evaluation.json')); policy = read(artifact(run_id, 'operating_policy.json'))
        verification = read(artifact(run_id, 'completion_verification.json'))
        if not verification['complete']: raise ValueError('Incomplete run verification')
        with np.load(path(artifact(run_id, 'test_predictions.npz')), allow_pickle=False) as z:
            for key in ('target', 'weight', 'tile_index', 'flat_index'):
                np.testing.assert_array_equal(z[key], baseline[key])
            actual = counts(y, z['logits'].astype(np.float64) >= policy['raw_logit_cutoff'])
        saved = report['test']['wide_frozen_policy']
        for a, b in [('TP', 'true_positives'), ('FP', 'false_positives'), ('FN', 'false_negatives'), ('F1', 'f1')]:
            np.testing.assert_allclose(actual[a], saved[b], rtol=1e-12)
        actual.update(run_id=run_id, parameters=report['training']['parameters'], dataset=run['dataset'],
            validation_AP=report['validation']['wide_frozen_policy']['average_precision_exact'],
            AP=saved['average_precision_exact'], native_cell=report['native_cell']['test']['model'])
        results.append(actual)
    lookup = {r['run_id']: r for r in results}
    contrasts = {}
    for name, a, b in [('size_effect_d1', 'reference50_d1', 'large200_d1'),
                      ('size_effect_d2', 'control50_d2', 'large200_d2'),
                      ('data_effect_50m', 'reference50_d1', 'control50_d2'),
                      ('data_effect_200m', 'large200_d1', 'large200_d2')]:
        contrasts[name] = {'from': a, 'to': b, 'delta_test_F1': lookup[b]['F1']-lookup[a]['F1'],
            'delta_validation_AP': lookup[b]['validation_AP']-lookup[a]['validation_AP'],
            'delta_test_AP': lookup[b]['AP']-lookup[a]['AP'],
            'delta_TP': lookup[b]['TP']-lookup[a]['TP'], 'delta_FP': lookup[b]['FP']-lookup[a]['FP']}
    result = {'query_scope': {'known_queries': len(y), 'positive_queries': int(y.sum()), 'benchmark_previously_examined': True},
        'conditions': results, 'contrasts': contrasts, 'second_expansion': read('data/feature_receipt.json'),
        'single_seed': cfg['training']['seed'],
        'interpretation': ['These are single-seed architecture/data contrasts with matched optimizer settings and evaluation rules.',
            'Early stopping and weight initialization differ by fitted model; this is not a statistical estimate of a universal size effect.',
            'Dataset d2 adds curated observations within the same four conservative cohorts; selection is fire-enriched.',
            'Original validation and test cohorts were previously examined; correlated queries are not independent fire incidents.'],
        'preserved_reference_sha256': sha(V1 / 'models/inference.pt')}
    write('results/comparison.json', result)
    lines = ['# Second expansion: model size and data comparison', '',
        f'New model: **201,797,633 parameters**, width 4,096, feed-forward expansion 8,192, three residual blocks.', '',
        f'Dataset d1 has **15,117,425 training queries / 8,132 positives**. Dataset d2 has **{result["second_expansion"]["combined_training_queries"]:,} queries / {result["second_expansion"]["combined_positive_queries"]:,} positives**.', '',
        'All rows use the same 585,464 benchmark query centers and 95 positive VIIRS-reference labels. Query thresholds and native-cell thresholds use validation only. This is a reused benchmark.', '',
        '| Condition | Parameters | Data | Precision | Recall | F1 | TP | FP | FN | AP | Validation AP |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["run_id"]} | {r["parameters"]:,} | {r["dataset"]} | {100*r["precision"]:.2f}% | {100*r["recall"]:.2f}% | {r["F1"]:.4f} | {r["TP"]} | {r["FP"]} | {r["FN"]} | {r["AP"]:.4f} | {r["validation_AP"]:.4f} |')
    lines += ['', '## Controlled contrasts', '', '| Contrast | Test F1 change | Test AP change | Matched-positive change | False-positive change |', '|---|---:|---:|---:|---:|']
    for name, r in contrasts.items():
        lines.append(f'| {name} | {r["delta_test_F1"]:+.4f} | {r["delta_test_AP"]:+.4f} | {r["delta_TP"]:+} | {r["delta_FP"]:+} |')
    lines += ['', '## Native-cell results', '', '| Condition | Precision | Recall | F1 | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in results[1:]:
        c = r['native_cell']
        lines.append(f'| {r["run_id"]} | {100*c["precision"]:.2f}% | {100*c["recall"]:.2f}% | {c["f1"]:.4f} | {c["true_positives"]:.0f} | {c["false_positives"]:.0f} | {c["false_negatives"]:.0f} |')
    lines += ['', *result['interpretation'], '']
    path('results/COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')
    return result
