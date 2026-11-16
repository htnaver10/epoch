"""Compare frozen fire models on matched reference queries; never fit a network.

The additional wide/tree native-cell thresholds are determined solely from
validation, using the pre-existing MODIS false-positive-budget rule.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLS = ROOT / 'fire-harmonization-PLSWORK'
WIDE = PLS / 'wide_diverse_v1'
FULL = ROOT / 'fire-harmonization-research-fullres'
YES = ROOT / 'fire-harmonization-research-YESSS'
spec = importlib.util.spec_from_file_location('existing_comparison', HERE.parent / 'build_visualizations.py')
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)
SOURCES = {}


def register(p):
    p = Path(p)
    with p.open('rb') as stream:
        SOURCES[p.relative_to(ROOT).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return p


def read(p):
    return json.loads(register(p).read_text())


def array(p):
    return np.load(register(p), allow_pickle=False)


def check_saved(actual, expected, style='standard'):
    keys = [('TP', 'true_positives'), ('FP', 'false_positives'), ('FN', 'false_negatives'), ('TN', 'true_negatives'),
            ('precision', 'precision'), ('recall', 'recall'), ('F1', 'f1')]
    for source, destination in keys:
        if style == 'short' and source in ('TP', 'FP', 'FN', 'TN'):
            destination = source
        np.testing.assert_allclose(actual[source], expected[destination], rtol=1e-12)


def aggregate(cell, y, baseline, score):
    n = int(cell.max()) + 1
    cy = np.zeros(n, bool)
    cb = np.zeros(n, bool)
    cs = np.full(n, -np.inf)
    np.maximum.at(cy, cell, y)
    np.maximum.at(cb, cell, baseline)
    np.maximum.at(cs, cell, score)
    assert np.isfinite(cs).all()
    return cy, cb, cs


def align_legacy(values, tile_records, parent):
    ledger = read(PLS / 'data/features/tiles.json')
    lookup = {r['tile_id']: r for r in ledger if r['split'] == 'test'}
    permutation = np.full(len(parent['target']), -1, np.int64)
    assert len(tile_records) == len(lookup)
    for index, tile in enumerate(tile_records):
        expected = lookup[tile['tile_id']]
        indices = np.flatnonzero(values['tile_index'] == index)
        start, end = expected['start'], expected['end']
        assert len(indices) == end - start
        assert {Path(s).name for s in tile['source_ids']} == {Path(s).name for s in expected['source_ids']}
        permutation[start:end] = indices
    assert np.all(permutation >= 0) and len(np.unique(permutation)) == len(permutation)
    aligned = {key: values[key][permutation] for key in values.files}
    for key in ('target', 'weight'):
        np.testing.assert_array_equal(aligned[key], parent[key])
    np.testing.assert_array_equal(aligned['modis'], parent['modis'])
    return aligned


def validation_cutoff(scores, y, baseline):
    budget = int(np.count_nonzero(baseline & ~y))
    ordered = np.sort(scores[~y])[::-1]
    cutoff = float(np.nextafter(ordered[budget], np.inf)) if budget < len(ordered) else float(scores.min() - 1)
    assert int(np.count_nonzero((scores >= cutoff) & ~y)) <= budget
    return {'fit_split': 'validation', 'test_used_for_fit': False, 'cutoff': cutoff,
            'false_positive_budget': budget, 'aggregation': 'maximum score among eligible reference centers per native MODIS cell'}


def main():
    e = existing.load_evidence()
    y = e['y'].astype(bool)
    baseline = e['predictions']['modis']
    assert len(y) == 585464 and int(y.sum()) == 95
    rows = []
    names = {'modis': 'Standard MODIS', 'mlp': 'Original PLSWORK MLP', 'tree': 'PLSWORK tree',
             'hybrid': 'Neural/tree hybrid', 'v2': 'Saturation V2 coarse', 'v2_localized': 'Saturation V2 localized'}
    for key in names:
        rows.append({'id': key, 'model': names[key], 'policy': 'frozen validation policy' if key not in ('modis', 'v2', 'v2_localized') else 'operational flags' if key == 'modis' else 'unlabeled frozen anomaly policy',
                     **e['metrics'][key]})
    parent = array(PLS / 'results/test_mlp_predictions.npz')
    wide_policy = read(WIDE / 'results/operating_policy.json')
    wide_report = read(WIDE / 'results/evaluation.json')
    wide = array(WIDE / 'results/test_predictions.npz')
    for name in ('target', 'weight', 'tile_index', 'flat_index'):
        np.testing.assert_array_equal(wide[name], parent[name])
    m = existing.metric(y, wide['logits'].astype(np.float64) >= wide_policy['raw_logit_cutoff'], wide['logits'])
    check_saved(m, wide_report['test']['wide_frozen_policy'])
    np.testing.assert_allclose(m['AP'], wide_report['test']['wide_frozen_policy']['average_precision_exact'], rtol=1e-12)
    rows.append({'id': 'wide', 'model': 'Wide diverse MLP (50.6M)', 'policy': 'frozen validation policy', **m})
    # Legacy models: verify labels, order, eligibility and MODIS flags before comparison.
    for folder, prefix, policy_path, method, cutoff_key in [
            (FULL / 'results/substantial_round_v3', '', FULL / 'results/substantial_round_v3/calibration.json', 'cnn', 'raw_logit_cutoff'),
            (FULL / 'results/teacher_student_v1', 'student_', FULL / 'results/teacher_student_v1/operating_policy.json', 'student', 'raw_logit_cutoff')]:
        old = read(folder / 'evaluation.json')
        z = align_legacy(array(folder / f'test_{prefix}predictions.npz'), old['test_tiles'], parent)
        policy = read(policy_path)
        for suffix, cutoff, saved_key in [('raw', 0., 'raw_model_at_0_5' if method == 'cnn' else 'distilled_student'),
                                          ('frozen', policy[cutoff_key], 'validation_adjusted_model_at_frozen_threshold' if method == 'cnn' else 'distilled_student_adjusted')]:
            score = existing.metric(y, z['logits'].astype(np.float64) >= cutoff, z['logits'])
            check_saved(score, old['test'][saved_key])
            rows.append({'id': method + '_' + suffix,
                         'model': ('Previous supervised CNN' if method == 'cnn' else 'Distilled MODIS student') + (' (raw 0.5)' if suffix == 'raw' else ' (frozen policy)'),
                         'policy': 'raw probability 0.5' if suffix == 'raw' else 'frozen validation policy', **score})
    v1 = array(YES / 'results/test_predictions.npz')
    np.testing.assert_array_equal(v1['y'], y)
    np.testing.assert_array_equal(v1['modis'], baseline)
    v2 = array(YES / 'saturation_aware_v2/results/test_predictions.npz')
    np.testing.assert_array_equal(v1['cell'], v2['cell'])
    v1_report = read(YES / 'results/evaluation.json')
    for key, predkey, saved_key in [('v1', 'coarse', 'unsupervised_coarse'), ('v1_localized', 'fine', 'unsupervised_two_stage')]:
        score = existing.metric(y, v1[predkey], v1['score'] if key == 'v1' else None)
        check_saved(score, v1_report['test']['query'][saved_key], 'short')
        rows.append({'id': key, 'model': 'Saturation V1 ' + ('coarse' if key == 'v1' else 'localized'),
                     'policy': 'legacy implementation with audited coverage defect', **score})
    rows.sort(key=lambda r: -r['F1'])
    # Native-cell sensitivity: preserve the original models' native policies and
    # freeze the two previously unavailable policies on validation only.
    vcell = array(PLS / 'hybrid_v1/data/validation_cell.npy')
    vy = array(PLS / 'data/features/validation_y.npy') > .5
    vb = array(PLS / 'data/features/validation_modis.npy') > .5
    validation_wide = array(WIDE / 'results/validation_predictions.npz')
    actual = array(PLS / 'results/validation_mlp_predictions.npz')
    for name in ('target', 'weight', 'tile_index', 'flat_index'):
        np.testing.assert_array_equal(validation_wide[name], actual[name])
    validation_tree = array(PLS / 'results/validation_tree_predictions.npz')
    policies = {}
    for key, score in [('wide', validation_wide['logits']), ('tree', validation_tree['logits'])]:
        cy, cb, cs = aggregate(vcell, vy, vb, score)
        policies[key] = validation_cutoff(cs, cy, cb)
    (HERE / 'native_cell_policies.json').write_text(json.dumps(policies, indent=2) + '\n')
    native = [{'id': key, 'model': names[key], **value} for key, value in e['native_metrics'].items() if key != 'viirs']
    for key, score in [('wide', wide['logits']), ('tree', e['scores']['tree'])]:
        cy, cb, cs = aggregate(v2['cell'], y, baseline, score)
        value = existing.metric(cy, cs >= policies[key]['cutoff'], cs)
        native.append({'id': key, 'model': 'Wide diverse MLP (50.6M)' if key == 'wide' else names[key], **value})
    native.sort(key=lambda r: -r['F1'])
    # Stored tensor counts are descriptive; boosted-tree size is not an NN parameter count.
    import torch
    counts = {}
    for key, filename in [('wide', WIDE / 'models/inference.pt'), ('mlp', PLS / 'models/mlp/inference.pt'),
                          ('cnn', FULL / 'models/substantial_round_v3/best.pt'),
                          ('student', FULL / 'models/teacher_student_v1_student/student_inference.pt'),
                          ('v2_coarse', YES / 'saturation_aware_v2/models/coarse/inference.pt'),
                          ('v2_fine', YES / 'saturation_aware_v2/models/fine/inference.pt')]:
        saved = torch.load(register(filename), map_location='cpu', weights_only=True, mmap=True)
        state = saved['model']
        counts[key] = sum(t.numel() for t in state.values() if isinstance(t, torch.Tensor))
        del saved, state
    output = {'scope': 'Previously examined 2024 Quebec benchmark; identical 585464 eligible VIIRS reference query centers, 95 positives, 84 tiles.',
              'query_metrics': rows, 'native_cell_metrics': native, 'neural_weight_counts': counts,
              'native_cell_scope': {'eligible_cells': len(cy), 'positive_cells': int(cy.sum()), 'wide_tree_policy_fit': 'new validation-only cell policies; existing other policies preserved'},
              'wide_training': {'available_queries': 15117425, 'positive_queries': 8132},
              'original_training': {'available_queries': 10371908, 'positive_queries': 1913},
              'caveats': ['This is a reused benchmark, not an independent blind test.',
                          'Query centers and native cells are correlated reference units, not independent wildfire incidents.',
                          'Capacity, training observations and sampling objective changed together; gains cannot be attributed to size alone.',
                          'Supervised models use validation labels; V2 thresholds use unlabeled anomaly-score distributions.',
                          'Raw 0.5 and frozen-policy legacy rows are separate pre-existing operating points; no test thresholds were optimized.',
                          'V1 had an audited saturation/coverage defect; its low scores do not establish that unsupervised learning fails.'],
              'excluded': ['VIIRS teacher receives VIIRS imagery, unlike deployed MODIS models; no matched teacher test result is available.',
                           'Earlier pilots and older experiments have different exposure/definitions and are not ranked here.'],
              'checks': {'matched_query_arrays': True, 'confusion_counts_independently_recomputed': True,
                         'wide_score_ranking_recomputed': True, 'existing_native_cell_alignment_checked': True},
              'source_sha256': {**existing.SOURCES, **SOURCES}}
    (HERE / 'comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: output[k] for k in ('query_metrics', 'native_cell_metrics', 'neural_weight_counts')}, indent=2))


if __name__ == '__main__':
    main()
