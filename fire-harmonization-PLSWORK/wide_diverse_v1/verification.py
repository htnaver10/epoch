"""Inspect source isolation, immutable parents and label-free deployment outputs."""
from __future__ import annotations

import json
import numpy as np
import torch

from dataset import check_extra_rows
from inference import infer
from settings import PARENT, path, rows, read, sha, write, freeze_sources


def verify():
    lock = freeze_sources()
    parent = rows(PARENT / 'data/manifests/source_v3.jsonl')
    extra = rows(path('data/manifests/extra_train.jsonl'))
    check_extra_rows(extra, parent)
    checked = 0
    for row in extra:
        if sha(path(row['tile'])) != row['sha256']:
            raise ValueError('Additional observation changed after fitting')
        checked += 1
    parent_receipt = json.loads((PARENT / 'data/features/receipt.json').read_text())
    for name, expected in parent_receipt['files'].items():
        if sha(PARENT / name) != expected:
            raise ValueError('Original feature cache differs from its frozen receipt')
    best = torch.load(path('models/best.pt'), map_location='cpu', weights_only=True)
    saved = torch.load(path('models/inference.pt'), map_location='cpu', weights_only=True)
    if best['model'].keys() != saved['model'].keys() or any(not torch.equal(value, best['model'][key]) for key, value in saved['model'].items()):
        raise ValueError('Export does not exactly match selected model')
    del best, saved
    # Select by the already evaluated reference metadata solely for a deployment
    # fixture; this never changes training, checkpoint choice or policy.
    example = next(row for row in parent if row['split'] == 'test' and row['positive_queries'] > 0)
    with np.load(PARENT / example['tile'], allow_pickle=False) as z:
        arrays = {key: z[key] for key in ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid', 'inference_valid')}
    with path('results/inference_input_only.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    deployment = infer('results/inference_input_only.npz', 'results/inference_output.npz', 'cuda')
    ledger = json.loads((PARENT / 'data/features/tiles.json').read_text())
    selected = next(row for row in ledger if row['tile_id'] == example['tile_id'])
    indices = np.load(PARENT / 'data/features/test_flat_index.npy', mmap_mode='r', allow_pickle=False)[selected['start']:selected['end']]
    with np.load(path('results/test_predictions.npz'), allow_pickle=False) as predictions:
        expected = predictions['logits'][selected['start']:selected['end']]
    with np.load(path('results/inference_output.npz'), allow_pickle=False) as output:
        actual = output['raw_logit'].ravel()[indices]
        max_difference = float(np.max(np.abs(actual - expected)))
        policy = read('results/operating_policy.json')
        different_decisions = int(np.count_nonzero((actual.astype(np.float64) >= policy['raw_logit_cutoff']) !=
                                                   (expected.astype(np.float64) >= policy['raw_logit_cutoff'])))
    if not np.allclose(actual, expected, rtol=0.02, atol=0.125):
        raise ValueError('Input-only deployment differs materially from evaluated model')
    result = {'complete': True, 'extra_tiles_hash_checked': checked,
              'source_overlap_with_parent_heldout': False, 'duplicate_native_windows': False,
              'parent_feature_files_hash_checked': len(parent_receipt['files']),
              'parent_manifest_sha256': lock['parent_manifest_sha256'],
              'parent_model_sha256': lock['parent_model_sha256'],
              'export_exactly_matches_selected_weights': True,
              'input_only_keys': sorted(arrays), 'label_free_inference_passed': True,
              'deployment_compared_known_queries': len(actual), 'maximum_bf16_logit_difference': max_difference,
              'deployment_different_decisions': different_decisions,
              'numerical_note': 'CUDA BF16 matrix batch shapes can change rounding; CPU float32 may also differ near thresholds.',
              'deployment': deployment}
    write('results/completion_verification.json', result)
    return result
