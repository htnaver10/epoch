"""Predict from MODIS arrays only; reference labels are never opened as inputs."""
from __future__ import annotations

import numpy as np

from feature_extraction import extract, query_indices
from settings import path, read, sha, feature_config
from training import load_model, predict
from firefullres.calibration import sigmoid


def infer(tile, output, device='cuda'):
    model, saved = load_model(device)
    with np.load(path(tile), allow_pickle=False) as z:
        inputs = {key: z[key] for key in ('thermal', 'reflective_500', 'reflective_250', 'geometry', 'query_grid')}
        grid = inputs['query_grid']
        shape = grid.shape[:2]
        if 'inference_valid' in z:
            valid = z['inference_valid'] > 0
        else:
            iy, ix, _, _ = query_indices(grid, *inputs['thermal'].shape[-2:])
            valid = (np.abs(grid) <= 1).all(-1) & (inputs['thermal'][6:, iy, ix].sum(0) >= 2)
    features, names = extract(inputs, feature_config())
    if names != saved['stats']['feature_names']:
        raise ValueError('Inference feature schema differs from trained weights')
    logits = predict(model, features.reshape(-1, len(names)), saved['stats'],
                     read('config.json')['training']['inference_batch_size'])
    arrays = {'raw_logit': logits.reshape(shape), 'raw_probability': sigmoid(logits).reshape(shape), 'valid': valid}
    policy = read('results/operating_policy.json')
    if policy['model_sha256'] != sha(path('models/inference.pt')):
        raise ValueError('Operating policy does not match model weights')
    arrays.update(adjusted_probability=sigmoid(logits + policy['logit_offset']).reshape(shape),
                  frozen_policy_detection=(logits.astype(np.float64) >= policy['raw_logit_cutoff']).reshape(shape) & valid)
    destination = path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return {'output': output, 'requires_viirs': False, 'requires_labels': False,
            'query_shape': list(shape), 'execution': 'CUDA BF16' if device == 'cuda' else 'CPU float32',
            'model_sha256': policy['model_sha256']}
