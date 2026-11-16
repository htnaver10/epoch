"""CUDA training, validation-only selection and reloadable inference export."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from model import WideFireMLP
from settings import ROOT, PARENT, V1, DEFAULT_RUN, path, read, write, sha, freeze_sources, artifact, run_config
from dataset import TrainingData, heldout, verify_previous_features
from firefullres.calibration import exact_ranking, sigmoid, fit_intercept, matched_false_alarm_cutoff, metrics


def atomic_save(filename, state):
    destination = path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    torch.save(state, temporary)
    temporary.replace(destination)


def load_model(device='cpu', run_id=DEFAULT_RUN):
    state = torch.load(path(artifact(run_id, 'inference.pt', 'models')), map_location=device, weights_only=True)
    if state['kind'] != 'wide_diverse_fire_v2' or state['requires_viirs_at_inference']:
        raise ValueError('Unsupported wide-model export')
    model = WideFireMLP(**state['architecture']).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    return model, state


def predict(model, x, stats, batch_size=2048):
    model.eval()
    device = next(model.parameters()).device
    mean = np.array(stats['mean'], np.float32)
    scale = np.array(stats['scale'], np.float32)
    output = np.empty(len(x), np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            values = np.clip((np.asarray(x[start:start + batch_size]) - mean) / scale, -10, 10)
            tensor = torch.from_numpy(values).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(tensor)
            output[start:start + len(values)] = logits.float().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite predictions')
    return output


def aggregate_cells(split, logits, raw):
    cell = np.load(PARENT / f'hybrid_v1/data/{split}_cell.npy', allow_pickle=False)
    if len(cell) != len(logits):
        raise ValueError('Native cell/query exposure differs')
    count = int(cell.max()) + 1
    target = np.zeros(count, bool); baseline = target.copy(); score = np.full(count, -np.inf)
    np.maximum.at(target, cell, raw['y'] > .5)
    np.maximum.at(baseline, cell, raw['modis'] > .5)
    np.maximum.at(score, cell, logits)
    if not np.isfinite(score).all(): raise ValueError('Unobserved native-cell index')
    return target, baseline, score


def profile(run_id=DEFAULT_RUN):
    cfg, run = run_config(run_id)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the existing CUDA environment')
    torch.set_num_threads(4)
    model = WideFireMLP(**run['architecture']).cuda()
    model.initialize_prior(0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    batch = cfg['training']['batch_size']
    x = torch.randn(batch, run['architecture']['feature_count'], device='cuda')
    y = (torch.arange(batch, device='cuda') % 4 == 0).float()
    weights = torch.where(y > 0, 0.004, 0.999 / 0.75)
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(x)
        loss = (F.binary_cross_entropy_with_logits(z.float(), y, reduction='none') * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        if not torch.isfinite(loss):
            raise ValueError('GPU preflight produced nonfinite loss')
    result = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'parameters': sum(p.numel() for p in model.parameters()), 'architecture': model.architecture,
              'batch_size': batch, 'median_step_seconds': float(np.median(elapsed[1:])),
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
              'finite_cuda_bf16_update': True}
    write(artifact(run_id, 'gpu_profile.json'), result)
    print(result, flush=True)
    return result


def train(run_id=DEFAULT_RUN):
    lock = freeze_sources(run_id)
    cfg, run = run_config(run_id)
    options = cfg['training']
    if path(artifact(run_id, 'completed.json', 'models')).exists():
        done = read(artifact(run_id, 'completed.json', 'models'))
        if done['inference_sha256'] != sha(path(artifact(run_id, 'inference.pt', 'models'))):
            raise ValueError('Completed inference export changed')
        return done
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the supplied training launcher')
    torch.set_num_threads(4)
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    rng = np.random.default_rng(options['seed'])
    verify_previous_features()
    data = TrainingData(run['dataset'])
    validation = heldout('validation')
    model = WideFireMLP(**run['architecture']).cuda()
    model.initialize_prior(data.prevalence)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    start_epoch, best_ap, stale, history = 0, -1., 0, []
    if path(artifact(run_id, 'last.pt', 'models')).exists():
        state = torch.load(path(artifact(run_id, 'last.pt', 'models')), map_location='cuda', weights_only=True)
        if state['software_lock'] != lock or state['config'] != cfg:
            raise ValueError('Resume requires identical code/configuration/data')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        start_epoch, best_ap, stale, history = state['epoch'] + 1, state['best_ap'], state['stale'], state['history']
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters; {len(data.y):,} real training queries; cohorts={data.receipt["groups"]}', flush=True)
    for epoch in range(start_epoch, options['epochs']):
        started = time.perf_counter()
        lr = options['learning_rate'] * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / options['epochs'])) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        losses = []
        cohort_draws = np.zeros(len(data.pools), np.int64)
        positive_draws = 0
        torch.cuda.reset_peak_memory_stats()
        for step in range(options['steps_per_epoch']):
            x, y, correction, groups = data.draw(rng, options['batch_size'], options['positive_sample_fraction'])
            cohort_draws += np.bincount(groups, minlength=len(data.pools))
            positive_draws += int(y.sum())
            x, y, correction = [torch.from_numpy(a).cuda() for a in (x, y, correction)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            # Mean, rather than sum(weights) normalization, preserves the exact
            # class-stratified importance-sampling expectation for each cohort.
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction='none') * correction).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; optimizer update refused')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        z = predict(model, validation['X'], data.stats, options['inference_batch_size'])
        ranking = exact_ranking(z, validation['y'], validation['w'])
        ap = ranking['average_precision_exact']
        if ap > best_ap:
            best_ap, stale = ap, 0
            improved = True
        else:
            stale += 1
            improved = False
        row = {'epoch': epoch + 1, 'training_group_balanced_logloss_estimate': float(np.mean(losses)),
               'validation_average_precision': ap,
               'validation_logloss': float(np.mean(np.logaddexp(0., z.astype(np.float64)) - validation['y'] * z)),
               'learning_rate': lr, 'seconds': time.perf_counter() - started,
               'sampled_queries': options['steps_per_epoch'] * options['batch_size'],
               'sampled_positive_queries': positive_draws, 'cohort_draws': cohort_draws.tolist(),
               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30}
        history.append(row)
        state = {'kind': 'wide_diverse_fire_v2', 'requires_viirs_at_inference': False,
                 'model': model.state_dict(), 'architecture': model.architecture, 'stats': data.stats,
                 'config': cfg, 'software_lock': lock, 'optimizer': optimizer.state_dict(),
                 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                 'numpy_rng': rng.bit_generator.state, 'epoch': epoch, 'best_ap': best_ap,
                 'stale': stale, 'history': history}
        if improved:
            atomic_save(artifact(run_id, 'best.pt', 'models'), state)
        atomic_save(artifact(run_id, 'last.pt', 'models'), state)
        write(artifact(run_id, 'training_history.json'), history)
        write(artifact(run_id, 'run_status.json'), {'stage': 'training', **row, 'best_validation_ap': best_ap})
        print(__import__('json').dumps(row), flush=True)
        if stale >= options['patience']:
            break
    selected = torch.load(path(artifact(run_id, 'best.pt', 'models')), map_location='cpu', weights_only=True)
    exported = {key: selected[key] for key in ('kind', 'requires_viirs_at_inference', 'model', 'architecture', 'stats', 'epoch', 'software_lock')}
    atomic_save(artifact(run_id, 'inference.pt', 'models'), exported)
    restored, saved = load_model(run_id=run_id)
    if any(not torch.equal(value, selected['model'][key]) for key, value in restored.state_dict().items()):
        raise ValueError('Inference export differs from selected weights')
    done = {'run_id': run_id, 'dataset': run['dataset'], 'parameters': sum(p.numel() for p in restored.parameters()), 'epochs_completed': len(history),
            'selected_epoch': selected['epoch'] + 1, 'best_validation_average_precision': best_ap,
            'inference_sha256': sha(path(artifact(run_id, 'inference.pt', 'models'))), 'inference_bytes': path(artifact(run_id, 'inference.pt', 'models')).stat().st_size,
            'exact_export_reload_passed': True, 'early_stopping': len(history) < options['epochs']}
    write(artifact(run_id, 'completed.json', 'models'), done)
    return done


def evaluate(run_id=DEFAULT_RUN):
    freeze_sources(run_id)
    cfg, run = run_config(run_id)
    model, saved = load_model('cuda', run_id)
    digest = sha(path(artifact(run_id, 'inference.pt', 'models')))
    policy_file = artifact(run_id, 'operating_policy.json')
    if path(policy_file).exists():
        policy = read(policy_file)
        if policy['model_sha256'] != digest:
            raise ValueError('Frozen validation policy belongs to different weights')
    else:
        validation = heldout('validation')
        z = predict(model, validation['X'], saved['stats'], read('config.json')['training']['inference_batch_size'])
        cutoff, budget = matched_false_alarm_cutoff(z, validation['y'], validation['w'], validation['modis'])
        cy, cb, cs = aggregate_cells('validation', z, validation)
        cell_cutoff, cell_budget = matched_false_alarm_cutoff(cs, cy, np.ones(len(cy)), cb)
        policy = {'fit_split': 'validation', 'test_used_for_fit': False, 'model_sha256': digest,
                  'raw_logit_cutoff': cutoff, 'logit_offset': fit_intercept(z, validation['y'], validation['w']),
                  'false_positive_budget': budget, 'benchmark_previously_examined': True,
                  'native_cell': {'fit_split': 'validation', 'cutoff': cell_cutoff, 'false_positive_budget': cell_budget,
                      'aggregation': 'maximum score per observed native MODIS cell; any-positive VIIRS reference'}}
        write(policy_file, policy)
        print('Validation operating policy frozen before test predictions.', flush=True)
    if path(artifact(run_id, 'evaluation.json')).exists():
        report = read(artifact(run_id, 'evaluation.json'))
        if report['policy'] != policy:
            raise ValueError('Sealed evaluation has another policy')
        return report
    parent_report = __import__('json').loads((PARENT / 'results/evaluation.json').read_text())
    report = {'policy': policy, 'new_blind_test': False, 'training': read(artifact(run_id, 'completed.json', 'models')),
              'run_id': run_id, 'dataset': run['dataset'], 'native_cell': {}, 'expansion': read('results/expansion_receipt.json') if run['dataset'] == 'd2' else None, 'features': read('data/feature_receipt.json') if run['dataset'] == 'd2' else __import__('json').loads((V1 / 'data/feature_receipt.json').read_text())}
    for split in ('validation', 'test'):
        raw = heldout(split)
        z = predict(model, raw['X'], saved['stats'], read('config.json')['training']['inference_batch_size'])
        score = sigmoid(z + policy['logit_offset'])
        cy, cb, cs = aggregate_cells(split, z, raw)
        report['native_cell'][split] = {'model': metrics(sigmoid(cs), cy, np.ones(len(cy)), cs >= policy['native_cell']['cutoff'], ranking_score=cs),
            'modis': metrics(cb.astype(float), cy, np.ones(len(cy)), cb)}
        report[split] = {'wide_frozen_policy': metrics(score, raw['y'], raw['w'], z.astype(np.float64) >= policy['raw_logit_cutoff'], ranking_score=z),
                         'parent_mlp_frozen_policy': parent_report[split]['mlp_frozen_policy'],
                         'previous50_d1': __import__('json').loads((V1 / 'results/evaluation.json').read_text())[split]['wide_frozen_policy'],
                         'regular_modis': parent_report[split]['regular_modis']}
        target = path(artifact(run_id, f'{split}_predictions.npz'))
        with target.open('wb') as stream:
            np.savez_compressed(stream, logits=z, target=np.asarray(raw['y']), weight=np.asarray(raw['w']),
                                tile_index=np.asarray(raw['tile_index']), flat_index=np.asarray(raw['flat_index']))
    write(artifact(run_id, 'evaluation.json'), report)
    lines = ['# Second expansion / ' + run_id, '',
             f'Parameters: **{report["training"]["parameters"]:,}**, selected epoch **{report["training"]["selected_epoch"]}**.', '',
             f'Dataset {run["dataset"]}: cached expansion windows: **{report["features"]["new_tiles"]:,}**; additional known queries: **{report["features"]["new_queries"]:,}**; additional positive reference queries: **{report["features"]["new_positive_queries"]:,.0f}**.', '',
             'Training cohorts: ' + ', '.join(report['features']['groups']) + '.', '',
             'Original validation/test exposure is preserved. Both cohorts were examined in previous experiments; this is a reused research benchmark. VIIRS reference queries are correlated and are not independent wildfire incidents.', '']
    for split in ('validation', 'test'):
        lines += ['## ' + split.capitalize(), '', '| Model | Precision | Recall | F1 | TP | FP | FN | AP |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, value in report[split].items():
            lines.append(f'| {name} | {100 * value["precision"]:.2f}% | {100 * value["recall"]:.2f}% | {value["f1"]:.4f} | {value["true_positives"]:.0f} | {value["false_positives"]:.0f} | {value["false_negatives"]:.0f} | {value["average_precision_exact"]:.4f} |')
        lines.append('')
    lines += ['## Native MODIS-cell test comparison', '', '| Model | Precision | Recall | F1 | TP | FP | FN | AP |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in report['native_cell']['test'].items():
        lines.append(f'| {name} | {100*value["precision"]:.2f}% | {100*value["recall"]:.2f}% | {value["f1"]:.4f} | {value["true_positives"]:.0f} | {value["false_positives"]:.0f} | {value["false_negatives"]:.0f} | {value["average_precision_exact"]:.4f} |')
    comparison = report['test']['wide_frozen_policy']['f1'] - report['test']['parent_mlp_frozen_policy']['f1']
    lines += [f'Wide-model benchmark F1 change versus parent: **{comparison:+.4f}**.', '',
              'The 50M/d2 versus 200M/d2 comparison holds the expanded dataset and optimizer/sampling rules fixed. The 200M/d1 versus 200M/d2 comparison holds architecture fixed. One seed per condition limits conclusions about random variation. The class-stratified sampler corrects importance weights to each cohort\'s actual class prevalence; cohorts contribute equally. Sampling with replacement does not promise every background query is visited.', '',
              'Native calibration, cross-sensor timing and geographic inversion are retained. Reference eligibility matches the parent: VIIRS mask classes 5/8/9, bowtie QA bit 22 excluded, valid mapped MODIS support and core/halo restrictions. QA bits 0-6 are reported for expansion observations rather than added as new label exclusions. Label-informed crop selection occurs only within training and makes the expanded training set fire-enriched. Importance correction applies to curated training rows; it does not undo the crop selection or describe deployment prevalence. No synthetic fire labels were created. New independent test incidents are still required to establish generalization.', '']
    path(artifact(run_id, 'TRAINING_SUMMARY.md')).write_text('\n'.join(lines), encoding='utf-8')
    write(artifact(run_id, 'run_status.json'), {'stage': 'complete', 'run_id': run_id, 'parameters': report['training']['parameters'],
                                    'test_f1': report['test']['wide_frozen_policy']['f1'],
                                    'parent_test_f1': report['test']['parent_mlp_frozen_policy']['f1']})
    print(__import__('json').dumps(report['test'], indent=2), flush=True)
    return report
