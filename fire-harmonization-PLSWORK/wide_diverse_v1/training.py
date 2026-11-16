"""CUDA training, validation-only selection and reloadable inference export."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from model import WideFireMLP
from settings import ROOT, PARENT, path, read, write, sha, freeze_sources
from dataset import TrainingData, heldout
from firefullres.calibration import exact_ranking, sigmoid, fit_intercept, matched_false_alarm_cutoff, metrics


def atomic_save(filename, state):
    destination = path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    torch.save(state, temporary)
    temporary.replace(destination)


def load_model(device='cpu'):
    state = torch.load(path('models/inference.pt'), map_location=device, weights_only=True)
    if state['kind'] != 'wide_diverse_fire_v1' or state['requires_viirs_at_inference']:
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


def profile():
    cfg = read('config.json')
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the existing CUDA environment')
    torch.set_num_threads(4)
    model = WideFireMLP(**cfg['architecture']).cuda()
    model.initialize_prior(0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'],
                                  weight_decay=cfg['training']['weight_decay'])
    batch = cfg['training']['batch_size']
    x = torch.randn(batch, cfg['architecture']['feature_count'], device='cuda')
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
    write('results/gpu_profile.json', result)
    print(result, flush=True)
    return result


def train():
    lock = freeze_sources()
    cfg = read('config.json')
    options = cfg['training']
    if path('models/completed.json').exists():
        done = read('models/completed.json')
        if done['inference_sha256'] != sha(path('models/inference.pt')):
            raise ValueError('Completed inference export changed')
        return done
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the supplied training launcher')
    torch.set_num_threads(4)
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    rng = np.random.default_rng(options['seed'])
    data = TrainingData()
    validation = heldout('validation')
    model = WideFireMLP(**cfg['architecture']).cuda()
    model.initialize_prior(data.prevalence)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    start_epoch, best_ap, stale, history = 0, -1., 0, []
    if path('models/last.pt').exists():
        state = torch.load(path('models/last.pt'), map_location='cuda', weights_only=True)
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
        state = {'kind': 'wide_diverse_fire_v1', 'requires_viirs_at_inference': False,
                 'model': model.state_dict(), 'architecture': model.architecture, 'stats': data.stats,
                 'config': cfg, 'software_lock': lock, 'optimizer': optimizer.state_dict(),
                 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                 'numpy_rng': rng.bit_generator.state, 'epoch': epoch, 'best_ap': best_ap,
                 'stale': stale, 'history': history}
        if improved:
            atomic_save('models/best.pt', state)
        atomic_save('models/last.pt', state)
        write('results/training_history.json', history)
        write('results/run_status.json', {'stage': 'training', **row, 'best_validation_ap': best_ap})
        print(__import__('json').dumps(row), flush=True)
        if stale >= options['patience']:
            break
    selected = torch.load(path('models/best.pt'), map_location='cpu', weights_only=True)
    exported = {key: selected[key] for key in ('kind', 'requires_viirs_at_inference', 'model', 'architecture', 'stats', 'epoch', 'software_lock')}
    atomic_save('models/inference.pt', exported)
    restored, saved = load_model()
    if any(not torch.equal(value, selected['model'][key]) for key, value in restored.state_dict().items()):
        raise ValueError('Inference export differs from selected weights')
    done = {'parameters': sum(p.numel() for p in restored.parameters()), 'epochs_completed': len(history),
            'selected_epoch': selected['epoch'] + 1, 'best_validation_average_precision': best_ap,
            'inference_sha256': sha(path('models/inference.pt')), 'inference_bytes': path('models/inference.pt').stat().st_size,
            'exact_export_reload_passed': True, 'early_stopping': len(history) < options['epochs']}
    write('models/completed.json', done)
    return done


def evaluate():
    freeze_sources()
    model, saved = load_model('cuda')
    digest = sha(path('models/inference.pt'))
    policy_file = 'results/operating_policy.json'
    if path(policy_file).exists():
        policy = read(policy_file)
        if policy['model_sha256'] != digest:
            raise ValueError('Frozen validation policy belongs to different weights')
    else:
        validation = heldout('validation')
        z = predict(model, validation['X'], saved['stats'], read('config.json')['training']['inference_batch_size'])
        cutoff, budget = matched_false_alarm_cutoff(z, validation['y'], validation['w'], validation['modis'])
        policy = {'fit_split': 'validation', 'test_used_for_fit': False, 'model_sha256': digest,
                  'raw_logit_cutoff': cutoff, 'logit_offset': fit_intercept(z, validation['y'], validation['w']),
                  'false_positive_budget': budget, 'benchmark_previously_examined': True}
        write(policy_file, policy)
        print('Validation operating policy frozen before test predictions.', flush=True)
    if path('results/evaluation.json').exists():
        report = read('results/evaluation.json')
        if report['policy'] != policy:
            raise ValueError('Sealed evaluation has another policy')
        return report
    parent_report = __import__('json').loads((PARENT / 'results/evaluation.json').read_text())
    report = {'policy': policy, 'new_blind_test': False, 'training': read('models/completed.json'),
              'expansion': read('results/expansion_receipt.json'), 'features': read('data/feature_receipt.json')}
    for split in ('validation', 'test'):
        raw = heldout(split)
        z = predict(model, raw['X'], saved['stats'], read('config.json')['training']['inference_batch_size'])
        score = sigmoid(z + policy['logit_offset'])
        report[split] = {'wide_frozen_policy': metrics(score, raw['y'], raw['w'], z.astype(np.float64) >= policy['raw_logit_cutoff'], ranking_score=z),
                         'parent_mlp_frozen_policy': parent_report[split]['mlp_frozen_policy'],
                         'regular_modis': parent_report[split]['regular_modis']}
        target = path(f'results/{split}_predictions.npz')
        with target.open('wb') as stream:
            np.savez_compressed(stream, logits=z, target=np.asarray(raw['y']), weight=np.asarray(raw['w']),
                                tile_index=np.asarray(raw['tile_index']), flat_index=np.asarray(raw['flat_index']))
    write('results/evaluation.json', report)
    lines = ['# Wide diverse MLP: completed experiment', '',
             f'Parameters: **{report["training"]["parameters"]:,}**, selected epoch **{report["training"]["selected_epoch"]}**.', '',
             f'Additional real training windows: **{report["features"]["new_tiles"]:,}**; additional known queries: **{report["features"]["new_queries"]:,}**; additional positive reference queries: **{report["features"]["new_positive_queries"]:,.0f}**.', '',
             'Training cohorts: ' + ', '.join(report['features']['groups']) + '.', '',
             'Original validation/test exposure is preserved. Both cohorts were examined in previous experiments; this is a reused research benchmark. VIIRS reference queries are correlated and are not independent wildfire incidents.', '']
    for split in ('validation', 'test'):
        lines += ['## ' + split.capitalize(), '', '| Model | Precision | Recall | F1 | TP | FP | FN | AP |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, value in report[split].items():
            lines.append(f'| {name} | {100 * value["precision"]:.2f}% | {100 * value["recall"]:.2f}% | {value["f1"]:.4f} | {value["true_positives"]:.0f} | {value["false_positives"]:.0f} | {value["false_negatives"]:.0f} | {value["average_precision_exact"]:.4f} |')
        lines.append('')
    comparison = report['test']['wide_frozen_policy']['f1'] - report['test']['parent_mlp_frozen_policy']['f1']
    lines += [f'Wide-model benchmark F1 change versus parent: **{comparison:+.4f}**.', '',
              'Model capacity, training coverage, and sampling objective changed together. This experiment cannot isolate a parameter-count effect. The class-stratified sampler corrects importance weights to each cohort\'s actual class prevalence; cohorts contribute equally. Sampling with replacement does not promise every background query is visited.', '',
              'Native calibration, cross-sensor timing and geographic inversion are retained. Reference eligibility matches the parent: VIIRS mask classes 5/8/9, bowtie QA bit 22 excluded, valid mapped MODIS support and core/halo restrictions. QA bits 0-6 are reported for expansion observations rather than added as new label exclusions. Label-informed crop selection occurs only within training and makes the expanded training set fire-enriched. Importance correction applies to curated training rows; it does not undo the crop selection or describe deployment prevalence. No synthetic fire labels were created. New independent test incidents are still required to establish generalization.', '']
    path('results/TRAINING_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    write('results/run_status.json', {'stage': 'complete', 'parameters': report['training']['parameters'],
                                    'test_f1': report['test']['wide_frozen_policy']['f1'],
                                    'parent_test_f1': report['test']['parent_mlp_frozen_policy']['f1']})
    print(__import__('json').dumps(report['test'], indent=2), flush=True)
    return report
