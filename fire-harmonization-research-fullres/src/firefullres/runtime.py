"""Configurable GPU execution, with FP32 geometry and canonical checkpoints."""
import contextlib
import os

import torch
from .core import path


IMAGE_KEYS = ('thermal', 'reflective_500', 'reflective_250', 'geometry')
LABEL_KEYS = ('target', 'weight', 'native_target', 'native_weight')
DEFAULTS = {
    'persistent_workers': True,
    'worker_start_method': 'spawn',
    'tile_cache_mib': 0,
    'query_bucket_size': 0,
    'memory_format': 'contiguous',
    'cudnn_benchmark': False,
    'optimizer_impl': 'default',
    'cuda_prefetch': False,
    'compile_mode': None,
    'compile_dynamic': False,
}


def resolve_config(config):
    result = {**DEFAULTS, **config}
    for name in ('microbatch', 'accumulation_steps', 'prefetch_factor'):
        if int(result.get(name, 1)) < 1:
            raise ValueError(name + ' must be positive')
    if int(result.get('workers', 0)) < 0 or float(result['tile_cache_mib']) < 0:
        raise ValueError('workers and tile_cache_mib must be nonnegative')
    if int(result['query_bucket_size']) < 0:
        raise ValueError('query_bucket_size must be nonnegative')
    if result['memory_format'] not in ('contiguous', 'channels_last'):
        raise ValueError('Unknown memory_format')
    if result['worker_start_method'] not in ('spawn', 'forkserver'):
        raise ValueError('worker_start_method must be spawn or forkserver; avoid forking initialized CUDA')
    if result['optimizer_impl'] not in ('default', 'auto', 'fused', 'foreach'):
        raise ValueError('Unknown optimizer_impl')
    if result['compile_mode'] not in (None, 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'):
        raise ValueError('Unknown compile_mode')
    if result['precision'] not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('Unknown precision')
    return result


def configure_model(model, config, device, compile_model=True):
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = bool(config['cudnn_benchmark'])
    if config['memory_format'] == 'channels_last':
        model.to(memory_format=torch.channels_last)
    if compile_model and config['compile_mode']:
        for variable, name in [('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TRITON_CACHE_DIR','triton')]:
            folder=path('.cache/'+name);folder.mkdir(parents=True,exist_ok=True)
            os.environ[variable]=str(folder)
        # Keep model itself for state_dict/optimizer; only wrap execution.
        return torch.compile(model, mode=config['compile_mode'], dynamic=config['compile_dynamic'])
    return model


def optimizer_for(model, config, device):
    implementation = config['optimizer_impl']
    if implementation == 'auto':
        implementation = 'fused' if device.type == 'cuda' else 'default'
    if implementation == 'fused' and device.type != 'cuda':
        raise ValueError('fused optimizer requires CUDA; use default/auto on CPU')
    options = {implementation: True} if implementation in ('fused', 'foreach') else {}
    return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                             weight_decay=config['weight_decay'], **options)


def move(batch, device, config=None, labels=True):
    inputs = {}
    channels_last = config and config['memory_format'] == 'channels_last'
    for key, value in batch['inputs'].items():
        kwargs = {'memory_format': torch.channels_last} if channels_last and key in IMAGE_KEYS else {}
        inputs[key] = value.to(device, non_blocking=True, **kwargs)
    return {**batch, 'inputs': inputs,
            **({key: batch[key].to(device, non_blocking=True) for key in LABEL_KEYS} if labels else {})}


def device_batches(batches, device, config):
    if not (device.type == 'cuda' and config['cuda_prefetch'] and config.get('pin_memory')):
        for batch in batches:
            yield move(batch, device, config)
        return
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    iterator = iter(batches)

    def load_next():
        raw = next(iterator, None)
        if raw is None:
            return None
        with torch.cuda.stream(stream):
            return move(raw, device, config)

    pending = load_next()
    try:
        while pending is not None:
            current = torch.cuda.current_stream(device)
            current.wait_stream(stream)
            batch = pending
            for value in [*batch['inputs'].values(), *(batch[k] for k in LABEL_KEYS)]:
                value.record_stream(current)
            pending = load_next()
            yield batch
    finally:
        # Also protects host buffers when a failed step aborts the generator.
        stream.synchronize()


def region(name, enabled):
    return torch.profiler.record_function(name) if enabled else contextlib.nullcontext()


def runtime_info(config, device):
    return {'device': str(device), 'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'memory_format': config['memory_format'], 'compile_mode': config['compile_mode'],
            'optimizer_impl': ('fused' if device.type == 'cuda' else 'default')
                if config['optimizer_impl'] == 'auto' else config['optimizer_impl'],
            'cuda_prefetch_active': bool(device.type == 'cuda' and config['cuda_prefetch'] and config.get('pin_memory'))}
