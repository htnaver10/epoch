"""Bounded, verified NumPy tile loading and deterministic shape-aware windows.

This module deliberately has no PyTorch dependency, so I/O correctness and
sampling coverage can be checked on a preparation-only machine.
"""
from collections import OrderedDict
import math
import random
import zipfile

import numpy as np

from .core import INPUT_KEYS, REQUIRED_KEYS, path, sha256, validate_arrays, normalize


def tile_shapes(row):
    """New manifests carry shapes; old manifests need only small NPY headers."""
    if 'native_shape' in row and 'query_shape' in row:
        shapes = tuple(row['native_shape']), tuple(row['query_shape'])
    else:
        shapes = []
        with zipfile.ZipFile(path(row['tile'])) as archive:
            for key in ('thermal', 'target'):
                with archive.open(key + '.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, _ = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported tile NPY header; regenerate the native tile')
                    shapes.append(tuple(shape[-2:]))
    if any(len(s) != 2 or any(not isinstance(n, int) or n <= 0 for n in s) for s in shapes):
        raise ValueError('Invalid manifest tile shapes')
    return tuple(shapes)


class TileStore:
    """One decompression per required member, with an optional per-worker LRU.

    Cache entries contain normalized FP32 inputs and labels only, not geographic
    export arrays. File signatures are checked even on cache hits. A new signature
    triggers checksum verification; modified data never silently enter a run.
    """
    def __init__(self, rows, stats, cache_mib=0):
        self.rows = rows
        self.stats = stats
        self.limit = int(float(cache_mib) * 1024**2)
        if self.limit < 0:
            raise ValueError('tile_cache_mib must be nonnegative')
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.verified = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        key = row['tile']
        cache_key = (key, index)
        stat = path(key).stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.verified.get(key) != signature:
            if sha256(key) != row['sha256']:
                raise ValueError('Tile changed since manifest freeze')
            self.verified[key] = signature
        if cache_key in self.cache:
            value, size = self.cache.pop(cache_key)
            self.cache[cache_key] = (value, size)
            return value
        with np.load(path(key), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in REQUIRED_KEYS}
        validate_arrays(arrays)
        for name, shape in [('native_shape', arrays['thermal'].shape[-2:]),
                            ('query_shape', arrays['target'].shape)]:
            if name in row and tuple(row[name]) != tuple(shape):
                raise ValueError('Manifest shape disagrees with frozen tile')
        inputs = {name: normalize(name, arrays[name], self.stats) if name != 'query_grid'
                  else arrays[name] for name in INPUT_KEYS}
        value = {'inputs': inputs,
                 **{name: arrays[name].astype(np.float32, copy=False)
                    for name in ('target', 'weight', 'native_target', 'native_weight')},
                 'record': row}
        size = sum(a.nbytes for a in inputs.values()) + sum(
            value[k].nbytes for k in ('target', 'weight', 'native_target', 'native_weight'))
        if self.limit and size <= self.limit:
            while self.cache_bytes + size > self.limit:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.cache_bytes -= old_size
            self.cache[cache_key] = (value, size)
            self.cache_bytes += size
        return value


class WindowSampler:
    """Never drop/duplicate a tile; each window stays in one shape bucket.

    A training window holds up to microbatch * accumulation_steps tiles. Bucket
    tails are smaller effective batches, normalized by their actual exposure.
    """
    def __init__(self, rows, window_size, bucket_size=0, shuffle=False, seed=0):
        self.window_size = int(window_size)
        self.bucket_size = int(bucket_size)
        if self.window_size < 1 or self.bucket_size < 0:
            raise ValueError('Invalid window/bucket size')
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.shapes = [tile_shapes(row) for row in rows]
        self.buckets = {}
        for i, (native, query) in enumerate(self.shapes):
            q = tuple(math.ceil(n / self.bucket_size) * self.bucket_size for n in query) if self.bucket_size else ()
            self.buckets.setdefault((*native, *q), []).append(i)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum(math.ceil(len(v) / self.window_size) for v in self.buckets.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        windows = []
        for original in self.buckets.values():
            indices = original.copy()
            if self.shuffle:
                rng.shuffle(indices)
            windows.extend(indices[i:i+self.window_size] for i in range(0, len(indices), self.window_size))
        if self.shuffle:
            rng.shuffle(windows)
        return iter(windows)
