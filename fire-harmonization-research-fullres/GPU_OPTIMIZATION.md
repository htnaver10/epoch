# GPU training optimizations and their justification

These changes address costs visible in the implementation. They do not establish
an RTX 5070 speedup, memory fit or fire-localization result. This host cannot run
PyTorch/CUDA, and package hosts fail DNS. GPU-dependent choices require the tests
and profiles below on the actual desktop RTX 5070 12 GB.

All scripts, caches, configurations, measurements and checkpoints stay in this
folder. Model inputs, source-resolution imagery, reference definitions, geographic
precision and exposure-weighted loss definitions are preserved.

## Implemented changes

| Change | Justification | Setting / tradeoff |
|---|---|---|
| Materialize each required NPZ member once | The former loader decompressed 51 members for 11 distinct arrays per tile | Always enabled; normalized inputs/labels match the former path bit for bit |
| Bounded normalized-tile LRU | Avoid repeat decompression/normalization when the working set fits | `tile_cache_mib`, **per worker per loader**; 0 disables; never an unbounded dataset cache |
| Persistent, reusable loaders | Avoid restarting workers and losing verification/cache state every epoch | `persistent_workers`; ignored with `workers: 0`; default worker start method is `spawn` to avoid forking initialized CUDA |
| Prefetch complete accumulation windows | Workers prepare the next effective batch while the GPU processes the current one, replacing main-thread collection of many separate loader results | `prefetch_factor` counts **windows** during training, physical batches during evaluation; start at 1 |
| Shape-aware windows and query padding | Reduce wasted work from mixing very different query sizes and limit convolution/compile shape variation | `query_bucket_size: 32` in RTX presets; 0 disables query bucketing; native crop shapes are always separated |
| Larger physical-batch candidate | Effective batch accumulation does not offer the same parallel work as a larger physical batch | Throughput preset is 4 tiles × 4 accumulated; baseline stays 1 × 16 until measured |
| Channels-last convolution inputs/model | Gives BF16 convolutions an alternative Tensor Core-friendly layout | `memory_format`; only NCHW image tensors are converted, **never** the NHW2 query grid; mixed-operation conversion overhead may outweigh benefits |
| cuDNN algorithm autotuning | Reuse faster convolution algorithms for recurring shapes | `cudnn_benchmark`; shape bucketing limits retuning; selected algorithms may vary between runs |
| Fused AdamW option | Reduce optimizer kernel overhead | `optimizer_impl`: `auto` chooses fused on CUDA, default on CPU; `default`, `foreach`, `fused` can be selected explicitly; parameters and moments remain FP32 |
| One finite check per optimizer update | Removes the Python/GPU scalar decision after every microbatch | Accumulate finite-loss flags on device, check loss and gradient norm before stepping; nonfinite updates still fail |
| Optional CUDA transfer prefetch | Overlap the next microbatch's pinned-memory copies with compute on a dedicated stream | `cuda_prefetch`; requires CUDA and pinned memory; wait/record stream dependencies protect tensor lifetimes; extra VRAM for an additional input batch |
| Optional model compilation | Reduce Python/pointwise kernel launch costs | `compile_mode`; disabled by default; eager canonical model owns checkpoint keys, optimizer and RNG state; caches live in `.cache/` here |
| Batched evaluation/inference | Reuse the training batch/layout settings; avoid copying targets to GPU and back during validation | Inference mode; predictions are cropped back to original query shapes before export; CPU exact/histogram metrics remain the reference implementation |
| Representative profiling and epoch logging | The old profile excluded loader waiting and used an optimizer step every microbatch | Same exposure-normalized update function as training; end-to-end tiles/s, loader wait, padding, memory, actual batch counts, validation/checkpoint time and optional trace |

Source files: [loading.py](src/firefullres/loading.py),
[runtime.py](src/firefullres/runtime.py), [engine.py](src/firefullres/engine.py).

## Correctness and research boundaries

No tile is dropped or duplicated by bucketing. Different native crop sizes never
share a physical batch. Padded queries have zero weight and out-of-support grid
coordinates; the query head does not normalize/mix across padded query positions.
Bucket tails produce smaller effective batches. A nominal 4 × 4 setting means
**up to** 16 tiles per update, not a promise that every update has 16. Reordering
and smaller tails can change the optimizer trajectory; compare held-out metrics
after retraining. Set query bucketing to 0 to avoid query-bucket fragmentation.

Each fine/native loss is normalized by its own actual exposure sum across the
whole window, including partial windows. Zero-fine-exposure windows are skipped;
nonfinite losses/gradients fail before an optimizer update. Source/event split,
checksum, synthetic opt-in and train-only normalization guards remain active.

The optional cache is tied to the frozen tile/normalizer run. File size/mtime are
checked even on cache hits; a changed signature triggers checksum verification.
Edit source files only by creating a new audited dataset version. The cache holds
no GPU tensors and excludes geographic export arrays. New manifests carry shapes;
older manifests remain readable through their small NPY headers and actual tile
shapes are validated when loaded.

Stored latitude/longitude stay FP64. Query-grid construction/sampling and loss
reductions stay FP32/FP64 as before. BF16 autocast is for neural operations;
there is no FP8/quantized optimizer state, coordinate downcasting, automatic
resolution reduction, teacher-feature leakage or change to positive/negative QA.

Checkpoints save the original model state, not a `torch.compile` wrapper. Resume
checks configuration and data and restarts at an epoch boundary using seed+epoch
shuffle. Training format 2 records the new window order; older checkpoints still
load for inference, but require a new training run rather than an inexact resume.
CUDA grid-sampling backward/autotuning can be nondeterministic.

## Presets and memory budgeting

- [train_rtx5070.json](configs/train_rtx5070.json): 1 × 16, 2 workers,
  window prefetch 1, no tile cache, channels-last/BF16, autotuning and automatic
  optimizer implementation. Compilation and CUDA copy prefetch are off.
- [train_rtx5070_throughput.json](configs/train_rtx5070_throughput.json): an
  **unmeasured candidate**, 4 × 4, 4 workers, 64 MiB cache per worker and CUDA copy
  prefetch. Compilation stays off. Profile before a long training run.
- [smoke_cpu.json](configs/smoke_cpu.json): small software-only model with no
  CUDA assumptions; missing new keys have portable defaults.

Training prefetch stores effective windows: workers × prefetch factor windows
can be queued, in addition to the current window, active worker batches and
temporary pinning/collation buffers. Persistent training **and validation**
loaders retain separate caches. The throughput preset permits up to 512 MiB
of configured tile cache across its 4+4 workers, in addition to data buffers,
metadata and Python/process memory. Start with cache 0/fewer workers if RAM is
limited. More workers/prefetch are not automatically faster. Keep tiles on a
local SSD, and avoid decoding raw HDF/downloads in the training loop.

Keep activation checkpointing off initially: recomputation trades speed for
memory. Enable it only if it enables a better measured batch/crop configuration.
Compilation and copy prefetch can increase VRAM; leave headroom for the largest
query grids, convolution workspaces and desktop display/other processes.

## Measurement procedure on the RTX host

First follow [RUNNING.md](RUNNING.md) to set up the supported CUDA environment,
audit real tiles, fit train-only normalizers and run all PyTorch tests. A skipped
CUDA test on the target RTX machine is not GPU validation.

```sh
python -m unittest discover -s fire-harmonization-research-fullres/tests -v
python fire-harmonization-research-fullres/run.py profile --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --warmup-steps 5 --steps 30 --output results/rtx_baseline_profile.json
python fire-harmonization-research-fullres/run.py profile --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070_throughput.json --warmup-steps 5 --steps 30 --output results/rtx_throughput_profile.json
python fire-harmonization-research-fullres/run.py profile --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070_throughput.json --worst-case --warmup-steps 5 --steps 30 --output results/rtx_large_queries_profile.json
```

`--steps` counts **accumulated optimizer windows**, not individual microbatches.
The report includes actual processed tiles/microbatches/updates because bucket
tails and small datasets differ from the nominal effective batch. It repeats
small datasets when needed and resets to the leading windows after warm-up.
Default sampling uses the training seed+epoch shuffle; `--worst-case` uses
largest-query-first bucket order instead.
The profile changes only its disposable model, not saved training checkpoints.

For a diagnostic timeline, run a separate short capture:

```sh
python fire-harmonization-research-fullres/run.py profile --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070_throughput.json --warmup-steps 5 --steps 5 --trace results/rtx_training_trace.json --output results/rtx_trace_profile.json
```

Open the Chrome trace in a local trace viewer. Capture overhead changes speed;
use **untraced runs** for throughput comparisons. CUDA event window times include
device timeline gaps and transfers, not pure kernel execution/SM occupancy.
`loader_wait_seconds` is host blocking and can partly overlap prior GPU work;
its fraction is not a direct GPU-idle percentage. Validation/checkpoint are
excluded from the short profile and timed separately in epoch logs. Setup/worker
startup and compilation cold starts are reported or included in warm-up/first
epoch; recurring shape compilation can still occur during measurement.

Use separate config files to compare 1×16, 2×8, 4×4 and 8×2 on the same manifest.
Change one execution option at a time after that, repeat measurements, compare
actual sample/update counts and evaluate the selected recipe on held-out data.
Increase warm-up and coverage for variable shapes or compilation. `--worst-case`
starts with largest query buckets; it is not exhaustive memory-fit proof.

To test compilation, copy a successful config, set `compile_mode` to `default`
or `reduce-overhead`, and profile a new process/output. `compile_dynamic: true`
can reduce shape recompilation, but may alter available optimizations. No CUDA
graph/compilation success or benefit is assumed. Set `compile_mode: null` if
unsupported or slower. Do not change a resumed run's execution recipe silently.
Compilation needs a compatible PyTorch/compiler/Triton environment; it is not
required for the normal eager training path.

## Local evidence and reproducible I/O benchmark

[optimization_tile_io.json](results/optimization_tile_io.json) records a
30-iteration, warm-cache synthetic 32-pixel fixture comparison: 51→11 compressed
member reads, approximately 6.1→1.9 ms median NumPy load/normalization and bitwise
equal inputs/labels. This does **not** measure RTX training or native NASA files.

```sh
python fire-harmonization-research-fullres/tools/benchmark_tile_io.py --manifest data/manifests/demo.jsonl --stats data/manifests/demo-stats.json --allow-synthetic --iterations 30 --output results/demo_io_benchmark.json
```

Remove `--allow-synthetic` and use audited native manifest/stats to benchmark a
real prepared tile. The benchmark excludes initial hash warm-up and CUDA.
Software tests cover one-read loading, old-manifest headers, cache eviction and
tamper rejection, complete deterministic sampling, partial-window weighted
updates, finite guards, shape/layout/export behavior, spawn worker reuse,
checkpoint/compile wrapper keys, CPU profile and a CUDA transfer/fused optimizer
comparison. PyTorch-dependent tests are present but could not run here.

Primary references: [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[DataLoader](https://docs.pytorch.org/docs/stable/data.html),
[CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams),
[compile options](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
[profiler](https://docs.pytorch.org/docs/stable/profiler.html).
