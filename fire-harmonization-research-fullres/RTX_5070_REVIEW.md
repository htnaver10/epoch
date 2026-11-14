# Native-resolution plan review: desktop RTX 5070, 12 GB

**Reviewed 2026-11-14.** Scope: the native MODIS tile model plan, scientific
validity, data engineering, and practical training on the user-confirmed desktop
RTX 5070 with 12 GB VRAM. OS, CPU, RAM, storage and driver are not yet specified.
This review does not benchmark that GPU or claim a trained native-image model.

## Decision

**A compact first model is a reasonable experiment on this card.** The earlier
plan's 16–24 GB suggestion was a headroom estimate, not a demonstrated minimum.
Twelve GB does require deliberate architecture, small microbatches and profiling.
The full optional temporal/teacher/transport architecture should not be the first
implementation.

The plan's larger unresolved risks are scientific: trustworthy paired native
observations, geographic alignment, one well-defined target, and whether MODIS
contains enough information to improve locations against an independent reference.
A more powerful GPU would not resolve those issues.

Use the [revised canonical plan](PLAN.md) and
[proposed initial profile](configs/rtx5070_proposed.json). All full-resolution
content, scripts, data and outputs belong in **`fire-harmonization-research-fullres/`**
at the project root. The existing coarse research and main app remain separate.

## 1. Evidence boundaries

Completed review work:

- Read the complete existing native plan and latest Experiment 04.5 report.
- Checked current NVIDIA specifications/compute capability and official PyTorch
  support, precision, memory and checkpoint documentation; checked data-reader
  and Windows/WSL considerations.
- Executed a reproducible standard-library memory/rounding worksheet and its
  numerical tests. Values below are component arithmetic, not GPU measurements.
- Prepared a portable CUDA compatibility preflight for the actual training host.

The review host is macOS/ARM and has no PyTorch installation or attached RTX 5070.
Its [local preflight](results/local_host_preflight.json) records that limitation.
No native data was acquired, no native architecture instantiated on CUDA, and
no training throughput, peak model VRAM or localization accuracy was measured.
The existing coarse models are not image-model initialization weights.

## 2. What the plan already gets right

The plan correctly calls for native thermal radiances, separately retained
250/500 m context, acquisition-level matching, QA and clear-sky/reference-validity
masks, MODIS-only inference, leakage-resistant event/region/time splits, and
independent location evaluation. It distinguishes output lattice spacing from
achieved precision and keeps multiple VIIRS pixels from being reduced to one
assumed fire center.

It also correctly stages data proof before model complexity and retains simple
native baselines. Those are important strengths. The necessary changes are to
make targets, geometry precision, 12 GB resource controls and the first executable
architecture more concrete.

## 3. Scientific/design issues, in priority order

| Priority | Issue | Consequence | Required resolution |
|---|---|---|---|
| Blocking | No native imagery/coverage dataset has been ingested | Existing count files cannot train the proposed image model | Open calibrated MODIS L1B/geolocation and VIIRS L2/geolocation bundles; audit real arrays and masks first |
| Blocking | Fine target/units remain partly ambiguous | Native-pixel fire status, overlap coverage, expected active cells and true fire existence can be mixed in losses/calibration | Freeze one target, validity denominator and map transformation before coding losses |
| High | Fine thermal locations are an ill-posed inverse problem | Many subpixel fire arrangements can produce similar 1 km thermal measurements | Predict conditional candidates/support with uncertainty; require measured added location information |
| High | VIIRS labels are an observation reference | Accurate reproduction of VIIRS pixels does not prove exact ground flame coordinates | Score at the reference scale and obtain close-time independent finer observations for stronger claims |
| High | Registration error can resemble localization error | The network can learn systematic map offsets or apparent sharpening | Quantify sensor registration first; independent stable-landmark checks and uncertainty-aware labels |
| High | A 15-minute pair is not necessarily a static scene | Growth/extinction between acquisitions can create contradictory labels | Audit actual scan/pixel time differences; strict-pair benchmark plus separate time-gap sensitivity |
| High | Narrow clean-pair selection can bias the cohort | Good clear-sky/nadir results may fail on cloudy, smoky or outer-angle observations | Report excluded populations and stratified/degraded-input tests; use an explicit support/abstention policy |
| High | Rare-fire sampling changes prevalence | High apparent precision on balanced crops may not survive full scenes | Record sampling/exposure weights and calibrate/evaluate on representative whole scenes |
| High | Earlier model outputs could leak labels | The final coarse model used 2012–2025 VIIRS labels, including proposed native evaluation years | Omit it from native features/teachers, or generate truly fold-isolated outputs without held-out event/year labels |
| High | Fine reflective features can paint forest priors | A detailed forest-shaped map can look plausible without improving thermal localization | Thermal-only versus multiresolution ablation, hard negatives and fixed-false-alarm localization scoring |
| High | Temporal context can persist old fires | Apparent recall increases can be stale detections | Current-image baseline first; acquisition-time-aware context and newly active/fire-extinction strata |
| Medium | Forest mask has its own resolution and latency | A small thermal anomaly is not confirmed as an unplanned forest wildfire | Prior-map availability/QA, ambiguous boundaries and explicit forest-context terminology |
| Medium | Too many auxiliary tasks/losses initially | Debugging and attribution become difficult, with unnecessary VRAM cost | Masked BCE-with-logits plus one well-defined native target first; add heads through ablation |
| Medium | 100 event groups/200–500 bundles is a pilot target | Counts alone do not establish enough independent diversity for global claims | Event-count learning curves, biome/time diversity and blocked uncertainty |
| Medium | Acceptance thresholds are provisional | A 20% matched-location improvement can hide lost recall or selection changes | Freeze thresholds, reference representation, point extraction, false-alarm policy and unmatched handling before scoring |

The latest coarse [partial-2026 diagnostic](../fire-harmonization-research/experiment-04.5/results/REPORT.md)
reinforces the coverage concern: reported MAE improvement varies from 63.8% under
its 51-day activity-conditioned mask to 11.7% under its broader 59-day assumption.
Those are different partial-archive populations, not evidence about the new image
model, but they show why observation validity must be solved before training.

### 3.1 Make the target contract unambiguous

Version 1 should begin with valid native VIIRS-pixel fire/non-fire supervision
at known target coordinates, and a precisely documented aggregation for the
native MODIS support. If a regular fine map is trained, define whether each target
is observed fire-pixel status or overlap-weighted reference-footprint coverage.
Fractional overlap is not the fraction of ground actually burning.

Define invalid/partially observed outputs separately. Do not present a score on
soft footprint-fraction targets as though it were a score on binary true-fire
labels. Calibration, point extraction and calendar units must all use the same
contract. Begin with a derived aggregate from the fine field rather than an
unnecessary independently learned count head if that simplifies consistency.

A raw MODIS hotspot does not imply one fine fire. Internal expected-activity
consistency is useful only when its units and partition are defined; it must not
force fine output to reproduce raw MODIS counts.

### 3.2 Alignment and pairing are likely the hardest data work

The 96/192/384 array sizes express nominal band sampling, not perfect physical
registration or fixed geographic extents. Off-nadir support, detector alignment,
geolocation approximation and residual scan effects must be measured. The
375 m query map needs real geographic transforms and bounded shapes.

Audit the distribution of actual Aqua–S-NPP time/angle matches before promising
global training volume. A strict criterion may leave a small or unrepresentative
set, especially when acquisition regimes change. As an illustrative calculation,
a hypothetical front moving 0.5 m/s moves 450 m in 15 minutes; this is not an
assumed typical speed, but it shows that a timing error can rival the output spacing.

NASA documents that 1 km MODIS detections and 375 m VIIRS detections differ in
sensitivity, and that cloud, smoke/canopy and sampling can prevent observations.
Native labels require observation masks rather than missing-row negatives.
[NASA FIRMS limitations](https://firms.modaps.eosdis.nasa.gov/alerts/notes.php)

## 4. RTX 5070 hardware and software compatibility

NVIDIA specifies the desktop RTX 5070 as Blackwell with **12 GB GDDR7**, **6,144
CUDA cores** and **250 W total graphics power**. Its CUDA compute capability is
**12.0 (`sm_120`)**. These facts establish the target platform; AI TOPS and gaming
frame rates do not predict this model's training speed.
[NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/),
[NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)

### 4.1 Use a current supported Blackwell binary

Current stable documentation resolves to PyTorch 2.15; its release matrix lists
CUDA 13.2 and Blackwell 12.0 support. The broader lesson is to use a verified
stable build for the training OS, with working `sm_120` kernels and a driver
compatible with that exact runtime. Older advice to install a 2025 nightly or
permanently pin CUDA 12.8 is stale as a default. The CUDA 12.6 legacy build is
not the appropriate Blackwell choice.
[PyTorch release/architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)

Install into a newly created full-resolution environment on the RTX host. Do not
copy the current macOS/ARM virtual environment to an x86 CUDA machine. Python
3.12 is a conservative proposed starting point for the combined geospatial stack;
pin the actual resolved packages only after CUDA and HDF4/HDF5 smoke tests pass.

The PyTorch wheel/runtime and NVIDIA driver must match. A separately installed
CUDA Toolkit is primarily needed for compiling extensions; simply changing
`nvcc` does not replace an incompatible installed PyTorch binary. The CUDA version
shown by a driver utility is also not the same as `torch.version.cuda`.

Check GPU name, total/free memory, compute capability, wheel CUDA version, compiled
architectures and cuDNN. Then execute real convolution forward/backward and an
optimizer step: `cuda.is_available()` alone is not a kernel-compatibility test.
Absence of an explicit architecture entry may involve PTX fallback, so the
operation tests and official support matrix matter more than a single string.
Third-party CUDA extensions need their own Blackwell compatibility checks.

### 4.2 OS-specific concerns

Native Linux is a straightforward proposed training target. Windows native CUDA
PyTorch is possible, but HDF4/GDAL and extension packaging must also work.
Windows DataLoader workers use process spawning; dataset code needs import-safe
entry points and worker-local file handles.

WSL2 is another option. Use the Windows host's supported NVIDIA driver; NVIDIA
warns against installing a Linux GPU driver inside WSL because it can overwrite
the mapped driver interface. Store training caches in the Linux filesystem when
using Linux tools rather than continuously crossing `/mnt/c` for small reads.
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
[Microsoft filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)

## 5. Why 12 GB can work, and where it can fail

### 5.1 Parameter count is not the main memory problem

For ordinary FP32 parameters/gradients and two FP32 Adam moment arrays, the
component arithmetic is 16 bytes per parameter:

| Parameters | Weights + gradients + two moments |
|---:|---:|
| 2 million | 30.5 MiB |
| 5 million | 76.3 MiB |
| 10 million | 152.6 MiB |

These values exclude activations, workspaces, temporary/cached weight casts,
CUDA context, allocator effects and other applications. They do not predict the
model's total peak. The plan's compact model weights are small; high-resolution
features and decoder/loss choices dominate uncertainty about fit.

One 384×384×64 BF16 feature tensor is **18 MiB at batch 1**, or 36 MiB at batch 2.
Multiple convolutions, skip connections and backward states keep many such
tensors live. Wide early stems, upsampling every branch to 250 m before fusion,
and storing temporal features at every scale multiply the cost.

The input tile payload, about 2.51 MiB in the plan's example, is a disk/transfer
size, not a training-memory estimate. Three input frames do not imply just three
times that small payload; their saved feature graphs also consume memory.

### 5.2 Unrestricted attention or transport can exceed the card by itself

For the plan's nominal 256×256 fine queries:

| Materialized matrix, batch 1 | Size alone |
|---|---:|
| 8-head BF16 attention from those queries to 96×96 tokens | **9 GiB** |
| 8-head BF16 attention to 384×384 tokens | **144 GiB** |
| Full 65,536×65,536 FP32 distance/transport matrix | **16 GiB** |

These are exact component formulas for materialized matrices, before gradients
or any model tensors. Memory-efficient fused attention can avoid materializing
scores, but does not remove all compute/compatibility constraints. No such kernel
is assumed here. Use convolutions and local coordinate sampling first; add only
bounded local/coarse attention if justified. A global optimal-transport/spatial
matching loss over every fine-grid cell is inappropriate for the initial 12 GB run.
Use sparse/local foreground matching and staged or evaluation-only point metrics.

See the executable [worksheet](tools/memory_worksheet.py) and
[computed results](results/memory_worksheet.json). It explicitly labels omitted
memory costs and does not claim a GPU measurement.

### 5.3 Further practical failure modes

- **Teacher/student concurrency:** storing two networks' graphs can exhaust VRAM.
  Precompute teacher targets or run a frozen teacher without gradient tracking;
  train ensemble members sequentially. Teacher inputs remain training-only.
- **Dynamic query extents:** outer-angle footprints can create unexpectedly large
  query maps. Bucket/pad and enforce a measured maximum, with documented exclusions.
- **Graph retention:** appending live loss/output tensors to Python lists can
  retain graphs. Log detached summaries, and stream inference/export instead of
  keeping an epoch's maps on the GPU.
- **Training-state initialization:** measuring only a forward pass misses backward,
  optimizer-state creation and cuDNN workspaces. Warm up actual optimizer steps.
- **Compilation/capture overhead:** begin eager; compile only after fixed/bucketed
  shapes work, and profile again. Compile-time caches and graph pools can add peaks.
- **Desktop/display load:** all advertised VRAM is not free for the process.
  Inspect actual free memory and leave room for shape/workspace variation.

PyTorch distinguishes allocated tensor memory from allocator-reserved memory;
freeing unused cache does not free live tensors or solve an oversized graph.
[CUDA memory documentation](https://docs.pytorch.org/docs/2.15/notes/cuda.html#memory-management)

## 6. Critical precision risks for a location model

### 6.1 Keep absolute coordinates out of BF16/FP16

The worksheet rounds the illustrative point **60.11°N, 120.24°W** to each format:

| Coordinate storage | Rounded latitude/longitude | Displacement caused by rounding |
|---|---|---:|
| FP32 | approximately original | 0.14 m |
| FP16 | 60.125, −120.25 | **1.76 km** |
| BF16 | 60.0, −120.0 | **18.08 km** |

This is floating-point quantization arithmetic, not a claim about actual satellite
geolocation accuracy. It demonstrates how an indiscriminate mixed-precision cast
can destroy the localization objective even when the CNN trains normally.

Use FP64 geospatial preprocessing and FP32-or-higher local metric coordinates,
normalized sampling grids, distances and uncertainty calculations. Keep them
outside low-precision autocast. Local normalization helps, but is not permission
to store a coarse quantized grid without a measured error bound.

Disabling autocast also does not by itself disable TF32 acceleration of float32
matrix multiplication. Prefer CPU FP64 geometry, or verify an IEEE-FP32/FP64
transform path for sensitive GPU geometry. Validate coordinate round trips in
meters before and after any numeric-performance change.

### 6.2 Preserve weak thermal contrasts

Around 300 K, representable spacing is 2 K in BF16 and 0.25 K in FP16. Casting
absolute temperatures before subtracting background can erase small contrasts.
Compute radiometric conversion, differences, background statistics and normalization
in FP32 first; then test AMP neural operations against an FP32 reference, especially
on weak VIIRS-only positives and hard negatives. Keep raw calibrated source arrays
and QA available for inspection.

Autocast is operation-specific: it should not be implemented by converting the
whole input record/model to half precision. Use BCE with logits for the initial
fire loss. Keep masked sums and sensitive spatial operations at higher precision;
profile the memory cost of these protected regions too.
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.15/amp.html)

## 7. Proposed first training profile for this card

| Choice | Proposed starting point |
|---|---|
| Model | Compact convolutional multiresolution student; target 2–5M parameters |
| Native tile arrays | Thermal 96×96; 500 m 192×192; 250 m 384×384, with masks and actual geolocation |
| Fine decoder | Local coordinate sampling/shared head; bounded query shapes derived from overlap |
| Context | Current acquisition only; no simultaneous VIIRS image teacher |
| Precision | BF16 autocast after parity checks; FP32 weights/state, protected FP32/FP64 geometry and thermal preprocessing |
| Microbatch | Start 1; profile 2; increase only with measured headroom |
| Effective optimizer batch | Provisional 16 tiles, through correctly normalized accumulation |
| Normalization | GroupNorm rather than relying on large batch statistics |
| Memory headroom | Roughly 75–80% of actual device memory as a soft peak-reserved target, adjusted to free/display memory |
| Initial loss | Masked BCE-with-logits plus a well-defined native auxiliary task; additional heads later |
| Loader | Start two CPU workers, measured prefetch/pinned-memory settings, cached local tiles |
| Fallback | Reduce width/microbatch or checkpoint; if needed use 64/128/256 native crops, changing extent rather than native sampling |

Gradient accumulation does not change BatchNorm's instantaneous statistics.
GroupNorm avoids a batch-statistics dependency, though its configuration and
behavior still need validation on sparse fire features.
[GroupNorm documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GroupNorm.html)

For varying valid coverage, divide each objective's accumulated weighted loss sum
by its actual total valid weight over the effective batch. Simply averaging
microbatch means gives different weights to equally sized tiles with different
valid area. Handle zero-valid tiles, final partial windows and any class/exposure
weights explicitly. Compare accumulation with a non-accumulated small reference
case before trusting equivalent-batch claims.

Activation checkpointing can trade recomputation for saved activations; use the
documented non-reentrant path where applicable and measure its real effect.
It does not remove all skip/output tensors or repair a quadratic-size loss.
[Checkpoint documentation](https://docs.pytorch.org/docs/2.15/checkpoint.html)

## 8. CPU, data readers, RAM and storage can bottleneck first

The entire training dataset does not need to fit in VRAM. Source decoding,
alignment and tile creation should run as a separate auditable CPU preparation
stage, with bounded batches streamed to the GPU. The example 100,000-tile tensor
cache including two thermal history frames is about **286 GiB**, before raw
granules, extra QA/geometry, splits, intermediate caches and checkpoints.

For planning, 32 GB system RAM is a practical starting point and 64 GB provides
more room for concurrent preprocessing/cache; neither is a measured hard minimum.
Use a fast local SSD with capacity determined from the pilot's actual bytes.
Avoid loading all raw granules or a full tile cache into RAM. The exact CPU/RAM
requirement remains unknown until the user's host and preparation workflow are known.

Verify that the chosen GDAL build includes HDF4/HDF4Image and can open actual MODIS
science subdatasets. HDF5 support alone is insufficient.
[GDAL HDF4 driver](https://gdal.org/en/stable/drivers/raster/hdf4.html)

For independent HDF5 multiprocessing readers, open files inside each worker rather
than inheriting an open handle across fork. MPI is not needed for ordinary
independent read-only workers. Prefer prebuilt tile caches for the GPU loop.
[h5py parallel-read guidance](https://docs.h5py.org/en/stable/mpi.html)

DataLoader workers/prefetch/pinned buffers consume host memory. More workers can
oversubscribe CPU/decompression threads and worsen throughput. Measure data wait,
CPU load and GPU occupancy before increasing worker count. Check host RAM as well
as GPU peaks. Do not perform remote NASA downloads on the critical training path.

At sustained training load, cooling/power limits can reduce clocks. The NVIDIA
250 W graphics-power specification is not guaranteed wall power or a requirement
to run constantly at that value. Use measured sustained throughput, not an initial
short burst, for runtime planning.

## 9. Concrete preflight and profiling protocol

1. **Host inventory:** GPU/model/VRAM, OS, driver, Python/PyTorch/CUDA/cuDNN, available
   system RAM, SSD/free space and hardware runtime limits.
2. **Compatibility smoke:** run [gpu_preflight.py](tools/gpu_preflight.py) on the
   RTX host. Convolution forward/backward/Adam and FP32 sampling backward must work.
   Its tiny synthetic memory/time values are not the native architecture's profile.
3. **Data smoke:** independently open real HDF4/HDF5 files and check units, masks,
   coordinates, bands, target type and matched acquisition times.
4. **Freeze V1:** explicit targets, narrow architecture, loss terms, all array dtypes,
   shape buckets and masked accumulation semantics. No teacher/history initially.
5. **Profile real training:** warm up enough full steps to initialize optimizer and
   kernels, then reset peaks. Time synchronized representative windows and record
   allocated/reserved memory, free memory, loader delay and wall-clock throughput.
6. **Stress cases:** largest query shape, densest targets, smallest/zero valid mask,
   checkpoint-save step and validation/export. Test several batches, not one easy tile.
7. **Numerical checks:** FP32 versus AMP loss/gradients, weak signals, coordinate
   round trips, aggregate consistency and accumulation equivalence.
8. **Scale cautiously:** overfit one genuine tile as an implementation check, then
   train/evaluate whole-event held-out feasibility data. Add independent events
   before adding expensive branches. Keep run configs and all profiles here.

CUDA geometry sampling backward may be nondeterministic due to atomic updates;
seeds do not guarantee bitwise identical training. Use numeric tolerances,
several seeds and documented environment/algorithm settings. A strict deterministic
mode may reject unsupported operations, so test it explicitly instead of promising
bitwise reproducibility.

### Runtime estimates must follow measurement

For 100,000 processed tiles × 30 epochs, hypothetical end-to-end throughput gives:

| Hypothetical tiles/s | Arithmetic runtime |
|---:|---:|
| 5 | 166.7 hours |
| 20 | 41.7 hours |
| 50 | 16.7 hours |

These are scenario calculations, **not predictions for the RTX 5070**. Include
actual data preparation, validation, checkpointing, ablations and seeds separately.
Gradient accumulation changes optimizer update frequency, not the number of
tiles that must be processed. The earlier 93-second coarse-tree run is irrelevant
as an image-training speed benchmark.

## 10. Recommended scope and readiness decision

Proceed with the native-data pilot and compact current-image baseline on the
existing RTX 5070. A hardware upgrade is not established as necessary. Keep the
full-resolution band measurements, but constrain feature width, geographic tile
extent, decoder complexity and simultaneous tasks to the measured budget.

Do not yet treat the design as a training-ready specification: no real source
bundles, stable target contract, resolved 5070 environment or full-model profile
exist here. The next concrete milestones are valid native tiles, native baseline
benefit and a measured 12 GB forward/backward fit. Fine-location and forest-fire
precision remain empirical questions after those gates.

The canonical plan has been updated with the folder requirement, 12 GB profile,
precision/accumulation protections and readiness gates. The legacy native-plan
path is now a navigation stub. No old model, experiment result or main-app code
has been modified by this review.
