# Run the native-resolution pipeline

The first implementation now exists: native NASA readers, pairing/preparation,
a multiresolution CNN, training/resume, CUDA profiling, prediction/evaluation and
alignment previews. Everything writes inside `fire-harmonization-research-fullres/`.
The existing app and coarse experiments are not imported or modified.

**Current validation limit:** this development host has NumPy but no PyTorch,
GDAL/HDF4, h5py, pyproj or earthaccess; its download hosts fail DNS. NumPy geometry,
calibration, manifest and metric tests run here. PyTorch tests are explicitly
skipped until installed. No real native NASA training or RTX memory/speed result
has been produced. Run the PyTorch tests and real-file reader/alignment smoke
checks on the training machine before a scientific run.

## 1. Path convention and entry points

Run the commands below from the **project root** (the directory containing
`backend`, `frontend`, and the two research folders):

```text
epoch/
  fire-harmonization-research/          existing coarse research
  fire-harmonization-research-fullres/  this pipeline
    run.py
    configs/train_rtx5070.json
    configs/smoke_cpu.json
    src/firefullres/
```

Every CLI argument such as `data/manifests/pilot.jsonl` or `models/native_v1/best.pt`
is relative to **the full-resolution folder**, regardless of the shell's working
directory. Absolute paths are accepted only inside that folder. No hard-coded
`/Users/...` path is required. Do not move outputs into the original research
folder. Do not copy an existing macOS virtual environment to the RTX machine.

```sh
python fire-harmonization-research-fullres/run.py --help
python fire-harmonization-research-fullres/run.py train --help
```

| Command | Purpose |
|---|---|
| `acquire` | Bounded NASA CMR inventory; `--download` authenticates and downloads science files |
| `propose-pairs` | Close-time, intersecting Aqua/S-NPP bundle proposals requiring human event/split review |
| `prepare` | Calibrated native bands, QA, geolocation inversion, reference labels, masks and NPZ tile manifest |
| `audit` | Hash, schema, precision and event/source leakage checks |
| `stats` | Fit normalizers on training tiles only |
| `preview` | Native band-21/reference overlay before training; requires Matplotlib |
| `train` | Train the CNN, select best validation log loss, save epoch checkpoints and resume |
| `profile` | Actual model CUDA forward/backward/optimizer profile |
| `evaluate` | Validation/test reference metrics and georeferenced prediction files |
| `infer` | Prediction without passing VIIRS labels to the model |
| `make-demo` | Synthetic software fixtures requiring explicit training opt-in |

## 2. Recommended RTX 5070 environment

The user-confirmed GPU is **desktop RTX 5070, 12 GB**, Blackwell `sm_120`.
Use an NVIDIA driver supported by the chosen CUDA PyTorch wheel. This is an
environment setup on the RTX host, not a command to run on the current Mac.

Linux or WSL2 with an isolated Conda/Miniforge environment is a practical path
because native MODIS needs GDAL's HDF4 driver. Install the geospatial stack before
adding pip packages. Example from the project root:

```sh
conda create --prefix ./fire-harmonization-research-fullres/.venv -c conda-forge python=3.12 pip numpy h5py pyproj earthaccess gdal libgdal-hdf4 matplotlib
conda activate ./fire-harmonization-research-fullres/.venv
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
```

The CUDA 13.2 wheel index is an example supported by the current release matrix.
Use the official current stable command appropriate for your OS/driver if it
changes; do not install a CPU-only or legacy CUDA build and ignore Blackwell
warnings. This pipeline does not require torchvision, torchaudio, a VIIRS image
teacher, or custom CUDA extensions for its first model.

Verify the environment:

```sh
python fire-harmonization-research-fullres/tools/gpu_preflight.py
python -c "from osgeo import gdal; import h5py, pyproj, earthaccess; assert gdal.GetDriverByName('HDF4'); assert gdal.GetDriverByName('HDF4Image'); print('Native readers available')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
python -m unittest discover -s fire-harmonization-research-fullres/tests -v
```

On the RTX host the neural tests must actually run; a skipped test is not a pass.
The preflight is a small compatibility check. Use `profile` below to measure the
real model. Pin the successful environment for reproduction:

```sh
python -m pip freeze > fire-harmonization-research-fullres/results/pip-environment.txt
conda list --explicit > fire-harmonization-research-fullres/results/conda-environment.txt
```

If using native Windows with prebuilt tiles, a Python 3.12 venv plus the correct
CUDA PyTorch wheel and NumPy can run training; raw-data preparation still needs
a working HDF4/GDAL environment. For WSL2 use the Windows NVIDIA driver, not a
Linux GPU driver inside WSL. Place heavy caches in the Linux filesystem.

References: [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/),
[Conda prefix environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html),
[GDAL HDF4 support](https://gdal.org/en/stable/drivers/raster/hdf4.html),
[conda-forge HDF4 plugin](https://anaconda.org/conda-forge/libgdal-hdf4),
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## 3. Software smoke test before downloading data

The following creates **synthetic** native-shaped arrays. It verifies wiring;
it is not NASA data and cannot establish fire performance. The manifest is
marked `synthetic_smoke_only`; training/evaluation reject it without opt-in.
Choose fresh manifest/run/output names on repeated runs; existing artifacts
are not silently overwritten.

```sh
python fire-harmonization-research-fullres/run.py make-demo
python fire-harmonization-research-fullres/run.py audit --manifest data/manifests/demo.jsonl --allow-synthetic --output results/demo-audit.json
python fire-harmonization-research-fullres/run.py stats --manifest data/manifests/demo.jsonl --allow-synthetic --output data/manifests/demo-stats.json
python fire-harmonization-research-fullres/run.py train --manifest data/manifests/demo.jsonl --stats data/manifests/demo-stats.json --config configs/smoke_cpu.json --run smoke_cpu --allow-synthetic
python fire-harmonization-research-fullres/run.py evaluate --manifest data/manifests/demo.jsonl --checkpoint models/smoke_cpu/best.pt --split test --output results/smoke_cpu_test --device cpu --allow-synthetic
```

To exercise CUDA instead, use the same demo manifest with
`--config configs/train_rtx5070.json --epochs 2 --run smoke_cuda --allow-synthetic`.
The demo uses smaller 32/64/128 native arrays, so a successful demo is not proof
that the real 96/192/384 configuration fits in VRAM. For CPU-only software testing,
install a supported CPU PyTorch build, use `smoke_cpu.json`, and omit the GPU
preflight. CPU runs do not test CUDA kernels or the RTX memory budget.

The repository's generated `software_smoke` fixture/audit are development evidence
only. Use `make-demo` to generate the complete deterministic fixtures on another
host; large binary tiles are ignored by Git.

## 4. Find and download actual NASA granules

The initial implementation supports **Aqua MODIS C061 + S-NPP VIIRS C002**.
Terra and NOAA-20/21 are future separately validated extensions. This is a
same-acquisition image model, not an FIRMS hotspot CSV model.

Required MODIS files: `MYD021KM`, `MYD03`, `MYD14`. Optional native reflective
companions: `MYD02HKM`, `MYD02QKM`. Required reference files: `VNP14IMG`, `VNP03IMG`.
Forest-only preparation also requires appropriate earlier `MCD12Q1` tiles.

Configure Earthdata credentials **locally in the training environment**, using
`EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` or `EARTHDATA_TOKEN` as supported by
earthaccess. Do not paste credentials into chat, commit them, or include them in
CLI URLs. The code uses environment authentication with persistence disabled.
A FIRMS `MAP_KEY` does not authenticate these native archive downloads.
[earthaccess authentication](https://github.com/earthaccess-dev/earthaccess/blob/main/docs/user/howto/authenticate-with-environment-variables.md)

Example bounded pilot inventory, followed by an explicit download:

```sh
python fire-harmonization-research-fullres/run.py acquire --start 2023-07-01T20:00:00Z --end 2023-07-01T23:59:59Z --bbox -125 50 -110 65 --output data/manifests/pilot_inventory.jsonl --max-per-product 50
python fire-harmonization-research-fullres/run.py acquire --start 2023-07-01T20:00:00Z --end 2023-07-01T23:59:59Z --bbox -125 50 -110 65 --output data/manifests/pilot_downloads.jsonl --max-per-product 50 --download
python fire-harmonization-research-fullres/run.py propose-pairs --inventory data/manifests/pilot_downloads.jsonl --output data/manifests/pilot_proposed_bundles.jsonl
```

This AOI/time is illustrative, not a verified available paired-fire dataset.
Inventory may return no suitable pairs. Inspect actual coverage and choose other
dates/regions as necessary. If a product exceeds the bound, the command fails
rather than quietly sampling the first files. Increasing the bound may download
substantial data. The download performs a fresh metadata search and saves its
own IDs/checksums; the earlier no-download inventory is advisory, not a locked
transfer order. Credentials and signed download URLs are not stored in manifests.

Download corresponding earlier MCD12Q1 files with a separate `acquire --products
MCD12Q1` inventory/download or NASA's portals. Audit the mapped year and verified
availability date. Collection IDs, filenames and checksums identify processing;
reprocessing changes require a new dataset version.

## 5. Review event groups and splits before preparation

Copy/edit the proposed bundle JSONL into a new version, for example
`data/manifests/reviewed_bundles.jsonl`. Each JSONL line is one bundle object.
The proposals deliberately leave these fields null:

- `event_group`: one reference event cluster across overlapping tiles, repeated
  overpasses and nearby context; not a unique arbitrary ID for each crop.
- `region_id`: reviewed geographic region/block.
- `split`: exactly `train`, `validation` or `test`.
- Forest context fields when forest-only training is intended.

See [the illustrative bundle schema](examples/bundles.example.jsonl). Its file
paths/availability text are placeholders and are not a downloadable dataset.
Each acquisition needs explicit UTC start/end intervals. Companion filenames
must match the acquisition and supported collection. The **worst-case separation
between the entire MODIS and VIIRS intervals must be <=15 minutes**; this is
conservative and is not a claim that per-pixel scan timing was measured.

Keep the same event and all shared source granules in one split. Do not assign
overlapping tiles randomly. Region-held-out experiments require additional
manual geographic partitioning/buffers; `region_id` is recorded, not an automatic
guarantee of geographic independence. Normalizers are fit on train only, and
test metrics are not used to choose the checkpoint.

For forest-only data, supply:

- `land_cover`: list of real prior-map MCD12Q1 file paths under `data/raw/`.
- `land_cover_year`: strictly earlier than the MODIS acquisition year and matching
  the chosen actual product; verify manually against source metadata.
- `land_cover_available_utc`: a verified publication/availability timestamp no
  later than the acquisition being simulated. Do not invent an availability date.

The reader uses IGBP classes 1–5 with QC 0 only. This strict rule excludes mixed
or less-certain contexts. It is a land-cover context filter, not proof that a
thermal detection is an unplanned wildfire or an exact forest boundary.

## 6. Prepare native tiles and inspect them

Forest-only mode is the default:

```sh
python fire-harmonization-research-fullres/run.py prepare --bundles data/manifests/reviewed_bundles.jsonl --output data/manifests/native_v1.jsonl
python fire-harmonization-research-fullres/run.py audit --manifest data/manifests/native_v1.jsonl --output results/native_v1_audit.json
python fire-harmonization-research-fullres/run.py stats --manifest data/manifests/native_v1.jsonl --output data/manifests/native_v1_stats.json
python fire-harmonization-research-fullres/run.py preview --manifest data/manifests/native_v1.jsonl --index 0 --output visualizations/native_v1_alignment_0.png
```

If forest maps are unavailable, `prepare ... --all-land` explicitly creates an
all-land thermal-activity baseline. Label the dataset/run accordingly; it cannot
be called a forest-only model.

Preparation preserves per-band calibrated radiance/reflectance and validity;
invalid values are masked rather than silently treated as cold measurements.
Missing optional reflective companions become unavailable masked channels.
Thermal normalization uses training-only signed-log radiance statistics, avoiding
an invented brightness-temperature conversion or clipping away hot tails.

Defaults: native thermal 96×96, reflective 192×192/384×384, halo 16 and stride 64.
The stride must equal tile size minus twice the halo to avoid duplicated/gapped
evaluation cores. The maximum native VIIRS query canvas is 384 per side. It is
derived from actual query geolocation, not an integer 1 km/375 m resizing ratio.
Oversized or invalid-geolocation windows are excluded and logged, not truncated.

Native MODIS usable classes are land/fire (5/7/8/9), with at least two usable
thermal bands and view zenith <=40°. Reference negatives are VIIRS observed land
class 5; primary positives are classes 8/9. Cloud/unknown/low-confidence/deleted
states and residual bow-tie QA are excluded. These conditions define a screened
paired-reference population, not performance in all clouds/outages.

The native auxiliary target is **mean known VIIRS fire status among query centers
assigned to a MODIS cell**, weighted by known query exposure. It is neither the
number of unique fires nor a true any-fire probability. Primary outputs are
reference-activity probabilities at geographic queries.

Inspect `results/native_v1_preparation.json`, including exclusions and source
hashes. If no tiles survive, see `*_failed_preparation.json`. Inspect actual
overlays, masks and calibration attributes before training. The 20 m inversion
residual is a consistency check of the local bilinear geolocation approximation;
it does not establish 20 m absolute ground accuracy. Finer-band registration
currently uses the native 2×/4× band lattice relationship and needs real-file
alignment QA. Exact PSFs/footprints and per-detector registration corrections
remain outside this first implementation.

## 7. Profile and train the actual model

Start by reading [train_rtx5070.json](configs/train_rtx5070.json). The initial
configuration is current-acquisition only, base width 32, branch width 8,
GroupNorm, BF16 autocast, microbatch 1 and 16 accumulated microbatches. Parameters
and optimizer state remain FP32. Geometry sampling, weighted reductions and stored
coordinates retain FP32/FP64. No full-resolution global attention, dense global
transport loss or simultaneous VIIRS image teacher is used.

The trainer now uses single-read verified tiles, persistent workers, prefetched
accumulation windows, shape bucketing, channels-last BF16 and automatic CUDA
optimizer selection. Optional bounded CPU caching, CUDA copy prefetch and model
compilation are available. See [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) for the
reasoning, switches, RAM/VRAM costs and a larger-batch candidate configuration.

Profile real tiles on the RTX machine:

```sh
python fire-harmonization-research-fullres/run.py profile --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --steps 20 --output results/native_v1_rtx5070_profile.json
```

The profile runs the same accumulated, exposure-normalized optimizer update as
training. `--steps` counts update windows, not microbatches. Its throughput includes
loader wait, transfers, guards, backward and optimizer work after warm-up. It
reports actual tile/update counts, query padding and memory for sampled tiles.
Add `--worst-case` to start with largest query buckets, and optionally `--trace
results/rtx_trace.json` for a separate short CPU/CUDA diagnostic capture. Tracing
adds overhead; compare untraced runs for throughput. Memory fit must be checked
across the real largest shapes; a short profile cannot guarantee all-data fit.
Epoch logs also record validation/checkpoint and full epoch time.

After the baseline works, profile `configs/train_rtx5070_throughput.json` in a
separate process/output. It proposes microbatch 4, accumulation 4 and more CPU
prefetch/cache; its 12 GB fit/speed have not been measured here. Compare 1×16,
2×8, 4×4 and 8×2 before choosing a long-run recipe. Bucket tails are smaller
effective batches, normalized by actual exposure; no tile is dropped/duplicated.

Train a short first run, then a longer version under a new name:

```sh
python fire-harmonization-research-fullres/run.py train --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --run native_v1_pilot --epochs 2
python fire-harmonization-research-fullres/run.py train --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --run native_v1_30epochs
```

Outputs:

```text
models/native_v1_30epochs/run.json   frozen run config, dataset hash, parameter count
models/native_v1_30epochs/last.pt    epoch-boundary model/optimizer/scaler/RNG checkpoint
models/native_v1_30epochs/best.pt    lowest validation log loss
logs/native_v1_30epochs/epochs.jsonl validation metrics, time, observed memory peaks
```

Resume the same run at the next completed epoch boundary:

```sh
python fire-harmonization-research-fullres/run.py train --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --run native_v1_30epochs --resume --epochs 40
```

`--epochs` is the total desired epoch count. Resume verifies manifest/statistics
and learning configuration; model/precision/optimizer changes require a new run.
Progress within an unfinished epoch is not restored. CUDA sampling backward can
be nondeterministic; matching seeds/checkpoints do not guarantee bitwise-identical
cross-device training.

Training format 2 records the new deterministic window order. Older checkpoint
formats remain loadable for inference but require a new training run rather
than an inexact resume. Compiled execution saves the canonical model's keys;
compilation/layout/cache settings and batch order are frozen with the run config.

If memory is tight, try activation checkpointing or narrower branches in a new
config/run. Smaller native crop fallback: tile size 64, halo 16, stride 32,
reflective sizes 128/256. This reduces geographic extent while preserving native
band sampling; regenerate the dataset and normalizers. Do not blindly downsample
the thermal source or cast absolute coordinates to half precision.

## 8. Evaluate a held-out partition

```sh
python fire-harmonization-research-fullres/run.py evaluate --manifest data/manifests/native_v1.jsonl --checkpoint models/native_v1_30epochs/best.pt --split test --output results/native_v1_test
```

The evaluator refuses training splits and checks that evaluation events/source
granules/tiles do not overlap the checkpoint's training provenance. The default
0.5 threshold is stored in the training config; do not tune it on test results.

`report.json` contains exposure-weighted Brier/log loss, precision/recall/F1,
4096-bin approximate ROC AUC/average precision, and the same-mask official MODIS
baseline. Per-tile NPZ files contain probabilities, valid masks and float64
latitude/longitude. GeoJSON contains strict local candidate peaks, **not verified
independent incidents, exact flame coordinates, or calibrated location intervals**.

These evaluate recorded VIIRS query status. They do not establish actual forest
fire accuracy. Point-distance matching, independent ASTER/Landsat validation,
uncertainty calibration, blocked confidence intervals and calendar acceptance
tests in the full plan still need separate implementation/evidence. No spatial
accuracy result should be invented from a sharp-looking prediction map.

## 9. MODIS-only inference

Create a new bundle JSONL with only the MODIS and permitted earlier forest inputs,
the MODIS interval, reviewed metadata and `split: "inference"`. Omit VIIRS files
and intervals. Then:

```sh
python fire-harmonization-research-fullres/run.py prepare --inference --bundles data/manifests/inference_bundles.jsonl --output data/manifests/inference_v1.jsonl
python fire-harmonization-research-fullres/run.py infer --manifest data/manifests/inference_v1.jsonl --checkpoint models/native_v1_30epochs/best.pt --split inference --output results/inference_v1
```

Inference queries are created in a local equal-area projection at nominal 375 m
spacing. The network sees MODIS inputs and query coordinates only; it is not
passed targets or VIIRS imagery. The same maximum-query-size and support/forest
rules apply. Use `--all-land` only for an explicitly corresponding activity model.
Outputs remain model estimates. Regular query spacing is not achieved fire-location
accuracy; native VIIRS training queries and regular map queries also require a
footprint/domain-shift assessment before a deployment claim.

## 10. Common failures and next work

| Symptom | Action |
|---|---|
| `No module named torch` | Activate the full-resolution environment and install the correct PyTorch wheel; inspect the active Python path |
| CUDA unavailable / unsupported kernels | Verify NVIDIA driver, CUDA wheel and `sm_120` support with the preflight; a standalone CUDA toolkit install does not replace a wrong wheel |
| HDF4/HDF4Image missing | Use a verified GDAL build/plugin; h5py and many generic raster wheels do not provide HDF4 support |
| No pairs/tiles | Inspect metadata, interval bounds, query caps, QA and preparation exclusions; do not silently relax criteria to obtain a favorable score |
| Missing calibration attributes | Inspect the actual product/schema; reader fails deliberately instead of fabricating radiance |
| Leakage error | Keep overlapping events and shared granules in one split; regenerate a versioned manifest |
| CUDA OOM | Profile worst-case queries; lower width/crop extent, use checkpointing, or remove simultaneous branches; use a new run for architecture changes |
| Slow training/GPU idle | Inspect HDF preparation versus cached-tile I/O, CPU workers, prefetch and SSD; do not download data in the training loop |
| Existing output/run error | Use a new dataset/output/run version, or `--resume` for the same saved configuration |
| Synthetic metrics look good | They are a software test, not satellite evidence; train/evaluate real audited native pairs |

Implemented now is the initial current-image baseline, not every optional research
idea. Temporal modules, VIIRS image-teacher distillation, physical forward models,
uncertainty regions, cross-platform transfer and app integration remain future
experiments. Native source readers require real-file scientific validation before
their outputs can support claims. The main app remains unchanged.
