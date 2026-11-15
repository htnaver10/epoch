# Training readiness on this Windows PC

**The real-data training pilot is now assembled.** Run this in PowerShell:

```powershell
cd "C:\Users\Darrien\Desktop\epoch1\epoch\fire-harmonization-research-fullres"
.\train-real-data.cmd
```

The first real CUDA epoch completed in about 30 seconds, including validation and
checkpointing. `best.pt` and `last.pt` were saved, and checkpoint reload was
verified. The launcher therefore resumes from that first completed epoch. See
[execution verification](results/real_training_pilot_v1_execution.json).

This trains to **two total epochs**, automatically resuming an existing pilot
checkpoint. It uses 54 western-North-America training tiles and 71 eastern-Canada
validation tiles from real NASA granules. Shared granules stay in one partition;
the complete native input footprints are geographically separated, and validation
is restricted to the selected eastern block. Normalization is fitted on training
only. The checksum/array/source/event audit is in
[real_training_pilot_v1_audit.json](results/real_training_pilot_v1_audit.json).
See the inspected [validation alignment preview](visualizations/real_training_pilot_v1_validation_alignment.png).

Checkpoints are `models/real_training_pilot_v1/last.pt` and `best.pt`;
progress is `logs/real_training_pilot_v1/epochs.jsonl`. To extend to five total
epochs, run `.\train-real-data.cmd --epochs 5`. The first epoch includes startup
and kernel setup; wait for its epoch summary. This is an **all-land execution
pilot**, with only seven positive training query centers and no final test set.
Model quality needs a substantially larger fire-positive training population;
forest-only training additionally needs prior land-cover context.

To reproduce dataset assembly after the two source tile manifests exist, run
`tools/assemble_real_pilot.py` in `.data-env` with a fresh output dataset version.
The builder retains the original prepared tiles and records the excluded
validation footprints; it does not create a random crop split.

**Real-data update:** the first authenticated pilot download is now present:
16 source files and one proposed pair. The MYD03 geolocation reader was corrected
to expose hidden raw SDS using GDAL's documented `LIST_SDS=YES` fallback. A
conservative, train-only **all-land reader-QA** manifest now contains 54 real
tiles, with seven positive and 382,427 known reference query centers. This is not
a reviewed forest-only training/validation dataset. The real-data CUDA profiling
path completed three optimizer updates; see
[QA summary](results/real_pilot_qa_summary.json),
[alignment preview](visualizations/real_pilot_all_land_qa_alignment.png) and
[real-data GPU profile](results/real_pilot_rtx5070_profile.json).

The next candidate validation download is a separate eastern-Canada acquisition.
Public metadata confirms 14 source granules across all seven products, with
candidate MODIS/VIIRS interval overlaps within the required limit:

```powershell
cd "C:\Users\Darrien\Desktop\epoch1\epoch\fire-harmonization-research-fullres"
.\download-real-data.cmd --start 2023-06-05T17:45:00Z --end 2023-06-05T18:10:00Z --bbox -80 45 -65 55 --tag validation_quebec_20230605
```

This command prompts for Earthdata credentials locally. It downloads candidate
validation inputs; preparation, event/source split checks and actual-file QA must
finish before a checkpointed real-data pilot run. Forest-only training additionally
requires prior land-cover context. The earlier setup assessment below is retained
as a dated record of the host before the authenticated download.

Checked 2026-11-14. **The computer can run the implemented CNN training pipeline.**
Both Python environments and the native NASA reader environment are installed.
CUDA training, checkpoint resume, evaluation and inference passed on full-size
synthetic tiles. **Scientific training is waiting for real NASA data, Earthdata
authentication, reviewed splits and calibration/alignment QA.**

**To start downloading with your new Earthdata account:** double-click
[`download-real-data.cmd`](download-real-data.cmd) in this folder. Enter your
Earthdata username and password in its local terminal; the password is hidden.
It downloads the bounded 2023-07-01 pilot, saves source checksums and proposes
pairs under fresh timestamped manifest names. Credentials are not saved to files.
The pilot is the next real-reader/alignment check; training still needs reviewed
independent events/splits and forest context. From PowerShell, the same launcher
is `.\fire-harmonization-research-fullres\download-real-data.cmd` at the project root.

| Item | Observed on this computer |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB, compute capability 12.0 |
| Driver | 610.88 |
| Training Python | 3.12.10, independent `.venv` |
| PyTorch | 2.11.0+cu128; explicit `sm_120` support; CUDA and BF16 passed |
| System RAM | 61.6 GiB reported usable by Windows |
| CPU | 32 logical processors |
| Free project-drive space | About 243 GiB at the final host check; no full-dataset storage estimate yet |
| Native preprocessing | Separate `.data-env`, Python 3.12.16, GDAL 3.13.3 with HDF4/HDF4Image |

Evidence: [training host](results/windows_host_preflight.json),
[CUDA compatibility](results/windows_gpu_preflight.json),
[reader environment](results/windows_reader_preflight.json),
[HDF4/HDF5 file checks](results/windows_reader_smoke.json).

The native reader environment passes synthetic HDF4/HDF5 round trips and PROJ
transforms. Real MODIS/VIIRS calibration, QA interpretation and alignment remain
unverified until actual source files are inspected. GDAL belongs in `.data-env`;
its absence from the GPU `.venv` is intentional. Use the appropriate launcher.

## Verified execution

- Full-resolution suite: **44 tests passed, zero skipped**, including CPU and CUDA
  neural checks, persistent Windows spawn workers, mixed precision and optimizer
  accumulation. See [test log](results/windows_unit_tests.log).
- Coarse research: **37 tests passed** across the four test directories. A tiny
  native LightGBM fit and loading the real 2023 count archive also passed.
- The actual 1,982,050-parameter CNN trained for one synthetic epoch, resumed for
  a second, wrote `last.pt`/`best.pt`, then evaluated and inferred two held-out
  synthetic tiles. See [epoch log](logs/windows_cuda_smoke/epochs.jsonl),
  [evaluation](results/windows_cuda_smoke_test/report.json) and
  [inference](results/windows_cuda_smoke_infer/report.json).
- Both pip environments pass `pip check`. Installed packages are recorded in
  [training lock](results/windows-requirements-lock.txt),
  [data-environment explicit lock](results/windows-data-explicit-lock.txt), and
  the coarse folder's corresponding lock. No global Python packages were changed.

The profiling fixture contains 16 training, two validation and two test tiles,
with thermal/context arrays of 96/192/384 pixels and a 384 x 384 query canvas.
All records and checkpoints are marked synthetic. These are execution checks,
not fire-quality measurements.

| Configuration | Microbatch / accumulation | Warm tiles/second | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| `train_rtx5070.json` | 1 / 16 | 26.4 | 185.4 MiB | 240 MiB |
| `train_rtx5070_throughput.json` | 4 / 4 | 33.1 | 657.6 MiB | 1,520 MiB |

See [default profile](results/windows_fullshape_profile.json) and
[larger-batch profile](results/windows_fullshape_throughput_profile.json).
Each measured five optimizer updates after two warmup updates. These timings
include tile loading and updates, exclude validation/checkpointing, and use a tiny
repeated synthetic dataset. Startup includes Windows worker launch and CUDA/cuDNN
initialization; a one-update cold epoch is much slower. GPU memory figures are
PyTorch allocator measurements, excluding desktop/driver allocations. Measure
again on the real prepared dataset before choosing batch size or estimating time.

## Deployment issues repaired

1. **NASA inventory silently missed required imagery.** LAADS CMR versions are
   `6.1` for `MYD021KM/MYD03/MYD02HKM/MYD02QKM` and `2` for `VNP03IMG`.
   LP DAAC fire products use `061`/`002`. Acquisition now uses explicit versions
   per product and records the distinct filename collection. The bounded pilot
   previously returned five records with missing companions; it now returns
   **16 records covering all seven required/optional products**. No science
   granules were downloaded. See [corrected inventory](data/manifests/windows_pilot_inventory_fixed.jsonl).
2. **MODIS SDS matching was ambiguous.** A radiance name also matched its
   `_Uncert_Indexes` companion. Exact name boundaries now distinguish them;
   regression tests retain failure on genuinely ambiguous datasets.
3. **Coarse isolation checks were tied to an earlier computer.** New runs capture
   a fresh main-app baseline, use portable path separators and exclude both
   research folders. Historical snapshots/results remain in place.
4. **PowerShell execution policy blocks `.ps1` files here.** The supplied `.cmd`
   launchers work with the current policy and locate their own environments.
5. Full-resolution manifests now write portable forward-slash paths. The demo
   generator accepts explicit native/query sizes and training-tile counts so
   hardware checks can exercise the actual planned dimensions.

CMR versions were checked against NASA's public collection records, including
[MODIS radiance](https://cmr.earthdata.nasa.gov/search/concepts/C1379758607-LAADS.html),
[VIIRS geolocation](https://cmr.earthdata.nasa.gov/search/concepts/C2105092163-LAADS.html)
and [MODIS fire](https://cmr.earthdata.nasa.gov/search/concepts/C2278858993-LPCLOUD.html).
The separate reader setup follows the
[official portable Micromamba instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
and uses the conda-forge HDF4 plugin required by the
[GDAL HDF4 reader](https://gdal.org/en/stable/drivers/raster/hdf4.html).

## Launch from the project root

```powershell
# Training, profiling, inference, audits and statistics:
.\fire-harmonization-research-fullres\run-windows.cmd --help

# NASA inventory/download and native preparation:
.\fire-harmonization-research-fullres\run-data-windows.cmd --help

# Repeat the local training-host check:
.\fire-harmonization-research-fullres\.venv\Scripts\python.exe fire-harmonization-research-fullres/tools/host_preflight.py
```

Full-resolution CLI paths are relative to this folder, regardless of the working
directory. Windows launchers do not require venv activation. WSL is not required
for the tested native Windows setup. Keep compilation disabled in the supplied
configs; the tested training path is eager CUDA.

The following **future real-data command requires files that do not exist yet**:

```powershell
.\fire-harmonization-research-fullres\run-windows.cmd train --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --run native_v1
```

Before that command:

1. Configure Earthdata credentials locally as described in [RUNNING.md](RUNNING.md).
   No `EARTHDATA_TOKEN` or complete username/password pair was present in the
   checked process environment. File-based credentials were not inspected and
   authentication was not attempted.
2. Download a bounded paired pilot with the data launcher. The saved inventory
   contains metadata only; it is not a locked transfer order or prepared dataset.
3. Review event groups, regions and train/validation/test splits. Forest-only
   preparation also needs verified earlier MCD12Q1 data and availability dates.
4. Prepare tiles with the data launcher; audit them, fit training-only statistics,
   and inspect real calibration, masks and geolocation overlays.
5. Profile those real tiles, then start training. The bundled old
   `software_smoke.jsonl` references eight ignored tile files absent from this
   checkout; use a newly generated fixture rather than that historical manifest.

To reconstruct environments, create fresh Python 3.12 venvs, install the verified
CUDA wheel from `https://download.pytorch.org/whl/cu128`, then install the respective
pip locks. Recreate preprocessing from `environment-data.yml` or the explicit
Windows package lock using Micromamba with a local prefix. Venvs, downloaded wheels,
data, checkpoints and package caches are local ignored files, not portable Git
artifacts. Commands for acquisition, preparation and scientific review remain in
[RUNNING.md](RUNNING.md).
