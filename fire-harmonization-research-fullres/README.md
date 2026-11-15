# Native-resolution fire harmonization

**Next substantive run:** [the expanded 30-epoch workflow](SUBSTANTIAL_ROUND.md)
collects fire-rich training scenes, retains MODIS as the baseline, and freezes
calibration/operating thresholds before separately reserved test predictions.
Run `run-substantial-round.cmd` and enter Earthdata credentials locally.

For the difference between executing the small pilot and training a model with
demonstrated benefit, see [useful-model training workflow](USEFUL_MODEL_TRAINING.md)
and [current readiness evidence](results/useful_training_readiness.json).

**Windows RTX-host verification (2026-11-14):** environments are prepared; 44 tests,
full-size synthetic CUDA training/resume/prediction and GDAL/HDF4 file checks pass.
Real NASA data and reviewed preparation remain required. Start with
[this computer's readiness report and Windows launchers](WINDOWS_TRAINING_READINESS.md).

This is the project-root folder for **all full-resolution work**: plans, research
notes, acquisition/preprocessing/training/evaluation scripts, configurations,
source manifests, native data/cache, checkpoints, logs, results and figures.
The existing `../fire-harmonization-research/` folder retains the coarse-grid
experiments. The main application is unchanged.

- [Full native-resolution model plan](PLAN.md).
- **[Installation and complete run instructions](RUNNING.md).**
- [GPU optimizations, justification and measurement procedure](GPU_OPTIMIZATION.md).
- [Executable CLI](run.py) and [source package](src/firefullres/).
- [RTX training configuration](configs/train_rtx5070.json) and [CPU smoke configuration](configs/smoke_cpu.json).
- [Larger-batch RTX throughput candidate](configs/train_rtx5070_throughput.json), requiring hardware profiling.
- [Complete plan evaluation for the desktop RTX 5070 12 GB](RTX_5070_REVIEW.md).
- [Proposed initial configuration](configs/rtx5070_proposed.json): a design profile,
  not a validated trainer configuration.
- [Analytical memory/precision worksheet](results/memory_worksheet.json).
- [Hardware preflight tool](tools/gpu_preflight.py): CUDA compatibility smoke test,
  not the native model and not a throughput benchmark.

## Current status

The user confirmed a **desktop RTX 5070 with 12 GB VRAM**. The first pipeline is
now implemented: NASA inventory/download, reviewed pairing, native readers and
tiling, a multiresolution CNN, training/checkpoint resume, profiling, evaluation,
inference and alignment previews. All paths stay in this folder. It implements
the current-acquisition baseline; optional temporal/teacher/uncertainty extensions
remain future work.

The original implementation-review host had NumPy but no PyTorch/native-reader stack or attached RTX GPU,
and package/download hosts fail DNS. NumPy pipeline checks run here; neural
execution tests are explicitly skipped until PyTorch is installed. Native readers
require actual-file calibration/alignment QA. No real native model has been
trained and no RTX memory/speed or fire-localization result is claimed.

The implemented first image model is a compact current-acquisition convolutional
student, with native MODIS thermal and optional 250/500 m context, trained using
aligned VIIRS masks. Start with microbatch 1, then profile 2; use effective batch
16 through correctly weighted accumulation if stable. Use mixed-precision neural
operations with FP32/FP64 geometry and numerically sensitive calculations. Add
temporal branches and any VIIRS image teacher only after the baseline is valid.

## Start the pipeline

See [RUNNING.md](RUNNING.md) for setup, software smoke tests and real NASA data.

```sh
python fire-harmonization-research-fullres/run.py --help
python -m unittest discover -s fire-harmonization-research-fullres/tests -v
```

After preparing and auditing the dataset on the configured RTX host:

```sh
python fire-harmonization-research-fullres/run.py train --manifest data/manifests/native_v1.jsonl --stats data/manifests/native_v1_stats.json --config configs/train_rtx5070.json --run native_v1
```

## Run the review tools

These commands run from the project root. The worksheet needs only Python's
standard library and writes exclusively inside this folder:

```sh
python3 fire-harmonization-research-fullres/tools/memory_worksheet.py
python3 -m unittest discover -s fire-harmonization-research-fullres/tests -v
```

On the RTX machine, after installing a verified CUDA-enabled PyTorch environment:

```sh
python fire-harmonization-research-fullres/tools/gpu_preflight.py
```

The preflight checks the GPU/build and runs a small synthetic convolutional
forward/backward/optimizer step and a geometry-sampling backward check. Its memory
numbers describe that tiny smoke test only. The real architecture still requires
its own measured worst-case training profile with all losses and optimizer states.

## Required full-resolution directory boundary

```text
fire-harmonization-research-fullres/
  PLAN.md, RTX_5070_REVIEW.md, README.md
  configs/                  proposed and eventually locked run configurations
  tools/                    environment checks and planning utilities
  src/firefullres/           acquisition, pairing, tiling, model, train, infer, evaluate
  tests/                    unit and real-data integration checks
  data/                     raw granules, manifests, pairs, tiles, masks, splits, caches
  models/                   checkpoints, calibration and model cards
  logs/                     acquisition and training logs
  results/                  audits, predictions, tables and profiling evidence
  visualizations/           georeferenced comparisons and exported figures
  .venv/                    optional isolated full-resolution environment
```

The implementation and software tests are present; native source files and trained
checkpoints are not bundled. Raw granules and caches are ignored by Git. An approved
external drive may mount at a data location under this folder and must be
manifested explicitly. Do not redirect full-resolution writes into the legacy
research folder or the main app.
