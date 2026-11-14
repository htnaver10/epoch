# Native-resolution fire harmonization

This is the project-root folder for **all full-resolution work**: plans, research
notes, acquisition/preprocessing/training/evaluation scripts, configurations,
source manifests, native data/cache, checkpoints, logs, results and figures.
The existing `../fire-harmonization-research/` folder retains the coarse-grid
experiments. The main application is unchanged.

- [Full native-resolution model plan](PLAN.md).
- [Complete plan evaluation for the desktop RTX 5070 12 GB](RTX_5070_REVIEW.md).
- [Proposed initial configuration](configs/rtx5070_proposed.json): a design profile,
  not a validated trainer configuration.
- [Analytical memory/precision worksheet](results/memory_worksheet.json).
- [Hardware preflight tool](tools/gpu_preflight.py): CUDA compatibility smoke test,
  not the native model and not a throughput benchmark.

## Current status

The user confirmed a **desktop RTX 5070 with 12 GB VRAM**. OS, CPU, system RAM,
storage and driver remain unspecified. The review host is macOS/ARM with no
PyTorch installation or attached RTX 5070, so no GPU training/profile is claimed.
Native NASA acquisition, the full image model and its trainer are not implemented.

The recommended first image model is a compact current-acquisition convolutional
student, with native MODIS thermal and optional 250/500 m context, trained using
aligned VIIRS masks. Start with microbatch 1, then profile 2; use effective batch
16 through correctly weighted accumulation if stable. Use mixed-precision neural
operations with FP32/FP64 geometry and numerically sensitive calculations. Add
temporal branches and any VIIRS image teacher only after the baseline is valid.

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
  src/                      future acquisition, pairing, tiling, model, train, infer, evaluate
  tests/                    unit and real-data integration checks
  data/                     raw granules, manifests, pairs, tiles, masks, splits, caches
  models/                   checkpoints, calibration and model cards
  logs/                     acquisition and training logs
  results/                  audits, predictions, tables and profiling evidence
  visualizations/           georeferenced comparisons and exported figures
  .venv/                    optional isolated full-resolution environment
```

Only existing tools/documents/results are present now; the tree also identifies
future locations. Raw granules and caches should be ignored by Git. An approved
external drive may mount at a data location under this folder and must be
manifested explicitly. Do not redirect full-resolution writes into the legacy
research folder or the main app.
