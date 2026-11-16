# Second expansion and 201.8M-parameter fire model

The new residual MLP has **201,797,633 parameters**, a hidden width of **4,096**,
three blocks expanding to **8,192** units, GELU, LayerNorm and dropout. It retains
the same 96 MODIS features and all original validation/test queries.

The experiment separates architecture and training-data changes using four
conditions: the preserved **50M/d1** run, a new **200M/d1** run, a new **50M/d2**
control, and the requested new **200M/d2** model. Dataset d1 contains the original
and first-expansion training observations. Dataset d2 adds only disjoint new
windows from the second expansion. Each new network starts from scratch.

All conditions retain the same seed, optimizer settings, batch size, sampling
objective, epoch limits and validation-selection rule. Their training samples
are drawn with replacement, using equal cohort contributions and exact
class-stratification correction within the curated data. This is one seed per
condition; it does not quantify sensitivity to initialization. Early stopping
can select different training lengths.

The second expansion uses additional real fire-near and geographically spread
control windows from already downloaded MODIS/VIIRS acquisitions. Both prior
training manifests and prior failed-window records are excluded. Native QA,
calibration, core/halo restrictions and 15-minute worst-case pairing are retained.
Four conservative cohorts remain: expanded coverage is not an independent-fire
count. The original validation and 2024 benchmark are previously examined.

## Run

From the workspace root:

```powershell
.\fire-harmonization-PLSWORK\run-wide-diverse-v2.cmd all
```

The launcher reuses the existing native-reader and CUDA environments, keeps
outputs here, and preserves both prior experiments. Individual CUDA stages are
`features`, `profile`, `train`, `evaluate`, `verify`, `fit`, and `compare`. Select a
condition with `--run large200_d1`, `--run control50_d2`, or `--run large200_d2`.
`fit` runs profile/training/evaluation/verification for one condition.

The new dataset's receipts are under `results/expansion_receipt.json` and
`data/feature_receipt.json`. The completed controlled comparison is
[results/COMPARISON.md](results/COMPARISON.md). Each run's history, policies,
evaluation and deployment verification are under `results/<run-id>/`.

## Inference

The requested large model is `models/large200_d2/inference.pt`. Its checkpoint
includes its architecture and training-only normalizer. Use matching model and
feature code plus `results/large200_d2/operating_policy.json`.

```powershell
.\fire-harmonization-PLSWORK\run-wide-diverse-v2.cmd infer --run large200_d2 --tile results/large200_d2/inference_input_only.npz --output results/large200_d2/my_prediction.npz
```

Paths are relative to this folder. Prepared inputs contain MODIS thermal,
reflectance, geometry, query grid and validity; VIIRS labels/imagery are not
inference inputs. CUDA BF16 and CPU float32 can differ near thresholds.

## Checks

```powershell
.\fire-harmonization-research-fullres\.venv\Scripts\python.exe -m unittest discover -s .\fire-harmonization-PLSWORK\wide_diverse_v2 -p "test*.py" -v
```

Each fitted run verifies exact checkpoint/export equality and input-only
inference against evaluated predictions. Source/data locks guard resumed runs.
All prior model and feature files are read-only inputs and hash checked.
