# Wide diverse fire model

A separate, executed research experiment inside PLSWORK. The architecture has
**50,567,169 trainable parameters**, compared with the original model's 8,321.
Its hidden width is **2,048**, with three residual blocks expanding to **4,096**
units, LayerNorm, GELU and 20% dropout. The input remains the same 96 explicit
MODIS thermal, neighborhood, reflective and geometry features.

The completed expansion adds **245 real native windows**, **4,745,517 known
queries** and **6,219 positive reference queries**. Combined training has
**15,117,425 queries and 8,132 positives**, across four conservative cohorts.
Positive reference coverage grows from **52 to 127 one-degree geographic
cells**, with 75 cells outside the original training support. These are
spatial coverage measures, not counts of independent wildfire incidents.

The original PLSWORK code, checkpoints, observations, validation and benchmark
are preserved. Parent training feature arrays are referenced read-only. New
native observations, their source copies, features, checkpoints and reports
belong to this directory.

## Run

From the workspace root:

```powershell
.\fire-harmonization-PLSWORK\run-wide-diverse.cmd all
```

This expands local training observations using the existing native-reader
environment, prepares additional features, profiles the GPU, trains or resumes,
freezes a validation-only operating policy, evaluates and verifies deployment.
No external downloads, new packages, accounts or user-supplied labels are needed.
Completed expansion/training stages are reused with integrity checks.

Individual stages are `expand`, `features`, `profile`, `train`, `evaluate`, and
`verify`. CUDA training reuses
`fire-harmonization-research-fullres/.venv`; native preparation reuses its
`.data-env` through the existing micromamba launcher. No environment is modified.

## Additional real observations

`expand_data.py` selects new windows from already downloaded MODIS/VIIRS
acquisitions, over broader North American geography, and the separate western
Canada July 2023 pilot acquisition. Original validation/test source identities
and cohorts are prohibited. Native windows already present in parent training
are excluded. Native radiometric calibration, the original reference eligibility rules, geolocation
inversion and a maximum 15-minute worst-case observation interval are retained.
Reference eligibility uses VIIRS classes 5/8/9, excludes residual bowtie QA bit
22, and requires valid mapped MODIS support inside the tile core. Per-input QA
bits 0-6 are audited for the expansion, rather than silently changing the
parent's label definition.

Selection is training-only and uses nominal/high-confidence VIIRS fire locations
plus spatially distributed controls. Every admitted query still uses the native
reference mask and eligibility rules; the selection hints never become fabricated labels.
Reference labels are not network inputs. No synthetic fires are counted as new
observations. The expansion receipt lists actual admitted counts and one-degree
spatial coverage. A cohort, query, date or geographic cell does not equal an
independent wildfire incident.

## Training and comparison

Normalization uses parent and extra **training observations only**. Cohorts
contribute equally to the loss. Batches oversample positives to 25%, with exact
importance correction back to each curated training cohort's actual class prevalence.
This correction does not undo the fire-enriched selection of training windows.
Sampling with replacement can revisit positive queries and does not guarantee
every background query is visited. The configured run allows 30 epochs of 256
minibatches and stops after eight epochs without improved validation AP.

Training uses CUDA BF16 execution, float32 weights, AdamW, gradient clipping,
dropout and a cosine learning-rate schedule. Checkpoints retain optimizer and
NumPy/PyTorch RNG states for epoch-boundary resume. Selection uses exact
validation average precision. The operating threshold and probability-intercept
adjustment use validation only, before test predictions. The threshold is
constrained by the MODIS validation false-positive budget.

See [training summary](results/TRAINING_SUMMARY.md),
[metrics and evaluation](results/evaluation.json),
[expansion receipt](results/expansion_receipt.json),
[GPU profile](results/gpu_profile.json), and
[completion verification](results/completion_verification.json).

The original validation cohort and 2024 benchmark were examined previously.
This is not a fresh blind test. Model capacity, coverage and training sampling
changed together, so this run cannot isolate the effect of parameter count.
More parameters do not establish better performance. VIIRS supplies reference
activity, not independently verified wildfire truth or operational validation.

## Inference

The verifier creates a fixture containing MODIS input arrays and an input-validity
mask, without VIIRS imagery, targets or reference weights:

```powershell
.\fire-harmonization-PLSWORK\run-wide-diverse.cmd infer --tile results/inference_input_only.npz --output results/my_prediction.npz --device cuda
```

Paths are relative to this directory. Inputs are prepared native NPZ tiles with
`thermal`, `reflective_500`, `reflective_250`, `geometry`, `query_grid`, and an
optional `inference_valid` mask. Outputs include logits, raw and adjusted scores,
validity and frozen-policy detections. This is a prepared-tile model, not a raw
HDF reader. CUDA BF16 and CPU float32 can differ near a decision threshold.

Deployment requires `models/inference.pt`, `results/operating_policy.json`, the
feature/model/inference code, `config.json` and the vendored calibration helper.
The checkpoint includes architecture and training-only normalization statistics.
`models/best.pt` and `models/last.pt` also contain training-resume information.

## Checks

```powershell
.\fire-harmonization-research-fullres\.venv\Scripts\python.exe -m unittest discover -s .\fire-harmonization-PLSWORK\wide_diverse_v1 -p "test*.py" -v
```

`test_model.py` adds focused forward/backward, parameter-count, exact reload and
dropout checks. `run-wide-diverse.cmd verify` verifies source/window isolation,
parent feature hashes, additional tile hashes, exact selected-weight export and
input-only predictions against the evaluated model.
