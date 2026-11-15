# Unsupervised MODIS fire-anomaly experiment

**Corrected model:** [SaturationAwareBackgroundV2](saturation_aware_v2/README.md) has its own rewritten pipeline, models, feature cache and evaluation under this same folder. Run it with `run-saturation-v2.cmd all`. The original pipeline described below is retained for reproducibility.

**Audit finding:** this run has a major saturation-handling flaw. Its input requirements exclude 75 of the 95 fire-positive test query locations despite band 21 being available. The saved poor scores are accurate for that implementation, but do not establish failure of unsupervised learning. [Audit and evidence](results/IMPLEMENTATION_AUDIT.md). The fitted models and predictions remain unchanged.

This folder implements two-stage, self-supervised background prediction using the existing local MODIS scenes. No new satellite data are downloaded. Two compact neural networks learn to reconstruct withheld measurements without VIIRS fire labels or MODIS fire flags. A native-cell detector identifies unusually high thermal residuals; a daytime 500 m branch refines supported locations inside the detected cell.

## Run the complete pipeline

From `C:\Users\Darrien\Desktop\epoch1\epoch` in PowerShell:

```powershell
.\fire-harmonization-research-YESSS\run-training.cmd all
```

The launcher uses the existing `fire-harmonization-research-fullres/.venv` Python environment with PyTorch and CUDA. It copies local prepared tiles from `fire-harmonization-PLSWORK`, separates MODIS inputs from reference arrays, prepares native-grid features, trains both models, freezes an unlabeled alert policy, evaluates against VIIRS, plots results and verifies inference. No credentials, new environments or package installations are required.

Re-running reuses checksum-verified completed stages. Training resumes from complete epoch checkpoints, including optimizer and random-generator state. Interrupted copy and preparation stages can be rebuilt without modifying earlier experiments. Source changes after the fitting lock is created require a new experiment version.

Artifacts:

- `results/TRAINING_SUMMARY.md`: completed run and its conclusion; this experiment did not improve on MODIS.
- `results/run_status.json`: current pipeline stage or completion.
- `results/EVALUATION.md`: methods, results, research references and limitations.
- `results/training_and_comparison.png`: learning curves and benchmark comparison.
- `results/completion_verification.json`: copy, provenance and inference checks.
- `models/coarse/inference.pt` and `models/fine/inference.pt`: portable model exports.
- `models/policy.json`: frozen residual normalization and alert thresholds.
- `data/features/coarse_stats.json` and `data/features/fine_stats.json`: training-only normalizers and feature schemas.

## Design

**Stage 1: native MODIS-cell anomaly detection.** A 64→64→32→2 network predicts central band-20 and band-22 signed-log radiances. Its inputs are annular neighborhood statistics, central longwave bands 31/32 and acquisition geometry. Thermal neighborhood summaries exclude the central 3x3 cells. The withheld central measurements are training targets, never predictor inputs. Robust SmoothL1 regression reduces the influence of unusually large errors; it does not ensure that fires are absent from the learned background.

**Stage 2: optional daytime localization.** A 44→64→32→1 network predicts central band-7 signed-log reflectance at 500 m. Its inputs are annular background summaries, visible bands, valid-only pooled 250 m measurements and geometry. The central 3x3 band-7 measurements are excluded from its context. A strong positive residual supplies possible within-cell fire-location evidence. Four valid child observations are required. Above the frozen shortwave-anomaly threshold, retain the two highest-scoring children, including ties. Otherwise retain the complete parent alert and mark its location unresolved. At night, the reflective-band branch is unavailable and the parent alert remains unresolved. This produces 500 m estimates, not new 375 m thermal measurements.

Relative weights sum to one among the four child samples. They express concentration of the localization evidence, **not calibrated probabilities that those samples contain fire**. The native anomaly score is also not a fire probability. Use the validity masks when interpreting outputs.

## Label separation and evaluation

The copied source tiles retain original provenance. Separate `data/modis` files contain only `thermal`, `reflective_500`, `reflective_250`, and `geometry`. Separate `data/references` files hold VIIRS target/weight/query coordinates and the MODIS baseline. Training and policy code cannot load reference arrays through the strict input loader.

Training uses valid native MODIS measurements, with no VIIRS-derived query mask. The main detector requires usable band-20/22/31/32 measurements, geometry and neighborhood support. This branch does not claim an independent MODIS cloud-screening algorithm; clouds and other backgrounds can produce anomalies. Previous paired-scene preparation still determined which tiles exist, so this is not an uncurated global MODIS sample.

Normalization uses training measurements only. Checkpoints minimize reconstruction error on unlabeled development scenes. The alert policy flags at most approximately 0.1% of development native anomaly scores separately by day/night, with conservative ties. The localization threshold uses the unlabeled 99th percentile. These are predeclared alert-volume choices, **not false-positive-rate guarantees**. If a day/night group has fewer than 1,000 samples, the policy records a pooled-distribution fallback.

Only after both models and policies are frozen does evaluation open VIIRS labels. It reports both exact query-center comparisons and any-reference-fire-in-native-cell comparisons. Model-unusable known reference queries remain in the denominator, counted as undetected. Native-cell assignment is approximate; it is not exact footprint intersection. Partial reference coverage does not prove an entire physical cell is fire-free. The old test cohort has been examined before and is **not a new blind test**. This experiment's score must not be treated as proof of operational accuracy or general superiority to MODIS.

## Inference

List the physically separated test inputs:

```powershell
$rows = Get-Content .\fire-harmonization-research-YESSS\data\modis_manifest.json -Raw | ConvertFrom-Json
$tile = ($rows | Where-Object split -eq 'test' | Select-Object -First 1).input
.\fire-harmonization-research-YESSS\run-training.cmd infer --tile $tile --output results\my_prediction.npz
```

Paths passed to inference are relative to this experiment folder and must stay inside it. Inputs are prepared, calibrated MODIS NPZ arrays, not raw HDF files. No VIIRS imagery, query grid, reference labels or MODIS fire flags are required. Outputs include native/fine anomaly scores, validity masks, detections, localization support, relative weights and an entropy summary. The entropy describes relative-weight spread, not validated uncertainty in real-world fire location.

## Tests

```powershell
& .\fire-harmonization-research-fullres\.venv\Scripts\python.exe .\fire-harmonization-research-YESSS\tests\test_pipeline.py
```

All 12 tests passed. They cover hidden-center invariance, forbidden label inputs, nighttime fallback, invalid measurements, conservative tied thresholds, consistent calibration/inference precision, path confinement, localization mass and alert preservation, and reconstruction-model training/export/reuse. Final verification checks all copied and prepared file hashes, original benchmark exposure, and identical input-only inference outputs.

The completed native detector recovered only 8 of the 95 positive reference queries, and the localized version recovered 6, compared with 76 for standard MODIS. This is a preserved unsuccessful experiment, not a recommended replacement. There are no nighttime development samples, so nighttime thresholds use the recorded pooled fallback. A numerical threshold-tie correction was applied after initial evaluation without changing model weights or test confusion counts; the original code/results and repair receipt are retained under `results`.

## Research basis

- [SatMAE](https://sustainlab-group.github.io/SatMAE/) supports learning satellite representations from withheld measurements. This experiment uses compact blind-spot regressors rather than SatMAE's transformer architecture.
- [Unsupervised thermal anomaly-detection feasibility study](https://www.mdpi.com/2313-433X/12/6/229) explores background reconstruction for a different sensor. Its reference-guided development and accuracy claims do not transfer to this experiment.
- [USGS MODIS daytime 500 m fire-localization prototype](https://www.usgs.gov/publications/prototype-downscaling-algorithm-modis-satellite-1-km-daytime-active-fire-detections) motivates finer band-7 evidence inside a coarse thermal alert. The neural residual rule here is an experimental adaptation, not a reproduction of that paper.

Learning a normal background does not establish that every anomaly is fire, and resampling cannot recover thermal information MODIS never measured. The experiment is intended to test this approach honestly on the existing data.
