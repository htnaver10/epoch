# SaturationAwareBackgroundV2

This is the separately named, corrected unsupervised model inside `fire-harmonization-research-YESSS`. The original V1 package, configuration, feature cache, checkpoints, predictions and scores are preserved. V2 reads the already copied MODIS-only files and writes all model-specific artifacts under this subfolder. No satellite downloads or new package installation are needed.

## Run

From `C:\Users\Darrien\Desktop\epoch1\epoch` in PowerShell:

```powershell
.\fire-harmonization-research-YESSS\run-saturation-v2.cmd all
```

This runs feature preparation, both model fits, unlabeled day/night calibration, frozen-policy evaluation, reporting and verification. The existing fullres Python environment supplies PyTorch/CUDA. The task-local PyTorch cache avoids writing to a restricted global cache directory.

Completed stages are reused after checksum checks. Training resumes at a saved complete epoch with optimizer and random states. The V2 source/configuration lock refuses changed code after fitting starts. The separate original `run-training.cmd` remains available to reproduce V1.

## Corrections

1. **Use band 21 when band 22 is unavailable.** Predict the two measurements separately, with missing targets masked from training loss and target statistics. Normalize each channel's reconstruction residual using its observed development distribution. Prefer band 22 when valid, otherwise band 21. Band 20 is background context only. V2 does not require both residuals to be positive or both target channels to be valid. Saturation/fill codes are not numeric radiances and are never imputed. The existing input mask records availability; it does not distinguish every raw reason for missingness. The original raw-source provenance and the saturation audit retain the observed code evidence.
2. **Reconcile query/cell boundaries.** Fitting stays within disjoint original halo-16 interiors. Prediction uses complete radius-10 native context, including cells covering the original continuous-query halo boundary. Reference eligibility is unchanged; invalid model locations still count as undetected.
3. **Supply actual nighttime development data.** Within each inherited training region/day-night stratum having at least two swaths, hold out `max(1, floor(0.2*n))` complete swaths by seeded SHA-256 order. Add the original validation swath. Exclude those held-out swaths from training statistics and optimizer updates. Test swaths are never read during fitting or calibration. The frozen partition is in `data/partition.json`. This is swath separation within the limited existing events, not independent regional validation.
4. **Measure the failure directly.** Report positive and negative coverage separately, selected-channel counts at reference locations, and independent confusion-count checks. Tests reproduce band-20/22 unavailability with a valid band-21 measurement, missing-target gradient masking, hidden-center invariance, boundary support and threshold precision.

The same small network widths, robust reconstruction loss, maximum 40 epochs, patience 7, coarse 0.1% alert-volume fraction and fine 1% fraction are retained. Whole-swath holdout changes the training/development composition. This is a corrective experiment with several necessary changes; it is not a one-factor ablation.

## Outputs

- `results/TRAINING_SUMMARY.md`: completed run, comparison and interpretation. V2 coarse recall is 87.37% with 136 false positives; it is much better than V1 but has lower precision/F1 than MODIS. Localization hurts the current benchmark.
- `models/coarse/inference.pt`: **SaturationAwareBackgroundV2**, native background regressor, predicting bands 21/22.
- `models/fine/inference.pt`: its separately retrained daytime 500 m localization background regressor.
- `models/policy.json`: frozen per-channel/day-night residual normalization and thresholds.
- `results/EVALUATION.md`: same query benchmark, native-cell comparison, coverage, training and limitations.
- `results/comparison.png`: training curves and benchmark results.
- `results/completion_verification.json`: end-to-end checks.
- `results/prior_artifact_lock.json`: checksums proving V1 models/code/results were preserved.
- `results/source_lock.json`: V2 source and input provenance.
- `results/run_status.json`: current stage or completed status.

The original 95-positive test benchmark has already informed the diagnosis and repair. Its new result is development evidence, not a fresh blind test. Thresholds are fitted from MODIS measurements only and are not tuned to reference labels. VIIRS is used only after freezing to measure agreement. Its self-comparison row is perfect by definition.

## Inference

Copy a MODIS-only prepared NPZ inside this V2 subfolder, or use the input-only example created by final verification:

```powershell
.\fire-harmonization-research-YESSS\run-saturation-v2.cmd infer --tile results/input_only_example.npz --output results/my_prediction.npz
```

Input/output arguments are relative to `saturation_aware_v2` and confined there. Required arrays are `thermal`, `reflective_500`, `reflective_250`, and `geometry`. This accepts calibrated prepared tiles, not raw HDF files. VIIRS, fire labels, MODIS fire flags and a VIIRS query grid are not required.

Outputs include native anomaly score, validity, detection, `selected_band` (21/22, or 0 if unusable), both channel-availability masks, fine scores/validity/detection and localization weights/support. Scores and relative weights are **not calibrated fire probabilities**. A saturated pixel is not automatically classified as fire; it needs a valid alternative measurement and an anomaly exceeding the frozen threshold.

The 500 m localization stage retains the original experimental rule: require four valid daytime child observations and strong shortwave evidence; retain the top two scores including ties. Otherwise the coarse alert remains unresolved and all child locations are retained. This branch cannot recover a missed native alert or create additional native alerts. It provides estimated locations using finer MODIS measurements, not new 375 m thermal information.

## Tests

```powershell
& .\fire-harmonization-research-fullres\.venv\Scripts\python.exe .\fire-harmonization-research-YESSS\saturation_aware_v2\tests.py
```

All 15 regression tests passed, covering corrected sensor behavior, training/export, and correct report/verification module resolution. Final verification additionally checks the shared physical copies, new feature-cache hashes, preserved V1 artifacts, disjoint fit swaths, nighttime calibration counts, positive-reference coverage, independent confusion counts and exact input-only deployment agreement.

A post-fit import-path correction changed only how the frozen evaluation helper is loaded, preventing it from redirecting report/verification imports to V1 tools. The original source snapshot and a repair receipt are preserved under `results`; models, thresholds, features and predictions were unchanged.

## Scientific limits

Correcting coverage does not guarantee accurate fire detection. Background reconstruction may model fires as ordinary observations or flag hot ground, reflections and cloud edges. A 0.1% alert-volume threshold is an engineering choice, not a known false-positive rate. No additional labeled data are used to select checkpoints or policies, but the overall corpus and repair were developed with knowledge of previously examined events. Further claims of improvement require independent events.

[NASA calibration ATBD](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/MODIS_L1B_ATBD_ver8_finalupload.pdf) describes low-gain fire band 21. [NASA L1B file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/5/MOD021KM.fs) defines saturation code 65533. [The USGS 500 m localization prototype](https://www.usgs.gov/publications/prototype-downscaling-algorithm-modis-satellite-1-km-daytime-active-fire-detections) supplies the physical rationale for finer shortwave evidence; this learned residual rule is an experimental adaptation.
