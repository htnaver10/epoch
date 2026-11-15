# MODIS contextual neural / residual-tree hybrid

This experiment combines physical context, a frozen neural network, new shallow boosted-tree corrections, and the operational MODIS detection flag. VIIRS supplies reference labels during training and evaluation. Inference requires only prepared MODIS inputs, an output query grid, MODIS detection flags and a MODIS-derived validity mask.

The parent experiment and its results remain unchanged. Satellite inputs come exclusively from the previously copied local dataset. No imagery downloads, account credentials, new Python environments or package installations are needed.

## Run

From `C:\Users\Darrien\Desktop\epoch1\epoch` in PowerShell:

```powershell
.\fire-harmonization-PLSWORK\run-hybrid.cmd all
```

The launcher uses the existing fullres Python environment for neural inference and the existing research Python environment for tree fitting. It verifies the parent source lock, prepares aligned features, computes the frozen neural prior, trains the residual trees, freezes validation policies, evaluates, plots, and verifies deployment. Completed stages are reused with artifact checks. Interrupted tree fitting restarts that stage from the fixed seed; this is not an iteration-level optimizer resume.

Results: [completed training summary](results/TRAINING_SUMMARY.md), [evaluation report](results/EVALUATION.md), [comparison figure](results/comparison.png), [spatial figure](results/spatial_comparison.png), [run status](results/run_status.json), [verification](results/completion_verification.json).

The completed run did not improve overall on the prior MLP. A post-fit JSONL parsing correction affected only final verification tooling; the original fitted source snapshot, source lock and explicit repair receipt are retained under `results`. Training parameters, models, thresholds and evaluated predictions were unchanged.

The exported model consists of **both** `models/base_mlp.pt` and `models/residual.txt`, plus `results/operating_policy.json`, `data/receipt.json` (feature schema), the parent feature-extraction code/configuration and this experiment's inference code. The tree file alone does not contain the neural prior.

## Inference example

The completed verification creates an input-only example containing no VIIRS values, target labels or reference weights:

```powershell
.\fire-harmonization-PLSWORK\run-hybrid.cmd infer --tile results/inference_input_only.npz --output results/my_prediction.npz
```

Input and output paths here are relative to `hybrid_v1`. Other inputs must be prepared MODIS NPZ files placed inside that directory. Required arrays are `thermal`, `reflective_500`, `reflective_250`, `geometry`, `query_grid`, `modis_baseline`, and `inference_valid`. This is not a raw-HDF reader. The supplied query grid can be a MODIS-only deployment grid; VIIRS is not needed to generate predictions. The example's grid preserves benchmark query coordinates for verification. The command is single-job: it uses temporary derived files under `data/infer_*`, so do not run concurrent inference jobs in the same experiment folder.

Outputs include raw logits, raw and validation-adjusted probabilities, the MODIS validity mask, and detections at the frozen query threshold. Scores in invalid cells must not be used; the detection array already applies the mask. Calibration on other regions is not guaranteed.

## Alignment and scientific design

- Preserve the existing calibrated radiances, native geolocation mapping, maximum 15-minute observation interval bound, and QA-based label exclusions. The geolocation inversion residual is numerical consistency, not a measure of absolute sensor accuracy.
- Pool valid 250/500 m reflectance measurements over aligned 4x4/2x2 blocks to match MODIS thermal-cell support. Keep the fine measurements and add pooled means, local-minus-pooled contrasts and observed fractions. No VIIRS label enters these features.
- Train a residual classifier on all 10,371,908 known queries, with the existing MLP logit as its initial score. Optimize ordinary binary log loss, with no negative subsampling or class-weight inflation. Select the number of boosting rounds by validation average precision.
- Freeze the query threshold using the validation MODIS false-positive budget. Fit the probability offset on validation. Report the original 95-positive query benchmark unchanged.
- Also group eligible VIIRS reference centers by native MODIS cell. Any positive center makes a cell reference-positive. Compare MODIS and each model on this shared observed support, with separate validation-only cell thresholds. This approximates support alignment; it is not an exact footprint intersection or sensor point-spread-function model.

The new preprocessing is physically motivated, but this experiment does not independently isolate its contribution from the residual model and MODIS flag. The old MLP is a zero-residual comparator. The test cohort was previously examined and is not a new blind test. VIIRS is a reference sensor, not ground truth; a perfect VIIRS self-comparison is not evidence of perfect fire detection.

## Research basis

1. [Maeda and Tonooka, contextual MOD14 plus machine learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823964/): physical background features can help a learned fire classifier. Their Himawari results do not establish the performance of this MODIS adaptation.
2. [Friedman, gradient boosting](https://boosting.org/papers/Fri99.pdf): stagewise additive corrections under classification loss. Here the initial function is a frozen MLP rather than a constant.
3. [Schroeder et al., active-fire validation with fine-resolution reference data](https://www.isprs.org/proceedings/2011/isrse-34/211104015Final00178.pdf): near-coincident reference observations and coarse-sensor support matter to valid comparisons.
4. [NASA's VIIRS active-fire description](https://viirsland.gsfc.nasa.gov/Products/NASA/FireESDR.html): 375 m VIIRS has greater small-fire sensitivity than coarser sensors. Resampling cannot create missing MODIS thermal information.
5. [LightGBM Dataset API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Dataset.html): initial-score interface. Tests check that validation includes the neural prior and that reloaded tree inference explicitly adds it.

## Tests

```powershell
& .\fire-harmonization-research\.venv\Scripts\python.exe .\fire-harmonization-PLSWORK\hybrid_v1\test_hybrid.py
```

Nine focused tests cover aligned valid-only pooling, unknown support, label-input rejection, native-cell aggregation, tied thresholds, path confinement, validation-before-test enforcement and residual export semantics. The full run also checks original-feature equivalence, split source separation, SHA-256 provenance, unchanged benchmark exposure, and input-only inference agreement.
