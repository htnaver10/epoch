# fire-harmonization-PLSWORK

A complete local training experiment using **contextual MODIS features → compact
MLP**, with a **LightGBM baseline** on the same features. Training, validation-only
calibration, model selection, benchmark evaluation and label-free inference are
included. No satellite data are downloaded by this pipeline.

**Completed:** data copying, feature preparation, both fits, calibration and
benchmark evaluation. Read the [results and charts](results/TRAINING_SUMMARY.md).

## Run on this computer

```powershell
cd "C:\Users\Darrien\Desktop\epoch1\epoch\fire-harmonization-PLSWORK"
.\run-training.cmd
```

The default runs all stages and reuses completed copy/preparation/training artifacts.
MLP training resumes at the last completed epoch if interrupted. An interrupted tree
fit restarts from its fixed seed; completed tree fits are reused. Configuration,
source hashes and feature-matrix hashes are frozen before training.

The launchers use the already installed GPU Python in
`../fire-harmonization-research-fullres/.venv` and the existing LightGBM Python in
`../fire-harmonization-research/.venv`. **Code, copied data, features, models, logs
and results are in this new folder; only installed Python runtimes are reused.**
No packages are installed and no data acquisition API is called.

Individual stages:

```powershell
.\run-training.cmd copy
.\run-training.cmd prepare
.\run-training.cmd train-mlp
.\run-training.cmd train-tree
.\run-training.cmd calibrate
.\run-training.cmd evaluate
```

The launcher dispatches `train-tree` to the existing LightGBM environment. To run
only that stage directly:

```powershell
..\fire-harmonization-research\.venv\Scripts\python.exe .\run.py train-tree
```

## Data: physically copied, with original partitions

`tools/bootstrap_copy.py` copied **1,218 local files, 10.97 GiB** from the prior
full-resolution folder, including the admitted native tiles and all raw granules
referenced by them, the frozen parent manifest, the prior supervised checkpoint,
and a small snapshot of its inference modules. Files were physically copied, not
linked; source and destination SHA-256 values were checked. The source folder was
not edited. Unused raw acquisitions and separate VIIRS-teacher imagery are not
needed and were not copied.

| Split | Tiles | Known reference queries | Positive query labels |
|---|---:|---:|---:|
| Train | 844 | 10,371,908 | 1,913 |
| Validation | 172 | 1,896,310 | 1,757 |
| Test benchmark | 84 | 585,464 | 95 |

All known queries are retained. There is no negative downsampling, synthetic
augmentation, hidden class multiplier, VIIRS imagery input, fire-mask input or
MODIS-detection feature. VIIRS labels are used as supervision and MODIS detections
as the benchmark. Acquisition groups and source identities retain their original
partitions. Split leakage is checked by the copied native manifest validator.

**The validation cohort and 2024 benchmark were examined in earlier runs. This
is not a new blind test.** Training has three regional cohorts; validation and
test each have one. Correlated query labels are not independent wildfire events.
Agreement with VIIRS is not independent ground-truth validation.

## Architecture and 96 features

Instead of asking another image model to discover all thermal context, this
experiment constructs explicit sensor features and learns a small classifier.

- Six MODIS thermal radiances are transformed with signed `log1p`; their validity
  masks are separate inputs.
- For bands 20, 22, 31 and 32, **7×7 and 21×21 annular windows**, excluding the
  central 3×3 region, provide background mean, standard deviation, center-minus-
  background contrast, clipped standardized contrast, valid fraction and support.
  Context uses only valid MODIS radiances, never reference labels. Eight valid
  background pixels are required; unsupported context has explicit indicators.
- Three between-band log-radiance contrasts and validity indicators describe
  spectral anomalies. These are radiance contrasts, not brightness temperatures.
- Seven reflective channels are sampled on their original 500 m/250 m lattices
  at the query coordinates, with separate validity indicators.
- Fourteen existing geometry/context channels and two offsets within the native
  MODIS cell complete the feature vector. Absolute latitude/longitude are not
  features. No label is used to construct these inputs.

The **8,321-parameter MLP** is `96 → Linear(64) → GELU → Dropout(0.1) →
Linear(32) → GELU → Dropout(0.1) → Linear(1)`. Its sigmoid output estimates
VIIRS-reference fire activity. Features are standardized from training queries
only and clipped at ±10 normalized units. The output bias starts at training
prevalence. Training uses weighted binary cross entropy without class-cost
reweighting, AdamW, BF16 on CUDA, and gradient clipping. It allows up to 25 epochs
with patience 8, selecting by exact validation average precision.

LightGBM uses the same unnormalized features with up to 300 trees, 15 leaves,
maximum depth 6, L2 regularization, deterministic CPU training and validation
early stopping. It also selects by validation average precision. No class weights
are added. Its native model is saved and reloaded for verification.

An MLP is a reasonable **baseline to test**, not an established remedy. Contextual
features plus learned decisions have published support, but those studies do not
prove this MLP beats MODIS or can reconstruct thermal detail that MODIS never
observed.

## Policies and evaluation

Both models are selected on validation; the primary model is the one with higher
exact validation average precision. For each model, a logit-intercept adjustment
and a conservative decision cutoff at the MODIS validation false-positive budget
are frozen **before** new benchmark predictions. Results include raw 0.5 and
frozen-policy predictions, regular MODIS, and the copied prior supervised CNN on
exactly the same eligible queries. No post-benchmark threshold tuning occurs.

Predictions are preserved with query indices and source/tile lineage. The report
includes precision, recall, TP/FP/FN, F1, AP, ROC AUC, Brier score, log loss and
score-sum bias. A favorable single metric does not automatically promote a model.

## Inference without labels or VIIRS

Use a prepared native MODIS tile inside this folder. It must contain `thermal`,
`reflective_500`, `reflective_250`, `geometry` and `query_grid`; it may also include
the label-independent `inference_valid` mask. Target, weight and VIIRS arrays are
not required and are not read by inference.

```powershell
.\run-training.cmd infer --tile results/inference_input_only.npz --output results/prediction.npz --method mlp
```

The completion verifier creates that input-only example from a copied test tile.
`--method tree` dispatches to the existing LightGBM runtime. Output contains raw
probabilities, a validity mask and, when calibration exists, adjusted scores and
frozen-policy detections. The GPU benchmark uses BF16 model execution, while the
standalone MLP inference command defaults to CPU float32. Small numerical changes
near decision boundaries can occur between these execution modes. A finer query canvas is not an increase in the thermal
sensor's true spatial resolution.

## Artifacts and checks

```text
results/local_copy_receipt.json
data/manifests/source_v3.jsonl
data/features/{train,validation,test}_{X,y,w,modis,tile_index,flat_index}.npy
data/features/statistics.json
data/features/receipt.json
models/mlp/{best,last,inference}.pt
models/tree/model.txt
results/mlp_history.json
results/tree_history.json
results/operating_policy.json
results/EVALUATION.md
results/evaluation.json
results/metrics.csv
results/training_and_benchmark.png
results/completion_verification.json
```

Run tests using either existing Python environment:

```powershell
..\fire-harmonization-research-fullres\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\fire-harmonization-research\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All 13 tests are covered across these environments. Tests cover annular/context
arithmetic and masks, input-label isolation, finer-grid reflectance sampling,
training-only statistics, retained query counts, tamper detection, validation-only
policy fitting, tree/MLP reload, label-free prediction and CUDA/BF16 updates.

## Why test this approach?

The operational MODIS algorithm uses spectral thermal anomalies and neighborhood
context: [Giglio et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6110110/).
A related Himawari study used MOD14-derived contextual features with a random
forest and reported gains in its own study setting:
[Maeda and Tonooka (2023 issue)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823964/).
That is supporting evidence for the feature design, not a result for this dataset.

MODIS daytime downscaling has also used finer reflective bands and local context,
with proof-of-concept validation at two sites:
[Kumar et al. (2019), USGS](https://www.usgs.gov/publications/prototype-downscaling-algorithm-modis-satellite-1-km-daytime-active-fire-detections).
The [DataCamp MLP article](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning)
explains the model family; it does not demonstrate fire-detection improvements.
