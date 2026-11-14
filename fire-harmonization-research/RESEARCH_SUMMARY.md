# Fire-harmonization research: complete current summary

**Status date: 2026-11-14.** This is the current overview of the research folder.
It separates completed work, measured findings, blocked work, and proposed work.
Earlier protocols and reports remain historical records of their own experiments.

**Completed:** isolated coarse-grid experiments, before/after figures, a global
archive acquisition plan and validator, and a trained larger boosted-tree model.
**Next objective:** an actual image-tile model using native-resolution MODIS
measurements, supervised by spatially and temporally aligned VIIRS observations.
**Not completed:** native-radiance acquisition, image-tile training, validated
fine-scale fire localization, global/full-year validation, or main-app integration.

The detailed next-model specification is
[Native-resolution MODIS tile model plan](docs/NATIVE_RESOLUTION_MODEL_PLAN.md).

## 1. Project goal and decisions

The user supplied the challenge objective: combine MODIS's historical record with
VIIRS continuity/detail to build a harmonized burning-activity calendar. A web
application should support regional/date selection, historical comparisons,
seasonal patterns, anomalies, visualizations and downloadable evidence. The
challenge Resources URL was investigated but could not be retrieved successfully;
the challenge wording in this folder came from the conversation.

The two work streams are data harmonization and the eventual application that
consumes it. Sensor differences should not dominate the apparent trend, while
sensor identity, processing version and observed-versus-estimated provenance
remain available for inspection.

The direction changed during discussion:

1. The existing app uses MODIS as its activity reference and translates VIIRS
   toward MODIS with a seasonal power-law bridge.
2. Initial ideas included that same direction and LightGBM/Tweedie calibration.
3. The user revised the ML objective to **MODIS → VIIRS-reference activity**.
   All executed research models discussed below follow this revised direction.
4. The user requested a separate research folder and explicitly prohibited
   implementation in the main app. That boundary has been maintained.
5. The user then asked about native resolution, upscaling, broader training and
   more precise locations. The latest objective is **native MODIS imagery as
   model input, VIIRS as training supervision**, rather than another model of
   coarse aggregate counts.

NASA-distributed observations are the agreed source restriction. NOAA-platform
VIIRS products distributed by NASA fit that interpretation. The existing
response/spread simulator is a separate system and was not changed. Active-fire
hotspots, unique incidents, ignition locations, burned area and future spread are
different quantities; no experiment here makes them interchangeable.

## 2. Existing pipeline inspected, without modification

The relevant existing application path is offline preparation followed by a
static frontend at runtime:

```text
NASA FIRMS downloads
    → sensor/year cell-day aggregates
    → frontend preparation scripts and JSON assets
    → regional/date aggregation, calendars, comparisons and exports
```

Inspection covered the season downloader, published season metadata and year
files, preparation logic, and the relationship between observed and harmonized
series. The downloader originally aggregated detections immediately and discarded
their original detailed records. Its intermediate schema included nighttime
counts; the published five-column schema does not retain them. Those richer
intermediate archives were not available as local input for these experiments.

The app's reference direction, inferred persistent-source masks, latitude
weighting, heuristic gap handling and static build architecture were documented
in [the integration proposal](docs/INTEGRATION.md). That document is a proposal,
not an implemented service or schema change. A FastAPI prediction endpoint,
browser model inference, automatic retraining and deployment were not added.

## 3. What data actually existed locally

| Property | Observed local archive |
|---|---|
| Files audited | 24 year JSON files plus one metadata JSON file |
| MODIS years | 2003–2026, June–August |
| Full MODIS/VIIRS overlap | 2012–2025: 14 summers |
| 2026 overlap | Partial: metadata reports 92 MODIS days and 61 VIIRS days; excluded from the main benchmarks |
| Geographic scope | 50–70°N, 140–60°W rectangular grid; not a national boundary mask |
| Grid | 0.25° latitude/longitude; 80 × 320 = 25,600 cells |
| Published fields | Date, cell-center latitude/longitude, MODIS count, VIIRS count |
| All archived raw detection totals | 1,206,005 MODIS and 5,081,318 VIIRS, before experiment masking/gap screening |
| Native imagery present | None: no MODIS or VIIRS L1B/L2 swath granules were used |

At 60°N, a 0.25° cell is approximately 28 km north–south by 14 km east–west.
That is far larger than either instrument's nominal active-fire pixel. A cell-day
record does not retain the positions or thermal structure of its constituent
detections. Interpolating this archive cannot recover native imagery.

Important missing fields include original coordinates within each cell, UTC
overpass time, radiances/brightness temperatures, FRP, confidence, MODIS platform,
pixel footprint, scan/track geometry, cloud/clear-sky exposure and detailed QA.
The original data audit, totals and checksums are in
[archive_audit.json](results/archive_audit.json),
[archive_audit.csv](results/archive_audit.csv) and
[input_hashes.json](results/input_hashes.json).

### Common target, units and screening

The daily target is whether S-NPP VIIRS recorded at least one detection in a
coarse cell that day. Summing expected daily occupancy gives expected occupied
cell-days. Calendar totals use area weights `cos(latitude)/cos(60°N)`.
These units are neither raw hotspot counts nor the number of independent fires.

A training-only persistence rule identified 31 cells for the temporal experiments:
activity in at least 80% of training summers and a median of at least five VIIRS
active days. This is a heuristic exclusion of persistent thermal sources, not a
verified industry/forest classification. The common mask keeps temporal model
comparisons on the same targets. Geographic transfer fits restrict source-mask
learning to their training area.

Metadata's suspected-gap dates were removed from both sensors instead of filling
labels. For the 2023–2025 comparison, 91, 88 and 89 matched days remain. The gap
flags themselves are heuristic. Downloaded-day coverage is not verified clear-sky
coverage, and a recorded-zero cell is not established ground-truth absence of fire.

## 4. Experiment 01: executable statistical baselines

**Location:** root `src/`, `tests/`, `models/`, `results/` and `visualizations/`.

Implemented and executed:

- Weighted NumPy logistic regression for daily VIIRS coarse-cell occupancy.
- Weighted NumPy Poisson regression for weekly occupied days.
- Raw MODIS, constant-ratio and log-space power-law activity baselines.
- A separate eastern geographic stress test, training west of 92°W and testing
  east of 90°W with a two-degree separation.
- Audits, hashes, diagnostics, serialized models, weekly/seasonal predictions,
  independent metric recalculation and five before/after figure families.

Features use MODIS counts, occupied days, neighborhood activity, preceding-six-day
activity, available-day counts, geographic coordinates and seasonal terms.
Standardization uses training data. The fixed history was 2012–2019 for fitting,
2020–2022 for validation, and initially 2023–2025 for temporal evaluation.
Training retained all MODIS- or VIIRS-active examples plus a 1% background sample
with inverse inclusion weights; the daily fit retained **251,968 rows**. Evaluation
covered the complete eligible grid. The logistic/Poisson implementation used L2
regularization and iteratively reweighted fitting.

The optional LightGBM binary/Tweedie branch was written but **never executed**:
PyPI DNS failed and LightGBM was not installed. Executed logistic/Poisson results
must not be described as LightGBM. See
[lightgbm_status.json](results/lightgbm_status.json).

### Whole-rectangle temporal results

All entries below use the original eight-year fit and the same 42 calendar bins
in 2023–2025. MAE is in area-weighted occupied cell-days per calendar bin.

| Method | Weekly MAE | Aggregate bias | Worst matched-summer absolute error |
|---|---:|---:|---:|
| Raw MODIS | 348.77 | −26.75% | 28.92% |
| Constant ratio | 210.5 | +15.8% | 23.6% |
| Power law | 109.38 | +4.78% | 15.07% |
| Daily logistic | 118.37 | −1.14% | 7.67% |
| Weekly Poisson | 454.3 | −34.3% | 39.7% |

The daily logistic model reduced error relative to raw MODIS but did not beat
the power law on weekly MAE. Its eastern transfer MAE was **201.59**, with
**+165.28%** aggregate bias, compared with raw eastern MODIS MAE **32.99**.
This was a clear extrapolation failure. Coordinate terms and a nonzero background
floor were possible contributors, but their causal roles were not established.

Decision: keep the model experimental. The initial temporal scores subsequently
became development evidence once examined and used to motivate new experiments.
See [the original report](results/REPORT.md) and
[visual evaluation](results/EVALUATION.md).

## 5. Experiment 02: bounded spatial response and rolling backtests

**Location:** [experiment-02](experiment-02/README.md).

This replaced the free spatial/logistic response with an established bounded
response applied to local current-day MODIS occupancy:

`p = 1 − exp(−alpha × signal^beta)`.

Zero MODIS neighborhood signal produces zero output. The scale was fitted to
weighted activity using full-grid sufficient-statistic histograms; this experiment
already used the full grid, rather than the sampled regression rows.
Validation selected among 27 combinations of spatial radius, response exponent
and regional pooling. The selected radius was two source cells (0.5°), beta 0.75,
and regional pooling weight 1.0. These are coarse degree-based neighborhoods,
not physical fire footprints or a super-resolution model.

| 2023–2025 retrospective comparison | Weekly MAE | Bias | Worst summer error |
|---|---:|---:|---:|
| Raw MODIS | 348.77 | −26.75% | 28.92% |
| Original logistic | 118.37 | −1.14% | 7.67% |
| Kernel response | 119.32 | +4.58% | 17.60% |
| Same-history power law | 109.38 | +4.78% | 15.07% |

The kernel improved eastern transfer substantially: MAE **28.27**, bias **+18.11%**.
Its worst eastern seasonal error still reached **96.65%**. The six chronological
outer backtests, 2020–2025, selected settings on earlier years and refitted on
history before each outer year. Pooled national MAE was **101.17** for the kernel
and **80.85** for matched-history power-law calibration. Those results are also
retrospective; they are not a prospective trial.

The kernel preserved the reference peak bin in all three 2023–2025 summers.
However, **11.7%–17.0%** of reference activity lay outside its MODIS support.
Matching totals can compensate at other locations and therefore does not prove
better spatial localization. Its aggregate fitting did not establish cell-level
probability calibration. Six figure families include the five original comparisons
and an additional rolling-backtest figure.

Decision: do not promote; retain simple calibration as a serious comparator.
See [report](experiment-02/results/REPORT.md),
[protocol](experiment-02/PROTOCOL.md) and
[validation](experiment-02/results/validation.json).

## 6. Experiment 03: larger NASA archive discovery and access attempt

**Location:** [experiment-03](experiment-03/README.md).

Official documentation established larger, global, year-round sources:

- NASA FIRMS bulk historical exports with explicit MODIS C6.1 and VIIRS coverage.
- Science-team monthly `MCD14ML` MODIS and `VNP14IMGML` VIIRS location files on
  UMD's public SFTP server, documented in NASA/USGS product guides.
- NASA Level-2 fire masks/geolocation and Level-1B measurements through DAACs,
  needed for observation-aware/native-resolution work.

The public monthly SFTP route does not require a FIRMS key. Its documented MODIS
directory has collection-006 filename examples, so current availability of
collection 061 at that location remains unverified. No silent collection fallback
is allowed. Monthly **location** files preserve individual detections but still
do not contain the full image and all observed non-fire pixels needed by the next
tile model.

Implemented:

- A **48 planned sensor-month pilot**: all months of 2019 and 2023, two products.
- A **288 planned sensor-month expansion**: all months of 2013–2024, two products.
- Interactive SFTP inspection/download command lists and planned inventory.
- A streaming gzip/schema/date/coordinate/confidence/version reader, SHA-256
  manifests, duplicate-month detection and an ingestion-readiness audit.
- Synthetic parser tests and access/isolation evidence.

Direct retries failed to resolve UMD SFTP, NASA FIRMS and LAADS before connection.
No FIRMS map key or Earthdata credential was configured. No authenticated API
success, remote inventory, file sizes or downloaded record counts were established.
The full audit records **zero valid new files and all 288 sensor-months missing**.
The web tool could read documentation; that is not a bulk-data download.

Experiment 03's `not_run_pending_acquisition` training status applies to its
proposed global expansion. It does not negate the later successful local-archive
training in Experiment 04. See
[archive assessment](experiment-03/LARGER_ARCHIVE.md),
[access evidence](experiment-03/results/access_status.json) and
[full audit](experiment-03/results/full_audit.json).

## 7. Experiment 04: larger model successfully trained

**Location:** [experiment-04](experiment-04/README.md).

A custom NumPy histogram gradient-boosted logistic model was implemented and
trained. It is **not LightGBM, XGBoost or an image neural network**. It uses ten
fixed-bin MODIS-only features: current counts; occupied 3×3 neighbors and 5×5/9×9
outer rings; 5×5 counts; prior-six-day counts and occupancy; prior-six-day 3×3
activity; available prior days; and season phase. It excludes coordinates as
predictors. Current 9×9 or prior-six-day 3×3 MODIS context is required for nonzero
output. No future observations or VIIRS predictors enter inference.

Identical binned inputs were collapsed into exact weighted exposure and
positive-label sums. This preserves the binned model's weighted logistic
objective and derivatives, without random background sampling. Unsupported
cells are included in the population accounting but have forced-zero output;
the supported exposure is the portion used to learn the conditional response.

Fixed settings: depth at most 5, learning rate 0.075, L2 leaf penalty 10, minimum
leaf exposure 500 weighted cell-days and clipped Newton leaf step. Selection
considered 25, 50, 75, 100, 150 and 200 trees using 2020–2022 only, after an
eight-year fit. The criterion equally weights national weekly WAPE and mean
regional weekly WAPE. It selected **200 trees** for the temporal model.

### Separate fits and their meaning

| Fit | Training years | Eligible cell-days processed | Supported weighted cell-days | Supported distinct patterns |
|---|---|---:|---:|---:|
| Initial selection fit | 2012–2019 | 18,818,784 | 1,533,702 | 57,163 |
| Evaluated temporal refit | 2012–2022 | 25,875,828 | 2,028,997 | 68,293 |
| Final research refit | 2012–2025 | 32,728,320 | 2,979,022 | 122,498 |

The final research ensemble has **200 trees and 5,319 leaves**. Its training
history is 75% longer than the original eight-year fit. Full cell-day counts,
area-weighted exposures and distinct patterns are different units. The raw input
is still the same reduced archive; no new NASA download occurred. These cell-days
are highly correlated and are not millions of independent fires.

The final fourteen-year model has **no fresh complete local holdout**. The
eleven-year model below is the separately evaluated artifact. No 2023–2025 score
of the fourteen-year final fit is presented as held-out performance.

### Whole-rectangle retrospective benchmark, 2023–2025

| Model | Fit history | Weekly MAE | RMSE | WAPE | Bias | Worst summer error |
|---|---|---:|---:|---:|---:|---:|
| Raw MODIS | None | 348.77 | 427.04 | 27.09% | −26.75% | 28.92% |
| Original logistic | 8 years | 118.37 | 161.36 | 9.19% | −1.14% | 7.67% |
| Kernel response | 8 years | 119.32 | 171.16 | 9.27% | +4.58% | 17.60% |
| Boosted model | 8 years | **94.82** | **129.30** | **7.36%** | −0.74% | **4.47%** |
| Power law | 11 years | 102.70 | 145.71 | 7.98% | +1.20% | 10.48% |
| Boosted temporal refit | 11 years | 100.83 | 135.30 | 7.83% | −2.09% | 4.73% |

The eleven-year boosted model reduces MAE **71.1% versus raw MODIS**, but only
**1.8% versus its matched-history power law**. It wins 38/42 bins against raw
MODIS and 21/42 against the power law. Statistical superiority was not established.

**Increasing history did not improve this metric:** eleven-year boosted MAE is
6.3% worse than the same selected architecture fitted on eight years. Capacity,
context and additional training years have different effects. These experiments
do not prove that the largest final fit is the best model.

### Daily recorded-occupancy metrics

These are area-weighted full-grid scores on 2023–2025 for the eleven-year model.
The precision/recall/F1 threshold is 0.5. The reference is recorded VIIRS
occupancy, not verified fire existence. ROC AUC is not an accuracy percentage.

| Metric | Raw MODIS | Eleven-year boosted |
|---|---:|---:|
| Brier score | 0.003605 | 0.002794 |
| Log loss | 0.099613 | 0.021053 |
| ROC AUC | 0.81916 | 0.96611 |
| Average precision | 0.56038 | 0.79821 |
| Precision | 0.87240 | 0.86814 |
| Recall | 0.63906 | 0.68067 |
| F1 | 0.73772 | 0.76306 |

Precision decreased slightly while recall improved. Experiment 01's reported
daily Brier/log-loss diagnostics use different weighting, so they should not be
directly ranked against this table without recalculation on identical definitions.

### Geography, support and timing

A separate model used only longitudes west of 92°W for fitting and checkpoint
selection, then evaluated east of 90°W. It selected 75 trees and refitted through
2022. No eastern labels were used to learn its persistent-source mask.

| Eastern transfer model | Weekly MAE | WAPE | Bias |
|---|---:|---:|---:|
| Raw MODIS | 32.99 | 27.05% | −23.01% |
| Training-area power law | 27.38 | 22.44% | +21.09% |
| Boosted transfer model | **14.53** | **11.91%** | **+2.34%** |

This is improved transfer inside the boreal pilot, not evidence of global or
tropical performance. The final fourteen-year temporal fit has **10.6%** of
training-reference activity outside its required MODIS support.

The eleven-year model matches two of three reference peak bins. It matches
August 3, 2023 and August 10, 2024; its 2025 peak is August 3, one bin later than
the July 27 VIIRS peak. Raw MODIS and the power law match all three. Better
aggregate activity does not establish better timing or precise coordinates.

Artifacts: [full report](experiment-04/results/REPORT.md),
[training ledgers](experiment-04/results/training_ledgers.json),
[weekly predictions](experiment-04/results/weekly_predictions.csv),
[daily metrics](experiment-04/results/binary_metrics.json),
[peak timing](experiment-04/results/peak_timing.csv),
[feature-gain diagnostics](experiment-04/results/final_model_feature_gain.csv),
[selection evidence](experiment-04/results/selection.json).

## 8. Models, figures and verification preserved

Important saved models:

- Original logistic, Poisson, geographic and simple-baseline JSON artifacts in
  [models](models/).
- Kernel fixed, eastern and six rolling-fold artifacts in
  [experiment-02/models](experiment-02/models/).
- [Eight-year selected boosted model](experiment-04/models/boosted_8year_selected.json).
- [Eleven-year evaluated boosted model](experiment-04/models/boosted_11year_benchmark.json).
- [Fourteen-year final research model](experiment-04/models/boosted_14year_final.json).
- [Eastern transfer boosted model](experiment-04/models/boosted_cold_east.json).

There are **17 completed figure families**, each exported to PNG and SVG:
five in the original experiment, six in Experiment 02, and six in Experiment 04.
They cover totals, weekly profiles, calendars, shared-scale spatial comparisons,
regional errors, rolling backtests and model-size selection. Figure data,
underlying map arrays and total reconciliation are retained. Spatial figures
continue to use the original 0.25° grid; visual smoothness is not new information.

Galleries:
[original](visualizations/README.md),
[Experiment 02](experiment-02/results/REPORT.md#figures),
[larger model](experiment-04/visualizations/README.md).

| Check set | Recorded result |
|---|---|
| Original model tests | 13 passed; 1,848 prediction rows and 44 metric groups independently checked; 25 input files hashed |
| Kernel tests | 8 passed; 3,192 prediction rows, six outer folds and six figures checked; 22 earlier artifacts preserved |
| Archive-reader tests | 7 passed, using synthetic format/integrity cases; no evidence of real downloaded observations |
| Larger-model tests | 8 passed, including exact aggregation, causal temporal features, split/loss behavior, serialization, probability ties and reconciliation |
| Larger-model independent audit | 2,352 prediction rows, 56 metric groups, full-grid probability metrics, maps, figures and frozen training code checked |
| App isolation | 183 original application files unchanged in the recorded audits |
| Earlier research preservation during Experiment 04 | 127 pre-existing research files unchanged at that run's validation |

No application dependency, environment, frontend code, backend code, deployment
artifact or build configuration was modified. Research uses a separate `.venv`
with existing NumPy/Matplotlib; it did not successfully install LightGBM, SciPy,
PyTorch or other training libraries. Documentation updates after an experiment
are intentional and do not retroactively change its saved validation evidence.

## 9. What the evidence supports

Supported: context-dependent models can improve recorded coarse-grid VIIRS
activity estimates from MODIS; the constrained and boosted models improve on the
first model's eastern extrapolation failure; the native-data upgrade needs richer
measurements and stronger labels than the local counts contain.

Not established: actual 375 m MODIS reconstruction, exact forest-fire locations,
unique-fire counts, forest-only classification, future ignition/spread forecasts,
cloud-obscured fire recovery, full-year/global transfer, independent operational
performance, reliable pre-2012 fine-scale reconstruction, or the superiority of
the final fourteen-year refit. No synthetic fixture substitutes for satellite
performance evidence. No model has been promoted into the app.

## 10. Latest objective and next deliverable

The next experiment must consume **real native MODIS image tiles**, including
thermal measurements, native high-resolution reflective context where usable,
quality masks, pixel geometry and acquisition times. VIIRS supplies spatial
supervision and observation masks during training. At inference, the MODIS-only
variant must work without VIIRS images, coordinates, fire masks or features.

The next output is a georeferenced activity/forest-fire candidate field with
calibrated location uncertainty, plus separately retained observations. Native
pixel detection and finer candidate localization need separate evaluation. A
fine output lattice is a model representation; achieved location accuracy must
be measured, and higher-resolution independent observations are needed to justify
claims beyond the VIIRS observation reference.

Proceed through the acquisition, matching, native tiling, model, validation and
acceptance gates in the [full native-resolution plan](docs/NATIVE_RESOLUTION_MODEL_PLAN.md).
The present request documents that work; it does not claim a native tile model
has already been trained or authorize integration with the main application.
