# Larger NumPy boosted-tree experiment

Frozen before inspecting new model scores. All code, caches, models and figures
remain in experiment-04. Main application files and prior model artifacts are
read-only. NASA download retry failed DNS on 2026-11-14; no new archive was acquired.

## Data and estimand

Use the existing NASA FIRMS-derived 0.25-degree cell-day archive, 50–70N,
140–60W, June–August. Full overlapping years: 2012–2025. Target is **recorded
S-NPP VIIRS occupancy**, not true fire, future ignition, or a 375m reconstruction.
Downloaded-date metadata does not establish clear-sky observation coverage.
Exclude the existing suspected-gap dates from both inputs and labels. Preserve
the earlier 31-cell training-only persistent-source mask for the temporal
comparison, so baseline/model target totals remain comparable.

## Larger model and larger training set

Implement histogram gradient boosting with logistic loss and Newton leaf values
in NumPy; this is not LightGBM. Fixed depth 5, learning rate .075, L2 leaf penalty
10, minimum leaf exposure 500 area-weighted cell-days, maximum leaf step 2.
Fit at most 200 trees. Select a tree count from 25,50,75,100,150,200 by
0.5 × national weekly WAPE + 0.5 × mean regional weekly WAPE on 2020–2022 only.
Resolve exact ties in favor of fewer trees. Do not change features/parameters in
response to the 2023–2025 scores.

Ten fixed-bin features, all MODIS-only: current detection count, occupied cells
in 3x3 and outer 5x5/9x9 rings, 5x5 detection count, previous-six-valid-date-window
detection count at the cell, previous-six-day cell occupancy, previous-six-day
3x3 occupied cell-days, available prior days, and season phase. Temporal windows
use the preceding six calendar days, excluding invalid dates; they never use
future days. No longitude/latitude predictors or VIIRS-derived input features.

Predictions are zero where neither current 9x9 nor previous-six-day 3x3 MODIS
activity exists. This predeclared support constraint avoids a learned background
floor across huge areas; report the VIIRS activity outside support. It also means
this model cannot recover isolated VIIRS detections lacking MODIS context.

Use **every eligible cell-day with no random background subsampling**. Collapse
identical binned feature vectors to exact exposure and positive-label sums.
These sufficient statistics preserve weighted logistic loss/gradients exactly
for the binned model. Report full-grid exposures, supported exposures and
distinct feature patterns separately; none is a count of independent fires.
Area weights are cos(latitude)/cos(60N), matching the reported activity unit.

## Fit and evaluation sequence

1. Train 2012–2019, choose tree count on 2020–2022. This reproduces the earlier
   temporal selection structure with a higher-capacity model and full grid.
2. Freeze that tree count; refit 2012–2022 (11 summers) and evaluate 2023–2025.
   Those years have been inspected in previous research: call this a
   **retrospective temporal benchmark**, not a new blind test.
3. Fit the same configuration on **2012–2025 (14 summers, 75% more years than
   the original eight-year training)** and save as the final research model.
   Its 2023–2025 predictions are in-sample and must not be reported as holdout
   evidence. No complete new local year is available to evaluate this final fit.
4. Run a separate geographic transfer fit using only longitude < -92 degrees,
   select tree count on the same western/central training area in 2020–2022,
   refit those regions through 2022, then evaluate longitude >= -90 in 2023–2025.
   Restrict the persistent-source mask to the geographic training area; east
   labels cannot affect it. The two-degree buffer exceeds the 1-degree maximum
   feature radius. The east evaluation includes persistent eastern sources that
   cannot be identified from western training labels. These are retrospective
   diagnostics and do not establish global transfer.

Comparators on identical windows/masks: raw MODIS, matched-history power-law
and constant-ratio calibration, the frozen original daily logistic model and
frozen eight-year kernel (only when their reference totals match). Also score
the eight-year boosted model before refitting to distinguish capacity effects
from adding the 2020–2022 training years.

## Outputs and checks

Weekly MAE/RMSE/WAPE, aggregate bias, seasonal errors, per-region errors and peak
timing. Daily area-weighted Brier/log loss, threshold-.5 precision/recall/F1,
ROC AUC and average precision refer only to recorded VIIRS occupancy. Clip
probabilities only for log calculations, not the activity estimates. Show
before/after/reference totals, weekly profiles, calendars, shared-scale maps,
regional errors, and selection/training curves. Plot only the temporal benchmark
as evidence; explicitly label the final 14-year model as awaiting a new holdout.

Meaningful synthetic tests must establish weighted aggregation equivalence,
no temporal/spatial wrap or future leakage, correct logistic gradients/splits,
serialization, zero support and conservation of prediction/observation totals.
Independently recompute metric tables from saved predictions, check figure data
against predictions, and fingerprint the main app and previous experiments.
