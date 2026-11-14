# Revised experiment plan: MODIS -> VIIRS

## Contract and boundaries

Read the bundled observation files without importing app modules or executing
their preparation scripts. Write every new file under this research folder.
Preserve the common 0.25-degree grid and cos(latitude)/cos(60N) weighting. Compare
MODIS and VIIRS on the same dates. Do not add either a browser inference engine or
a prediction service to the app.

Daily target: VIIRS Suomi-NPP recorded occupancy (0/1) per grid cell/day.
Daily output: expected occupancy, aggregated by summing probability × area weight.
Weekly alternative: recorded VIIRS occupied days per grid cell/week. The short
final week uses its actual length; evaluation uses totals, not normalized seven-day
display intensity. These are observation-reference targets, not true-fire labels.

## Available-data preliminary experiment

Use only fields available locally: MODIS counts, occupied days, neighboring MODIS
counts/occupancy, the preceding six days of MODIS activity, latitude/longitude and
seasonal sine/cosine. No VIIRS-derived feature and no future-day feature is allowed.
Leading June days have shorter context, represented by `previous_valid_days`.
Before claiming full-year service, ingest May context as well as all other months.

The statistical population is the entire rectangular grid, including recorded-zero
cells. These zeros mean no row in the provided archive, not verified clear-sky
absence of burning. Training keeps all MODIS- or VIIRS-active examples and samples
1% of both-zero background. Inverse sampling weights preserve population prevalence;
validation/test aggregate evaluation covers the full grid. This avoids reporting
results only on cases both instruments detected.

Fit inferred persistent-source masks on training years only (at least 80% of
training seasons active; median >=5 VIIRS days). Apply the resulting fixed mask to
all periods in that experiment. Do not use the app's all-years static mask.

Exclude the metadata's suspected-gap dates from both instruments; never train on
filled labels. The existing flags are heuristic and use reference activity, so
this is a screened-archive benchmark, not proof of unbiased performance during
clouds/outages. Keep the excluded dates and counts visible. Exclude 2026 because
the local overlap is incomplete.

## Fixed splits and models

- Training: 2012-2019. Validation: 2020-2022. Untouched temporal test: 2023-2025.
- Geographic stress test: train/validate west of 92W, then test east of 90W in
  2023-2025. The two-degree buffer reduces adjacent-cell leakage. Learn the mask
  without east-region labels. Geographic distribution shift is explicitly tested.
- Baselines: uncorrected MODIS occupancy, a training-fitted constant ratio, and a
  training-fitted power law. Fit the latter two per region at the evaluated weekly
  aggregation scale, directly in the MODIS -> VIIRS direction. The existing app's
  reverse, seasonal bridge is not silently relabeled as this benchmark.
- Executable fallback models: weighted NumPy logistic regression for daily
  occupancy and weighted NumPy Poisson regression for weekly occupied days.
  Both use fixed L2=0.0001, training-only standardization and no test tuning.
- Planned library models: native LightGBM binary and Tweedie objectives, shallow
  trees, early stopping using validation data only, feature-order checks and
  native model serialization. Their results are separate from the NumPy models.

## Evaluation and decision

Record weekly MAE/RMSE, aggregate bias, matched-season relative errors and daily
Brier/log loss. Include region and extreme-year breakdowns; a quiet-background
classifier score is not evidence of a useful calendar. A calendar may have good
season totals but poor peak timing; that requires additional acceptance tests.

Before promotion, require improved held-out weekly activity error and no worse
worst-season relative error than the strongest relevant simple baseline, plus
geographic and sensor-transition validation. Preliminary runs are not promotion.
No cell-level probability interval should be summed as an aggregate confidence
interval; use appropriately blocked held-out residuals when designing uncertainty.

The models saved here are trained only on training years, not refitted on the test
years. Future archival publication should use out-of-fold historical predictions;
future dates can use a separately approved final model. This folder does not
publish any reconstructed 2003-2011 series.

## Next experiments if supported

1. Run the optional LightGBM branch in an environment with the dependencies.
2. Verify authenticated NASA availability and ingest original MODIS/VIIRS rows.
3. Add brightness, FRP, pixel geometry, acquisition-time and confidence summaries
   available for MODIS history; keep collection/processing provenance.
4. Test verified coverage masks and unscreened-gap sensitivity; compare weekly
   peak timing, calendar rank preservation, blocked uncertainty and ablations.
5. Extend platform calibration to NOAA-20/21 with independent validation.
6. Extend geography/full-year coverage and test pre-2012 feature distribution shift.

Stop before app integration. Review results and the revised product metric first.
