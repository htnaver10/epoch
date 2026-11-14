# Preliminary findings: MODIS -> VIIRS-reference activity

**The daily ML baseline improves on uncorrected MODIS, but does not pass the promotion criteria.** The power-law baseline has lower held-out weekly error, and the daily model fails the cold geographic transfer test. No model was integrated with the app.

## What actually ran

- NumPy weighted logistic regression for daily VIIRS occupancy, using MODIS-only predictors.
- NumPy weighted Poisson regression for weekly occupied days.
- Uncorrected MODIS, constant-ratio and power-law baselines in the revised MODIS -> VIIRS direction.
- A separate east-region transfer test, excluding east-region training/validation labels and a two-degree buffer.
- 44 metric groups independently recalculated from 1,848 saved prediction rows.
- Integrity checks of all 25 source JSON files and the existing app files.

The planned **LightGBM binary/Tweedie experiments did not run**. Installing dependencies failed because PyPI DNS resolution was unavailable. The optional runner records this explicitly; these results are logistic/Poisson baselines, not boosted trees.

## Population and evaluation

Training: 2012-2019. Validation: 2020-2022. Temporal test: 2023-2025. Parameters were fixed before the temporal test; test data was not used for tuning. The pilot covers June-August in the 50-70N, 140-60W rectangle, not all Canada or the full year.

Targets are **recorded Suomi-NPP VIIRS occupancy on the common 0.25-degree grid**. Metrics use cos(latitude)/cos(60N)-weighted occupied cell-days. They measure agreement with a satellite reference, not true wildfire detection accuracy or reconstructed 375-m locations.

The training-only static-source rule excluded 31 cells. Existing heuristic gap dates were omitted from both instruments. The test seasons therefore contain **91, 88 and 89 matched days**, respectively; these are screened-period totals, not complete summer estimates. The local 2026 overlap is partial and was excluded.

Training retains all active examples plus 1% of both-zero background, with inverse inclusion weights. Evaluation uses the entire eligible grid. Recorded zeros remain coverage-uncertain; no clear-sky denominator is available.

## Temporal holdout: entire rectangle

Weekly MAE is the mean absolute error across 42 calendar bins (including the actual short final bin) over 2023-2025. Lower is better. Seasonal errors sum the same matched days. Country and sector results are reported separately to avoid double counting.

| Method | Weekly MAE | Aggregate bias | Median absolute seasonal error | Worst absolute seasonal error |
|---|---:|---:|---:|---:|
| Uncorrected MODIS | 348.8 | -26.7% | 27.0% | 28.9% |
| Constant ratio | 210.5 | +15.8% | 15.4% | 23.6% |
| Power law | 109.4 | +4.8% | 6.4% | 15.1% |
| Daily logistic ML | 118.4 | -1.1% | 6.2% | 7.7% |
| Weekly Poisson ML | 454.3 | -34.3% | 35.6% | 39.7% |

Daily logistic ML reduces weekly MAE by **66.1% versus uncorrected MODIS**, but its weekly MAE is **8.2% higher than the power law**. It has a smaller worst matched-season error than the power law. These mixed outcomes do not establish that ML is the best harmonizer.

## Geographic stress test

The daily model trained west of 92W and was tested east of 90W in 2023-2025. The geographic baselines also use only western/central training labels. This is deliberately harder than interpolation inside the training geography.

| Method | East weekly MAE | Aggregate east bias |
|---|---:|---:|
| Uncorrected MODIS | 33.0 | -23.0% |
| Constant ratio | 31.2 | +23.3% |
| Power law | 35.0 | +28.1% |
| Daily logistic ML, east held out | 201.6 | +165.3% |

The daily model substantially overpredicts in the unseen east. Geographic/seasonal terms and the dense background may contribute, but this experiment does not establish a cause. This is evidence against applying this fitted model outside its validated support. The simpler power-law transfer is also imperfect; a flexible model is not automatically an improvement.

## Access, checks and limitations

- Unit tests: see [unit_tests.log](unit_tests.log). Independent result validation: **passed**.
- Main-app isolation: **passed**, 183 pre-existing files unchanged; no added files outside research.
- Authenticated NASA probe: **not_run**. No FIRMS_MAP_KEY configured; no authenticated request was attempted.
- LightGBM: **blocked**; no LightGBM performance claim is supported.
- No original acquisition times, FRP, brightness, confidence, footprint geometry, clear-sky masks or NOAA-20/21 training data were used.
- No pre-2012 reconstruction, full-year expansion, peak-week acceptance test, aggregate uncertainty interval, statistical significance test, or operational validation was completed.
- No synthetic data is used as evidence of satellite performance. Synthetic unit-test fixtures only check code behavior.
- Only three temporal test seasons are available here; cell-days are spatially and temporally correlated.
- Raw recorded occupancy and simple aggregate translations are different representations. All comparisons here use the same common-grid activity target and matched dates.

## Decision and next step

**Do not promote this preliminary model.** Keep the research separate. Run the planned LightGBM experiment when dependencies are available; then test richer MODIS fields and verified coverage, retaining both simple baselines and untouched new holdouts. Investigate geographic support before extrapolating. New model changes following inspection of these results make 2023-2025 a development benchmark; future claims need additional untouched holdouts or nested blocked evaluation.

## Reproducible evidence

- [Input audit](archive_audit.json), [input hashes](input_hashes.json), [fixed configuration](numpy_config.json).
- [All metrics](numpy_metrics.csv), [weekly predictions](numpy_weekly_predictions.csv), [seasonal predictions](numpy_seasonal_predictions.csv).
- [Model diagnostics](numpy_diagnostics.json), [run log](experiment.log), [independent validation](result_validation.json).
- [NASA check](nasa_access_check.json), [LightGBM status](lightgbm_status.json), [isolation check](isolation_check.json).
