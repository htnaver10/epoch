# Larger model: actual training and evaluation

Trained and saved a **200-tree, maximum-depth-5 NumPy gradient-boosted model** on all 14 local overlapping summers (2012–2025). The final ensemble has **5,319 leaves** and uses ten MODIS-only features. A separately fitted 11-year model supplies the retrospective temporal benchmark below; the final 14-year fit has no new complete holdout.

| Fit | Years | Full eligible cell-days processed | Supported weighted cell-days | Distinct supported patterns |
|---|---:|---:|---:|---:|
| Selection fit | 8 | 18,818,784 | 1,533,702 | 57,163 |
| Temporal benchmark fit | 11 | 25,875,828 | 2,028,997 | 68,293 |
| Final research fit | 14 | 32,728,320 | 2,979,022 | 122,498 |

Every eligible cell-day is accounted for, without background sampling. Identical binned input patterns are collapsed into exact weighted sufficient statistics. The full-grid total includes unsupported cells with forced-zero output; supported exposure is the portion used to learn the conditional response. Area-weighted exposures and distinct patterns are different units. The earlier kernel also used the full grid; the expansion relative to it is training years, features and model capacity.

## Retrospective temporal benchmark: 2023–2025

Train through 2022; tree count selected using only 2020–2022 after an initial 2012–2019 fit. The dates, masks and references match within each comparison. These years have been inspected in earlier research and are not a new blind test.

| Model | Weekly MAE | Weekly RMSE | Weekly WAPE | Aggregate bias | Worst summer error |
|---|---:|---:|---:|---:|---:|
| uncorrected_modis | 348.77 | 427.04 | 27.09% | -26.75% | 28.92% |
| numpy_daily_occupancy | 118.37 | 161.36 | 9.19% | -1.14% | 7.67% |
| kernel_occupancy | 119.32 | 171.16 | 9.27% | +4.58% | 17.60% |
| boosted_8year | 94.82 | 129.30 | 7.36% | -0.74% | 4.47% |
| power_law | 102.70 | 145.71 | 7.98% | +1.20% | 10.48% |
| boosted_11year | 100.83 | 135.30 | 7.83% | -2.09% | 4.73% |

The larger benchmark fit has **71.1% lower weekly MAE than raw MODIS** and **-1.8% MAE change relative to matched-history power-law calibration**. It improves 38/42 bins over raw MODIS, 21/42 over power law, and 14/42 over the eight-year boosted fit. The old logistic and kernel models use eight-year fits; the displayed power-law baseline uses the same eleven-year history as the larger benchmark.

**More years did not improve national weekly MAE in this comparison:** the same boosted architecture fitted on eight years has MAE 94.82, compared with 100.83 after refitting eleven years (+6.3%). Higher capacity improved on the old regression/kernel; adding the 2020–2022 years did not produce a monotonic improvement. Do not interpret the fourteen-year refit as the best-performing model without a new holdout.

### Peak timing

Peak means the calendar bin with the highest mean activity per matched day, including the one-day final bin. The eleven-year model matches **2/3** reference peak bins; details are in `peak_timing.csv`.

| Year | VIIRS peak-bin start | Larger-model peak-bin start | Timing error |
|---|---|---|---:|
| 2023 | 2023-08-03 | 2023-08-03 | +0 days |
| 2024 | 2024-08-10 | 2024-08-10 | +0 days |
| 2025 | 2025-07-27 | 2025-08-03 | +7 days |

## Daily recorded-occupancy metrics

Area-weighted, full eligible grid, 2023–2025. These scores compare recorded VIIRS occupancy; they do not measure true-fire detection or predict future ignition.

| Metric | Raw MODIS | Larger benchmark |
|---|---:|---:|
| brier | 0.00361 | 0.00279 |
| logloss | 0.09961 | 0.02105 |
| roc_auc | 0.81916 | 0.96611 |
| average_precision | 0.56038 | 0.79821 |
| precision_at_0_5 | 0.87240 | 0.86814 |
| recall_at_0_5 | 0.63906 | 0.68067 |
| f1_at_0_5 | 0.73772 | 0.76306 |

## Geographic transfer

The east (longitude >= -90) was excluded from fitting and checkpoint selection, with training longitude < -92. Only training-region labels identify persistent sources. Eastern persistent sources remain in the reference because no eastern labels can be used to identify them.

| Model | East weekly MAE | WAPE | Bias |
|---|---:|---:|---:|
| boosted_cold_east | 14.53 | 11.91% | +2.34% |
| constant_ratio | 28.52 | 23.38% | +20.81% |
| power_law | 27.38 | 22.44% | +21.09% |
| uncorrected_modis | 32.99 | 27.05% | -23.01% |

## Limits and saved deliverables

- The final fit processes **32,728,320** eligible cell-days and covers **75% more training years** than the original eight-year fit. No new NASA files were downloaded.
- Its MODIS-context support excludes **10.6%** of recorded VIIRS activity during training; isolated detections cannot be recovered under this model.
- The original archive has aggregated counts only: no native radiances, FRP, confidence, exact overpass times or clear-sky coverage. More model capacity cannot restore those fields or create native 375m resolution.
- Geographic transfer is limited to parts of the same boreal rectangle. Global and year-round generalization remain untested. No main-app integration or promotion was performed.
- `models/boosted_14year_final.json` is the final larger research model; `models/boosted_11year_benchmark.json` is the evaluated temporal model. `results/weekly_predictions.csv`, `binary_metrics.json`, `training_ledgers.json` and `selection_curve.csv` retain evidence.

## Figures

- [Matched-summer totals for the eleven-year benchmark fit.](../visualizations/01_summer_before_after.png) · [SVG](../visualizations/01_summer_before_after.svg)
- [Weekly activity includes eight-year and eleven-year boosted models on the same dates.](../visualizations/02_weekly_before_after.png) · [SVG](../visualizations/02_weekly_before_after.svg)
- [Before, after and reference calendars on a shared linear scale.](../visualizations/03_calendar_before_after.png) · [SVG](../visualizations/03_calendar_before_after.svg)
- [Shared-scale coarse maps reconcile exactly to the saved seasonal predictions.](../visualizations/04_spatial_before_after_2023.png) · [SVG](../visualizations/04_spatial_before_after_2023.svg)
- [Regional temporal errors and a separately trained eastern transfer test.](../visualizations/05_regional_errors.png) · [SVG](../visualizations/05_regional_errors.svg)
- [Validation checkpoint selection and supported-cell training loss.](../visualizations/06_training_selection.png) · [SVG](../visualizations/06_training_selection.svg)
