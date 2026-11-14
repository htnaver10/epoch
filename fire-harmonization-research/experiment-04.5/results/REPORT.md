# Experiment 04.5: frozen model evaluated outside its training years

**The saved 2012–2025 model was evaluated on 2026 data without retraining, threshold selection or calibration.** This is an out-of-time partial-archive diagnostic, with substantial coverage limitations.

## What was unseen

Every scoring date is in 2026. The frozen 14-year model trained on 2012–2025; the comparator models end in 2022 and 2019. No 2026 labels fitted their trees or selected their checkpoints. Raw archive coverage had been inspected, so this is not a prospective blinded trial. All three model hashes stayed unchanged.

## Coverage policies, frozen before prediction

- **Primary: 51 dates** between June 3 and July 31 with at least one archived detection from each sensor anywhere in the rectangle. This excludes globally empty sensor dates but conditions the population on recorded activity. It is not a representative all-days evaluation.
- **Sensitivity: 59 dates** in the inferred June–July download window after removing the two existing gap flags. It includes eight additional dates with one globally empty sensor archive and treats absent rows as recorded zeros.
- The original per-request manifest is missing, so contiguous coverage of the inferred window cannot be proven. Both scenarios lack verified per-cell clear-sky coverage. August was not scored.
- The masks affect scoring only. Features were computed once from current/past MODIS and its coverage flags, without VIIRS-derived input selection. The original frozen 31-cell static mask was retained.

## Whole-rectangle results

Units: area-weighted occupied cell-days per calendar bin. Nine partial bins in each scenario. These MAEs should not be compared directly with Experiment 04's longer three-summer benchmark.

| Model | 51-day MAE | 51-day WAPE | 51-day bias | 59-day MAE | 59-day WAPE | 59-day bias |
|---|---:|---:|---:|---:|---:|---:|
| raw_modis | 167.54 | 22.29% | -22.29% | 188.51 | 25.04% | -9.91% |
| constant_ratio_14year | 110.55 | 14.70% | +14.35% | 248.53 | 33.01% | +32.57% |
| power_14year_rate | 71.36 | 9.49% | +9.45% | 200.88 | 26.68% | +26.68% |
| boosted_8year | 65.90 | 8.77% | +2.35% | 175.92 | 23.37% | +17.20% |
| boosted_11year | 59.75 | 7.95% | +0.97% | 166.18 | 22.07% | +15.61% |
| boosted_14year | 60.70 | 8.07% | +1.21% | 166.50 | 22.12% | +15.80% |

The frozen final model lowers MAE by **63.8%** against raw MODIS on the 51-day mask, and by **11.7%** on the broader 59-day assumption. The eleven-year model is slightly better than the fourteen-year model on both masks; this run does not select or replace any model.

The ratio and power-law comparators are fitted only on 2012–2025 observations. The power-law fit uses weekly mean daily activity and scales back to each scored bin length, so it is a different comparator from Experiment 04's weekly-total power law.

## Daily recorded-occupancy metrics

| Mask | Model | Precision | Recall | F1 | Brier | ROC AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|
| paired_active_dates | raw_modis | 0.824 | 0.640 | 0.721 | 0.002591 | 0.820 | 0.529 |
| paired_active_dates | boosted_14year | 0.841 | 0.663 | 0.741 | 0.002044 | 0.953 | 0.762 |
| metadata_window | raw_modis | 0.710 | 0.639 | 0.673 | 0.002809 | 0.819 | 0.455 |
| metadata_window | boosted_14year | 0.730 | 0.662 | 0.694 | 0.002246 | 0.952 | 0.634 |

These metrics use the full eligible coarse grid on included dates, area weighting, and threshold 0.5. They measure recorded VIIRS occupancy; they are not true-fire, forest-only or future-ignition accuracy.

## Interpretation

The model was genuinely evaluated outside its training years. The useful agreement on the primary mask is encouraging, but the large sensitivity to ambiguous dates means this is not a definitive clean holdout validation. A complete verified 2026 overlap or independently acquired regional/year archive is still required. Excluding globally empty archives also excludes possible genuine quiet days and cannot be treated as an unbiased fix.

No native-image data, new geographic domain, precise fire locations, complete summer, or calibrated ground-truth fire probability is demonstrated. No model or main-app integration was changed.

## Figures

- [Coverage sensitivity](../visualizations/01_holdout_error_and_coverage.png).
- [Daily before/after/reference](../visualizations/02_daily_before_after_2026.png).
- [Spatial before/after/reference](../visualizations/03_spatial_before_after_2026.png).

## Inspectable evidence

- [Frozen protocol](../PROTOCOL.md), [model hashes and training years](frozen_models.json), [date audit](date_audit.csv).
- [Metrics](metrics.csv), [weekly predictions](weekly_predictions.csv), [daily predictions](daily_predictions.csv), [probability metrics](binary_metrics.csv), [support diagnostics](support_diagnostics.csv).
- [Validation](validation.json), [isolation](isolation.json), [input hashes](input_hashes.json).

Validation passed: 432 weekly rows, 48 metric groups, 3 unchanged models and 183 unchanged main-app files.
