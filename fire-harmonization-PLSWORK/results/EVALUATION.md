# Contextual MLP and tree: reused VIIRS-reference benchmark

**This benchmark and development cohort were examined in earlier experiments; it is not a new blind test.**

Selected using validation average precision: **mlp**.
Features use MODIS radiances, annular background contrast, reflective measurements and geometry. VIIRS labels and MODIS detections are not feature inputs.

## Validation

| Method | Precision | Recall | TP | FP | FN | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| regular_modis | 35.58% | 63.29% | 1,112 | 2,013 | 645 | 0.4556 | 0.2256 |
| previous_cnn_raw_0.5 | 45.86% | 3.47% | 61 | 72 | 1,696 | 0.0646 | 0.2194 |
| mlp_raw_0.5 | 55.42% | 20.09% | 353 | 284 | 1,404 | 0.2949 | 0.4242 |
| mlp_frozen_policy | 36.94% | 66.82% | 1,174 | 2,004 | 583 | 0.4758 | 0.4242 |
| tree_raw_0.5 | 45.45% | 44.68% | 785 | 942 | 972 | 0.4506 | 0.3740 |
| tree_frozen_policy | 35.67% | 63.52% | 1,116 | 2,013 | 641 | 0.4568 | 0.3740 |

## Test

| Method | Precision | Recall | TP | FP | FN | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| regular_modis | 44.97% | 80.00% | 76 | 93 | 19 | 0.5758 | 0.3598 |
| previous_cnn_raw_0.5 | 53.85% | 58.95% | 56 | 48 | 39 | 0.5628 | 0.5003 |
| mlp_raw_0.5 | 68.75% | 11.58% | 11 | 5 | 84 | 0.1982 | 0.5556 |
| mlp_frozen_policy | 49.30% | 73.68% | 70 | 72 | 25 | 0.5907 | 0.5556 |
| tree_raw_0.5 | 53.12% | 53.68% | 51 | 45 | 44 | 0.5340 | 0.5586 |
| tree_frozen_policy | 39.44% | 74.74% | 71 | 109 | 24 | 0.5164 | 0.5586 |

## Interpretation limits

The operating policies and model selection were fitted on validation before these benchmark predictions. No post-test threshold tuning was performed.
Training has three regional cohorts; validation and test each have one. Query labels are correlated and do not represent independent wildfire incidents.
VIIRS provides reference observations, not verified ground-truth wildfire. Not matching a VIIRS detection does not alone establish that a MODIS alert is false.
Raw and validation-adjusted probabilities may remain miscalibrated on other regions. The target is active-fire reference detection, not forecasting or burned area.
A finer query grid does not create new thermal spatial resolution or guarantee detection of fires absent from MODIS measurements.
All imagery was physically copied from the previous research folder. No new satellite data were downloaded.
