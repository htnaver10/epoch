# Expanded real-data round: substantial_round_v3

The operating threshold and intercept adjustment were fitted on development validation, then frozen before test predictions.
This is an all-land VIIRS-reference experiment. It is not a forest-only or operational wildfire model.

## Validation

| Method | Precision | Recall | False positives | False negatives | Brier | Score-sum activity |
|---|---:|---:|---:|---:|---:|---:|
| MODIS | 35.584% | 63.290% | 2,013 | 645 | 0.001402 | 3,125.0 |
| Raw model at 0.5 | 45.865% | 3.472% | 72 | 1,696 | 0.000839 | 638.7 |
| Validation-adjusted model, frozen operating point | 27.063% | 41.434% | 1,962 | 1,029 | 0.000822 | 1,757.0 |
| Always-clear reference | 0.000% | 0.000% | 0 | 1,757 | 0.000927 | 0.0 |

## Test

| Method | Precision | Recall | False positives | False negatives | Brier | Score-sum activity |
|---|---:|---:|---:|---:|---:|---:|
| MODIS | 44.970% | 80.000% | 93 | 19 | 0.000191 | 169.0 |
| Raw model at 0.5 | 53.846% | 58.947% | 48 | 39 | 0.000126 | 282.2 |
| Validation-adjusted model, frozen operating point | 11.126% | 87.368% | 663 | 12 | 0.000236 | 715.7 |
| Always-clear reference | 0.000% | 0.000% | 0 | 95 | 0.000162 | 0.0 |

## Interpretation

Matching validation activity totals is a consequence of the fitted intercept, not independent calibration evidence.
The threshold permits at most the MODIS validation false-positive exposure; test false alarms can reveal a failure of that policy to transfer.
An always-clear baseline can have low Brier loss in rare-fire data while detecting no fires.
Compare models on the same new test exposures. Pooled counts are not directly comparable with the earlier one-scene pilot.
Read evaluation.json for exact tied-score ranking metrics, activity bias, checkpoint hashes and coverage limitations.

No automatic usefulness or deployment approval is made. More independent events and uncertainty analysis are needed before a broad claim.
