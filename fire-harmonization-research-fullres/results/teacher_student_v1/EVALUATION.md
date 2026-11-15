# VIIRS teacher to MODIS student: completed experiment

**The 2024 test benchmark was examined in the previous experiment. This is a reused benchmark comparison, not a new blind test.**

The teacher receives calibrated VIIRS I04/I05 thermal imagery and validity masks. The student receives MODIS inputs only.
Teacher soft targets are restricted to training tiles. Both checkpoints are selected using development validation.
New student calibration and its operating threshold are frozen on validation before test prediction.

Teacher epochs completed: 30; student epochs completed: 30.
Selected teacher epoch: 26; selected student epoch: 23.

## Validation

| Method | Precision | Recall | TP | FP | FN | Brier | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regular MODIS | 35.58% | 63.29% | 1,112 | 2,013 | 645 | 0.00140167 | 0.2256 |
| Previous MODIS ML, raw 0.5 | 45.86% | 3.47% | 61 | 72 | 1,696 | 0.00083884 | 0.2194 |
| Distilled MODIS student, raw 0.5 | 46.01% | 7.23% | 127 | 149 | 1,630 | 0.00083429 | 0.2672 |
| Distilled student, frozen validation policy | 29.73% | 48.38% | 850 | 2,009 | 907 | 0.00081713 | 0.2672 |

## Test

| Method | Precision | Recall | TP | FP | FN | Brier | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regular MODIS | 44.97% | 80.00% | 76 | 93 | 19 | 0.00019130 | 0.3598 |
| Previous MODIS ML, raw 0.5 | 53.85% | 58.95% | 56 | 48 | 39 | 0.00012577 | 0.5002 |
| Distilled MODIS student, raw 0.5 | 50.00% | 53.68% | 51 | 51 | 44 | 0.00013003 | 0.4543 |
| Distilled student, frozen validation policy | 11.10% | 83.16% | 79 | 633 | 16 | 0.00015906 | 0.4543 |

## Interpretation limits

A student gain must be assessed against both MODIS and the previous supervised model at the stated operating policies.
Changing a threshold after seeing this benchmark would be exploratory. No post-test retuning was performed.
The validation cohort remains limited. Positive query labels within scenes are correlated and are not independent wildfire incidents.
VIIRS provides reference labels, not independently verified ground truth. This experiment does not establish operational usefulness.
The exported student uses the ordinary NativeFireNet loader; teacher models, radiances and soft-target files are unnecessary for inference.

[Frozen operating policy](operating_policy.json) | [Teacher validation quality](teacher_quality.json) | [Full results](evaluation.json)
