# Completed hybrid experiment

The hybrid pipeline completed training, validation-only operating-point selection, evaluation, scientific plots, and input-only inference verification. It used the existing copied dataset and downloaded no new satellite data. Previous model weights, training code and results remain intact.

The model combines a frozen 8,321-parameter contextual MLP with newly trained logistic residual trees. New preprocessing aligns valid 250/500 m reflectance measurements to the native MODIS thermal-cell scale and retains local contrasts. The residual model has 118 inputs, including the MODIS detection flag, and used all 10,371,908 known training queries with 1,913 positive VIIRS reference labels. Early stopping selected round 9 after 49 rounds. Training the new residual stage took 14.7 seconds, excluding feature preparation, prior inference and verification. The frozen MLP was reused, not retrained.

| Method | Precision | Recall | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| VIIRS reference, self-comparison | 100.00% | 100.00% | 95 | 0 | 0 | 1.0000 |
| Standard MODIS | 44.97% | 80.00% | 76 | 93 | 19 | 0.5758 |
| Previous MLP, frozen policy | 49.30% | 73.68% | 70 | 72 | 25 | 0.5907 |
| Hybrid, frozen policy | 49.29% | 72.63% | 69 | 71 | 26 | 0.5872 |

VIIRS defines the labels, so its perfect row is tautological; it does not measure real-world VIIRS accuracy. All methods use the same 585,464 eligible query centers. Thresholds were selected on validation, not adjusted after this benchmark.

**The hybrid does not improve overall on the previous MLP.** It removes one false positive but loses one true positive. Compared with standard MODIS, it removes 22 false positives but loses seven net true positives. It recovers five positives missed by MODIS while dropping twelve that MODIS detected. These tradeoffs do not establish an overall advance.

The support-alignment audit found 526 of 596 positive training native cells contained a mixture of positive and clear VIIRS centers. The secondary native-cell benchmark has 39 positive cells. Standard MODIS detects 29 with 9 false positives (76.32% precision, 74.36% recall, F1 0.7532); the hybrid detects 25 with 7 false positives (78.12% precision, 64.10% recall, F1 0.7042). Standard MODIS has the higher F1 at that scale. These cell metrics answer a different question from exact query-center localization and cannot be substituted into the table above.

Nine focused tests passed. Final checks confirmed unchanged parent sources and benchmark exposure, matching feature/model hashes, no split source overlap, validation false-positive budgets respected, correct additive model reload, and deployment predictions matching the evaluated model without VIIRS values or labels. A JSONL parsing error in the final verifier was corrected after fitting; the original fitted source snapshot and an explicit repair receipt are preserved. Model weights, preprocessing, thresholds and evaluated predictions were unchanged by that tooling repair.

The conclusion is to retain the previous model as the development comparator, not promote this hybrid. Further architectural choices need more independently reserved events; repeatedly optimizing this already examined cohort will not establish generalization. Alignment-aware training targets may be worth a separate predeclared experiment, but changing reference labels merely to increase agreement would invalidate the current comparison.

See [full methods, research references and both evaluation scales](EVALUATION.md), [verification](completion_verification.json), and [run instructions](../README.md).
