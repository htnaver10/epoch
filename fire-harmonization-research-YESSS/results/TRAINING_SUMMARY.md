# Completed unsupervised two-stage experiment

**Audit correction:** a major sensor-handling flaw excludes 75 of the 95 positive test query locations before scoring. Band 20 and/or 22 saturation accounts for 74 of these exclusions, while valid band-21 measurements are available at all 95 positive locations. The detector's maximum possible recall was therefore 21.05%. The scores below remain correct for this flawed implementation; they are insufficient to judge the unsupervised method. See the [implementation audit](IMPLEMENTATION_AUDIT.md).

The requested experiment is implemented and has completed training, reference evaluation, figure generation and inference verification. It uses only copied local data; no satellite downloads or new package installations were needed. Previous experiments remain unchanged.

**The resulting detector performs much worse than standard MODIS on the existing benchmark. Do not promote it as an improved fire detector.** The experiment tests unsupervised background prediction; successful reconstruction training does not establish useful fire detection.

| Method | Precision | Recall | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| VIIRS reference (self-comparison) | 100.00% | 100.00% | 95 | 0 | 0 | 1.0000 |
| Standard MODIS | 44.97% | 80.00% | 76 | 93 | 19 | 0.5758 |
| Unsupervised native-cell detector | 12.90% | 8.42% | 8 | 54 | 87 | 0.1019 |
| Unsupervised two-stage detector | 12.00% | 6.32% | 6 | 44 | 89 | 0.0828 |

This table uses the same 585,464 eligible query centers and 95 positive VIIRS reference labels as earlier runs. VIIRS's perfect row is a self-comparison, not evidence of real-world sensor accuracy. Test events were examined before this experiment and are not newly independent evidence.

At the native-cell scale, MODIS detects 29 of 39 reference-positive cells with 9 false positives (F1 0.7532). The unsupervised detector detects 5 of 39 with 12 false positives (F1 0.1786). The localization stage preserves those coarse alerts, only refining their within-cell location.

## What was trained

- A 6,306-parameter native-cell network predicts two withheld MODIS thermal measurements from surrounding context and longwave/geometry inputs. It used 3,204,188 training samples and 689,675 unlabeled development samples. Training completed 31 epochs and selected epoch 24 by reconstruction loss.
- A 4,993-parameter daytime network predicts withheld 500 m shortwave measurements from surrounding and visible-band context. It used 4,605,619 training samples and 2,761,434 unlabeled development samples. Training completed 13 epochs and selected epoch 6 by reconstruction loss.
- Both models start from scratch. No prior supervised checkpoint is reused. VIIRS labels and MODIS fire flags are absent from fitting, early stopping, normalization and threshold selection.
- The native alert fraction was fixed at 0.1% of unlabeled development scores; localization requires strong shortwave evidence above the unlabeled 99th percentile. These are alert-volume policies, not measured false-positive guarantees.
- Localization is supported for 33 of the 61 native alerts produced over the complete test tiles. On eligible benchmark queries, it removes 10 false positives but also 2 true positives. Without supporting evidence or at night, the coarse location remains unresolved.

The feature cache is fitted from native MODIS inputs rather than the VIIRS reference-valid query mask. Model-valid coverage of the unchanged query benchmark is 98.57%; uncovered known reference queries remain in the denominator as undetected. Training has 2,052,882 eligible nighttime native samples, but development has none. The recorded nighttime policy uses the pooled daytime fallback, so nighttime performance is not independently calibrated.

## Verification

All 12 focused tests passed. Final verification checked 3,303 copied/separated files, all prepared feature hashes, fit provenance, independent confusion-count calculations, frozen unlabeled alert budgets, unchanged benchmark exposure, and identical input-only inference. Localization preserves native-cell alerts and its relative child weights sum to one. Those weights are not calibrated fire probabilities.

A numerical threshold-tie bug was found by final verification and corrected by using consistent float64 residual-score arithmetic for calibration and inference. The original source and preliminary results are archived under `pre_precision_fix`; `precision_repair.json` records the change and unchanged model hashes. The correction did not change test confusion counts. No VIIRS-based threshold tuning or architectural revision followed evaluation.

## Interpretation

The lower false-positive count does not represent useful improvement because recall collapses. Native anomaly-score average precision is only 0.0169 on the query benchmark, but that ranking is itself confounded by assigning excluded positive locations an unusable score. The saturation-handling defect must be corrected before this experiment can support a conclusion about the underlying unsupervised method. Within the 20 scoreable positive locations, only eight are detected, so fixing eligibility alone does not guarantee good performance.

Self-supervised representation learning followed by limited fire-specific supervision is a distinct future experiment. It has not been implemented or demonstrated by this run. The two-stage architecture remains available for controlled experiments, with both coarse detection and localization evaluated separately.

See [full evaluation](EVALUATION.md), [verification](completion_verification.json), [figure](training_and_comparison.png) and [run instructions](../README.md).
