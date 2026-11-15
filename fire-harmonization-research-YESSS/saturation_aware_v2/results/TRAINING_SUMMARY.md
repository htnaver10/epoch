# Completed: SaturationAwareBackgroundV2

**The corrected model is trained and verified. The major saturation-handling defect is repaired: V2 scores all 95 positive reference query locations, compared with only 20 for V1.** This establishes a substantial improvement over the flawed V1 run, but the frozen operating policy still trades more false alarms for higher recall than standard MODIS.

| Method | Precision | Recall | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| VIIRS reference (self-comparison) | 100.00% | 100.00% | 95 | 0 | 0 | 1.0000 |
| Standard MODIS | 44.97% | 80.00% | 76 | 93 | 19 | 0.5758 |
| Original V1, coarse | 12.90% | 8.42% | 8 | 54 | 87 | 0.1019 |
| Original V1, two-stage | 12.00% | 6.32% | 6 | 44 | 89 | 0.0828 |
| **SaturationAwareBackgroundV2, coarse** | **37.90%** | **87.37%** | **83** | **136** | **12** | **0.5287** |
| SaturationAwareBackgroundV2, two-stage | 37.67% | 57.89% | 55 | 91 | 40 | 0.4564 |

All query comparisons use the same 585,464 known reference locations. VIIRS supplies the labels, so its perfect row is a self-comparison, not a measurement of real-world VIIRS accuracy. This benchmark has already informed the diagnosis and repair; these results are development evidence, not a new blind test.

Relative to standard MODIS, V2's coarse detector finds seven more positive query labels in total but produces 43 more false positives. Its recall increases by 7.37 percentage points, while precision and F1 decrease. It is a higher-sensitivity experimental detector, not an established overall improvement over MODIS. Average precision increases from V1's 0.0169 to 0.5317, but the original ranking was confounded by excluded fire locations, and several corrective changes occurred together.

At the native-cell scale, standard MODIS detects 29 of 39 reference-positive cells with 9 false positives (precision 76.32%, recall 74.36%, F1 0.7532). V2 detects 32 of 39 with 20 false positives (precision 61.54%, recall 82.05%, F1 0.7033). These approximate common-support metrics answer a different question from fine-grid localization.

## Corrections implemented

- Separate prediction heads for bands 21 and 22. Missing measurements are masked out of regression loss and target-normalization statistics. Scoring uses the normalized band-22 residual when available and falls back to band 21 otherwise. At the 95 positive query locations, 68 use band 21 and 27 use band 22. Neither channel is numerically fabricated from a saturation/fill code.
- Prediction uses complete radius-10 background context, while fitting remains restricted to original disjoint halo-16 interiors. This retains cells covering the inherited continuous-query boundary. Both positive and negative benchmark coverage are now 100%.
- Whole-swath unlabeled holdout produces 8 training swaths, 5 development swaths and 3 unchanged test swaths. Four of the original training swaths moved into development by a deterministic region/day-night rule. Nighttime calibration now has 841,856 native samples, avoiding the V1 daytime fallback.
- The same 0.1% coarse and 1% localization alert-volume choices remain fixed. They are not false-positive guarantees and were not fitted to VIIRS agreement. Source partitions, normalization, checkpoint selection and policies use only MODIS measurements.

## Training and localization

The coarse network has 6,306 parameters and trained on 1,696,366 native samples. It completed the configured 40 epochs, selecting epoch 40 by unlabeled development reconstruction loss. The loss was still improving near the cap; this run does not establish that a longer fit would improve fire discrimination.

The finer network has 4,993 parameters and trained on 1,941,171 daytime 500 m samples. It completed 21 epochs and selected epoch 14 by the same type of unlabeled objective. Both models were trained from scratch; V1 weights were not reused.

The localization stage removes 45 false-positive queries but also 28 true-positive queries from V2's coarse alerts. It has supported evidence for 69 of 118 native alerts across complete test tile outputs. The rule retains the top two child scores including ties when all four children have valid strong daytime shortwave evidence; otherwise it leaves the parent alert unresolved. It preserves coarse alert decisions, but its fine-grid mask performs worse on this benchmark. Treat localization as experimental rather than a recommended filtering step. Scores, child weights and their entropy are not calibrated fire probabilities or validated uncertainty intervals.

## Verification and preserved artifacts

All **15 regression tests** passed. End-to-end verification checked 3,303 shared copied files, V2 feature hashes, V1 artifact checksums, disjoint whole-swath assignments, observed target masks, actual nighttime development data, positive/negative coverage, independent confusion counts, strict checkpoint exports and exact input-only inference agreement. No satellite data were downloaded.

A post-fit import-path issue affected which reporting/verification modules the orchestrator selected. The identical evaluation metric helper is now loaded by explicit file location without altering Python's module search path. A regression test checks the correct module resolution. The original source lock, policy and results are archived under `results/pre_import_repair`; `results/import_repair.json` records the correction. Training code, models, feature matrices, thresholds and predictions were unchanged. Only source/policy provenance checksums were updated; the genuine V2 report and verification then completed successfully.

The original V1 package, configuration, feature cache, checkpoints, policy and evaluated scores remain in the parent directory. V2 source, configuration, artifacts and run launcher are separate within the same requested folder.

See [full evaluation and figure](EVALUATION.md), [verification](completion_verification.json) and [run instructions](../README.md).
