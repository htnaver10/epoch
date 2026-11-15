# Saturation handling invalidates a broad method-level conclusion

**The observed low scores are real, but this run is not an adequate test of whether unsupervised fire detection can work. A major input-eligibility mistake prevents the model from scoring most positive reference locations.**

The audit traces the saved test inputs and predictions back to original MODIS Level-1B integer measurements. It changes no models, thresholds or predictions. Its reproducible calculation is `tools/audit_eligibility.py`, with counts and raw-code samples in `implementation_audit.json`.

| Quantity | Result |
|---|---:|
| Positive VIIRS reference query locations | 95 |
| Positive locations the model could score | 20 (21.05%) |
| Positive locations excluded before scoring | 75 (78.95%) |
| Positive locations with band 20 saturated | 72 |
| Positive locations with band 22 saturated | 68 |
| Positive locations with either band 20 or 22 saturated | 74 |
| Positive locations outside the new native-cell interior mask | 5 (4 also saturated) |
| Positive locations with valid band 21 | 95 |
| Positive locations with valid bands 31 and 32 | 95 |

The masks overlap, so the per-requirement exclusions must not be summed. Among the 75 excluded positives, 74 have a saturation failure and one has only the native-cell boundary-mask failure. All raw invalid values inspected for bands 20 and 22 are integer **65533**. The NASA [Level-1B file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/5/MOD021KM.fs) identifies that code as detector saturation. The [2012 MODIS Level-1B data dictionary](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/M1055_PDD_D_072712final.pdf) provides the same definition.

The reader correctly excluded these codes from numeric radiance calibration: saturation flags must not be converted into ordinary radiances. The mistake is downstream. The detector requires valid unsaturated measurements in **both bands 20 and 22** and has no band-21 alternative. Consequently, many actual hot locations become automatically undetected. NASA's [MODIS calibration ATBD](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/MODIS_L1B_ATBD_ver8_finalupload.pdf) describes band 21 as a low-gain band primarily used for fire detection, with a specified maximum scene temperature of 500 K.

The original detector's theoretical recall ceiling on this benchmark is **20/95 = 21.05%**, even if every scoreable positive were detected. It detects 8 of those 20. Of its 87 false negatives, 75 are excluded before scoring and 12 are eligible but below threshold. Eligibility explains most missed positives in this run, but does not explain all its false positives or establish that reconstruction residuals are a good fire score.

Overall query coverage of 98.57% was misleading for judging fire coverage because negatives dominate the dataset. Negative coverage is 98.58%, whereas positive coverage is only 21.05%. Previous software tests checked arithmetic, leakage prevention, output consistency and label separation; they did not check class-specific sensor eligibility. Passing them did not establish domain correctness.

A counterfactual eligibility check using measured band 21 with bands 31/32 and the same interior/background requirements admits **90 of 95** positive queries. This is only evidence that a usable measured channel exists. It is **not a prediction that a corrected model will detect 90 positives**, nor a claim about performance on new events. Five query locations still fail the current interior mask, which differs from the inherited continuous-query halo rule.

Other design choices also remain unvalidated:

- The 0.1% unlabeled alert budget controls alert volume and is unrelated to a desired fire recall.
- Checkpoint selection minimizes background reconstruction error; better reconstruction need not improve anomaly discrimination.
- The coarse score requires positive anomalies in both target bands. This restrictive aggregation needs a physically justified alternative when measurements saturate or are missing.
- The fine stage retains two of four supported child samples. This heuristic removed 10 false positives but also two true positives; it is not evidence of improved localization.
- There are zero nighttime development samples. The pooled daytime fallback does not establish nighttime calibration.

Before comparing methods, a corrected experiment should retain explicit sensor-quality reasons, use the fire-appropriate available channel or a documented band-22/band-21 fallback, reconcile halo eligibility, and report positive/negative coverage separately. Saturation alone must not be declared fire, and invalid numeric measurements must not be filled with invented radiances. New checkpoints and policies should be trained and frozen in a separate version. The existing benchmark has now informed this diagnosis, so it remains development evidence; independent events are needed for a generalization claim.

**Conclusion:** the strongest demonstrated problem is the implementation's sensor handling. The usefulness of this unsupervised design remains unresolved until that problem is corrected and the experiment is rerun. The original scores should remain visible as the behavior of the flawed implementation.
