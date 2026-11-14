# Experiment 04.5 — frozen models on partial 2026 data

Written before generating 2026 predictions or scores. The objective is an
out-of-time evaluation of the saved Experiment 04 models, especially the final
2012–2025 fit, on dates excluded from training and checkpoint selection.
All new files stay in experiment-04.5. No prior model, source, result, app or
documentation file is modified; no boosted model is retrained or selected here.

## Available holdout and limitations

Only partial 2026 MODIS/VIIRS overlap is available locally. Metadata gives 61
VIIRS downloaded days, but no original request manifest proving exactly which
days. Positive VIIRS records occur June 3–July 31, with none July 11–16; MODIS
has no records June 11–12. Existing VIIRS gap flags are day indices 40 and 45.
Treat August as unavailable for scoring. The June 1–July 31 interval is an
inferred candidate window, not established complete observation coverage.

This is an out-of-training-year **partial-archive diagnostic**, not a clean
full-summer/prospective/global validation. Raw 2026 coverage was inspected before
scoring. No clear-sky exposure or native-resolution imagery is available.

## Predeclared scoring scenarios

1. Primary `paired_active_dates`: candidate dates with at least one archived
   detection from each sensor anywhere in the rectangle, excluding existing gap
   flags. This avoids silently treating globally empty sensor archives as verified
   negatives. It conditions evaluation on globally active days, which changes the
   population and can bias results; it does not verify per-cell clear-sky coverage.
2. Sensitivity `metadata_window`: every June–July candidate date except the
   existing sensor gap flags. This treats other absent rows as recorded zeros
   under the inferred contiguous-window assumption. It is deliberately less
   conservative about ambiguous dates. Report it regardless of whether it helps
   or hurts the model. Do not choose between scenarios based on their scores.

All grids retain the frozen 31-cell Experiment 04 static-source mask and area
weights. Construct MODIS features once from current/prior MODIS only, using the
MODIS coverage flags (not VIIRS activity or scenario masks). Scoring masks never
change model inputs. Same fixed bins, feature function, probabilities, tree
count, threshold .5 and original calibration. No future data or 2026 fitted mask.

## Frozen models and comparators

- Main model: boosted_14year_final.json, trained 2012–2025, 200 trees.
- Frozen 8- and 11-year boosted fits provide a predeclared history comparison.
- Raw MODIS occupancy.
- Constant-ratio baseline fitted only on 2012–2025 matched historical observations.
- Power-law baseline fitted only on 2012–2025 **mean daily activity within weekly
  bins**, then multiplied by holdout bin length. Day normalization accommodates
  irregular partial bins and is a distinct comparator from the original weekly-total
  power law. No 2026 labels enter these baseline fits.

## Reporting and checks

For each mask and region: weekly MAE/RMSE/WAPE, aggregate bias and period totals.
For raw MODIS and frozen boosted fits: full-grid area-weighted daily Brier,
log loss, precision/recall/F1 at .5, ROC AUC and average precision, all against
recorded VIIRS occupancy. Report monthly diagnostics and outside-support labels.
No exact-fire-location or future-ignition claim follows from these metrics.

Save date audit, model/source/protocol hashes, predictions, scores, comparison
figures and a report. Independently recalculate metrics from saved predictions,
reconcile grid totals, verify test year exceeds every model's training years,
check feature invariance to future MODIS perturbations, and verify model/input
hashes and main-app isolation. Do not edit or promote a model after seeing results.
