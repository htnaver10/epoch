# Expanded real-data training round

This is a fresh experiment responding to the pilot's severe false-alarm and
calibration failures. It retains the compact CNN architecture and the frozen
pilot artifacts, collects substantially more training observations, reserves new
test scenes, trains for 30 epochs, and compares the result with MODIS.

**Completed on 2026-11-15 UTC: authenticated downloads, native preparation,
30 training epochs and reserved-test evaluation.** Read the
[training summary and charts](results/substantial_round_v3/TRAINING_SUMMARY.md).
The raw model trades higher precision for lower recall than MODIS. The
validation-adjusted policy produces excessive test false alarms.

From PowerShell, run:

```powershell
cd "C:\Users\Darrien\Desktop\epoch1\epoch\fire-harmonization-research-fullres"
.\run-substantial-round.cmd
```

The completed local run is reused when rerun; no login is needed for its cached
files. A fresh acquisition prompts locally for your NASA Earthdata login, with
the password hidden and excluded from the campaign journal. Native preparation
can take substantially longer than GPU training.

Version 2 stopped before training: its 384-pixel query canvas excluded valid image windows. Version 3 raises only this preparation limit to 768, retains the same frozen scenes, and reuses the downloaded sources. The earlier source snapshot and failed admission audit remain in results/substantial_round_v2.

## What runs automatically

1. **Frozen public inventory.** The checked inventory contains 12 candidate days
   in BC (2021), 12 in central Canada (2021), and 10 in NWT (2022). Each candidate
   satisfies the existing full-interval <=15-minute pairing bound. Metadata also
   confirms the native companion products for a sample in each region.
2. **Training-only fire screening.** Candidate VIIRS sparse arrays rank nominal/
   high-confidence pixels inside the region and MODIS metadata footprint.
   Actual C002 `FP_confidence` codes are 8/9; final labels still come from full
   masks, QA and geometry. Each region selects up to six fire-rich acquisitions
   plus two calendar controls, retaining day and night candidates when available.
   This plans **up to 24 training acquisitions across three regional cohorts**,
   not 24 verified independent wildfire incidents.
3. **Separate held-out acquisitions.** Validation uses four metadata/calendar-
   selected Ontario 2023 candidates plus the existing Quebec development scene.
   Five metadata/calendar-selected Quebec 2024 acquisitions are reserved for test.
   Held-out scenes are not chosen by their fire counts. Training precedes
   validation and test in time; shared raw sources cannot cross partitions.
4. **Bounded, journaled download.** New transfers have a planned **32 GiB** budget,
   estimated from metadata, with actual bytes checked after each file. The
   workflow keeps a **40 GiB free-space reserve**. Optional reflective companions
   may be absent at night; required thermal, geolocation and fire products remain
   mandatory. Science files, checksums and metadata stay in this research folder.
5. **Native preparation and admission checks.** Complete native input windows
   must lie inside the declared geographic block. Completed source-unit manifests
   are reused; interrupted tiles are reused only if their arrays and dtypes match
   exactly. Event/cohort groups linked by shared sources are merged conservatively.
   The round stops before training unless there are at least **500 training tiles,
   1,000 positive training reference pixels and three training regions**, plus
   eligible validation and test exposure. These are expansion guards, not proof
   of usefulness or replacements for the full project's independent-event target.
6. **Training-only preprocessing and initialization.** Normalizers and output
   priors are fitted from training data only. The final output heads start at
   training prevalence, rather than near 0.5 everywhere. This is a startup choice,
   not an established explanation of the pilot's failure. The loss remains masked,
   exposure-normalized BCE; no hidden class-cost multiplier is added. A real-data
   worst-shape profile precedes the fresh **30-epoch** CUDA/BF16 run. The best
   checkpoint is selected by development-validation log loss.
7. **Frozen operating policy and test comparison.** After checkpoint selection,
   a single logit-intercept adjustment is fitted on development validation. A
   conservative decision cutoff allows at most the MODIS validation false-positive
   exposure. Both are frozen before test predictions. The test report compares
   MODIS, the raw model, the adjusted model and an always-clear reference on the
   same eligible queries. It reports precision/recall, false alarms, exact tied-
   score ranking metrics, Brier/log loss and score-sum activity bias.

Matching calibration totals on validation is by construction and is not evidence
that probabilities are reliable elsewhere. Test false alarms can exceed the
validation budget if the policy does not transfer. Raw `run.py infer` scores are
unchanged; the new adjustment is applied by this round's evaluation tool and
recorded separately in `calibration.json`.

## Outputs and resuming

Run the same command again to resume. Completed transfers and source preparation
units are retained; neural training resumes at the last completed epoch.

```text
results/substantial_round_v3/inventory.json
results/substantial_round_v3/selection.json
results/substantial_round_v3/downloads.json
results/substantial_round_v3/software_lock.json
results/substantial_round_v3/dataset_audit.json
data/manifests/substantial_round_v3.jsonl
data/manifests/substantial_round_v3_stats.json
models/substantial_round_v3/best.pt
models/substantial_round_v3/last.pt
logs/substantial_round_v3/epochs.jsonl
results/substantial_round_v3/calibration.json
results/substantial_round_v3/EVALUATION.md
results/substantial_round_v3/metrics.csv
results/substantial_round_v3/evaluation.json
```

Individual stages are available as `inventory`, `download`, `prepare`, `train`,
or `evaluate`, for example `.\run-substantial-round.cmd inventory`. Inventory
uses public metadata and does not need authentication. Configuration, source
hashes and package versions are locked for the campaign. Meaningful changes need
a new version; changing a campaign name alone does not make a previously examined
test set blind again.

If the data targets are not met, inspect `dataset_audit.json` and expand a newly
versioned acquisition plan. The script does not lower the targets or fill missing
labels to force training. If a provider requires additional account authorization,
complete it locally and rerun; the cached transfers remain available.

## Limits and verification

This is an **expanded all-land VIIRS-reference experiment**. It is not the full
10,000–50,000-tile/100-event project target, a forest-only model, verified wildfire
truth, an ignition/perimeter model or an operational warning system. It may still
perform worse than MODIS. Geographic cohorts are not independently verified
incident identities, and correlated pixels cannot stand in for independent events.

The original pilot checkpoint, its one-scene evaluation and its threshold remain
unchanged. New pooled metrics are not directly comparable with the earlier
71-tile pilot counts; use the new matched MODIS comparison and the separately
reserved test observations.

Software tests cover pairing bounds, budget guards, cached-source tampering,
nighttime optional bands, native confidence encoding, zero-fire controls,
interrupted tiles, rare-event initialization, calibration, tied-score thresholds,
checkpoint round trips and freezing calibration before test collection.
A GPU update check passed with the new configuration on the old pilot; that is
a software check, not a benchmark or fire-performance result for the expansion.
Authenticated acquisition, native preparation, admission checks and the actual
30-epoch GPU run passed. The best checkpoint reloaded successfully and matches
the sealed evaluation. The scientific coverage limitations above still apply.

Campaign: [configuration](configs/substantial_round_v3.json).
Trainer: [configuration](configs/train_substantial_v2.json).
Evidence: [test log](results/substantial_round_v2/software_tests.log),
[metadata companion check](results/substantial_round_v2/companion_metadata_preflight.json),
[GPU software check](results/substantial_round_v2/software_smoke_profile_on_old_pilot.json).
