# Training toward a useful fire model

**An expanded experiment is now prepared:** [SUBSTANTIAL_ROUND.md](SUBSTANTIAL_ROUND.md)
provides one command for bounded data collection, preparation, a fresh 30-epoch
run, validation-only intercept/threshold fitting, and reserved-test comparison.
It is a first expansion, not a guarantee of usefulness or the full forest dataset.

**The next requirement is a substantially better dataset.** The RTX host and
trainer work, but the current all-land execution pilot is not a useful fire model.
Its training partition has 54 tiles, one acquisition/event group and seven
positive VIIRS reference query centers, occurring in two tiles. There is no
independent test partition or forest land-cover context.

The recorded first epoch, evaluated on the same known validation support as the
official MODIS baseline, gives:

| Validation measure | Current CNN, threshold 0.5 | Native MODIS baseline |
|---|---:|---:|
| Precision | 0.28% | 32.55% |
| Recall | 55.37% | 57.02% |
| F1 | 0.56% | 41.44% |

These are pilot validation observations, not sealed-test results or an operating
threshold recommendation. They do not establish wildfire truth or ground-location
accuracy. See [readiness evidence](results/useful_training_readiness.json).

## 1. Collect a fire-rich, diverse native dataset

Use the staged targets already proposed in [PLAN.md](PLAN.md): an initial
feasibility dataset of **10,000–50,000 native tiles and at least 100 distinct
reference fire-event groups**, together with real non-fire scenes. These are
planning targets, not a guarantee of usefulness or universal minimums. Include
many thousands of positive reference pixels across different events, rather than
many overlapping crops of the same few pixels.

For the first Canada-focused model, include multiple fire regions and seasons,
day and night, different fire sizes, viewing angles and clear/cloud/smoke
conditions. Keep warm land, agricultural and industrial edges, water/glint and
other difficult backgrounds in the acquisition plan. Invalid or unknown reference
states remain masked; a missing hotspot row is not a verified negative.

Use FIRMS or the existing NASA count archive to find **training** candidate
locations/dates. Then obtain actual native `MYD021KM`, `MYD03`, `MYD14`,
`VNP14IMG` and `VNP03IMG` granules, with optional `MYD02HKM/MYD02QKM` context.
The imagery and complete observed reference masks are the training inputs and
labels. Hotspot CSVs alone cannot train this image model.

The implemented population requires Aqua/S-NPP companion files, the supported
collections and a worst-case acquisition-interval separation of at most 15
minutes. Inventory and check actual overlap before downloading large batches.
The pilot showed that source availability does not imply a usable close-time pair.

The observed 30 local raw granules occupy about 2.70 GiB, with about 238 GiB free
on the project drive at this assessment. Actual granule sizes and usable tile
yields vary. Stage a bounded campaign, measure bytes, positive-event yield and
preprocessing time, then expand. Do not infer the total campaign storage or runtime
from the tiny synthetic benchmark.

## 2. Freeze event, geography and time partitions

Create reviewed bundle manifests before fitting normalizers or selecting
checkpoints. Keep a reference event, nearby context, repeated overpasses and
shared source granules in one partition. Merge linked acquisition components
conservatively; an arbitrary ID per tile is not event independence.

Reserve **training**, **validation**, and a genuinely untouched **test** partition.
Use separate geographic blocks with buffers and a later time period where
supported close-time imagery exists. Select representative validation/test
acquisition windows before inspecting their model scores; do not make these
partitions artificially fire-balanced. Document that historical dates already
examined during earlier research are retrospective development data.

For forest-specific claims, add prior MCD12Q1 land-cover maps and verified
availability dates that satisfy the existing reader's temporal and collection
checks. The implemented forest definition is IGBP classes 1–5 with strict QC.
Collection availability must be checked for the chosen historical years; do not
invent publication dates to make a historical experiment pass. A forest mask
identifies context, not whether an observed thermal source is an unplanned fire.

The reference positives remain VIIRS nominal/high-confidence fire classes 8/9;
observed land class 5 supplies negatives. Cloud, unknown and residual bow-tie
states are screened as specified in the
[NASA Collection 2 user guide](https://www.earthdata.nasa.gov/s3fs-public/2025-06/VIIRS_C2_AF-375m_User_Guide_1.2.pdf).

## 3. Prepare and audit a new dataset version

The following commands are supported now, **after** a real, reviewed forest bundle
file named `data/manifests/forest_v1_reviewed_bundles.jsonl` has been created.
That file and the large forest dataset do not exist in this checkout yet.
Run from `fire-harmonization-research-fullres`:

```powershell
.\run-data-windows.cmd prepare --bundles data/manifests/forest_v1_reviewed_bundles.jsonl --output data/manifests/forest_v1.jsonl
.\run-windows.cmd audit --manifest data/manifests/forest_v1.jsonl --output results/forest_v1_audit.json
.\run-windows.cmd stats --manifest data/manifests/forest_v1.jsonl --output data/manifests/forest_v1_stats.json
.\run-windows.cmd preview --manifest data/manifests/forest_v1.jsonl --index 0 --output visualizations/forest_v1_alignment_0.png
```

Inspect multiple positive, negative and difficult tiles across every source type,
region and partition. Check radiance scaling, invalid masks, geolocation, reference
alignment, exclusions and source/event leakage. Fit statistics on training only.
Counts of positive labels must be reported after all masks, not just from source
hotspot inventories. Do not duplicate positive pixels to claim more independent
training evidence.

## 4. Address class imbalance and train

Keep both positive examples and representative difficult backgrounds. If
fire-centered or hard-negative sampling is introduced, record inclusion
probabilities and correct the learning objective or calibrate on representative
full-scene validation exposure. A cost-weighted loss changes the interpretation
of raw probabilities and needs calibration. Do not silently add class weights
and treat the resulting scores as calibrated fire probabilities.

**Current implementation limit:** the trainer uses masked, exposure-normalized
BCE over supplied tile windows. The expanded-round workflow now provides
training-scene fire screening, controls and validation-only intercept/threshold
fitting. It does not provide a dedicated positive-aware tile sampler, verified
incident discovery or a complete large forest-dataset builder. Those still need
implementation and verification for the chosen scientific protocol.

Once the real dataset and sampling approach are ready, measure this model on the
real largest tile/query shapes and start a fresh run:

```powershell
.\run-windows.cmd profile --manifest data/manifests/forest_v1.jsonl --stats data/manifests/forest_v1_stats.json --config configs/train_rtx5070.json --steps 20 --worst-case --output results/forest_v1_profile.json
.\run-windows.cmd train --manifest data/manifests/forest_v1.jsonl --stats data/manifests/forest_v1_stats.json --config configs/train_rtx5070.json --run forest_v1 --epochs 30
```

Thirty epochs is a starting experiment, not a usefulness threshold. Select the
checkpoint using validation, inspect learning curves and stop or revise the
experiment when the evidence warrants it. Changing model/optimizer/sampling
configuration requires a new run. Extending epochs alone does not create new
events or positive labels.

## 5. Establish usefulness on untouched data

Choose operating thresholds and any calibration using validation only. Then
evaluate the frozen model once on the sealed test partition:

```powershell
.\run-windows.cmd evaluate --manifest data/manifests/forest_v1.jsonl --checkpoint models/forest_v1/best.pt --split test --output results/forest_v1_test
```

Compare with official MODIS on exactly the same eligible reference support.
Measure precision/recall curves, recall at matched false-alarm rate, calibration,
and geographic/day-night/view-angle failure strata. Use event/block uncertainty
estimates, not millions of correlated pixels as independent evidence. Useful
harmonization also requires checks of aggregated calendar activity. Fine-location
claims need the additional localization and independent-reference work in PLAN.md.

The existing evaluator supplies point metrics and MODIS comparisons. Matched
operating-point selection, calibration, event-level uncertainty, richer physical
localization metrics and a complete sealed-test report remain additional work.
Promote a model only when its held-out results improve the intended task without
unacceptable false alarms; a larger dataset, lower training loss or more epochs
does not by itself establish that result.

To recheck the current pilot evidence:

```powershell
.\.venv\Scripts\python.exe tools/assess_training_readiness.py
```
