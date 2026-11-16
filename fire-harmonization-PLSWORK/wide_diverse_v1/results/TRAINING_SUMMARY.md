# Wide diverse MLP: completed experiment

Parameters: **50,567,169**, selected epoch **10**.

Additional real training windows: **245**; additional known queries: **4,745,517**; additional positive reference queries: **6,219**.

Training cohorts: bc_2021, central_2021, nwt_2022, pilot_20230701_all_shared_sources.

Original validation/test exposure is preserved. Both cohorts were examined in previous experiments; this is a reused research benchmark. VIIRS reference queries are correlated and are not independent wildfire incidents.

## Validation

| Model | Precision | Recall | F1 | TP | FP | FN | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| wide_frozen_policy | 37.02% | 67.33% | 0.4777 | 1183 | 2013 | 574 | 0.4378 |
| parent_mlp_frozen_policy | 36.94% | 66.82% | 0.4758 | 1174 | 2004 | 583 | 0.4242 |
| regular_modis | 35.58% | 63.29% | 0.4556 | 1112 | 2013 | 645 | 0.2256 |

## Test

| Model | Precision | Recall | F1 | TP | FP | FN | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| wide_frozen_policy | 57.41% | 65.26% | 0.6108 | 62 | 46 | 33 | 0.5955 |
| parent_mlp_frozen_policy | 49.30% | 73.68% | 0.5907 | 70 | 72 | 25 | 0.5556 |
| regular_modis | 44.97% | 80.00% | 0.5758 | 76 | 93 | 19 | 0.3598 |

Wide-model benchmark F1 change versus parent: **+0.0201**.

Model capacity, training coverage, and sampling objective changed together. This experiment cannot isolate a parameter-count effect. The class-stratified sampler corrects importance weights to each cohort's actual class prevalence; cohorts contribute equally. Sampling with replacement does not promise every background query is visited.

Native calibration, cross-sensor timing and geographic inversion are retained. Reference eligibility matches the parent: VIIRS mask classes 5/8/9, bowtie QA bit 22 excluded, valid mapped MODIS support and core/halo restrictions. QA bits 0-6 are reported for expansion observations rather than added as new label exclusions. Label-informed crop selection occurs only within training and makes the expanded training set fire-enriched. Importance correction applies to curated training rows; it does not undo the crop selection or describe deployment prevalence. No synthetic fire labels were created. New independent test incidents are still required to establish generalization.
