# Next model: native MODIS image tiles supervised by VIIRS

**Specification/review date: 2026-11-14. Status: proposed; no native-image model has
been trained.** This plan follows the user's latest objective: a nuanced spatial
model that takes actual MODIS tiles at native band resolution and learns from
VIIRS data. It replaces coarse counts as the next experiment's primary input.

**Implementation update:** the initial current-image pipeline, native readers,
CNN, trainer/resume, evaluator and inference scripts now exist. Follow
[RUNNING.md](RUNNING.md). Their native-data science checks and RTX execution
remain pending; optional temporal, image-teacher and uncertainty extensions are
still proposed. No real native-resolution model has been trained.
The [legacy research summary](../fire-harmonization-research/RESEARCH_SUMMARY.md)
records the earlier coarse-grid experiments. The latest partial-2026 diagnostic
is [Experiment 04.5](../fire-harmonization-research/experiment-04.5/results/REPORT.md).

**Full-resolution directory contract:** all native-resolution plans, code,
acquisition/preprocessing/training/inference/evaluation scripts, configurations,
data/manifests/caches, checkpoints, logs, results and figures belong under the
project-root **`fire-harmonization-research-fullres/`** folder. This document is
its canonical plan. The existing research folder retains the coarse-grid work
and navigation links. No full-resolution outputs may default to legacy research
or application directories.

The user confirmed a **desktop RTX 5070 with 12 GB VRAM**. See the
[full scientific and hardware review](RTX_5070_REVIEW.md) and
[proposed first profile](configs/rtx5070_proposed.json). Initial feasibility is
conditional on real data, a CUDA-compatible environment and measured memory;
the 12 GB card has not been benchmarked from this macOS review host.

## 1. Objective and measurable outputs

Build a **MODIS-only spatial inference model**, trained with near-coincident
VIIRS observations, that estimates active-fire activity in forest contexts at
the time of a MODIS acquisition. It should use thermal signal, fine reflective
context, acquisition geometry, quality and recent MODIS history to distinguish
plausible fire locations from background and persistent thermal sources.

The model should produce:

1. A georeferenced native-scale detection field, calibrated against the VIIRS
   observation reference on valid paired scenes.
2. A finer candidate-location field, initially queried on a nominal 375 m map
   lattice, with per-candidate probability, position uncertainty and supporting
   MODIS footprint/quality information.
3. Forest-context probability or an explicit ambiguous/non-forest classification.
4. Additive summaries for the harmonized activity calendar, preserving observed
   versus estimated provenance.

**Output spacing and achieved accuracy are separate.** A 375 m output lattice
does not establish 375 m ground-location accuracy. The localization claim must be
earned by held-out spatial error and uncertainty coverage. NASA describes MODIS
hotspot coordinates as pixel centers, which may differ from the actual fire
position; multiple fires can occur within one pixel.
[NASA MODIS hotspot description](https://modaps.modaps.eosdis.nasa.gov/services/about/products/c61-nrt/MCD14DL-NRT.html)

The primary target is fire activity observable around the matched satellite
acquisitions. Ignition coordinates, future fire movement, exact perimeters and
all fires hidden by cloud/canopy require additional targets and evidence. The
current archive cannot establish those outcomes.

## 2. What “full resolution” means here

Use the instrument measurements before the existing 0.25° aggregation:

| Input | Native nominal nadir resolution | Contribution |
|---|---:|---|
| MODIS thermal bands | 1 km | Heat and thermal-background evidence |
| MODIS bands 1–2 | 250 m | Daytime visible/NIR spatial context |
| MODIS bands 3–7 | 500 m | Daytime spectral/vegetation and SWIR context |
| MODIS geolocation and QA | Product-specific | Actual coordinates, geometry, missingness and validity |

These instrument resolutions are documented by
[NASA MODIS specifications](https://modis.gsfc.nasa.gov/about/specifications.php).
Thermal information does not become 250 m because reflective channels are available
at that spacing. Pixel footprints also vary with viewing geometry.

“Tiles” will be ML patches cut from individual acquisition swaths, retaining
native measurements and geolocation. The source L1B products are swaths; the
training patch is our own data organization. Daily/8-day composited products,
browser image tiles, RGB screenshots, and interpolation of the current cell-day
counts cannot supply the required acquisition-level radiance evidence.

**Recommended representation:** retain separate native-resolution band arrays
and a coordinate-aware decoder. Create fine map queries using actual geography;
do not assume an integer MODIS-to-VIIRS pixel ratio. One kilometer divided by
375 m is not an integer, and neither swath is a perfectly aligned square lattice.

## 3. NASA data required

### 3.1 Primary inputs and labels

Collection 061 is the proposed MODIS starting point for continuity with the
existing research, subject to actual availability of all required products.
Use one verified collection consistently; a migration to a newer collection is
a separate versioned experiment. Current generic landing pages can describe
different collection generations. Bind actual collection concept IDs, product
versions and granule identifiers in the manifest rather than relying on a page
title or a filename prefix alone.

| Role | Products | Required content and use |
|---|---|---|
| Current MODIS thermal imagery | `MOD021KM`, `MYD021KM` | Calibrated radiances, band metadata and uncertainty/quality information. Proposed initial thermal set: 20, 21, 22, 23, 31, 32; start a smaller ablation with 21/22, 31, 32. |
| MODIS 250 m imagery | `MOD02QKM`, `MYD02QKM` | Native bands 1–2 when illuminated and valid. Do not replace with the 1 km aggregated versions. |
| MODIS 500 m imagery | `MOD02HKM`, `MYD02HKM` | Native bands 3–7 where usable, preserving detector/band quality and missing channels. |
| MODIS geometry | `MOD03`, `MYD03` | Geolocation, acquisition/scan timing available from source metadata, viewing/solar geometry and other relevant fields. Resolve finer-band geolocation with the product's documented alignment procedure. |
| Existing MODIS detection/QA | `MOD14`, `MYD14` | Same-acquisition fire classes and QA, for baseline comparison, auxiliary context and a supervised ablation. Inputs must not be limited to pixels this algorithm already called fire. |
| VIIRS supervision | `VNP14IMG` C2 | Native fire mask and algorithm QA, including observed land, fire, cloud and unprocessed states. This is the primary teacher-label source. |
| VIIRS label geometry | `VNP03IMG` C2 | Terrain-corrected geolocation and acquisition geometry for the target pixels. |
| Optional richer VIIRS teacher | `VNP02IMG`; possibly `VNP02MOD` and matching moderate-resolution geolocation | Training-only radiances/quality for a later teacher model and saturation analysis; not student inference inputs. |
| Forest context | `MCD12Q1` with QA | Annual 500 m land-cover classes and uncertainty/land-water context; use an acquisition-appropriate earlier map. |

Primary product references:
[MODIS C6.1 calibrated-radiance description](https://modaps.modaps.eosdis.nasa.gov/services/about/products/c61/MYD021KM.html),
[250 m L1B](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD02QKM/),
[500 m L1B](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD02HKM/),
[MODIS geolocation](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD03),
[MODIS L2 fire swath](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD14/),
[VIIRS fire product](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP14IMG/),
[VIIRS geolocation](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP03IMG/),
[VIIRS L1B](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP02IMG),
[land-cover product](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD12Q1/).

FIRMS and monthly location files remain useful for finding candidate events,
cross-checking sparse detection attributes and validating ingestion totals.
They do not replace full scene imagery or coverage masks. The existing
[Experiment 03 reader](../fire-harmonization-research/experiment-03/src/archive_tools.py) reads monthly
location products, not the L1B/L2 granules above; native readers must be added.

### 3.2 Forest and reference interpretation

Use an explicit forest definition, initially the IGBP forest classes 1–5 in
MCD12Q1, and retain mixed/uncertain boundaries as a separate stratum. The annual
classification's resolution and QA limit what it can say about a small hotspot.
Use the latest permitted pre-acquisition map, not a map incorporating the later
burn scar. For an operational simulation, enforce product publication/availability
time as well as the mapped year.
[MCD12Q1 guide](https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf)

A vegetation hotspot type is not automatically a forest wildfire or an unplanned
incident. Use available thermal-source typing as label QA/stratification; keep
agricultural, industrial, volcanic, offshore and ambiguous cases identifiable.
VIIRS type/confidence fields must not become MODIS inference features. Source
typing derived from future repeated observations also requires an as-of-time
check before use as an operational predictor.

### 3.3 Platforms and time scope

Begin with **Aqua MODIS and S-NPP VIIRS**, using actual near-coincident acquisition
times. Add Terra as a separately evaluated acquisition regime; do not supervise
a morning image with a later afternoon fire location as though the scene were
unchanged. Model platform, viewing geometry and elapsed time explicitly.

Keep the VIIRS target platform fixed in the first comparison. Add NOAA-20 and
NOAA-21 only with separate observation geometry, quality and cross-platform tests.
Their observations cannot be stacked into a count and called increased fire
activity. S-NPP transition notices make historical supervision and future label
continuity separate inventory questions; check current product availability.
[NASA product-delivery notice](https://ladsweb.modaps.eosdis.nasa.gov/alerts-and-issues/199516)

## 4. Acquisition and reproducibility

1. Restore working network/DNS access on the chosen training host. The current
   runtime failed to resolve NASA/UMD download hosts. That is the recorded blocker,
   not evidence that the archives are unavailable globally.
2. Use NASA CMR/DAAC discovery, via `earthaccess` where appropriate, to identify
   collection concept IDs and list granules intersecting the proposed regions
   and time intervals. Enumerate the complete result set; a bounded search limit
   is only a pilot, not a full inventory.
3. Configure Earthdata/DAAC credentials through a research-scoped environment or
   approved credential mechanism. A FIRMS `MAP_KEY` is not a substitute for
   authenticated L1B/L2 access. Do not write tokens into manifests or URLs in logs.
4. Before a large transfer, download **10–20 proposed matched granule bundles**
   spanning daytime, nighttime, clear non-fire, detected fire and awkward geometry.
   This is an ingestion test, not a sufficient training set.
5. Validate file signatures, product/version, dimensions, calibration attributes,
   checksums and matching orbit/time identifiers. Open both an HDF4 MODIS file
   and a VIIRS NetCDF/HDF5 file with the intended readers.
6. Record checksum, bytes, source identifier, collection, processing timestamp,
   acquisition interval, bounding geometry, retrieval time, status and retry history.
   Cache by immutable granule ID and checksum; handle reprocessing explicitly.
7. Download full required granules or a verified science-data subset. Preserve the
   original source objects. A server subset must retain calibration, QA,
   geolocation and enough halo context to reconstruct a tile.

The acquisition approach follows the package's documented distinction between
public metadata search and authenticated data access, and its collection/granule
identifiers. The exact client/environment will be pinned during the reader smoke
test; no installation or authenticated download is claimed in this document.
[earthaccess search documentation](https://github.com/earthaccess-dev/earthaccess/blob/main/docs/user/explanation/search.md)

MODIS HDF4 and VIIRS HDF5 are different formats. Plan a verified HDF4-capable
reader, such as an appropriate GDAL build, plus a compatible NetCDF/HDF5 reader.
An HDF5-only reader cannot open MODIS HDF4 merely because both filenames contain
“HDF.” Pin supported dependencies in an isolated research environment after the
first real files open successfully. The existing Python 3.14 environment has not
been verified for the proposed deep-learning/data stack.

## 5. Radiometric and geometric preprocessing

### 5.1 Preserve physical meaning

- Decode each band's scale/offset and units from the exact product. Keep radiance
  and, where derived correctly, brightness-temperature channels distinct.
- Preserve fill, invalid, saturation and uncertainty states as masks/features.
  An invalid or clipped measurement must not silently become a cold zero.
- Represent bands 21 and 22 separately, with their validity/saturation information.
  Do not blindly average their values. Verify all proposed channels on actual files.
- Treat missing/damaged reflective detectors with explicit per-band masks and
  tested handling; optional context must remain optional when unreliable.
- Represent nighttime reflective data as unavailable with a mask, not as measured
  zero reflectance. Train and report day/night behavior separately.
- Fit normalizers on training data only. Preserve rare hot tails; arbitrary
  percentile clipping can remove the very signal being detected.

The thermal/reflective groups, calibration requirements and product-specific
scaled representations are documented in NASA's
[L1B data dictionary](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/M1055_PDD_D_2022_1005_V6.2.2_Terra_V6.2.3_Aqua.pdf).

### 5.2 Retain actual geometry

Keep source pixel centers, scan/sample indices, viewing/solar angles, timestamps
and estimated ground support. A nominal 1 km square drawn around a center is not
an exact off-nadir footprint. Interpolated centers alone do not provide a precise
point-spread function. Use documented detector/scan geometry or validated footprint
approximations, recording the method and uncertainty when exact corners are absent.

Check residual alignment against stable geographic structure independent of fire
labels. A target-driven warp must not be fitted to each test fire and then used
to improve its apparent score. Record projection/WKT, geolocation transform and
pixel support; retain the original geographic coordinates through every conversion.

For labels, retain categorical class semantics or explicit footprint-overlap
fractions. Interpolating class codes into intermediate numbers is not a valid
fire-probability label. Fine-grid fractional coverage of flagged VIIRS pixels is
also not the fraction of ground actually burning.

## 6. MODIS–VIIRS pairing and trustworthy labels

### 6.1 Match observation opportunities before detections

Find overlapping swath acquisitions by their spatial footprints and time intervals.
Then determine per-pixel or scan-level time differences where the products permit.
Match using actual times, not UTC dates or an assumed fixed local overpass hour.

**Proposed initial clean-pair rule:** absolute time difference at most 15 minutes,
with at-most-30-minute pairs in a separate sensitivity set. Start with good-quality
moderate-view-angle data, provisionally <=40° where that metadata is reliable,
and report narrower/outer-angle strata. These are design defaults to verify on
the inventory before freezing the scoring protocol. Do not relax time criteria
until a poor result disappears. If too few clean pairs exist, change the data
scope or explicitly model temporal change; do not silently change the estimand.

Use the union of applicable VIIRS granules for a matched footprint, preserving
coverage and choosing a deterministic observation when acquisitions overlap.
Resolve duplicate and residual bow-tie observations using QA; keep separate
overpasses separate. VIIRS's guide explicitly distinguishes fire, land, cloud,
unprocessed and bow-tie states.
[VIIRS C2 guide](https://ladsweb.modaps.eosdis.nasa.gov/archive/Document%20Archive/Science%20Data%20Product%20Documentation/VIIRS_C2_AF-375m_User_Guide_1.2.pdf)

### 6.2 Label states

| VIIRS reference state | Training treatment |
|---|---|
| Valid nominal/high-confidence fire, appropriate scene/type QA | Primary positive reference |
| Low-confidence fire | Separate uncertain-label/sensitivity stratum; never silently convert to negative |
| Valid observed clear land with no detected fire | Reference negative, conditional on observation and timing; not proof of no real fire |
| Cloud, unknown, fill, deleted bow-tie, unprocessed or no matched acquisition | Mask out of the primary supervised loss |
| Partial footprint coverage or larger time mismatch | Preserve separate coverage/temporal uncertainty; secondary experiment only |

The primary clean benchmark requires usable MODIS measurements and a usable
VIIRS reference for the same physical scene. Cloud-adjacent and degraded MODIS
inputs get explicit secondary robustness analyses. Keep MODIS coverage and VIIRS
label-validity masks separate: the label-validity mask is training/evaluation
information, not an extra VIIRS feature available to the deployed student.

Training must contain all four detector-disagreement cases: both detect fire,
MODIS-only, VIIRS-only and neither. A MODIS algorithm-negative pixel can still
contain weak useful radiance. Restricting training/inference to existing MODIS
hotspot points would discard exactly the missed cases the next model should test.

### 6.3 Spatial uncertainty in supervision

Retain native VIIRS pixel labels at their actual coordinate queries as the
cleanest supervision path. When creating a regular output map, construct targets
and coverage from documented/approximated VIIRS footprints and retain the source
membership. Treat label-center location, footprint extent, registration error and
fire change during the time gap as separate uncertainties.

Compare a hard-label baseline with a predeclared uncertainty-aware spatial loss.
Any soft target kernel must be derived from measured/assumed-and-tested geometry
uncertainty, not an arbitrary blur tuned on test locations. Keep the unblurred
reference for scoring. A VIIRS fire-pixel center is still not an exact subpixel
ground-truth flame coordinate.

### 6.4 Freeze one target before implementing the image losses

Native VIIRS-pixel fire status, overlap-weighted fine-map reference coverage,
unique incident count and physical burned area are different targets. Version 1
must choose and name the representation, validity denominator and units. Start
with valid native VIIRS fire/non-fire observations for detection supervision;
explicitly define any map-raster overlap target and its partial-coverage policy
before scoring a fine field or applying an activity-consistency loss. A probability
against that observation reference is not automatically a calibrated probability
of a real wildfire. Derive calendar units only after the mapping is validated.

## 7. Native tile specification and data contract

### 7.1 Proposed patch layout

Use native MODIS patches, initially **96×96 thermal samples**, with corresponding
native **192×192 500 m** and **384×384 250 m** arrays where band registration
permits. Retain a 16-thermal-sample halo on each side and score the central area
to reduce boundary artifacts. These dimensions are an initial engineering choice.
Off-nadir geographic extents differ; “96 samples” must not be assumed to mean
exactly 96 km on the ground.

A nominal 96 km, near-nadir map extent corresponds to 256×256 queries at 375 m.
This is a sizing illustration, not a fixed warp between sensors. The actual target
queries and their valid footprint intersection follow geolocation. Larger or
irregular extents may use padded/masked query batches. The model's inputs remain
native arrays; the coordinate decoder samples their learned features at physical
query locations using recorded transforms.

Store original arrays and a versioned training cache separately. Every tile must
be traceable to its source granules and source pixel windows. No preprocessed
array is accepted without units, masks and source provenance.

### 7.2 Required records

| Record | Required fields |
|---|---|
| Source manifest | Product, platform, collection concept ID, granule ID, processing version, source checksum/bytes, acquisition interval, retrieval status |
| Tile identity | Stable tile ID, source windows, region block, event group, orbit/acquisition group, partition/fold, preprocessing version |
| Native MODIS tensors | Per-band arrays and band order, units, scales applied, native dimensions, fill/valid/saturation masks |
| Geometry | Latitude/longitude or coordinate transforms, CRS/WKT, scan/sample/time, viewing/solar geometry, footprint approximation and uncertainty |
| Temporal context | Earlier MODIS granule IDs, elapsed times, coverage and quality masks; missing-context flags |
| Forest context | Land-cover source/version/year/availability, taxonomy, QA, fractional/ambiguous forest membership |
| Training-only targets | VIIRS class/QA, target coordinates/footprints, reference-valid mask, time difference, label confidence/source type, target platform |
| Sampling | Scene/tile selection stratum, inclusion probability, effective exposure and loss weight |
| Audit fields | Deduplication decision, exclusion reasons, matching method/version, registration checks, content hashes |

Make MODIS inference inputs and VIIRS training targets distinct objects in the
loader and model signature. Test that running inference requires no VIIRS files.
Audit annual land-cover availability, temporal context and persistent-source
features for future-information leakage as well.

## 8. Data volume, diversity and partitioning

### 8.1 Staged acquisition

1. **Reader/alignment pilot:** 10–20 matched bundles, including hard cases. Inspect
   arrays, calibration, masks and geographic overlays; do not publish model scores.
2. **Model feasibility set:** provisionally 200–500 matched acquisition bundles
   and at least 100 distinct reference fire-event groups, plus non-fire scenes.
   Actual counts depend on clean-pair availability; they are targets, not acquired data.
3. **Diverse expansion:** forest regions in boreal North America and Eurasia,
   temperate regions, tropical South America, Southeast Asia and other supported
   forest regimes; full seasonal cycles, day/night, small/large signals,
   different backgrounds and viewing angles. Retain agricultural/industrial edges
   as important negative or alternative-type cases.
4. **Scale only after feasibility:** increase independent events, regions and
   matched overpasses, rather than counting heavily overlapping crops as new data.

Balance difficult cases for learning but record inclusion probabilities. If
positive-centered/hard-negative sampling alters prevalence, correct training or
calibrate on representative full-scene exposures. Score complete eligible held-out
scenes, not a balanced sample that makes precision appear artificially high.

### 8.2 Leakage-resistant splits

Define event groups in space and time for partitioning. These are reference
clusters, not verified incident identities. Keep an event's tiles, nearby context,
overlapping patches, repeated overpasses and source-granule duplicates within
one partition. Use spatial buffers at least as wide as the input/label context.
Partition before fitted preprocessing, calibration and any label-driven cropping.

Proposed historical development split: train 2013–2019, select 2020–2021, and
evaluate 2022–2025 with whole-event and geographic blocks. Those years already
have coarse research results; describe the corresponding native-data assessment
as retrospective. Freeze the exact native holdout inventory before inspecting
its prediction scores.

Also reserve entire newly acquired forest regions/event groups outside the
present boreal rectangle for geographic transfer, and a newly available
standard-processed time period after model freeze for prospective evaluation.
Its dates must come from actual archive availability, not a promised SP cutoff.
Do not reuse the final fourteen-year coarse fit's in-sample scores as evidence
for a native model. Unsupervised pretraining must respect held-out region/event
boundaries or be explicitly reported as transductive.

## 9. Model architecture: nuanced inputs, testable complexity

### 9.1 Necessary native baselines

Build before the large image model:

- Official MODIS fire pixels/centers on the same observation windows.
- A geometry-only localization baseline: map detected MODIS footprints to the
  output representation with uniform/uncertainty-weighted support. This measures
  the benefit of geometry alone and exposes misleading cosmetic sharpening.
- A native-pixel classifier using calibrated MODIS thermal anomalies, local
  thermal background, geometry and QA; aggregate VIIRS supervision to each
  parent region. This tests whether the raw measurements add information beyond
  coarse counts before requesting fine location detail.
- A small thermal-only spatial encoder/decoder, then a multiresolution version.

Reprojected coarse tree predictions are a calendar baseline, not a substitute
for these native measurement baselines.

### 9.2 Recommended first image model

Use a **multiresolution U-Net-style student with geometry conditioning and a
coordinate-aware output decoder**, initially a compact model provisionally in
the **2–5 million parameter range for the first RTX 5070 profile**. Larger
variants, up to the earlier proposed 10 million range, need a fresh memory and
accuracy comparison. Measure actual size after implementation.
This is a proposed architecture, not an already trained network.

```text
MODIS 1 km thermal + per-band QA ── thermal/context encoder ─┐
MODIS 500 m native bands + QA ───── optional spatial encoder ┤
MODIS 250 m native bands + QA ───── optional spatial encoder ┤
MODIS geometry/platform/time ────── conditioning features ──┼─ fusion
Earlier MODIS acquisitions + Δt ─── causal context encoder ─┤    │
Prior forest context + QA ───────── context features ───────┘    │
                                                               ▼
                       georeferenced native/fine coordinate decoder
                         │                │                 │
                 native detection   candidate field   uncertainty/quality
                         └───────── activity summaries ─────┘

Aligned VIIRS masks/QA/coordinates ───────── training losses only
```

Use actual native-scale receptive fields and skip connections. Preserve masks
through fusion so the model can use thermal evidence without missing night-time
reflective context. Condition on platform and geometry to handle different
measurement regimes without learning a location-specific geographic intercept.
Broad geographic IDs should be optional, explicitly ablated, and never the sole
source of a fine location.

On the 12 GB card, start with a convolutional implementation, narrow native-band
stems, GroupNorm and microbatch 1, then profile 2. Gradient accumulation can
provide a larger effective optimizer batch; it does not supply larger BatchNorm
statistics. Use local coordinate sampling/shared heads or coarse/local attention;
do not materialize full-resolution global query-by-pixel attention. Bucket/pad
query shapes to a bounded maximum and profile the largest actual overlap.

Start with current-acquisition inputs. Add the previous two usable MODIS
acquisitions through a masked temporal module only after the current-image model
is sound. Each context frame needs its elapsed time and geometry; a prior
acquisition is not automatically the previous calendar day. Newly ignited and
short-lived fire cases must be evaluated separately so temporal persistence does
not merely smear old detections into new locations.

The fine head estimates a spatial field and possibly multiple candidate peaks.
A single regressed point is inadequate when one MODIS footprint contains multiple
VIIRS fire pixels. Output probabilities and support regions; any optional
within-cell coordinate-offset head must have independent spatial validation.
Reflective texture can help allocate plausible locations but may also create a
forest-shaped prior without heat evidence. Test that explicitly.

### 9.3 Proposed distinguishing idea

Test **geometry-aware probabilistic disaggregation**: condition fine-location
predictions on the actual MODIS observation footprint and thermal evidence,
while retaining the richer 250/500 m context and learned uncertainty. Keep a
native/coarse activity head and a fine field consistent in expected-activity
units over non-overlapping analysis regions.

For a fine probability field `p_j`, an auxiliary regional head should agree with
`sum(a_Pj × p_j)`, where `a_Pj` assigns each fine cell to non-overlapping parent
analysis regions by overlap. This is expected **reference-active-cell activity**,
not raw MODIS detection-count conservation. Actual sensor footprints can overlap;
use them for observation geometry, not as an assumed non-overlapping accounting grid.

As a later ablation, add a radiance/FRP forward-consistency model with documented
point-spread/footprint assumptions and saturation/noise handling. Its latent heat
field and temperature/area tradeoffs are an inverse problem; do not impose an
unsupported exact count-to-radiance conservation law.

Multiscale segmentation, teacher–student learning and sensor fusion are known
techniques. The proposed combination is a research hypothesis, not an established
new algorithm or a demonstrated precision gain. Existing
[MODIS–VIIRS footprint-scale FRP fusion research](https://www.mdpi.com/2072-4292/18/10/1650)
uses both instruments during fusion; that does not establish MODIS-only historical
fine-location recovery.

### 9.4 Optional VIIRS image teacher

After supervised mask training works, test a separate teacher that sees VIIRS
radiances/QA and produces calibrated spatial targets or latent features. Distill
only on matched training scenes, with quality/uncertainty weighting. The student
must still infer from MODIS alone. Evaluate the teacher itself against an
independent reference where possible; an inaccurate teacher is not a truth source.

Hard fire-mask supervision remains the anchor. Fine VIIRS image structure may
be unidentifiable in MODIS; forcing an exact visual copy can generate confident
but unsupported locations. The student should express conditional uncertainty
when several fine configurations fit the same coarse thermal evidence. Do not
feed VIIRS-derived encodings to the student during its held-out inference.

## 10. Losses, training and calibration

Use a staged training objective, tuning weights on development validation only:

`L = L_valid_fire + w_native L_native + w_location L_location`
`    + w_count L_activity_consistency + w_aux L_forest/quality`.

- **Valid fire loss:** masked weighted binary cross-entropy at native VIIRS
  queries or explicitly defined fine-map targets. Compare focal loss only if
  representative-scene calibration is retained; it changes probability behavior.
- **Native auxiliary loss:** VIIRS-reference activity aggregated to the native
  analysis support, ensuring useful detection even if subpixel allocation is weak.
- **Location loss:** uncertainty-aware likelihood/transport or spatial matching
  term using geolocation and temporal uncertainty. Preserve the hard reference
  for evaluation; exclude unobserved targets.
- **Activity consistency:** agree between fine expected activity and the auxiliary
  analysis-region output. Do not force one VIIRS detection per MODIS hotspot.
- **Forest/quality tasks:** optional auxiliary outputs with their own valid masks;
  land-cover confidence is not fire confidence.

Measure gradient behavior on synthetic alignment cases, then overfit one real
clean tile as an implementation check before scaling. Use actual small held-out
scenes to verify that normalization/masks/resampling remain consistent. Physical
units must survive augmentation; rotations/flips must transform coordinates,
view directions and labels consistently. Artificial RGB transformations are not
valid thermal radiance augmentations.

Train several independent seeds after choosing the architecture. Calibrate
probabilities on a separate representative validation partition. Use event/region
blocks for uncertainty assessment; spatially correlated pixels are not independent
calibration samples. An ensemble can reveal model disagreement, but does not
guarantee calibrated geographic coverage. Report empirical uncertainty coverage
and its loss under domain shift.

### 10.1 Numerical precision and small-batch rules for RTX training

Use BF16 autocast for supported neural operations only after an FP32 parity check
on representative weak and strong signals. Keep parameters/optimizer state in
FP32 initially. FP16 is a tested alternative with gradient scaling; FP4/FP8 and
custom kernels are not initial prerequisites.

Keep geographic transforms/footprints in FP64 during preprocessing, local metric
coordinates and sampling grids in FP32 or higher, and perform coordinate distances,
thermal contrasts/normalization and weighted loss reductions outside low-precision
autocast. Never cast an entire tile record, including absolute latitude/longitude,
to BF16/FP16. The [precision worksheet](results/memory_worksheet.json) demonstrates
kilometer-scale coordinate displacement from such a cast.

For accumulated masked losses, normalize by total valid weighted exposure for
each objective across the actual accumulation window. Averaging microbatch means
is not equivalent when coverage differs. Handle all-invalid batches and partial
final accumulation windows explicitly. Begin with masked BCE with logits and a
well-defined native auxiliary target; add spatial transport and other heads only
after their units, gradients, memory and added value are demonstrated.

## 11. Evaluation that tests actual location usefulness

### 11.1 Native detection and fine reference agreement

Report full-scene precision/recall curves, average precision, Brier/log loss,
thresholded precision/recall/F1, false candidates per observed forest area per
overpass, and calibration. Score at native support and at 375/750/1000 m analysis
scales. Report denominators and exclusion counts. Overall pixel accuracy is a
poor headline for a rare-fire task.

Separate MODIS-confirmed and VIIRS-only reference detections, small/weak signals,
day/night, view-angle/time-gap strata, forest types, geographic regions, season,
smoke/cloud adjacency, newly active areas and persistent-source negatives.

### 11.2 Point and region localization

For exported candidate points, use a frozen extraction procedure and one-to-one
matching to reference VIIRS fire-pixel centers, with predeclared distance gates
at 375, 500 and 1000 m. Include unmatched detections and references in precision
and recall; reporting distance only on successful matches hides misses.

Report median/90th/95th distance errors, recall within each tolerance, duplicate
candidates, and distances to the union of reference fire-pixel footprints. The
latter is a support-consistency measure; zero distance within a footprint is
not proof that the predicted point is on a flame. Treat neighboring VIIRS pixels
as a reference patch in an additional component-level evaluation, rather than
calling every pixel an independent incident.

For predicted uncertainty regions, report reference-patch coverage, region area,
and probability/coverage reliability. An uncertainty region covering the entire
scene is not useful localization. An exact-looking point with poor coverage is
also not useful.

### 11.3 Independent higher-resolution checks

VIIRS-only testing establishes agreement with VIIRS at its observation scale.
To substantiate better ground localization, reserve independent, sufficiently
near-coincident NASA-distributed observations where available:

- ASTER thermal imagery can provide a finer thermal reference for suitable
  scenes; its TIR sampling is 90 m. Same-platform Terra opportunities are valuable
  for later Terra validation, but coverage and valid/saturated pixels must be audited.
  [NASA ASTER L1T product](https://doi.org/10.5067/ASTER/AST_L1T.004)
- NASA FIRMS distributes Landsat OLI active-fire detections at 30 m. Use only
  quality-controlled close-time overlaps; the sampling label is not automatically
  ground-location truth and later burn scars do not identify the earlier flame position.
  [NASA FIRMS active-fire downloads](https://firms.modaps.eosdis.nasa.gov/content/usfs/active_fire/)

These are optional validation acquisitions, not guaranteed available labels for
every event. If sufficiently close independent observations cannot be obtained,
limit the claim to VIIRS-reference localization with measured uncertainty.

### 11.4 Calendar compatibility

Aggregate the evaluated native model into daily/weekly/monthly activity on the
same valid observation windows. Measure count/activity bias, WAPE, peak timing,
seasonal totals and rank/anomaly stability. Report exposure changes. Repeated
overpasses and finer pixels must not inflate a calendar merely because more
samples were acquired. Preserve the distinction between acquisition-level
expected active pixels and daily occupancy; do not assume independence of
repeated views when converting one into the other.

## 12. Proposed acceptance gates

These are **proposed decision rules**, not achieved results or promises of meter
accuracy. Final thresholds should be frozen after the ingestion/label audit and
before holdout prediction scores are inspected.

| Gate | Required evidence |
|---|---|
| Data readiness | All pilot source bundles decode; calibration/QA and coordinate round trips pass; label-negative/unknown states audited; no unresolved product-version mismatch |
| Pair readiness | Actual time-gap/angle distributions and overlap documented; alignment residuals measured; enough independent events and negatives for held-out evaluation |
| Native-model benefit | Better reference detection at matched false-candidate rate than native MODIS and native thermal baselines, without a major region/day-night failure |
| Fine-location benefit | Provisional goal: at least 20% lower median matched reference-center error than the MODIS-center/geometry baseline, improved recall within the predeclared 500 m gate at matched false-candidate rate, and no worse 90th-percentile error; include unmatched cases |
| Honest uncertainty | Predicted high-coverage regions attain the predeclared empirical coverage on held-out events while remaining materially smaller than the coarse baseline support; report uncertainty of that estimate |
| Geographic/time transfer | Whole-region and later-period performance meets predeclared non-inferiority margins; use blocked intervals and disclose underpowered strata |
| Calendar value | No material degradation in weekly/seasonal agreement, peak timing or observed-versus-estimated provenance; better locations alone do not repair an inconsistent trend |
| Product readiness | Model/data versions, coverage/abstention rules, failure cases and validated metric scope are documented; no unsupported forest/ignition/perimeter claim |

If fine localization fails while native detection improves, retain the useful
native detector and broad uncertainty support. If extra reflective/temporal
branches add no benefit, use the simpler model. The acceptance process should
reward demonstrated location information and calibration, not parameter count.

## 13. Required ablations and failure investigations

Run comparable partitions/threshold policies for:

1. Thermal-only versus thermal + native 250/500 m context.
2. Current image versus causal temporal context, especially newly active fires.
3. Nominal pixel geometry versus measured footprint/angle conditioning.
4. With/without MODIS algorithm fire-mask context; explicitly retain VIIRS-only cases.
5. With/without forest context; test warm non-forest/industrial negatives.
6. Hard versus uncertainty-aware labels at strict/relaxed time gaps.
7. Native detection head alone versus additional fine-location head.
8. Single supervised student versus a training-only VIIRS image teacher.
9. Separate versus shared day/night and platform conditioning.
10. Pre-2012-like domain stress: early-overlap years, changed calibration/geometry
    and missing-context inputs, with properly separated labels and clear chronology.

Investigate any apparent location gain that vanishes after registration correction,
changes in sampling prevalence or removal of overlapping events. Inspect whether
the model simply paints forest edges or persists yesterday's fires. Compare the
fine decoder with a geometry-only uncertainty footprint on exactly the same cases.

## 14. Resources, milestones and concrete artifacts

### 14.1 Resource planning

The user-confirmed target is a **desktop RTX 5070, 12 GB**. A compact native tile
model is a reasonable first experiment on that card. The earlier 16–24 GB planning
suggestion was headroom, not a demonstrated minimum. The current review machine
is macOS/ARM with no PyTorch or attached target GPU; no measured RTX fit or speed
is available. Configure and verify an isolated environment on the training host.

Start with the existing 96/192/384 native patch sizes, current acquisition only,
microbatch 1 then 2, and a 2–5M-parameter convolutional model. Use accumulation
toward effective batch 16 if appropriate. Target roughly 75–80% of actual device
memory as a planning ceiling for peak reserved training memory, then adjust to
measured free memory/display load and worst-case tiles. A 64/128/256 native-pixel
crop is a fallback that reduces geographic extent while retaining band resolution.
Add history, wider decoders and teachers only after profiling.

Measure after optimizer-state initialization, with forward/backward, all losses,
geometry sampling, optimizer step and representative loader. Include variable-shape
and validation/export peaks. Checkpoint activations or reduce microbatch/width
before reducing native input resolution. Do not provision a paid host or assume
a training budget from this document.

Measure storage before scaling. For a nominal near-nadir 96-sample patch, a rough
uncompressed sizing example is six float32 thermal channels, two native 250 m
channels, five native 500 m channels, eight 1 km geometry/context channels and
three 256×256 byte masks. That is about **2.51 MiB** before extra QA, coordinate
arrays, history, caches and raw granules. Adding two six-channel thermal history
frames brings that example to about **2.93 MiB**. At 100,000 patches, those tensors
alone are roughly 286 GiB. This is arithmetic for planning, not an observed
download size; actual swath extents, masks, compression and raw-file overhead differ.

Deduplicate raw granules; crop lazily or cache reusable tiles. Report model
training time, peak memory, effective independent-event count and evaluation
cost from actual runs. The 93-second coarse-tree experiment is not a runtime
estimate for full-resolution image training.

### 14.2 Execution order and deliverables

| Phase | Work | Exit artifact/gate |
|---|---|---|
| A: native-source proof | Network/auth, collection inventory, 10–20 bundles, HDF4/HDF5 readers | Checksummed manifest, decoded-array checks, measured access/storage; Gate: data readiness |
| B: observation pairs | Acquisition-time/footprint joins, QA states, forest context, deduplication | Pair ledger, coverage audit, geographic overlays, exclusions; Gate: pair readiness |
| C: tile dataset | Native band arrays, coordinate queries, causal context, event/region partitions | Versioned tile manifest/data contract, fixed splits, loader leakage tests |
| D: native baselines | Official MODIS, geometry baseline, native thermal classifier and small encoder | Native detection/localization baselines and failure gallery |
| E: multiresolution student | Thermal + 250/500 m + geometry, then fine head | Frozen configuration, validation selection, checkpoint and native/fine benchmark |
| F: nuance ablations | Temporal module, forest/QA auxiliary tasks, uncertainty, optional VIIRS teacher | Ablation table, calibration, independent-reference subset, resource measurements |
| G: sealed evaluation | Whole events/regions/time, representative scenes, uncertainty and calendar checks | Model card, full metric arrays, before/after location figures and promotion decision |
| H: offline product proposal | Versioned inference partitions and calendar contract | Research-only export and integration proposal; actual app changes require a later task |

Required project-root location: **`fire-harmonization-research-fullres/`**. Keep immutable source
manifests and checksums, source granules, preprocessing code, tile/pair ledgers,
split definitions, training configs/logs, model checkpoints, calibration artifacts,
predictions, evaluation tables and figures together. These are planned artifacts,
not files claimed to exist now. Use the subdirectory contract in [README](README.md).
Do not create native training scripts, datasets or results inside the old
`fire-harmonization-research/` folder.

### 14.3 Before/after figures for the next model

Recreate the calendar/totals diagnostics for continuity, and add location-focused
figures that can actually evaluate the new objective:

- Same-scene thermal radiance with calibrated units and acquisition timestamps.
- MODIS footprint/center, fine model field, and VIIRS reference overlaid in the
  same projection; show time gap, viewing angle and label-validity mask.
- Day/night and small/weak-fire examples including misses and false alarms.
- Candidate points with uncertainty contours and an independent-reference subset.
- Detection/localization precision–recall and distance-recall curves.
- Error/coverage by geometry, region, forest type and reference-availability strata.
- Ablation results and native-to-calendar aggregation reconciliation.

Use common color scales for like quantities. Keep radiance, probability,
reference fire class and inferred activity in their own correctly labeled units.
Selected attractive scenes must be accompanied by systematic held-out results.

## 15. Inference, history and eventual application contract

The student inference call accepts MODIS tiles, MODIS-derived acquisition
information, allowed earlier MODIS context and permitted static context. VIIRS
data is unnecessary at inference for this variant. Batch inference keeps overlap
halos, reconciles duplicate views and returns coverage/abstention reasons where
inputs or domain support are inadequate.

Each exported estimate should retain model/data version, acquisition time,
MODIS platform and granule IDs, candidate geometry, probability, uncertainty
geometry, forest-context status, supporting footprint, QA/domain-support flag
and basis `model_estimate`. Retain observed MODIS and VIIRS points as separate
records with their own source provenance. Avoid presenting several decimal
places in a coordinate as evidence of corresponding physical precision.

Pre-2012 inference is a domain-transfer application: VIIRS labels do not exist
for those years. Earlier MODIS calibration, Terra-only periods, geometry, land
cover availability and changing overpass conditions require audits and uncertainty.
Assess early-overlap holdouts and independent historical references where possible.
Fine historical maps remain estimates even when their calendars align well.

The calendar can consume validated offline summaries after a future integration
decision. The location layer can display observed footprints/points and model
candidate regions with uncertainty. Keep derived activity exports separate from
original hotspot CSVs. This plan makes **no modification to the main app**.

## 16. Immediate next experiment

**Acquire and align the first real native MODIS/VIIRS swath bundles, then prove
that a tile retains calibrated native measurements, valid observation labels and
correct geography.** The first success criterion is trustworthy paired image
data. Train the thermal-only native baseline next, followed by the multiresolution
student and measured fine-location experiment. This sequence directly pursues
the user's image-tile objective while making every precision claim testable.

## 17. RTX 5070 readiness gates

1. Verify GPU identity, capability 12.0, current supported driver and a stable
   CUDA-enabled PyTorch build with working Blackwell kernels. Use the official
   release matrix; a 2025 CUDA 12.8/nightly recommendation is not a permanent pin.
2. Run [the compatibility preflight](tools/gpu_preflight.py) on the RTX host.
   It checks small convolution and sampling backward operations, not the full model.
3. Demonstrate readable native HDF4/HDF5 products and valid matched labels; the
   GPU cannot fix an unavailable or incorrectly joined training set.
4. Freeze the first architecture/target and measure full training peaks with
   microbatches 1/2, numerical parity, masked accumulation and no teacher/history.
5. Estimate runtime from measured end-to-end tiles/second and actual dataset size.
   Scale independent events and coverage after an overfit-one-tile and held-out
   feasibility check. No speed or localization-accuracy guarantee is inferred
   from the RTX 5070's gaming or AI marketing specifications.
