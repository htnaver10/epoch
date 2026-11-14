# Future integration proposal — NOT IMPLEMENTED

## Direction correction

The earlier plan preferred observed MODIS and translated VIIRS onto its scale.
The revised VIIRS-reference objective instead prefers **observed VIIRS**, then
estimates VIIRS-equivalent activity from MODIS where supported. Do not integrate
the earlier precedence unchanged or splice raw MODIS counts into a VIIRS series.

## Offline architecture

```text
NASA FIRMS original observations + provenance
        -> validate / preserve raw rows / observation availability
        -> common daily grid + features
        -> train, validate and version the MODIS -> VIIRS model
        -> batch inference and historical out-of-fold estimates
        -> versioned daily regional summaries + spatial prediction partitions
        -> web app's area/date aggregation and calendar
```

The current app is static at runtime: its downloader and data-preparation script
produce JSON; the frontend builds custom-area daily series and computes calendars,
rankings and comparisons. Add ML to that offline build only after approval. No
FastAPI prediction endpoint, browser inference, automatic retraining or model
publication is needed for the initial design.

## Proposed data contract

Retain original point observations for raw CSV/map detail. Publish model estimates
separately with model/feature/data versions, platform, supported dates, quality
status, training years and held-out validation. Partition daily grid predictions
by spatial tile/year for custom-area loading, and precompute built-in regions.

Use a single selected, validated VIIRS reference platform per day. Do not sum
Suomi NPP, NOAA-20 and NOAA-21 translations as independent activity. Validate
platform transitions before changing the reference. Spatial occupancy is not a
375-m reconstructed hotspot position.

Preferred source order for an approved VIIRS-reference product:

1. Reference VIIRS observation when the requested data is available.
2. Approved MODIS -> VIIRS ML estimate when inputs and coverage are supported.
3. A separately validated MODIS -> VIIRS simple calibration fallback.
4. Missing with a reason if none is supported.

Every resolved daily value needs its basis. Summing those values must produce
consistent selected-period, weekly, monthly, seasonal and custom-area totals.
Changing the source must not silently relabel an estimate as observed. Preserve
raw observations for sensor disagreement and provenance inspection. Show held-out
errors rather than the original single power-law curve as the ML explanation.

The new reference target also requires revising historical baselines, anomalies,
trend interpretations, units, exports and acceptance tests together. The current
MODIS-reference trend cannot be presented as a newly calibrated VIIRS-reference
trend by renaming it.

## What remains outside this folder's execution

No changes to downloader/preparation code, frontend analysis hooks, data schemas,
calendar UI, package manifests, build scripts, serving configuration, deployed
assets, ready-to-run bundle or response simulator. A later integration is a
separate task after the evidence and model direction are reviewed.
