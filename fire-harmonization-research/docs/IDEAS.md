# Discussion record and decisions

## Goal and audience

Build a harmonized burning-activity calendar from the early MODIS record through
VIIRS continuity. Emergency managers, scientists and land managers should select
a region/date range, inspect recurring peaks and unusual years, compare a recent
period with the same dates historically, and download evidence and visualizations.
This compares regional activity; incident-level fire size and spread need other
information. Hotspots are neither unique fires nor measured burned area.

The work has two parts: (1) produce a defensible common activity record, and (2)
build the interactive application that consumes it. Sensor changes should not
create artificial trends. Sensor identity and provenance must still be retained.

## Source and scope decisions

- Use NASA-distributed data for hotspot detection and historical harmonization.
  NOAA-platform VIIRS data distributed by NASA FIRMS meets this interpretation.
- NASA's thermal products already perform detection. An ML harmonizer translates
  observation characteristics; it is not a new satellite thermal detector.
- Keep the existing response/spread simulator separate. Replacing its weather,
  terrain, roads, fuel assumptions or aircraft model was outside the agreed scope.
- The current rectangle (50-70N, 140-60W) and June-August record are pilot scope.
  The challenge-facing calendar should eventually cover the full annual cycle
  within an explicitly supported geography. Global support is not yet established.
- Preserve original detection coordinates and attributes for a genuine hotspot
  CSV. The existing grid-cell-day CSV is a derived summary, not the original rows.

## How harmonization evolved in the conversation

1. Existing EPOCH: daily 0.25-degree occupied cells, latitude weighting, inferred
   persistent-source exclusions, heuristic gap filling, MODIS as reference, and
   a leave-one-season-out power-law bridge from VIIRS to MODIS.
2. Practical improvements: retain quality/provenance, distinguish missing data
   from zero, calibrate platforms separately, validate at the displayed timescale,
   and make observed vs estimated activity visible.
3. Initial ML candidate: LightGBM/Tweedie on cell-weeks, translating VIIRS to MODIS.
   Poisson is an alternative for integer counts; Tweedie suits nonnegative skewed
   activity targets. Neither has been proved better on this project.
4. User chose a practical approach rather than a full research observation model.
   A probabilistic latent-fire model with observation operators was considered,
   but would require Level-2 masks, detectability assumptions and more validation.
5. User chose ML as the primary translation method and support for all existing
   date selections. Daily expected occupancy can aggregate into any period; a
   weekly estimate should not be divided into fabricated daily observations.
6. Latest revision: use **VIIRS as the reference to improve MODIS-derived activity**.
   This reverses the original calibration direction. The current research predicts
   VIIRS occupancy/activity from MODIS-only predictors; it does not invert a fitted
   regression algebraically or reuse VIIRS fields as historical MODIS inputs.
7. User explicitly requested an isolated folder and preliminary tests only.
   No app integration, production model promotion or main-app changes are authorized.

## What ML can and cannot mean

ML may improve agreement with VIIRS and reduce systematic activity bias. It cannot
reliably restore precise locations, ignition times, or every small fire invisible
to MODIS. VIIRS is an imperfect observation reference, not ground truth. Different
overpasses, clouds, smoke, canopy and acquisition geometry cause real disagreement.
No raw count multiplier, confidence weight or resolution ratio eliminates those
differences. A smaller pixel size does not prove every individual detection is
more accurate.

Publish original fine-detail observations separately from the common-grid historical
estimate. The primary unit considered here is **VIIRS-equivalent area-weighted
occupied cell-days**, not raw detections. A binary daily model estimates the expected
VIIRS occupancy of a coarse cell conditional on MODIS inputs. This is not a
probability that a wildfire exists.

Pre-VIIRS reconstructions assume the learned relationship transfers to earlier
conditions. Validate feature ranges, sensor collection changes and overpass drift;
report uncertainty and reject unsupported inputs rather than guarantee recovery.

## Candidate differentiators

- Context-dependent calibration rather than a constant count multiplier.
- Satellite-aware calibration rather than silently adding new platforms.
- A sensor-removal test: reconstruct an overlap-period calendar using MODIS alone,
  then test whether VIIRS-observed peaks, rankings and anomalies are preserved.
- Show where agreement is robust and where sensors disagree; preserve provenance.

These are potentially useful product contributions, not claims of a new scientific
algorithm. MODIS/VIIRS observation models and FRP fusion already have published work.

## Web deliverable ideas

Region/radius selection; dates, months and dragged week selections; historical
calendar; seasonal baselines and anomalies; current-vs-historical matched periods;
accessible sequential color ramps with numeric equivalents; observation maps;
raw-hotspot CSV, derived-calendar CSV, and downloadable charts. Distinguish activity
amount from anomalies and use a common color scale for comparable periods.

## Historical facts found during inspection

The published archive has 24 MODIS summers (2003-2026), 14 full overlapping
Suomi-NPP summers (2012-2025), and partial local 2026 VIIRS coverage (61 days).
The files retain date, cell center and MODIS/VIIRS detection counts. They do not
retain the richer attributes proposed for the eventual model. Counts and hashes
are recomputed by the experiment, not copied from potentially stale README figures.
