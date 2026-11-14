# Larger NASA archives and the next training experiment

Assessed 2026-11-14. Documentation was accessed through web research; direct
SFTP failed before authentication because this runtime cannot resolve its host.
Remote inventory, file sizes and actual record counts remain unverified.

## Sources

| Source | Scope and role | Access and verification |
|---|---|---|
| NASA FIRMS bulk archive | Global MODIS C6.1 since November 2000; VIIRS S-NPP since January 2012; NOAA-20 since April 2018; NOAA-21 since March 2023. Individual detections in CSV/JSON/shapefile. | Earthdata or email authentication. Portal documentation verified; no authenticated export requested. Prefer standard science-quality observations for historical fitting; recent NRT is a separate processing tier. |
| MCD14ML monthly locations | Global Terra + Aqua MODIS detections, with date, UTC time, satellite, brightness, FRP, confidence, hotspot type, day/night and sample number. | NASA/USGS guide §4.3 documents public SFTP `fuoco.geog.umd.edu`, user `fire`, public password `burnt`, `data/MODIS/C6/MCD14ML`. Guide examples are collection **006**, not evidence of current **061** inventory. Listing and collection verification are required. |
| VNP14IMGML monthly locations | Global S-NPP VIIRS 375 m detections. C2 version 03 adds rich detection/background fields, cloud adjacency, viewing geometry, type and pixel area. A monthly container retains individual detection dates/times. | NASA guide §4.2 documents the same public login and `data/VIIRS/C2/VNP14IMGML`. Its example filenames use `C2`; naming text also says `02`. Reader accepts those equivalent C2 tokens, never C1. Current file inventory unverified. |
| MOD14/MYD14 and VNP14IMG Level 2 | Fire masks, QA and observation context; needed to distinguish observed non-fire, cloud, unprocessed and missing coverage. | NASA LAADS/LP DAAC route; separate authenticated acquisition may be needed. These are more demanding than location CSVs but scientifically valuable for detection modeling. |

NASA's monthly products are distributed by the instrument science team at UMD;
they are not a different fire-label provider. If the requirement is strictly a
NASA-hosted download endpoint, use FIRMS or LAADS instead of the UMD host.

### Primary documentation

- [NASA FIRMS archive portal](https://firms.modaps.eosdis.nasa.gov/download/): export formats, processing lag, temporal coverage and authentication.
- [MODIS C6/C6.1 guide, May 2021, §§4.3 and 5.5](https://lpdaac.usgs.gov/documents/1005/MOD14_User_Guide_V61.pdf): public monthly-file access and 12-field schema.
- [NASA-hosted VIIRS C2 guide v1.2, June 2025, §§3.2 and 4.2](https://ladsweb.modaps.eosdis.nasa.gov/archive/Document%20Archive/Science%20Data%20Product%20Documentation/VIIRS_C2_AF-375m_User_Guide_1.2.pdf): monthly data, C2 version 03, attributes and public SFTP access.
- [NASA LAADS VNP14IMG archive](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/): Level-2 counterpart linked in the guide.

## Size and usefulness

The local input is 14 overlapping boreal summers on a 0.25-degree grid. The
proposed new archive covers all climates and all months globally. The first
download is 2019 and 2023: 24 months × two products = **48 planned files**. The
full 2013–2024 design is 144 months × two products = **288 planned files**,
excluding optional NOAA-20 continuity checks and coverage granules. Those are
planned sensor-month counts, not measured downloaded sizes or training samples.
The product guide warns that converting a VIIRS monthly CSV to a shapefile can
exceed its 2 GB limit, so preserve compressed source files and stream the data.

Why this is better: retain exact coordinates and acquisition time instead of
discarding them, test wet/dry seasons and agricultural/savanna/forest regimes,
and use FRP, confidence, satellite and geometry when constructing the pairs.
MODIS sample number is available; MODIS pixel footprint/scan/track and full
clear-sky exposure cannot be assumed to exist in its monthly text file.

## Training design, before opening new model scores

1. **Target:** VIIRS-equivalent recorded activity, with separate observed and
   estimated columns. This is harmonization, not future ignition forecasting.
   Begin with common equal-area 10 km weekly cells; assess 5 km only after the
   coverage/matching audit. These are modeling choices, not claims that native
   MODIS resolves small fires.
2. **Sources:** fix MODIS collection 061 and VIIRS C2 version 03 throughout the
   primary run. Use S-NPP as the target platform throughout; keep NOAA-20 as a
   separate continuity test instead of increasing counts by stacking sensors.
   Retain original rows and manifest processing tiers and versions. No silent
   C6 fallback or mixing standard and NRT labels.
3. **Pairing and coverage:** separate Aqua/Terra and day/night. For an initial
   Aqua–S-NPP comparison, assess near-synchronous acquisition windows with a
   predeclared 30-minute matching rule and view-angle sensitivity. Do not assign
   unobserved cells a true negative. Acquire observation masks or explicitly
   limit the estimand to recorded counts with missing-coverage sensitivity.
   Check repeat observations and overlapping granules before aggregation.
4. **Model:** compare power-law calibration and the existing bounded spatial
   response against gradient-boosted trees predicting nonnegative activity from
   MODIS-only counts, FRP/brightness summaries, confidence, acquisition context,
   and multiscale spatial context. VIIRS-derived predictors are forbidden at
   inference. Do not start with a super-resolution neural network: these data
   do not justify unique 375 m historical reconstructions.
5. **Splits:** train 2013–2019, select 2020–2021, test 2022–2024. Hold out entire
   broad geographic blocks in a separate transfer experiment with buffers around
   boundaries and events. Fit masks, normalization and calibration only on each
   training partition. Audit fire-event spillover and resample uncertainty by
   year/region/event, not independent rows. Holdout scores are valid only once
   after choices are frozen; the 2019/2023 pilot is an ingestion/coverage test.
6. **Report:** MAE, RMSE, WAPE, signed bias, seasonal totals and peak timing;
   stratify by region, season, source platform, day/night and activity level.
   Probability metrics require a separate exposure-aware occupancy target.
   Compare on identical masks/windows against raw MODIS and simple calibration.
   Recreate totals, weekly curves, calendars, shared-scale maps and regional
   error plots only after new predictions exist.

## Access result and honest boundary

`sftp -oBatchMode=yes -oConnectTimeout=10 fire@fuoco.geog.umd.edu`
returned exit 255 with `Could not resolve hostname`. DNS checks also failed for
NASA FIRMS, LAADS and PyPI; it is not evidence that the remote archive is down.
The web research tool could read the official guides but does not supply a
general SFTP download facility. No SSH verification was disabled and no account
or email action was performed. No larger archive has been downloaded, no fresh
training has run, and no new before/after figure is presented as measured output.
