# NASA data availability assessment

Public documentation reviewed on 2026-11-14. Authenticated status is recorded by
`results/nasa_access_check.json`; documentation access is not a successful API probe.

## Sources and coverage

NASA FIRMS distributes MODIS and VIIRS active-fire observations, including VIIRS
from NOAA satellites. Its documented archive starts in November 2000 for MODIS,
January 2012 for Suomi NPP, April 2018 for NOAA-20 and March 2023 for NOAA-21.
Suomi NPP delivery ended November 2, 2026. Archive bounds combine processing tiers;
query current availability before selecting standard-processing training windows.

Use SP archives for retrospective training. NRT is useful for recent observations
but must be tested against SP before sharing a calibration model. Recent SP data
has a multi-month lag, so a documented archive end does not prove finalized SP
coverage for every date or a complete swath inventory.

## API and limits

The FIRMS Area API supports a bounding box, source and historical start date, with
1-5 days per request. A free `MAP_KEY` is required. The published allowance is 5,000
transactions per ten-minute interval; large responses consume multiple transactions.
Cache backend downloads and keep keys outside browser artifacts and logs. The
data-availability API reports source-specific min/max dates.

Fourteen June-August seasons × two sources × ceil(92/5) = **532 requests** before
retries. This is a request estimate, not a transaction or runtime guarantee.

Documented hotspot fields include coordinates, UTC acquisition date/time, satellite,
confidence, brightness, FRP, scan/track dimensions, version and day/night. Confidence
representations differ by product and must not be treated as interchangeable
probabilities. The local reduced archive lacks most of these fields; do not fill
them with fabricated numbers to claim an enriched-model test.

## Coverage information and claims

FIRMS hotspot CSVs describe detected thermal anomalies, not all observation
opportunities. Zero rows do not establish clear skies. NASA MODIS Level-2 fire masks
distinguish fire, non-fire, cloud, unknown and unprocessed pixels. VIIRS Level-2
products provide the corresponding fuller masks and metadata. LAADS exposes archive
APIs and downloads; authenticated Level-2 access is separate from FIRMS's map key.

These products can support an observation-aware extension, but their ingestion and
alignment have not been implemented or tested here. No burned-area product is used
as a substitute for active-fire labels.

## Inspectable primary references

- [FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)
- [FIRMS source availability](https://firms.modaps.eosdis.nasa.gov/api/data_availability/)
- [FIRMS archive coverage and processing notes](https://firms.modaps.eosdis.nasa.gov/download/)
- [NASA API tutorial with actual response schemas](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html)
- [NASA fire detection limitations](https://firms.modaps.eosdis.nasa.gov/alerts/notes.php)
- [Suomi NPP transition notice](https://firms2.modaps.eosdis.nasa.gov/content/notifications/firms2/news.html)
- [MODIS Collection 6/6.1 guide](https://www.earthdata.nasa.gov/s3fs-public/2023-09/MODIS_C6_C6.1_Fire_User_Guide_1.0.pdf)
- [VIIRS Collection 2 guide](https://www.earthdata.nasa.gov/s3fs-public/2025-06/VIIRS_C2_AF-375m_User_Guide_1.2.pdf)
- [LAADS archive API](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/api-v2/specs/)
- [FRP intercomparison research](https://doi.org/10.1029/2017JD027823)
- [Observation operators and detection limits](https://gmd.copernicus.org/articles/19/10123/2026/)
- [Challenge resource link supplied by the user](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/?tab=resources)

The challenge page could not be retrieved during this work. Its intended goal was
supplied in the conversation. No successful authenticated live API read is claimed
without the probe recording one.
