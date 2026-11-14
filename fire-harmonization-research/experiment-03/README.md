# Experiment 03 — global archive expansion

**Status: archive located; acquisition blocked by local DNS; larger-data training
has not run.** No new observations or model results are claimed. Nothing is
connected to the main app.

The best expansion is global, year-round **individual fire detections**, rather
than more samples of the existing Canadian June–August grid. NASA's guides
document monthly `MCD14ML` (MODIS) and `VNP14IMGML` (S-NPP VIIRS) files on the
University of Maryland's public SFTP server. A FIRMS API key is not needed for
that documented route. Its current remote inventory could not be inspected here.

See [archive assessment and training design](LARGER_ARCHIVE.md),
[source/access evidence](results/access_status.json), and the
[planned monthly inventory](acquisition/planned_months.csv).

## Acquire the data

1. From the project root, run `python3 fire-harmonization-research/experiment-03/src/archive_tools.py prepare`.
   It creates local input directories and interactive SFTP command lists. It
   does not connect or download. These lists contain download patterns, not
   verified remote filenames.
2. On a machine with network access, run `sftp fire@fuoco.geog.umd.edu`.
   The publicly documented password is `burnt`. Keep normal SSH host verification.
3. Paste the commands from `acquisition/inspect.sftp` first. Confirm which MODIS
   collection and product versions are actually present. The 2021 guide documents
   `data/MODIS/C6/MCD14ML` with **006** example filenames; it does not verify that
   **061** files are currently in that directory. Do not substitute collections.
4. The generated `pilot.sftp` requests 2019 and 2023 globally (48 sensor-months).
   `full.sftp` requests all months of 2013–2024 (288 sensor-months). Both request
   MODIS **061**, VIIRS **C2/02**, and within-collection version **03**. If the
   MODIS 061 files are elsewhere, verify their directory before changing the
   request. If 061 is unavailable, use the NASA FIRMS archive alternative below.
5. After actual downloads, run:

   ```sh
   python3 fire-harmonization-research/experiment-03/src/archive_tools.py audit --stage pilot
   # After the full acquisition:
   python3 fire-harmonization-research/experiment-03/src/archive_tools.py audit --stage full
   ```

The audit streams gzip files, checks collection/version, essential schema,
coordinates, date/month, time, confidence and type, records row totals, daily
counts and SHA-256 hashes, and requires exactly one file per requested
sensor-month. Synthetic tests check the reader, not NASA service availability.
The audit is an ingestion check: it cannot establish clear-sky coverage, complete
upstream processing, absence of duplicates, or model readiness on its own.

An alternative with explicit MODIS C6.1 coverage is the
[NASA FIRMS bulk archive](https://firms.modaps.eosdis.nasa.gov/download/).
It supports CSV exports after Earthdata/email authentication. No email was sent
or account created. FIRMS CSV has a different schema from these monthly files;
this reader intentionally does not silently ingest it as MCD14ML.

## What remains

Acquire and audit real files, establish observation coverage and deduplication,
build paired features, then train and evaluate the model in this folder. The
previous model remains unchanged. The current evidence supports choosing a
larger data source; it does **not** demonstrate global model generalization.
