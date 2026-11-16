# Sparse globe verification

The compiled site was tested with `npx serve` from `ready-to-run`, with the Python
backend stopped and API/external requests blocked.

- [Canada overview](canada-overview.png)
- [Regional 3D tilt](regional-3d.png)
- [Mobile layout](mobile.png)
- [Downloaded overview GeoJSON](overview.geojson)
- [Browser check receipt](browser-checks.json)

Seven browser checks passed with zero runtime errors and zero API requests. The
preparation receipt at `backend/data/ml_overview/verification.json` reconciles
22,174,168 source samples across 1,608 patches, reduced to 9,460 display locations.
The JSON bundle is 738,017 bytes. Source NPZ hashes, model versions, and source
counts were verified. Training, inference, and satellite downloads were not run.

See [ML-OVERVIEW.md](../../ML-OVERVIEW.md) for launch instructions and the precise
meaning of the coarse display cells and historical composite.
