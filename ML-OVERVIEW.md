# Sparse globe from the static site

From the project root:

```sh
cd ready-to-run
npx serve
```

In PowerShell on this computer, use `npx.cmd serve` if script execution blocks
`npx.ps1`. Use the Local URL printed by serve (normally port 3000), then open:

<http://localhost:3000/?mode=ml>

The app's **MLP research** mode opens **Canada overview / 3D** by default. It
works entirely from the built files: no Python server, model checkpoint, GPU,
live inference, or external imagery is required. The Calendar, Scenarios,
pre-rendered model presentation, and original MLP inspection view remain available.

Select a date, dataset, region, and MLP/MODIS/VIIRS/difference layer. Scroll to
zoom, drag to rotate, and click a dot to inspect its cached samples. Fit Canada
returns to the national view; Fit samples frames the current selection. The 3D
button tilts toward the horizon at regional zoom. An accessible table provides
the same selection without map picking. Download overview GeoJSON exports the
selected alert cells, filters, dates, model version, and sample-count semantics.
The URL preserves the filters.

## Data and interpretation

The frozen 50,567,169-parameter `wide-mlp-80a8b0e65bba` model's existing scene cache
supplies 22,174,168 eligible query samples across 1,608 patches and 20 dates in
2021–2024. This task performs no training, inference, or satellite download.

The overview aggregates those samples into **9,460 unique 0.25° display cells**.
The bundled JSON is 738,017 bytes. Each point is the display-cell center, not an
individual query location or a measured fire footprint. The complete source
queries remain available in the scene atlas.

Counts are sums of cached query samples; overlapping patches can repeat samples.
They are not counts of independent fires. Compare indicates which methods have
positive samples in a cell. Difference retains cells with any original query-level
MLP/MODIS disagreement, even if both methods also detect other queries in that cell.
VIIRS remains the reference, with its own layer.

The default is explicitly a historical composite across different days and years.
It includes training scenes, so it is not a new model evaluation. Choose a date
and dataset to narrow the view. The camera covers Canada and surrounding regions;
the model data covers only the existing sampled patches. Unmarked places are
unavailable, not evidence that no fire occurred. Aggregation does not improve the
measured spatial resolution of MODIS or establish continuous Canadian coverage.

## Rebuild and verify

The preparation script verifies each source NPZ checksum and model version,
checks per-scene counts, and reconciles every source query and alert total:

```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\prepare_ml_overview.py
cd frontend
npm.cmd run build
cd ..
.\backend\.venv\Scripts\python.exe backend\scripts\publish_frontend.py
```

The script writes `frontend/public/data/ml-overview.json` and a receipt in
`backend/data/ml_overview/verification.json`. Publishing includes the JSON in
`ready-to-run/data/` along with the application build.

With `npx serve` running on port 3000:

```powershell
cd frontend
npm.cmd run test:overview
```

Set `EPOCH_VIEWER_URL` for a different server URL. Browser verification blocks
the API and external network and checks rendering, all layers, dataset/date/region
filters, empty-state handling, links and reload, selection, GeoJSON, 3D tilt and
return, mobile layout, mode navigation, and a recoverable missing-data error.
Screenshots and the receipt are in `data-visualization/ml-overview/`.
