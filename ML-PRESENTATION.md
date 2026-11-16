# Ready-to-present 50.6M model

Double-click `run-ml-viewer.cmd`, or run it from PowerShell. Open
<http://127.0.0.1:8000/?mode=ml>. The ready-to-run site contains the complete
presentation; no training, inference, external images or API calls are required
by this mode. Optional web fonts fall back to system fonts offline.

For a static-only presentation:

```powershell
python -m http.server 8080 --directory ready-to-run
```

Open <http://localhost:8080/?mode=ml>. Keep the server open during the demo.

## Included data

The frozen **50,567,169-parameter** `wide_diverse_v1` MLP consumes 96 MODIS
features. The checkpoint is unchanged:
`80a8b0e65bba826868ccff828cff8b1e44f1000276d852e0ef8c4f7b4c747899`.
Its original training-only normalizer and validation-only policy are reused.

The input snapshot contains **1,608 prepared patches** and **22,174,168 eligible
query centers**:

- All 1,100 original prepared patches, including the original training,
  validation and 84-patch benchmark sets.
- All 245 completed first-expansion training patches.
- 263 additional, already admitted second-expansion patches available when
  this presentation snapshot started. The second expansion was still running;
  unfinished/unadmitted windows are not included. These patches were not used
  to fit the selected 50.6M model, but share training acquisitions and are not
  independent test incidents.

Every eligible query has a cached raw logit, adjusted score, coordinates and
MLP/MODIS/VIIRS flags, including background queries. The original saved CUDA BF16
validation/benchmark logits are reused exactly. Training and additional patches
were inferred with the completed checkpoint using CUDA BF16. No model was
retrained, no threshold was refitted, and no satellite data was downloaded.

The archive contains **8,040 rendered PNGs**: a four-panel comparison plus
individual MLP, MODIS, VIIRS and difference views for every patch. Each map uses
the same local geographic frame and latitude/longitude labels. Gray marks show
the actual eligible query locations; surrounding empty areas are unavailable.
The offline maps are query plots, without satellite imagery as a background.

The season calendar's 2003–2026 aggregate cell-day archive does not contain the
native MODIS feature arrays required by this model. It remains a separate
statistical analysis, rather than being presented as neural-model inference.

## Suggested demonstration

1. Open MLP research and introduce the completed 50.6M model and full data counts.
2. Leave **Benchmark** selected. The default patch has the most VIIRS positives.
3. Select **50.6M MLP**, **MODIS**, **VIIRS reference**, and **Difference** to
   compare identical geographic frames. **Compare all** shows all four together.
4. Choose **Present scenes**, then **Play scene tour**. The tour waits for an
   image to load before advancing every 6.5 seconds. Arrow keys or the Previous
   and Next buttons move through every filtered scene; Escape exits the tour.
5. Read the frozen benchmark and training chart below the explorer.
6. Explore Training, Validation and Additional windows with their visible scope
   labels. These are not new test evidence.
7. Download an image for slides, GeoJSON for GIS, all query scores as NPZ, or the
   complete scene-summary CSV. The full rendered atlas lists every patch.

The benchmark remains **62 TP / 46 FP / 33 FN**, **57.41% precision**,
**65.26% recall**, and **0.6108 F1**. MODIS has 44.97% precision and 80.00% recall.
The tradeoff is visible in the comparison table. These are previously examined
2024 patches, with VIIRS reference labels rather than independent field truth;
query labels are not counts of independent fires. Scores are not guaranteed
real-world fire probabilities, and the model does not create finer measured
MODIS thermal resolution or establish operational validation.

## Files and reproducibility

- `frontend/public/data/ml-presentation/catalog.json`: model hashes, snapshot
  manifests, per-scene metadata, split counts, benchmark and asset hashes.
- `frontend/public/data/ml-presentation/scenes/<id>/`: all five images,
  `scene.json`, alert-union `predictions.geojson`, and complete `queries.npz`.
- `frontend/public/data/ml-presentation/gallery.html`: standalone rendered atlas.
- `frontend/public/data/ml-presentation/scene-summary.csv`: every scene's counts.
- `backend/data/ml_presentation/inputs.json`: immutable prepared-input snapshot.
- `data-visualization/ml-presentation/verification.json`: independent checks of
  every asset, score, exported feature, image and benchmark count.
- `data-visualization/ml-presentation/browser-checks.json`: browser checks and
  screenshots for desktop, presentation and mobile layouts.

Rebuild with the existing CUDA environment from the workspace root:

```powershell
.\fire-harmonization-research-fullres\.venv\Scripts\python.exe backend\scripts\prepare_ml_presentation.py
.\fire-harmonization-research-fullres\.venv\Scripts\python.exe backend\scripts\validate_ml_presentation.py
cd frontend
npm.cmd test
npm.cmd run lint
npm.cmd run build
cd ..
.\backend\.venv\Scripts\python.exe backend\scripts\publish_frontend.py
cd frontend
npm.cmd run test:presentation
npm.cmd run test:browser
```

The builder verifies frozen source tile and feature hashes and resumes from
per-scene receipts. `--refresh-snapshot` explicitly captures newly admitted
prepared patches; it does not start the unfinished model experiment. The
completed presentation never reads a growing manifest in the browser.

The original contextual MLP's globe and `/api/ml/*` routes retain their original
model and 84-patch cache. The 50.6M presentation loads its separately versioned
static archive at `/data/ml-presentation/`. Model identities are not mixed.
