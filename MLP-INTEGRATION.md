# Running the PLSWORK MLP viewer

From this folder in PowerShell:

```powershell
.\run-ml-viewer.cmd
```

Or double-click `run-ml-viewer.cmd`. It opens <http://127.0.0.1:8000/?mode=ml>.
Keep the server window open; Ctrl+C stops it. The environment is installed on this
computer. The compiled frontend and API run together on localhost; Node and a GPU
are not required to view the results. If port 8000 is occupied, use
`.\run-ml-viewer.cmd --port 8001`. The launcher will not stop other services.

Select a date and scene patch, then Compare, MLP, MODIS, VIIRS reference, or
MLP / MODIS difference. Click an alert, or open the accessible prediction table and
press Inspect. Download GeoJSON exports the selected layer with coordinates,
scores, sensor flags, observation time, model version, and coverage limitations.
The URL preserves the selected scene, layer, and coverage toggle.

The Calendar and Scenarios buttons open the existing app modes. Their statistics
and simulation continue to use their existing data.

# What is integrated

This is the **cached research viewer**: the frozen PLSWORK contextual MLP's actual
saved predictions, not synthetic demonstration points. No new satellite data was
downloaded and no checkpoint was retrained or changed.

- Model version: `plswork-mlp-35535575bd95`; 96 features, 8,321 parameters.
- 84 patches across June 5, 8, and 13, 2024.
- 585,464 eligible, reference-matched query centers, including 95 VIIRS positive labels.
- Saved CUDA BF16 logits and the existing operating policy are preserved. The
  displayed score is sigmoid(logit + saved offset); the alert decision uses the
  saved raw-logit threshold, before display rounding.

| Existing benchmark | Precision | Recall | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLSWORK MLP | 49.30% | 73.68% | 70 | 72 | 25 |
| Original MODIS | 44.97% | 80.00% | 76 | 93 | 19 |

These previously examined patches are not a new independent test. VIIRS provides
the reference labels, not independent field truth. The viewer does not establish
improvement on arbitrary new scenes, create finer measured MODIS resolution, or
provide live monitoring. Scores are not guaranteed real-world probabilities.
Gray coverage marks show where cached reference-eligible queries exist; empty
surrounding areas are unavailable. Exported alert features are the union of MLP
alerts, MODIS detections, and VIIRS reference positives within that coverage.
The coverage payload includes all eligible query centers, not only positives.

# Backend and data flow

`backend/scripts/export_mlp_cache.py` reads only existing files from
`fire-harmonization-PLSWORK`. It verifies the checkpoint and source tile hashes,
checks saved prediction ordering, reconciles targets and MODIS flags with source
tiles, attaches their real geolocation, and verifies the benchmark totals.
It writes identical versioned caches to:

- `backend/data/ml_cache/` — canonical API cache.
- `frontend/public/data/ml/` — bundled static snapshot.

Each catalogue records checkpoint, operating-policy, prediction-file, and scene
hashes. The backend verifies all cached scene and coverage files at startup.
Missing or corrupt data returns 503, unknown scenes return 404, and invalid date
or layer parameters return 422. It never starts training or imports PyTorch in a
request. `EPOCH_ML_CACHE_DIR` can select another exported cache.

The single-server entry point `backend/viewer.py` serves the compiled site and
mounts the existing API under `/api`. Its routes are:

| Route | Response |
| --- | --- |
| `/viewer-health` | Viewer identity, cache readiness, version, patch count |
| `/api/ml/model` | Frozen model, policy, provenance, benchmark, limitations |
| `/api/ml/scenes?date=2024-06-08` | Available scene catalogue, optionally filtered |
| `/api/ml/scenes/{id}` | Scene metadata and counts |
| `/api/ml/scenes/{id}/predictions?layer=mlp` | GeoJSON; all, mlp, modis, viirs, difference |
| `/api/ml/scenes/{id}/coverage` | GeoJSON MultiPoint of eligible query centers |

`uvicorn main:app --app-dir backend` still exposes the original routes and `/ml/*`
without the `/api` prefix. The existing `/sample`, `/history/summary`, and root
API response are preserved. API documentation is at `/api/docs` in the viewer.

If the API is unavailable, the frontend explicitly labels its source
**Bundled snapshot · API unavailable**. It uses that snapshot for the entire scene;
it does not mix API and snapshot scene data. If neither source loads, the view
shows Scene unavailable and a Retry control.

# Setup on another computer

Python 3.12 is used on this computer. From the project root:

```powershell
py -3.12 -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\run-ml-viewer.cmd
```

The included cache and `ready-to-run` site are sufficient. The research model
folders are required only to regenerate the cache. A static-only alternative is
`python -m http.server 8080 --directory ready-to-run`, then
<http://localhost:8080/?mode=ml>; it displays the labelled bundled snapshot.

# Development and verification

To update the source UI, use Node 22.12+ (22.22.3 was tested):

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Vite proxies `/api/ml` to `http://127.0.0.1:8000/ml`. Start the API separately with
`backend\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend` from
the root when developing. `frontend/.env.example` documents the optional API URL.

After a UI change, rebuild and refresh the runnable site:

```powershell
cd frontend
npm.cmd run build
cd ..
.\backend\.venv\Scripts\python.exe backend\scripts\publish_frontend.py
```

The publisher verifies every copied file and retains existing unrelated assets.
Reload the browser after publishing. To regenerate both prediction caches from
the existing PLSWORK files:

```powershell
.\backend\.venv\Scripts\python.exe backend\scripts\export_mlp_cache.py
```

Restart the server after changing its cache. To run the checks:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
.\backend\.venv\Scripts\python.exe -m unittest discover -s backend\tests -p "test_*.py" -v
cd frontend
npm.cmd test
npm.cmd run lint
npm.cmd run build
npm.cmd run test:browser
```

Browser checks require the local viewer running on port 8000 and Playwright's
Chromium, already available here. On another machine, install it with
`npx.cmd playwright install chromium`. Set `EPOCH_VIEWER_URL` for another port.
The browser checks cover real API data, layers, query inspection, GeoJSON download,
dates, links, an empty patch, mobile width, mode navigation, API outage, and Retry.
Screenshots and the check receipt are in `data-visualization/mlp-integration/`.
