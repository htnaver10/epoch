# EPOCH

For the sparse 3D model overview, run `npx serve` inside `ready-to-run` and open
<http://localhost:3000/?mode=ml&view=overview> (or the port printed by serve).
The globe and its cached data are bundled; a Python backend is not required.
See [ML-OVERVIEW.md](ML-OVERVIEW.md) for controls and coverage limits.

The **50.6M-parameter model presentation** is available with `run-ml-viewer.cmd`,
then <http://127.0.0.1:8000/?mode=ml>. All **1,608 processed patches**, **22,174,168
eligible query scores**, and **8,040 comparison/layer images** are prepared in
advance. The viewer includes a scene tour, the frozen benchmark, training history,
and PNG, GeoJSON, NPZ and CSV downloads. It works with the bundled static site and
does not need a GPU, API key, or inference server during a presentation.
See [ML-PRESENTATION.md](ML-PRESENTATION.md) for the demo sequence, scope and rebuild
instructions. The original 8,321-parameter MLP globe remains at
<http://127.0.0.1:8000/?mode=ml&view=inspect>; its API and provenance are documented
in [MLP-INTEGRATION.md](MLP-INTEGRATION.md). These scene predictions are separate
from the historical season analysis and response simulation below.

Satellite fire counts over Canada have climbed since 2012. Part of that rise is real, and part of it is that NASA's newer VIIRS instrument sees fires the older MODIS instruments miss. EPOCH separates the two. It ranks every fire season against MODIS's own record, which has used the same two instruments since the early 2000s, so a sharper satellite can't make a season look worse than it was.

The bundled archive covers **June 1–August 31, 2003–2026** for MODIS and **2012–2026** for Suomi NPP VIIRS. The study area is a rectangle spanning **50–70°N and 140–60°W**. It covers much of boreal Canada and extends across national borders.

## Run the demo

Serve the checked-in build with Python 3. The calendar and scenario data are bundled, and opening the app requires no API key.

```sh
git clone https://github.com/htnaver10/epoch.git
cd epoch/ready-to-run
python3 -m http.server 8080
```

Open the [burning activity calendar](http://localhost:8080/) or the [response scenarios](http://localhost:8080/?mode=scenarios). Use a desktop browser for the full layout. The optional satellite imagery and web font require internet access.

## Explore the burning activity calendar

| Challenge objective | What to try in EPOCH |
| --- | --- |
| Compare historical burning activity | Start with the weekly calendar. Each row is a season, with a typical week shown below the corresponding column. Select a year or sort the rows by rank. |
| Select an area and time range | Choose a preset sector or search for a place or coordinates, or click the globe to set a circular area of interest with a 50–400 km radius. Select a month or drag over week headings to compare the same dates across years. |
| Find anomalies and critical periods | Read the season assessment for the activity index, rank, percentile, and robust z-score. The calendar marks the area's usual peak weeks, and the assessment compares the selected season's busiest week with its historical baseline. |
| See the impact of sensor differences | Review the MODIS-scale assessment and the independent VIIRS check. Hold **R** to see the uncorrected view or press **D** to inspect the sensor-disagreement layer. |
| Read and export the results | The calendar uses a sequential color ramp and exposes numeric cell values. Download a calendar CSV with weekly activity and season summaries, or a hotspot CSV with one row per observed grid cell and date and separate MODIS and VIIRS detection counts. |

For a quick example, open **2023** in the default study area and compare the full June–August season. The app ranks it **first of 24 seasons**, with about **4.0×** the median activity of the other seasons on the matched MODIS record. Comparing raw VIIRS counts with the historical MODIS baseline gives about **53×** for the same season. The activity index measures relative hotspot activity on a common grid.

## How the calendar harmonizes the records

1. **Use common spatial and temporal units.** Bin detections by UTC date on a 0.25° grid. Each sensor contributes at most one occupied cell-day per grid cell and date. Weight cell-days by the cosine of latitude, normalized at 60°N, to account for the smaller area of northern grid cells.
2. **Exclude persistent heat sources and flag likely archive gaps.** The bundled data identifies 30 persistently hot cells and excludes them from both sensors' analysis. When one archive drops sharply and the other maintains activity, estimate the missing daily total from the other sensor using the ratio on surrounding days. The bundled archive identifies ten such VIIRS days. These estimates apply to the activity series.
3. **Use MODIS as the comparison scale.** The calendar uses MODIS where it has coverage and supports a VIIRS-derived estimate when only VIIRS is available. The log–log bridge is fitted on overlapping seasons, holding out the year being evaluated. For the full study area and summer window, the median held-out absolute percentage error is **5.4%** for the power-law bridge and **9.3%** for a constant-ratio bridge.
4. **Compare the same dates and show disagreement.** Season assessments account for partial archive coverage, compare activity with the median of comparable seasons, and check VIIRS against its own history. A strong high or low assessment can become **Sensors disagree** when VIIRS places the season on the opposite side of its median.

The calendar is limited to the bundled summer archive and study bounds. Custom areas are circular and the interactive interface exports CSV. Hotspot cell-days indicate where and when satellites detected active fire. They cannot establish the number of individual fires or the area burned. Cloud cover, overpass timing, day/night sampling, and Terra's orbit drift can affect the record. Masking persistent heat sources also removes any wildfire activity within those cells.

## Response scenarios

Open **Scenarios** for a separate response-planning demonstration using historical cases from Jasper, Yellowknife, and Hay River. Select a fire to compare simulated tactics, preview aircraft and ground-crew assignments, and dispatch a plan inside the simulation. The model uses terrain, weather, and satellite detections to initialize each case. Its output is illustrative and has not been validated for incident operations. See the [scenario definitions](frontend/src/ops/engine/scenario.js) and [tactical assumptions and sources](frontend/src/ops/engine/doctrine.js).

## Develop from source

Use Node.js **20.19+ on the 20.x line or 22.12+**, as specified by the locked Vite dependency.

```sh
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173/](http://localhost:5173/). From `frontend/`, run `npm test` for the JavaScript tests, `npm run lint` for linting, and `npm run build` to generate `frontend/dist/`. The React, Vite, and three.js frontend uses bundled data and can run without the Python API.

| Path | Contents |
| --- | --- |
| [`ready-to-run/`](ready-to-run/) | Checked-in static demo and its data |
| [`frontend/src/`](frontend/src/) | Globe UI, calendar, exports, and scenario simulator |
| [`frontend/src/lib/seasons.js`](frontend/src/lib/seasons.js) | Season comparison, overlap calibration, and weekly calendar calculations |
| [`frontend/public/data/`](frontend/public/data/) | Processed season summaries, per-year cell-days, and scenario assets |
| [`frontend/scripts/`](frontend/scripts/) | Data preparation and Python tests |
| [`backend/`](backend/) | FIRMS download and validation utilities and a FastAPI app |

### Rebuild the calendar data

From the repository root, recompute the summary from the checked-in cell-day files and run the Python preparation tests.

```sh
python3 frontend/scripts/prepare_seasons.py --rebuild
python3 -m unittest discover -s frontend/scripts -p 'test_*.py'
```

To refresh the source archive, put a FIRMS MAP_KEY in `backend/.env` as `FIRMS_MAP_KEY=...`. Run `python3 backend/download_seasons.py` to populate `data/seasons/`, then run `python3 frontend/scripts/prepare_seasons.py` to regenerate the frontend season files. The download is resumable. The [scenario preparation script](frontend/scripts/prepare_ops.py) builds the separate terrain, weather, and scenario assets.

## Harmonization research

The browser calendar uses the statistical method above. The research directories contain separate experiments on estimating VIIRS-referenced activity from MODIS and on native-resolution detection. Their benchmark figures describe those experiments.

- [Harmonization research](fire-harmonization-research/README.md) covers grid-level activity and calendar experiments.
- [Full-resolution research](fire-harmonization-research-fullres/README.md) covers swath processing and teacher–student experiments.
- [PLSWORK](fire-harmonization-PLSWORK/README.md) and [Hybrid V1](fire-harmonization-PLSWORK/hybrid_v1/README.md) cover contextual MLP, tree, and hybrid models.
- [YESSS](fire-harmonization-research-YESSS/README.md) and [SaturationAwareBackgroundV2](fire-harmonization-research-YESSS/saturation_aware_v2/README.md) cover background reconstruction and localization experiments.
- [Model comparison](data-visualization/COMPARISON.md) and [figure gallery](data-visualization/README.md) document the benchmark, its limitations, and the PNG and PDF exports.

## Data credits

- Active-fire detections come from NASA FIRMS standard-processing MODIS Collection 6.1 and Suomi NPP VIIRS 375 m products. [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- Globe and satellite imagery include NASA Blue Marble and GIBS. Imagery close-ups use EOX Sentinel-2 cloudless, which contains modified Copernicus Sentinel data 2020 (CC BY-NC-SA 4.0). [NASA GIBS](https://www.earthdata.nasa.gov/data/tools/gibs), [Sentinel-2 cloudless](https://s2maps.eu/)
- Geographic boundaries use Natural Earth. Scenario source credits are available in the app and scenario code. [Natural Earth](https://www.naturalearthdata.com/)
