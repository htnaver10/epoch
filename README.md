# EPOCH

The integrated PLSWORK MLP research viewer is available with
`run-ml-viewer.cmd`, then <http://127.0.0.1:8000/?mode=ml>. It compares the frozen
model's cached predictions with MODIS and VIIRS reference observations across
84 processed patches. See [MLP-INTEGRATION.md](MLP-INTEGRATION.md) for setup,
API routes, data provenance, and verification. These scene predictions are
separate from the historical season analysis and response simulation below.

Satellite fire counts over Canada have climbed since 2012. Part of that rise is real, and part of it is that NASA's newer VIIRS instrument sees fires the older MODIS instruments miss. EPOCH separates the two. It ranks every fire season against MODIS's own record, which has used the same two instruments since the early 2000s, so a sharper satellite can't make a season look worse than it was.

It covers June to August, 2003 to 2026, over Canada between 50°N and 70°N, using NASA FIRMS hotspot archives (about 2.4 million detections).

## What it finds

| Season | Raw VIIRS vs. old MODIS record | Same sensor, same grid, same window | Rank of 24 |
|---|---|---|---|
| 2023 | 53× | 4.0× | 1 |
| 2025 | 33× | 2.6× | 2 |
| 2024 | 31× | 2.2× | 3 |
| 2016 | 2.0× | 0.36× | 23 |
| 2020 | 0.35× | 0.17× | 24 |

Comparing raw counts across instruments overstates a season by roughly ten times. Compared like for like, 2023 is still the record by a wide margin, and VIIRS ranks it first on its own record too.

The same split shows up in the long-run trend. On the consistent MODIS record, activity has risen about 54% per decade (Theil–Sen slope, Mann–Kendall p = 0.035). Splice raw VIIRS counts onto the older MODIS record and the same years appear to rise 273% per decade.

The results line up with known seasons: Québec and the Northwest Territories in 2023, Manitoba and Saskatchewan in 2025 (central sector, ranked first), Eastmain in 2013, and a very quiet 2020. 2016 ranks low because the Fort McMurray fire burned in May, before the June window.

## How it works

1. Hotspots are binned into a 0.25° grid by UTC date. Each sensor counts a cell at most once per day, which removes repeat detections of the same fire.
2. Each cell-day is weighted by cos(latitude), scaled to 1 at 60°N. Cells are equal in degrees, not kilometres: one at 70°N covers about half the ground of one at 50°N, so plain counts overstate northern fires.
3. A season's MODIS total is ranked against every other MODIS season. The top 10% is reported as unusually high, the bottom 10% as unusually quiet. A robust z-score on the log of the total says how far out it sits.
4. VIIRS is checked independently against its own 2012 to 2026 record. If the two sensors put a season on opposite sides of their medians, the app says they disagree instead of picking one.
5. To show VIIRS on the MODIS scale, EPOCH fits a power law, ln(MODIS) = a + b·ln(VIIRS), on the seasons both satellites flew. VIIRS's edge is widest in quiet seasons and narrows in extreme ones, so a single ratio misfits both ends. The fit is always redone without the season being judged. Held out this way the median error is 5.4%, against 9.3% for a constant ratio (worst season 21% against 39%).
6. **Static heat sources are left out.** Thirty grid cells are hot almost every day of almost every summer whatever the fire season does: the oil sands upgraders north of Fort McMurray, the refineries near Regina and Edmonton, the cement plant at Exshaw, and other industrial sites in Alberta, Saskatchewan, Manitoba and northeastern British Columbia. VIIRS sees these small, steady sources far more often than MODIS (558 days against 1 at Exshaw), so left in they inflate VIIRS against MODIS, most of all in quiet seasons and near those towns. A cell counts as static when VIIRS saw it in at least 80% of its full seasons, on a median of at least 5 days of each; a wildfire cell burns in one or a few seasons and a farm burn on a day or two a year, so neither qualifies. The list is in `seasons.json` (`static_sources`) and is applied to every sector, every area of interest and the map. FIRMS flags static sources itself (`type` 2), but the archive aggregates here do not carry that field, so EPOCH finds them from the record.
7. **Archive gaps are filled.** On ten days the VIIRS archive is empty or nearly so while MODIS sees a normal day (26 July 2023 has no VIIRS detections at all against 170 MODIS cells; four days in July 2024, three in July 2025, two in July 2026). These are missing data, not quiet days, and they pulled recent VIIRS totals down. A day is a gap when one sensor falls under 15% of its own median for the three days either side while the other sensor holds at least half of its own. Each gap day is filled with what the other sensor saw that day, scaled by the ratio between the two over the neighbouring days; the list is in `seasons.json` (`gaps`). One sensor alone cannot tell a missing day from a cloudy one, so before VIIRS nothing is filled.
8. A season still being archived, or a sensor whose archive stops early, is compared over the same calendar days of every other year, never against full seasons.

What it does not correct: cloud cover, overpass timing, day/night sampling, and Terra's orbit drift after 2020. Leaving out the static cells also leaves out any real fire in those thirty cells.

**What the two corrections changed.** Held-out bridge error, median and worst season, before and after, on the app's own code:

| Sector, period | Before | After |
|---|---|---|
| Canada, Jun–Aug | 5.7% / 31% | 5.4% / 21% |
| Canada, July | 14.1% / 49% | 8.6% / 30% |
| West, Jun–Aug | 7.4% / 45% | 5.1% / 18% |
| West, July | 11.1% / 75% | 9.4% / 30% |
| Central, Jun–Aug | 5.4% / 33% | 4.7% / 27% |
| East, Jun–Aug | 11.1% / 28% | 9.5% / 28% |

The exponent of the Canada-wide bridge falls from 1.26 to 1.15: part of what looked like VIIRS's extra reach in quiet seasons was industry. One place got slightly worse (central sector, June: 13.3% against 12.0%). Every season keeps its rank, and 2023 is still the record at 4.0×. The effect is largest for an area of interest near industry: within 150 km of Fort McMurray in 2020, VIIRS appeared to see 15 times what MODIS saw, and 6 times once the upgraders are out; the area's bridge exponent falls from 2.8 to 1.5, and Edmonton's from 1.9 to 1.05. Yellowknife, with no such sources, is unchanged. Percentiles describe this 24-season archive, not a probability. Hotspots are not individual fires or burned area.

## How it answers the challenge

EPOCH is an entry for the NASA Space Apps 2026 challenge "Harmonization of MODIS and VIIRS Hot Spots", which asks for a web application that harmonizes the two sensors' records into a burning-activity calendar for a selected area of interest.

| The challenge asks for | In EPOCH |
|---|---|
| The two records harmonized, "so the sensor of origin no longer matters" | One 0.25° grid, cell-days weighted by ground area, static industrial heat sources left out of both sensors, days missing from one archive filled from the other, MODIS as the consistent record and VIIRS translated onto its scale by a bridge with a held-out error figure. The uncorrected comparison is shown beside the corrected one. |
| A burning-activity calendar from the early 2000s to the present | The calendar docked on the right: every season from 2003, one column per week, with the usual week under it. |
| A selected area of interest | Click the globe, type a place or coordinates, and that circle becomes the area in scope. Its radius steps from 50 to 400 km. Every figure, the calendar and every panel re-run for it. |
| A selected date range | A month, or any span of weeks dragged across the calendar, becomes the period: the same days of every season are compared. |
| Which periods and which years burned most; how a season compares | The calendar, each season's index against the usual and its rank, and the season bars. |
| Anomalies against multi-year averages | The activity index (multiples of the median of the other seasons), percentile and robust z, for the area and period in scope. |
| Critical periods within the area of interest | The assessment card names the usual peak period for the area in scope and the selected season's busiest week against the usual for that week; the calendar marks the usual peak weeks under its headings. |
| Colour ramps, accessible | One ramp for amount, dark to light, tested to get steadily lighter; every cell's value is also readable as a number. |
| A CSV of the hotspots in a search | The cell-days of the area and period in scope, and the calendar itself. |

Not there yet: the data covers June to August and a 50–70°N, 140–60°W rectangle, so the calendar has three months and the south of the country is missing; areas are circles, not free-drawn shapes; charts do not export as images.

## Scenarios: response planning demo

A second mode of the same app (the `Scenarios` switch in the top bar, or `?mode=scenarios`). It takes one town and the fire near it and asks what to send.

- **Places.** Jasper (24 July 2024), Yellowknife (15 August 2023) and Hay River (13 August 2023): a 64 km square each, with real terrain, water, roads and buildings from OpenStreetMap, real hourly weather, and the fire seeded from the satellite hotspots up to the opening time.
- **Fire growth.** The Canadian Forest Fire Behavior Prediction System on a 160 m grid: rate of spread and intensity by fuel type, an elliptical spread with the wind, slope, hourly fine-fuel moisture, ember spotting. Run forward with no action, it brings the Jasper fire to the town about 17 hours after the scenario opens on the morning of 24 July; the real fire arrived that evening, about 11 hours after.
- **Crews and aircraft.** A simulated set of real aircraft types and ground crews at real bases. Aircraft are small 3D models that bank and climb; they taxi out and take off one at a time, fly Dubins tracks (no turn tighter than their speed and a 28 degree bank allow), line up before a drop or a scoop, and land on the runway into the wind. Each keeps its own lookout: if two tracks converge, the one with less right of way turns a circle and falls in behind. Ground crews drive the road network. What they all do is illustrative, and the app says so.
- **The planner.** Select a fire and it lays out up to three plans: direct attack, a control line tied into lakes and roads (chosen by dynamic programming over bearings from the town), or point protection at the town edge. Each is scored by running copies of the world forward under an ensemble of winds, against doing nothing, and the one that loses least is recommended.
- **Plan first.** Before anything moves, the chosen plan is flown once on a copy of the world. The map shows the route every crew and aircraft would take to its first piece of work, with the drop runs and their times, and the board lists each one's first task. Dispatch sends only the crews and aircraft the plan was seen to use.
- **Ahead of the fire, not only on it.** The forecast is a tree of which cell lights which. For every cell not yet burned the planner counts how much burns downstream of it; aircraft wet the cells with the most downstream that the fire reaches inside the time the water lasts, and airtankers lay retardant across the same path further out. Where wetted fuel brings the edge under the limit, drops and crews then knock it down. Water only works for minutes (25 here, after a Canadian drop test), so it goes down right ahead of the fire; a hot edge takes several stacked loads, so the scoring is per load. On the fire nearest Jasper this turned direct attack from losing the town in one run of five to holding the fire at about 210 ha in all five.
- **Who does what.** The heavy skimmers and the airtankers carry the work. Helicopters, with a quarter of the load, take small work only: quiet edge, spot fires, edge a crew is about to mop up. With none of that to do they go back to base, and they always give way to fixed-wing aircraft.
- **Ignite.** Light a new fire anywhere and plan against it. A fresh start near town is held at one cell by direct attack; the same plan fails against a crown fire, and the scoring shows it.

The numbers the planner rests on (intensity limits for crews, machines and aircraft, line production rates, aircraft loads and speeds) are in `frontend/src/ops/engine/doctrine.js` with their sources; the ones that are assumptions are marked. Two adjustments to the weather are stated in the scenario's sources panel: a share of the gust is added to the reanalysis wind, and at Jasper the wind is turned along the valley.

It is a simulation for a demonstration, not an operational tool. The fuel mosaic inside mapped forest is modelled, hotspots are pixels hundreds of metres across, and nothing here has been validated against a managed incident.

## Using it

The app is one screen: a 3D globe with summary cards docked around it. It opens from orbit, spins in to the analysis area, and lands on the record season.

- **Top left, the season's assessment.** A one-sentence summary ("2023 saw 4.0× the burning of a typical season here, the highest of 24 seasons"), then the activity index (multiples of a median season), the uncorrected figure a comparison of raw counts would give, the sensor effect between them, robust z and percentile. Under them are the **critical periods** for the area in scope: when burning usually peaks (the weeks whose usual activity is within 75% of the busiest usual week) and the selected season's busiest week against the usual for that week. Last come the assessment and whether VIIRS's own record agrees.
- **Top right, the burning activity calendar.** Every season by week for the area in scope, each with its index and rank; the usual week sits under it, and orange ticks under the week headings mark the usual peak period. Click a row to select that season, order it by year or by rank, drag across the week headings to set a period, and download it as CSV.
- **The toolbar under the header** states exactly what is on screen (area, season, period, day, layer, view, satellite image, marked areas) and is where each is chosen.
- **Header.** What the tool is reading, the search box, and the five panel tabs.

Everything else is on request:

- **Click the globe.** The camera spins and zooms to the spot, and when it arrives a numbered report pops out beside the range ring. From far out a click takes the 150 km zone around it, named after the nearest town; up close it takes the hotspot cell, or 100 km of open ground. Clicking a hotspot works the way it does on NASA FIRMS: the camera goes in close over sharp imagery, a yellow ring marks the cell, and the report lists every day it burned with the MODIS and VIIRS detections for each (click a day to show it on the map). The archive holds counts per cell and day rather than each pixel's brightness or FRP, so a link opens FIRMS itself at that spot and day for the individual detections. The report is a table: active days, cells, first and last detection, and the two sensors side by side on the common grid, with raw detections listed under it and labelled uncorrected. The area of interest also becomes the area in scope: the assessment, the calendar and every panel switch to that circle's own history, and the − and + on the report change its radius. Every area stays on the globe as a numbered marker until you clear it.
- **Search** (press `/`). It takes a place (`fort mcmurray`, `jasper`, `calgary`), coordinates (`58.4 -112.1`), a season (`2023`), a day (`jul 14`, `peak`) or a sector (`west`). The search stays in the box, and the up arrow recalls earlier ones. A place outside the analysis area still opens, and its report says so.
- **Open a panel** with the numbered tabs or keys 1 to 5: the overview, all seasons from 2003 to 2026 and their trend, the season day by day, the two sensors, and the method.
- **Hold `R`** for the uncorrected view: every cell drawn by its raw detection count instead of the days it burned, which is the inflation the tool corrects.

The map is drawn like NASA FIRMS: true-colour imagery, thin grey coasts, borders and provincial lines, and each hotspot cell as a red dot that runs to orange and yellow the more days it burned. On the Diff layer the shape says which sensor saw a cell instead: a triangle for MODIS only, a ring for VIIRS only, an open diamond for both but on different days, and colour is the sensor (white for MODIS, violet for VIIRS). A cell's true 0.25° footprint is drawn when you click it.

The imagery works like Google Earth: the globe is cut into tiles, and as you zoom the tiles in view are replaced by finer ones until one image pixel is no larger than a screen pixel, so the map stays sharp from orbit down to about 16 km up. Tiles come from Sentinel-2 cloudless at 10 m, or NASA Blue Marble at 500 m if that service does not answer, over a bundled summer Earth that needs no network. A tile still loading is covered by its parent, and each new tile fades in over what was there, so the map never goes blank; images are decoded off the main thread so the globe keeps turning while they arrive. Scroll zooms toward the spot under the pointer, and right-drag (or the 3D button) tilts the view toward the horizon.

The bar along the bottom is the record on the left (every season, click to open) and the selected season's days on the right, scaled to the busiest day in the whole record so a quiet season looks quiet.

| Key | Does |
|---|---|
| `/` | Search |
| `←` `→` · `space` | Step a day · play the season once, or pause |
| `[` `]` | Previous or next season |
| `1`–`5` | Open a panel |
| `D` · `S` | Sensor-disagreement layer · that day's NASA satellite image |
| `R` (hold) | Uncorrected view |
| `C` | Clear area markers |
| `0` | Summary: the record season with the Overview panel open |
| `+` `-` | Make everything larger or smaller; it stops at what the window can show |
| `H` · `Esc` | Home · close whatever was opened last |

The interface is sized from the window, so it reads on a large display without zooming the browser. Every view, including the focused area, the view and the size, is in the URL.

The analysis area is a rectangle, 50–70°N by 140–60°W, not the national border: southern Ontario and Quebec, the Maritimes and Newfoundland are outside it. The app says so wherever it names the area.

```sh
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173
npm test           # analysis, geography and command-line tests
npm run build      # static site in frontend/dist
```

## Building on it

From `frontend/`: `npm install`, then `npm run dev` (the app), `npm test` (94 tests on the statistics and the scenario engine), `npm run lint`, `npm run build`. React 19, Vite, three.js; no backend at run time. Everything the app reads is in `frontend/public/data/`.

Where things are, under `frontend/src/`:

| File | What it holds |
|---|---|
| `Root.jsx` | The switch between the two modes. Scenarios are loaded only when opened. |
| `App.jsx`, `components/`, `lib/` | Analysis mode: the globe, the season statistics, the calendar, the panels. |
| `ops/Ops.jsx` | The Scenarios shell: loads a place, runs the clock, and owns the select, plan, dispatch flow. |
| `ops/panels.jsx`, `ops/ops.css` | The boards: fires, crews and aircraft, plan, the plan in progress, one crew or aircraft, log, sources. |
| `ops/tacscene.js` | The three.js 3D map: terrain, the fire shader, forecast lines, plans, labels, picking. |
| `ops/aircraft3d.js`, `ops/icons.js` | The 3D aircraft models, and the flat symbols used in tables and for ground units. |
| `ops/engine/fbp.js` | Fire behaviour: the Canadian FBP equations (spread rate, intensity, ellipse, fuel moisture). |
| `ops/engine/fire.js` | Fire growth on the grid, and what retardant, guard, water, mop-up and burning off do to it. |
| `ops/engine/theatre.js` | Map data turned into fuels, breaks, lakes to scoop on, and a road network to route on. |
| `ops/engine/forecast.js` | The projection of the fire, the route to the town, and the downstream weight of each cell. |
| `ops/engine/line.js` | The control line: least-effort line between town and fire, by dynamic programming. |
| `ops/engine/units.js` | Unit types and specifications, flight tracks (Dubins paths), and how a unit follows its legs. |
| `ops/engine/world.js` | The running world: what every crew and aircraft does, where drops go, separation between aircraft, plans in progress. |
| `ops/engine/planner.js` | The plans, their scoring over an ensemble, and the preview shown before dispatch. |
| `ops/engine/scenario.js` | Each place's opening time, crews and aircraft, and weather, and the seeding of the fire from hotspots. |
| `ops/engine/doctrine.js` | Every limit and rate the tactics rest on, with its source. Tune behaviour here first. |

The engine has no dependence on React or three.js and runs under Node, which is how the tests and the planner's copies use it.

Common changes:

- **A new place:** add it to `THEATRES` in `frontend/scripts/prepare_ops.py` and run the script, then give it an entry in `SCENARIOS` in `scenario.js` (opening time, fuel profile, roster).
- **A new unit type:** a line in `SPECS` in `units.js`, a model in `aircraft3d.js` or a designator in `icons.js`, and its behaviour in `think()` in `world.js`.
- **Different tactics:** thresholds and rates are in `doctrine.js`; where aircraft put their loads is `waterTarget()` and `aheadTarget()` in `world.js`; which plans are offered is `plan()` in `planner.js`.

## Rebuilding the data

The bundled data is already processed. To download it again you need a free FIRMS map key in `backend/.env` as `FIRMS_MAP_KEY=...`.

```sh
backend/.venv/bin/python backend/download_seasons.py    # resumable, waits out FIRMS rate limits
python3 frontend/scripts/prepare_seasons.py
python3 -m unittest discover -s frontend/scripts -p 'test_*.py'
```

To recompute the daily series from the cell files already in the repo (no key needed, for example after changing the weighting):

```sh
python3 frontend/scripts/prepare_seasons.py --rebuild
```

The download keeps only 0.25° cell-day totals (`data/seasons/`, not committed); the app reads `frontend/public/data/`.

The scenario places are built by `frontend/scripts/prepare_ops.py` (needs `osmium shapely numpy pyproj pyshp pillow`; downloads about 1.3 GB to a cache). It writes `frontend/public/data/ops/`.

## Sources

- Fire detections: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), MODIS Collection 6.1 and VIIRS 375 m standard processing.
- Imagery: [NASA GIBS](https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api) (Blue Marble: Next Generation, and the daily corrected reflectance true colour); close-ups from [Sentinel-2 cloudless](https://s2maps.eu) by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2020, CC BY-NC-SA 4.0); the bundled globe texture is the summer Earth from the three.js example textures, derived from NASA Blue Marble imagery.
- Coastlines, borders and provincial and state lines: [Natural Earth](https://www.naturalearthdata.com/) (public domain), via world-atlas and the 1:50m admin-1 lines.
- Built with React, Vite and three.js. Charts are hand-drawn SVG.
