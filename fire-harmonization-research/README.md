# Fire harmonization research — isolated from EPOCH

This folder contains the discussion record, revised technical plan and executable
preliminary experiments. **Nothing here is connected to the main app.** Existing
app data is read-only input. All generated files, models and dependencies stay here.

## Start here: current status and next model

- [Comprehensive research summary](RESEARCH_SUMMARY.md): completed experiments,
  measured results, saved artifacts, access failures, limitations and current status.
- [Full native-resolution tile-model plan](../fire-harmonization-research-fullres/PLAN.md):
  actual MODIS imagery at native band resolutions, aligned VIIRS supervision,
  multiresolution spatial modeling, location uncertainty and acceptance gates.
- [Full-resolution folder and RTX 5070 review](../fire-harmonization-research-fullres/README.md):
  all native-resolution contents and scripts now belong in that separate folder.
- [Experiment 04: larger model actually trained](experiment-04/README.md):
  200-tree NumPy boosting, all 14 local overlap summers, separate temporal and
  geographic evaluation, saved models and six figure families.

The latest requested next-model objective is **real MODIS image tiles supervised
by VIIRS**, with directly evaluated fire-location estimates. Existing models use
the reduced 0.25-degree count archive; no native-image model has been trained.
The summary and native-resolution plan are the current overview. Earlier protocols
and reports describe the state of their own experiments.

## Current objective

Learn **MODIS -> VIIRS-reference burning activity**, combining MODIS history with
VIIRS continuity in a comparable calendar. The completed experiments estimate
VIIRS-equivalent activity at the existing common grid. The next planned model
uses native imagery for detection and probabilistic fine-location estimation,
with measured spatial error and uncertainty; observations retain their provenance.

Earlier discussion proposed the opposite direction, VIIRS -> MODIS. That remains
a useful baseline and describes the existing app, but **is not this experiment's
target**. The latest direction supersedes it.

## Read the record

- [Experiment 03: larger global NASA archives and acquisition status](experiment-03/README.md): public monthly products, staged download plan and input audit; new training pending actual acquisition.
- [Experiment 02: rationale, results and repeated visualizations](experiment-02/README.md): bounded spatial calibration, validation-only selection, and rolling-year backtests.
- [Challenge brief and user decisions](docs/CHALLENGE_BRIEF.md).
- [Discussion and decisions](docs/IDEAS.md): all major ideas, alternatives and limitations.
- [Original preliminary experiment plan](docs/PLAN.md): historical features, targets,
  splits and promotion criteria; later experiments have their own protocols.
- [NASA data assessment](docs/NASA_DATA.md): APIs, fields, coverage and access gaps.
- [Future integration design](docs/INTEGRATION.md): a proposal only; no app changes.
- [Preliminary findings](results/REPORT.md): actual results and what was not tested.
- [Before/after visual evaluation](results/EVALUATION.md) and [figure gallery](visualizations/README.md): summer totals, weekly profiles, calendar, spatial comparison, and regional errors.
- [Experiment source](src/experiment.py) and [tests](tests/test_experiment.py).

## Reproduce the preliminary tests

NumPy is the only dependency required for the executed model experiments. The
local `.venv` uses the existing system NumPy. From the **project root**:

```sh
fire-harmonization-research/.venv/bin/python -m unittest discover -s fire-harmonization-research/tests -v
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/experiment.py --engine numpy
fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/probe_nasa.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/validate_results.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/summarize_results.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/visualize_results.py
```

Outputs include audit hashes, model coefficients, per-week/per-season predictions,
validation/test metrics, diagnostics and the app-isolation check. Models predict
**recorded VIIRS activity**, not true-fire probabilities. The raw input already
comes from NASA; no new NASA download is needed for these reduced-feature tests.

Figure generation additionally uses Matplotlib, available in the current research
environment. `requirements-visualization.txt` lists its dependencies separately
from the optional LightGBM model dependencies. Every figure is exported as PNG
and SVG inside `visualizations/`; plotting reloads frozen coefficients and does
not retrain a model.

The environment could not resolve PyPI, so LightGBM could not be installed. An
optional native LightGBM branch is included, but its execution is explicitly
recorded as blocked. After installing this folder's requirements into an isolated
environment, run:

```sh
fire-harmonization-research/.venv/bin/python fire-harmonization-research/src/experiment.py --engine lightgbm
```

That branch uses daily binary boosting and weekly Tweedie boosting. **The executed
NumPy logistic/Poisson results must not be described as LightGBM results.** No
dependency or script was added to the app's package/build configuration.

The NASA probe uses `FIRMS_MAP_KEY` from the environment or an existing local
`.env` file. It does not create a key, send email, modify the app's environment,
log secrets or download an archive. Without a key, it records `not_run`.

## Isolation and status

`results/main_app_before.json` fingerprints existing project files before the
research work. `results/isolation_check.json` compares the current files against
that snapshot. The workspace has no Git metadata, so hashes provide the explicit
verification. Dependencies/caches are excluded from this content comparison;
none are installed into the app.

This is a research prototype, not an operational fire-warning system, a validated
historical replacement, or permission to integrate with EPOCH.
