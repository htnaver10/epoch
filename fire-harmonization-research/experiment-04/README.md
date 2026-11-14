# Larger model — trained in isolation

This experiment actually trains a NumPy histogram gradient-boosted model. The
final research fit uses all 14 local NASA overlap summers (2012–2025), with no
background sampling. It has 200 trees of maximum depth 5 and ten MODIS-only
spatial/recent-activity features. It predicts recorded VIIRS occupancy on the
existing coarse grid. Nothing is integrated with the main app.

Read the [results and limitations](results/REPORT.md),
[before/after figure gallery](visualizations/README.md), and
[frozen experiment protocol](PROTOCOL.md).

The final fourteen-year fit has no new complete local holdout. All before/after
benchmark figures use a separate model fitted through 2022 and evaluated on
2023–2025. Those years were inspected in prior research, so the assessment is
retrospective. A separate buffered eastern-region transfer fit also excludes
eastern labels from selection and training. No global archive was downloaded;
the SFTP/FIRMS hosts still failed DNS on the retry.

## Saved models

- [Final 14-year, 200-tree research model](models/boosted_14year_final.json).
- [Evaluated 11-year temporal model](models/boosted_11year_benchmark.json).
- [Selected 8-year model](models/boosted_8year_selected.json), useful for measuring the effect of adding years.
- [Eastern transfer model](models/boosted_cold_east.json).

The fourteen-year fit processes 32,728,320 eligible cell-days. It fits the
conditional response using 122,498 distinct supported feature patterns,
representing 2,979,022 area-weighted cell-days. Remaining cells are accounted
for by the explicit zero-support rule. These counts are different units and
are not counts of independent fires. The earlier kernel experiment also used
the full grid; this experiment increases training years and model capacity.

## Reproduce from the project root

NumPy and Matplotlib are already available in the isolated research environment.
This is a custom NumPy implementation, not LightGBM, XGBoost or scikit-learn.

```sh
fire-harmonization-research/.venv/bin/python -m unittest discover -s fire-harmonization-research/experiment-04/tests -v
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04/src/larger_model.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04/src/plot_and_report.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04/src/validate_run.py
```

Training logs, exact exposure ledgers, checkpoint scores, weekly predictions,
daily occupancy metrics, source/code hashes, independent checks and app-isolation
evidence are in `results/`. Compressed feature caches are keyed by the source
data, code and static mask; they are not extra NASA observations.
