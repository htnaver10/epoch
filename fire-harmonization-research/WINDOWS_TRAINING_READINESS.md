# Coarse-research training readiness

Checked 2026-11-14. This folder is ready for its existing CPU/NumPy and optional
LightGBM experiments on this Windows computer. Full native-image training belongs
in the [separate full-resolution folder](../fire-harmonization-research-fullres/WINDOWS_TRAINING_READINESS.md).

- Independent `.venv`: Python 3.12.10, NumPy 2.5.4, LightGBM 4.7.0,
  SciPy 1.18.1 and Matplotlib 3.11.2. `pip check` passes.
- **37 tests pass** across `tests`, `experiment-02/tests`, `experiment-03/tests`
  and `experiment-04/tests`. See [test log](results/windows_unit_tests.log).
- A tiny native LightGBM fit passed. The existing real 2023 archive loads as
  `(92, 80, 320)` MODIS/VIIRS arrays; all 24 local season files are present.
  See [smoke report](results/windows_training_smoke.json).
- New training runs fingerprint the current app before execution. Portable paths
  and exclusion of both research folders avoid false Windows isolation failures
  and expensive hashing of the sibling training environments.
- Historical results and the original development-host snapshot were retained.
  A newly installed LightGBM package does not turn prior NumPy findings into
  LightGBM findings. No complete coarse scientific rerun was performed here.

Run checks from the project root without activating the environment:

```powershell
.\fire-harmonization-research\.venv\Scripts\python.exe -m unittest discover -s fire-harmonization-research/tests -v
.\fire-harmonization-research\.venv\Scripts\python.exe -m pip check
```

The existing training entry points work with that Python executable:

```powershell
.\fire-harmonization-research\.venv\Scripts\python.exe fire-harmonization-research/src/experiment.py --engine numpy
.\fire-harmonization-research\.venv\Scripts\python.exe fire-harmonization-research/src/experiment.py --engine lightgbm
.\fire-harmonization-research\.venv\Scripts\python.exe fire-harmonization-research/experiment-04/src/larger_model.py
```

Those scientific runs write to existing result/model locations. Preserve a copy
of the historical research artifacts before intentionally reproducing a run.
The setup assessment only ran tests, a tiny dependency smoke fit and read-only
archive loading. It saved new Windows assessment files under `results/`.

Rebuild with a fresh Python 3.12 venv and
`pip install -r fire-harmonization-research/results/windows-requirements-lock.txt`.
The independent environment does not require the RTX GPU. Its statistical count
models do not produce native-resolution location estimates.
