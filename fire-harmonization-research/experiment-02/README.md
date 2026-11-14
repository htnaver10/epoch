# Experiment 02 — isolated research

[Read the evaluation and all figures](results/REPORT.md).

[Frozen rationale and protocol](PROTOCOL.md). The model is a bounded local-signal calibration, not a native-resolution upscaler.

Run from the project root:

```sh
fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/kernel_experiment.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/plot_experiment.py
fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/validate_experiment.py
fire-harmonization-research/.venv/bin/python -m unittest discover -s fire-harmonization-research/experiment-02/tests -v
```

All output is confined to this folder. Previous experiment models, prediction tables and figures are preserved. Main app unchanged.
