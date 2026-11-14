# Experiment 04.5 — frozen out-of-time evaluation

Evaluates the existing Experiment 04 models on partial 2026 data outside all their training years. No boosted model is retrained. All outputs are isolated here.

[Results and limitations](results/REPORT.md) · [Frozen protocol](PROTOCOL.md) · [Validation](results/validation.json)

From the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04.5/src/evaluate.py
PYTHONDONTWRITEBYTECODE=1 fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04.5/src/report.py
```
