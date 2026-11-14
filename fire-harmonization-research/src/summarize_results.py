"""Create the research readout from the saved experiment outputs, without app writes."""
import csv
import json
import platform
from pathlib import Path

import numpy as np

from experiment import RESEARCH, write_json, output_path


def main():
    result = RESEARCH / "results"
    metrics = json.loads((result / "numpy_metrics.json").read_text())
    config = json.loads((result / "numpy_config.json").read_text())
    audit = json.loads((result / "archive_audit.json").read_text())
    checks = json.loads((result / "result_validation.json").read_text())
    api = json.loads((result / "nasa_access_check.json").read_text())
    library = json.loads((result / "lightgbm_status.json").read_text())
    isolation = json.loads((result / "isolation_check.json").read_text())
    country = {r["model"]: r for r in metrics if r["phase"] == "test" and r["region"] == "canada_rectangle"}
    geo = {r["model"]: r for r in metrics if r["phase"] == "spatial_test"}
    names = {"uncorrected_modis": "Uncorrected MODIS", "constant_ratio": "Constant ratio",
             "power_law": "Power law", "numpy_daily_occupancy": "Daily logistic ML",
             "numpy_weekly_activity": "Weekly Poisson ML"}
    lines = ["# Preliminary findings: MODIS -> VIIRS-reference activity", "",
             "**The daily ML baseline improves on uncorrected MODIS, but does not pass the promotion criteria.** "
             "The power-law baseline has lower held-out weekly error, and the daily model fails the cold geographic transfer test. "
             "No model was integrated with the app.", "",
             "## What actually ran", "",
             "- NumPy weighted logistic regression for daily VIIRS occupancy, using MODIS-only predictors.",
             "- NumPy weighted Poisson regression for weekly occupied days.",
             "- Uncorrected MODIS, constant-ratio and power-law baselines in the revised MODIS -> VIIRS direction.",
             "- A separate east-region transfer test, excluding east-region training/validation labels and a two-degree buffer.",
             f"- {checks['metric_groups_recomputed']} metric groups independently recalculated from {checks['weekly_prediction_rows_checked']:,} saved prediction rows.",
             "- Integrity checks of all 25 source JSON files and the existing app files.", "",
             "The planned **LightGBM binary/Tweedie experiments did not run**. Installing dependencies failed because PyPI DNS resolution was unavailable. "
             "The optional runner records this explicitly; these results are logistic/Poisson baselines, not boosted trees.", "",
             "## Population and evaluation", "",
             "Training: 2012-2019. Validation: 2020-2022. Temporal test: 2023-2025. "
             "Parameters were fixed before the temporal test; test data was not used for tuning. "
             "The pilot covers June-August in the 50-70N, 140-60W rectangle, not all Canada or the full year.", "",
             "Targets are **recorded Suomi-NPP VIIRS occupancy on the common 0.25-degree grid**. "
             "Metrics use cos(latitude)/cos(60N)-weighted occupied cell-days. "
             "They measure agreement with a satellite reference, not true wildfire detection accuracy or reconstructed 375-m locations.", "",
             f"The training-only static-source rule excluded {config['training_static_cells']} cells. "
             "Existing heuristic gap dates were omitted from both instruments. The test seasons therefore contain **91, 88 and 89 matched days**, respectively; "
             "these are screened-period totals, not complete summer estimates. The local 2026 overlap is partial and was excluded.", "",
             "Training retains all active examples plus 1% of both-zero background, with inverse inclusion weights. "
             "Evaluation uses the entire eligible grid. Recorded zeros remain coverage-uncertain; no clear-sky denominator is available.", "",
             "## Temporal holdout: entire rectangle", "",
             "Weekly MAE is the mean absolute error across 42 calendar bins (including the actual short final bin) over 2023-2025. Lower is better. "
             "Seasonal errors sum the same matched days. Country and sector results are reported separately to avoid double counting.", "",
             "| Method | Weekly MAE | Aggregate bias | Median absolute seasonal error | Worst absolute seasonal error |",
             "|---|---:|---:|---:|---:|"]
    for name in names:
        r = country[name]
        lines.append(f"| {names[name]} | {r['weekly_mae']:.1f} | {r['aggregate_bias']:+.1%} | {r['median_season_abs_relative_error']:.1%} | {r['worst_season_abs_relative_error']:.1%} |")
    daily, power, raw = (country[n] for n in ["numpy_daily_occupancy", "power_law", "uncorrected_modis"])
    improvement = 1 - daily["weekly_mae"] / raw["weekly_mae"]
    regression = daily["weekly_mae"] / power["weekly_mae"] - 1
    lines += ["", f"Daily logistic ML reduces weekly MAE by **{improvement:.1%} versus uncorrected MODIS**, "
              f"but its weekly MAE is **{regression:.1%} higher than the power law**. "
              "It has a smaller worst matched-season error than the power law. These mixed outcomes do not establish that ML is the best harmonizer.", "",
              "## Geographic stress test", "",
              "The daily model trained west of 92W and was tested east of 90W in 2023-2025. "
              "The geographic baselines also use only western/central training labels. This is deliberately harder than interpolation inside the training geography.", "",
              "| Method | East weekly MAE | Aggregate east bias |",
              "|---|---:|---:|"]
    for name in ["uncorrected_modis", "constant_ratio", "power_law", "numpy_daily_east_holdout"]:
        r = geo[name]
        label = names.get(name, "Daily logistic ML, east held out")
        lines.append(f"| {label} | {r['weekly_mae']:.1f} | {r['aggregate_bias']:+.1%} |")
    lines += ["", "The daily model substantially overpredicts in the unseen east. Geographic/seasonal terms and "
              "the dense background may contribute, but this experiment does not establish a cause. "
              "This is evidence against applying this fitted model outside its validated support. "
              "The simpler power-law transfer is also imperfect; a flexible model is not automatically an improvement.", "",
              "## Access, checks and limitations", "",
              f"- Unit tests: see [unit_tests.log](unit_tests.log). Independent result validation: **{'passed' if checks['passed'] else 'failed'}**.",
              f"- Main-app isolation: **{'passed' if isolation['passed'] else 'failed'}**, {isolation['files_checked']} pre-existing files unchanged; no added files outside research.",
              f"- Authenticated NASA probe: **{api['status']}**. {api.get('reason', '')}",
              f"- LightGBM: **{library['status']}**; no LightGBM performance claim is supported.",
              "- No original acquisition times, FRP, brightness, confidence, footprint geometry, clear-sky masks or NOAA-20/21 training data were used.",
              "- No pre-2012 reconstruction, full-year expansion, peak-week acceptance test, aggregate uncertainty interval, statistical significance test, or operational validation was completed.",
              "- No synthetic data is used as evidence of satellite performance. Synthetic unit-test fixtures only check code behavior.",
              "- Only three temporal test seasons are available here; cell-days are spatially and temporally correlated.",
              "- Raw recorded occupancy and simple aggregate translations are different representations. All comparisons here use the same common-grid activity target and matched dates.", "",
              "## Decision and next step", "",
              "**Do not promote this preliminary model.** Keep the research separate. "
              "Run the planned LightGBM experiment when dependencies are available; then test richer MODIS fields and verified coverage, "
              "retaining both simple baselines and untouched new holdouts. Investigate geographic support before extrapolating. "
              "New model changes following inspection of these results make 2023-2025 a development benchmark; future claims need additional untouched holdouts or nested blocked evaluation.", "",
              "## Reproducible evidence", "",
              "- [Input audit](archive_audit.json), [input hashes](input_hashes.json), [fixed configuration](numpy_config.json).",
              "- [All metrics](numpy_metrics.csv), [weekly predictions](numpy_weekly_predictions.csv), [seasonal predictions](numpy_seasonal_predictions.csv).",
              "- [Model diagnostics](numpy_diagnostics.json), [run log](experiment.log), [independent validation](result_validation.json).",
              "- [NASA check](nasa_access_check.json), [LightGBM status](lightgbm_status.json), [isolation check](isolation_check.json).", ""]
    output_path("results/REPORT.md").write_text("\n".join(lines))
    write_json("results/promotion_decision.json", {
        "approved_for_app_integration": False,
        "daily_weekly_error_beats_power_law": daily["weekly_mae"] < power["weekly_mae"],
        "daily_worst_season_error_not_worse_than_power_law": daily["worst_season_abs_relative_error"] <= power["worst_season_abs_relative_error"],
        "daily_geographic_error_beats_ratio": geo["numpy_daily_east_holdout"]["weekly_mae"] < geo["constant_ratio"]["weekly_mae"],
        "lightgbm_executed": False, "authenticated_nasa_verified": api["status"] == "passed",
        "reason": "Daily model misses weekly-error gate and geographic transfer gate; planned LightGBM remains untested. User authorized research only."})
    write_json("results/environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "lightgbm": "not installed", "scipy": "not installed",
        "execution": "Isolated venv with read-only system NumPy; no app dependencies changed"})
    print("Wrote results/REPORT.md and promotion_decision.json")


if __name__ == "__main__":
    main()
