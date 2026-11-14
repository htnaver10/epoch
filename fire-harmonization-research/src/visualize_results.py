"""Evaluate frozen predictions and export scientific before/after figures.

No training, app writes, new NASA requests, or model promotion occur here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from experiment import (Archive, AREA_WEIGHT, FEATURES, NumpyGLM, RESEARCH,
                        SHAPE, TRAIN, output_path, verify_isolation, windows,
                        write_csv, write_json)

RESULTS = ROOT / "results"
FIGURES = ROOT / "visualizations"
BEFORE = "uncorrected_modis"
AFTER = "numpy_daily_occupancy"
POWER = "power_law"
REFERENCE = "observed_viirs"
YEARS = (2023, 2024, 2025)
COLORS = {BEFORE: "#2878B5", AFTER: "#D55E00", POWER: "#8060A0", REFERENCE: "#1F2933"}
NAMES = {BEFORE: "Before: observed MODIS", AFTER: "After: daily ML estimate",
         POWER: "Power-law baseline", REFERENCE: "Observed VIIRS reference"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 12, "axes.labelsize": 10.5,
    "figure.titlesize": 18, "figure.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#D1D5DB", "axes.labelcolor": "#374151",
    "text.color": "#17212B", "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    "grid.color": "#E5E7EB", "grid.linewidth": .6,
    "savefig.facecolor": "white", "svg.fonttype": "none",
})


def read_rows(name):
    with (RESULTS / name).open() as stream:
        return list(csv.DictReader(stream))


def paired_rows(rows, phase="test", region="canada_rectangle"):
    """Require one-to-one pairs, identical reference values and identical windows."""
    by_key = defaultdict(dict)
    for row in rows:
        if row["phase"] != phase or row["region"] != region:
            continue
        key = int(row["year"]), int(row["week"])
        if row["model"] in by_key[key]:
            raise ValueError("Duplicate model/year/week")
        by_key[key][row["model"]] = row
    out = []
    for (year, week), models in sorted(by_key.items()):
        chosen = [models[m] for m in (BEFORE, AFTER, POWER)]
        reference = float(chosen[0]["observed_viirs"])
        days = int(chosen[0]["valid_days"])
        assert all(math.isclose(float(r["observed_viirs"]), reference, rel_tol=1e-10) for r in chosen)
        assert all(int(r["valid_days"]) == days for r in chosen)
        out.append({"year": year, "week": week, "valid_days": days,
                    REFERENCE: reference,
                    **{m: float(models[m]["predicted_viirs"]) for m in (BEFORE, AFTER, POWER)}})
    assert len(out) == 42
    return out


def evaluate(pairs, metrics, seasonal):
    wins = {"versus_uncorrected": 0, "versus_power_law": 0}
    by_year = []
    for row in pairs:
        error = lambda model: abs(row[model] - row[REFERENCE])
        wins["versus_uncorrected"] += error(AFTER) < error(BEFORE)
        wins["versus_power_law"] += error(AFTER) < error(POWER)
    for year in YEARS:
        selected = [r for r in pairs if r["year"] == year]
        peak = lambda model: max(selected, key=lambda r: r[model] / r["valid_days"])["week"]
        item = {"year": year, "matched_days": sum(r["valid_days"] for r in selected),
                "reference_peak_week": peak(REFERENCE), "before_peak_week": peak(BEFORE),
                "after_peak_week": peak(AFTER), "power_peak_week": peak(POWER)}
        for model in (BEFORE, AFTER, POWER):
            item[model + "_weekly_mae"] = float(np.mean([abs(r[model]-r[REFERENCE]) for r in selected]))
        by_year.append(item)
    region_rows = []
    for phase in ("validation", "test"):
        for region in ("west", "central", "east"):
            lookup = {r["model"]: r for r in metrics if r["phase"] == phase and r["region"] == region}
            raw, adjusted, power = (lookup[m] for m in (BEFORE, AFTER, POWER))
            region_rows.append({"phase": phase, "region": region,
                "before_weekly_mae": raw["weekly_mae"], "after_weekly_mae": adjusted["weekly_mae"],
                "power_weekly_mae": power["weekly_mae"],
                "after_vs_before_error_reduction": 1-adjusted["weekly_mae"]/raw["weekly_mae"],
                "after_vs_power_error_change": adjusted["weekly_mae"]/power["weekly_mae"]-1,
                "after_aggregate_bias": adjusted["aggregate_bias"]})
    summary = {"weekly_comparisons": len(pairs), "weekly_wins": wins,
        "year_breakdown": by_year, "region_breakdown": region_rows,
        "peak_definition": "Highest mean daily activity within a calendar bin; partial-bin estimates are not full-week observations.",
        "promotion_supported": False,
        "interpretation": "Improved pooled MODIS/VIIRS agreement; not a proven universal improvement over simple calibration. East weakness already appears in validation."}
    write_json("results/visual_evaluation.json", summary)
    write_csv("results/visual_evaluation_by_year.csv", by_year)
    write_csv("results/visual_evaluation_by_region.csv", region_rows)
    write_csv("visualizations/weekly_figure_data.csv", pairs)
    return summary


def export(fig, stem, caption):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headers = [t for t in fig.texts if t.get_position()[1] >= .98]
    for ax in fig.axes:
        # ax.set_title(loc='left') uses a separate artist from ax.title.
        for title in (ax.title, ax._left_title, ax._right_title):
            if title.get_text():
                assert not any(title.get_window_extent(renderer).overlaps(t.get_window_extent(renderer))
                               for t in headers), "Panel title overlaps figure header"
    metadata = {"Title": caption,
                "Description": "Source: NASA FIRMS-derived bundled EPOCH archive. After = frozen NumPy logistic estimate, not observed hotspots."}
    fig.savefig(output_path(f"visualizations/{stem}.png"), dpi=180, bbox_inches="tight", pad_inches=.22, metadata=metadata)
    fig.savefig(output_path(f"visualizations/{stem}.svg"), bbox_inches="tight", pad_inches=.22, metadata=metadata)
    plt.close(fig)
    return {"stem": stem, "caption": caption}


def header(fig, title, subtitle, note):
    # Reserve header/footer space explicitly; panel titles must not collide with
    # the figure-level subtitle under constrained layout.
    fig.get_layout_engine().set(rect=(0, .04, 1, .83))
    fig.suptitle(title, x=.02, ha="left", y=1.04)
    fig.text(.02, .99, subtitle, va="top", fontsize=10.5, color="#4B5563")
    fig.text(.02, -.025, note + "\nSource: bundled NASA FIRMS-derived archive; frozen NumPy model estimates.",
             va="top", fontsize=9.2, color="#4B5563")


def season_figure(seasonal):
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    x = np.arange(3); width = .19
    models = (BEFORE, AFTER, POWER, REFERENCE)
    for j, model in enumerate(models):
        rows = [next(r for r in seasonal if r["phase"] == "test" and r["region"] == "canada_rectangle"
                     and int(r["year"]) == y and r["model"] == (AFTER if model == REFERENCE else model)) for y in YEARS]
        values = [float(r["observed_viirs"] if model == REFERENCE else r["predicted_viirs"]) for r in rows]
        bars = ax.bar(x + (j-1.5)*width, values, width, color=COLORS[model], label=NAMES[model])
        for bar, row, value in zip(bars, rows, values):
            text = f"{value:,.0f}" if model == REFERENCE else f"{float(row['relative_error']):+.1%}"
            ax.annotate(text, (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10,
                        fontweight="bold" if model in (AFTER, REFERENCE) else "normal")
    ax.set_xticks(x, ["2023\n91 matched days", "2024\n88 matched days", "2025\n89 matched days"])
    ax.set_ylabel("Area-weighted occupied cell-days")
    ax.set_ylim(0, 32000); ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1000:g}k"))
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9.3)
    header(fig, "ML brings the summer totals closer to VIIRS",
           "Held-out 2023–2025 · boreal rectangle, 50–70°N / 140–60°W · June–August",
           "Percent labels are error relative to observed VIIRS; black labels are reference totals.\n"
           "Matched dates exclude suspected archive gaps. These are activity estimates, not counts of unique fires.")
    return export(fig, "01_summer_before_after", "Matched-summer totals: observed MODIS, daily ML estimate, power-law estimate and observed VIIRS.")


def weekly_figure(pairs):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True, layout="constrained")
    ymax = max(r[m]/r["valid_days"] for r in pairs for m in (BEFORE, AFTER, POWER, REFERENCE)) * 1.12
    styles = {REFERENCE: ("-", "o", 2.1), BEFORE: ("--", "s", 1.6), AFTER: ("-", "^", 1.9), POWER: (":", "D", 1.5)}
    for ax, year in zip(axes, YEARS):
        rows = [r for r in pairs if r["year"] == year]
        for model in (REFERENCE, BEFORE, POWER, AFTER):
            ls, marker, lw = styles[model]
            ax.plot([r["week"] for r in rows], [r[model]/r["valid_days"] for r in rows],
                    color=COLORS[model], linestyle=ls, marker=marker, markersize=3.7,
                    linewidth=lw, label=NAMES[model])
        ax.set_title(str(year), loc="left", fontweight="bold")
        ax.set_ylim(0, ymax); ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.axvspan(12.65, 13.35, color="#F3F4F6", zorder=-1)
    labels = [(date(2001, 6, 1)+timedelta(days=7*w)).strftime("%b %d") for w in range(14)]
    axes[-1].set_xticks([0, 2, 4, 6, 8, 10, 12, 13], [labels[w] for w in [0, 2, 4, 6, 8, 10, 12, 13]])
    axes[-1].set_xlabel("Calendar-bin start date")
    fig.supylabel("Mean weighted occupied cells per matched day", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(.02, .955),
               frameon=False, fontsize=9)
    header(fig, "Better totals do not guarantee the closest weekly pattern",
           "Same axes in every panel · observations versus frozen out-of-time predictions",
           "Values divide each bin's activity by its matched-day count; gap-screened bins have fewer than seven days.\n"
           "Shaded final bin contains August 31 only. No missing days have been filled or smoothed.")
    return export(fig, "02_weekly_before_after", "Weekly profiles normalized by matched days; common y-axis across all held-out years.")


def calendar_figure(pairs):
    models = (BEFORE, AFTER, REFERENCE)
    matrices = [np.array([[next(r for r in pairs if r["year"] == y and r["week"] == w)[m]
                          / next(r for r in pairs if r["year"] == y and r["week"] == w)["valid_days"]
                          for w in range(14)] for y in YEARS]) for m in models]
    norm = Normalize(0, max(float(m.max()) for m in matrices))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, layout="constrained")
    for ax, model, matrix in zip(axes, models, matrices):
        image = ax.imshow(matrix, cmap="magma", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(3), YEARS)
        ax.set_title(NAMES[model], loc="left", fontweight="bold", fontsize=12)
        ax.set_xticks(np.arange(-.5, 14, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for iy in range(3):
            for ix in range(14):
                val = matrix[iy, ix]
                ax.text(ix, iy, f"{val:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if norm(val) < .65 else "#17212B")
    starts = [(date(2001, 6, 1)+timedelta(days=7*w)).strftime("%b %d") for w in range(14)]
    axes[-1].set_xticks(range(14), starts, rotation=40, ha="right")
    bar = fig.colorbar(image, ax=list(axes), location="right", shrink=.85, pad=.02)
    bar.set_label("Mean weighted occupied cells / matched day")
    header(fig, "The burning calendar, before and after calibration",
           "Held-out 2023–2025 · every panel uses the same color scale and observed-date window",
           "Numbers show mean activity per matched day. Final column is August 31 only; suspected-gap dates are omitted.\n"
           "The middle panel contains model estimates; it is not an additional set of observed hotspots.")
    return export(fig, "03_calendar_before_after", "Before/after/reference calendars on one common color scale, with exact per-day values.")


def map_figure(seasonal):
    archive = Archive(); static = archive.static_mask(TRAIN)
    saved = json.loads((ROOT / "models/numpy_daily_occupancy.json").read_text())
    assert saved["features"] == FEATURES and saved["objective"] == "binary"
    model = NumpyGLM("binary")
    model.center = np.array(saved["center"]); model.scale = np.array(saved["scale"])
    model.beta = np.array(saved["beta"])
    before = np.zeros(SHAPE); after = np.zeros(SHAPE); reference = np.zeros(SHAPE)
    days = 0
    for _, length, x, y, m in windows(archive, 2023, 1, static):
        before += m.reshape(SHAPE); after += model.predict(x).reshape(SHAPE)
        reference += y.reshape(SHAPE); days += length
    before[static] = after[static] = reference[static] = 0
    actual = next(r for r in seasonal if r["phase"] == "test" and r["region"] == "canada_rectangle"
                  and r["year"] == "2023" and r["model"] == AFTER)
    checks = {"observed_modis_weighted": float(np.sum(before*AREA_WEIGHT)),
              "estimated_viirs_weighted": float(np.sum(after*AREA_WEIGHT)),
              "observed_viirs_weighted": float(np.sum(reference*AREA_WEIGHT)),
              "matched_days": days, "static_cells": int(static.sum())}
    raw = next(r for r in seasonal if r["phase"] == "test" and r["region"] == "canada_rectangle"
               and r["year"] == "2023" and r["model"] == BEFORE)
    assert math.isclose(checks["observed_modis_weighted"], float(raw["predicted_viirs"]), rel_tol=1e-10)
    assert math.isclose(checks["estimated_viirs_weighted"], float(actual["predicted_viirs"]), rel_tol=1e-10)
    assert math.isclose(checks["observed_viirs_weighted"], float(actual["observed_viirs"]), rel_tol=1e-10)
    np.savez_compressed(output_path("visualizations/map_2023_arrays.npz"),
                        before=before, after=after, reference=reference, excluded=static)
    write_json("results/map_reconciliation.json", checks)
    arrays = (before, after, reference)
    # Nonlinear color normalization is shared and explicitly labelled, never data clipping.
    maximum = max(float(a.max()) for a in arrays)
    norm = PowerNorm(gamma=.5, vmin=0, vmax=maximum)
    cmap = plt.colormaps["magma"].with_extremes(bad="#A7ADB5")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True, sharey=True, layout="constrained")
    for ax, values, model_name in zip(axes, arrays, (BEFORE, AFTER, REFERENCE)):
        image = ax.imshow(np.ma.array(values, mask=static), origin="lower", extent=(-140, -60, 50, 70),
                          aspect=2, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(NAMES[model_name], loc="left", fontsize=11.5, pad=12)
        ax.set_xticks([-140, -120, -100, -80, -60], ["140°W", "120°W", "100°W", "80°W", "60°W"], fontsize=9)
        ax.set_yticks([50, 60, 70], ["50°N", "60°N", "70°N"])
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    bar = fig.colorbar(image, ax=list(axes), location="bottom", shrink=.7, pad=.1, aspect=45)
    ticks = [t for t in [0, 1, 3, 7, 14, 30, 60, 90] if t <= maximum]
    bar.set_ticks(ticks)
    bar.set_label("Occupied days per 0.25° cell · after = expected days · shared square-root color scale")
    header(fig, "Where the model adds activity: the 2023 spatial comparison",
           "91 matched days · frozen daily model trained on 2012–2019 · common grid and color normalization",
           "Gray = 31 training-inferred persistent-source exclusions. Empty cells have no recorded hotspot, not verified clear skies.\n"
           "The estimate is coarse-cell activity; it does not reconstruct VIIRS-resolution hotspot locations.")
    return export(fig, "04_spatial_before_after_2023", "2023 MODIS, ML-estimated VIIRS-equivalent activity, and observed VIIRS on the same 0.25-degree grid.")


def regional_figure(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, layout="constrained")
    for ax, phase, title in zip(axes, ("validation", "test"),
                               ("Validation · 2020–2022", "Temporal test · 2023–2025")):
        for i, model in enumerate((BEFORE, AFTER, POWER)):
            values = [next(r for r in metrics if r["phase"] == phase and r["model"] == model and r["region"] == reg)["weekly_mae"]
                      for reg in ("west", "central", "east")]
            bars = ax.bar(np.arange(3)+(i-1)*.24, values, .24, color=COLORS[model], label=NAMES[model])
            ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(3), ["West", "Central", "East"])
        ax.set_ylim(0, 240)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Weekly MAE · weighted occupied cell-days")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    header(fig, "National improvement hides a persistent eastern-region problem",
           "Lower weekly error is better · these models trained on all three regions in 2012–2019",
           "Eastern weakness is already visible in validation. This is separate from the harder cold-geography test:\n"
           "when the east is entirely held out from training, daily ML has +165.3% aggregate bias.")
    return export(fig, "05_regional_errors", "Regional MAE exposes eastern failure even before the cold-geographic stress test.")


def write_readout(summary, figures, metrics, seasonal):
    national = {r["model"]: r for r in metrics if r["phase"] == "test" and r["region"] == "canada_rectangle"}
    before, after, power = (national[m] for m in (BEFORE, AFTER, POWER))
    east_validation = next(r for r in summary["region_breakdown"] if r["phase"] == "validation" and r["region"] == "east")
    lines = ["# Before and after MODIS calibration: evaluation", "",
             "**The daily ML baseline improves agreement with observed VIIRS at the regional summer-total level. "
             "It is not yet a reliable general replacement for the power-law calibration.**", "",
             "The before series is observed MODIS occupied-cell activity. The after series is the frozen "
             "NumPy logistic model's VIIRS-equivalent expected activity, computed from MODIS inputs. "
             "Observed VIIRS is a reference, not independent wildfire ground truth. No new detections were created.", "",
             "## What the results support", "",
             f"- Weekly MAE falls from **{before['weekly_mae']:.1f} to {after['weekly_mae']:.1f}** weighted cell-days "
             f"across the held-out rectangle: **{1-after['weekly_mae']/before['weekly_mae']:.1%} lower** than uncorrected MODIS.",
             f"- ML beats uncorrected MODIS in **{summary['weekly_wins']['versus_uncorrected']} of 42** matched calendar bins, "
             f"but beats the power law in only **{summary['weekly_wins']['versus_power_law']} of 42**.",
             f"- Aggregate bias changes from **{before['aggregate_bias']:+.1%} to {after['aggregate_bias']:+.1%}**. "
             "The near-zero aggregate can conceal errors with opposite signs in different weeks and regions.",
             f"- Worst matched-summer error falls from **{before['worst_season_abs_relative_error']:.1%} to "
             f"{after['worst_season_abs_relative_error']:.1%}**; the power law's worst error is {power['worst_season_abs_relative_error']:.1%}.", "",
             "## Why the model should remain experimental", "",
             f"- The power law still has lower national weekly MAE: **{power['weekly_mae']:.1f}** versus "
             f"**{after['weekly_mae']:.1f}**, making ML **{after['weekly_mae']/power['weekly_mae']-1:.1%} worse** on that measure.",
             f"- Eastern validation MAE is **{east_validation['after_weekly_mae']:.1f}** for ML versus "
             f"**{east_validation['before_weekly_mae']:.1f}** for uncorrected MODIS and "
             f"**{east_validation['power_weekly_mae']:.1f}** for the power law. "
             "This weakness was already observable without using the temporal test set to make a decision.",
             "- When the east is completely excluded from training, ML aggregate bias is **+165.3%**. "
             "Spatial generalization is not established. Feature extrapolation/background predictions are plausible contributors, not proven causes.",
             "- The weekly Poisson experiment underperforms even uncorrected MODIS; model family and target choice matter.",
             "- LightGBM/Tweedie was not executed, so these plots do not establish whether boosted trees help.", "",
             "## Peak-bin comparison", "",
             "Peak means the bin with the highest activity per matched day. Bins are seven calendar days except the final August 31 bin; "
             "screened gap dates reduce some bins' observed-day count. This is descriptive timing, not an operational forecast test.", "",
             "| Summer | VIIRS reference peak starts | MODIS peak starts | ML peak starts | Power-law peak starts |",
             "|---|---|---|---|---|"]
    for r in summary["year_breakdown"]:
        label = lambda key: (date(r["year"], 6, 1)+timedelta(days=7*r[key])).isoformat()
        lines.append(f"| {r['year']} | {label('reference_peak_week')} | {label('before_peak_week')} | {label('after_peak_week')} | {label('power_peak_week')} |")
    lines += ["", "ML moves the peak bin **three weeks early in 2023** and **one week late in 2025**. "
              "Uncorrected MODIS and the power-law baseline preserve the reference peak bin in all three summers. "
              "This descriptive test is particularly relevant to a burning-calendar product: better summer totals do not establish better critical-period timing.",
              "", "## Figures", "",
              "All PNGs are accompanied by editable SVGs. Heatmap color scales and before/after map scales are shared. "
              "Season totals use actual matched-day sums; time profiles and calendars divide by matched days to avoid making short bins look artificially quiet.", ""]
    for item in figures:
        name = item["stem"]
        lines += [f"### {item['caption']}", "", f"![{item['caption']}](../visualizations/{name}.png)", "",
                  f"[PNG](../visualizations/{name}.png) · [SVG](../visualizations/{name}.svg)", ""]
    lines += ["## Interpretation limits and next step", "",
              "Only 2023-2025 are shown as temporal holdouts. Dates are matched after suspected archive gaps are screened; "
              "the summer windows contain 91, 88 and 89 days. Static exclusions were fitted using training years only. "
              "The 0.25-degree pilot covers the 50-70N, 140-60W rectangle during June-August. "
              "Do not interpret colors as burned hectares or unique fires, the expected map as observed VIIRS points, "
              "or these results as a validated reconstruction of the pre-VIIRS era. No aggregate uncertainty intervals were established.", "",
              "Keep the current model out of the main app. Use the power law as the serious benchmark while testing richer "
              "MODIS features, proper observation coverage and the planned boosted model. Regional validation must gate release; "
              "pooled national totals are insufficient. These results have now been inspected, so subsequent tuning requires fresh "
              "or nested blocked evaluation rather than treating the same years as untouched tests.", "",
              "## Evidence and reproducibility", "",
              "Run `src/validate_results.py`, then `src/visualize_results.py` from this research environment. "
              "No model fitting or app integration occurs during plotting.", "",
              "- [Structured evaluation](visual_evaluation.json) and [regional breakdown](visual_evaluation_by_region.csv).",
              "- [Weekly chart values](../visualizations/weekly_figure_data.csv).",
              "- [Map reconciliation](map_reconciliation.json) and [figure/input manifest](visualization_manifest.json).",
              "- [Original preliminary report](REPORT.md), [saved metrics](numpy_metrics.csv), [app isolation check](isolation_check.json).", ""]
    output_path("results/EVALUATION.md").write_text("\n".join(lines))
    gallery = ["# MODIS before / after visualizations", "",
               "[Read the evaluation and interpretation](../results/EVALUATION.md). "
               "Before = observed MODIS activity; after = model-estimated VIIRS-equivalent activity; reference = observed VIIRS.", ""]
    for item in figures:
        gallery += [f"## {item['caption']}", "", f"![{item['caption']}]({item['stem']}.png)", "",
                    f"[PNG]({item['stem']}.png) · [SVG]({item['stem']}.svg)", ""]
    output_path("visualizations/README.md").write_text("\n".join(gallery))


def main():
    rows = read_rows("numpy_weekly_predictions.csv")
    seasonal = read_rows("numpy_seasonal_predictions.csv")
    metrics = json.loads((RESULTS / "numpy_metrics.json").read_text())
    pairs = paired_rows(rows)
    summary = evaluate(pairs, metrics, seasonal)
    figures = [season_figure(seasonal), weekly_figure(pairs), calendar_figure(pairs),
               map_figure(seasonal), regional_figure(metrics)]
    manifest = {"matplotlib": matplotlib.__version__, "years": YEARS,
                "model_retrained": False, "model_engine": "NumPy weighted logistic regression",
                "sources": {}, "figures": figures,
                "map_model_outputs_reconciled_to_saved_results": True}
    manifest["observation_source_hashes"] = json.loads((RESULTS / "input_hashes.json").read_text())
    for name in ["results/numpy_weekly_predictions.csv", "results/numpy_seasonal_predictions.csv",
                 "results/numpy_metrics.json", "models/numpy_daily_occupancy.json"]:
        manifest["sources"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    write_json("results/visualization_manifest.json", manifest)
    write_readout(summary, figures, metrics, seasonal)
    isolation = verify_isolation()
    assert isolation["passed"]
    print(json.dumps({"figures": len(figures), "formats": ["PNG", "SVG"], "app_unchanged": True,
                      "weekly_wins": summary["weekly_wins"], "evaluation": str(RESULTS / "EVALUATION.md")}, indent=2))


if __name__ == "__main__":
    main()
