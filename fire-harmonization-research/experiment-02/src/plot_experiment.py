"""Same five figure families as Experiment 01, plus rolling-origin results."""
import csv
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from kernel_experiment import (ROOT, RESEARCH, Engine, MODEL, SECTORS, alpha_for,
                               output_path, response, write_csv, write_json)
from experiment import AREA_WEIGHT, REGIONS, SHAPE, verify_isolation
import visualize_results as style

plt = style.plt
BEFORE = "uncorrected_modis"
POWER = "power_law"
OLD = "numpy_daily_occupancy"
REF = "observed_viirs"
YEARS = (2023, 2024, 2025)
CREATED = []


def read_csv(name):
    with (ROOT / "results" / name).open() as f:
        return list(csv.DictReader(f))


def header(fig, title, subtitle, note):
    fig.get_layout_engine().set(rect=(0, .04, 1, .83))
    title = {"ML brings the summer totals closer to VIIRS": "Signal-conditioned calibration: before and after",
             "Better totals do not guarantee the closest weekly pattern": "Weekly timing after signal-conditioned calibration"}.get(title, title)
    subtitle = subtitle.replace("Held-out", "Development benchmark")
    fig.suptitle(title, x=.02, ha="left", y=1.04)
    fig.text(.02, .99, subtitle, va="top", fontsize=10.5, color="#4B5563")
    fig.text(.02, -.025, note + "\nSource: bundled NASA FIRMS archive; learned kernel-response estimates, not new observations.",
             va="top", fontsize=9.2, color="#4B5563")


def export(fig, stem, caption):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headers = [t for t in fig.texts if t.get_position()[1] >= .98]
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            if title.get_text():
                assert not any(title.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in headers)
    meta = {"Title": caption, "Description": "MODIS -> VIIRS-equivalent activity. Experiment 02 bounded kernel response; 0.25-degree source grid. No native-resolution reconstruction."}
    for extension in ("png", "svg"):
        fig.savefig(output_path(f"visualizations/{stem}.{extension}"), dpi=180,
                    bbox_inches="tight", pad_inches=.22, metadata=meta)
    plt.close(fig)
    result = {"stem": stem, "caption": caption}; CREATED.append(result)
    return result


def configure_style():
    # Reuse only the plot functions/style in a fresh process. Redirect every writer;
    # do not invoke the old main(), evaluation writer or old model/map loader.
    style.output_path = output_path
    style.AFTER = MODEL
    style.NAMES[MODEL] = "After: kernel-response estimate"
    style.COLORS[MODEL] = "#D55E00"
    style.header = header
    style.export = export


def paired(rows):
    pairs = style.paired_rows(rows)
    write_csv("visualizations/weekly_figure_data.csv", pairs)
    return pairs


def map_figure(seasonal):
    engine = Engine()
    selected = json.loads((ROOT/"models/fixed_selected.json").read_text())
    model = selected["model"]
    static = np.zeros(SHAPE, bool)
    for y, x in selected["static_cells"]:
        static[y, x] = True
    m, v, valid = engine.load(2023)
    counts, denominator = engine.signals(2023, model["radius"], static)
    fractions, lookup = engine.tables[model["radius"]]
    before = m[valid].sum(axis=0).astype(float)
    reference = v[valid].sum(axis=0).astype(float)
    after = np.zeros(SHAPE)
    for day in np.flatnonzero(valid):
        signal = fractions[lookup[counts[day], denominator]]
        for region in SECTORS:
            mask = REGIONS[region] & ~static
            p = response(signal, alpha_for(model, region, selected["pooling"]), model["beta"])
            after[mask] += p[mask]
    before[static] = reference[static] = after[static] = 0
    expected = next(r for r in seasonal if r["phase"] == "test" and r["model"] == MODEL
                    and r["region"] == "canada_rectangle" and r["year"] == "2023")
    total = float(np.sum(after*AREA_WEIGHT))
    assert math.isclose(total, float(expected["predicted_viirs"]), rel_tol=1e-10)
    assert math.isclose(float(np.sum(reference*AREA_WEIGHT)), float(expected["observed_viirs"]), rel_tol=1e-10)
    np.savez_compressed(output_path("visualizations/map_2023_arrays.npz"), before=before,
                        after=after, reference=reference, excluded=static)
    write_json("results/map_reconciliation.json", {"year": 2023, "matched_days": int(valid.sum()),
        "predicted_weighted_cell_days": total, "observed_weighted_cell_days": float(np.sum(reference*AREA_WEIGHT)),
        "saved_total_match": True, "radius_cells": model["radius"], "native_resolution_reconstructed": False})
    arrays = (before, after, reference)
    norm = style.PowerNorm(gamma=.5, vmin=0, vmax=max(float(x.max()) for x in arrays))
    cmap = plt.colormaps["magma"].with_extremes(bad="#A7ADB5")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True, sharey=True, layout="constrained")
    for ax, values, name in zip(axes, arrays, (BEFORE, MODEL, REF)):
        image = ax.imshow(np.ma.array(values, mask=static), origin="lower", extent=(-140,-60,50,70),
                          aspect=2, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(style.NAMES[name], loc="left", fontsize=11.2, pad=12)
        ax.set_xticks([-140,-120,-100,-80,-60], ["140°W","120°W","100°W","80°W","60°W"], fontsize=9)
        ax.set_yticks([50,60,70], ["50°N","60°N","70°N"]); ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    bar = fig.colorbar(image, ax=list(axes), location="bottom", shrink=.7, pad=.1, aspect=45)
    bar.set_ticks([x for x in [0,1,3,7,14,30,60,90] if x <= norm.vmax])
    bar.set_label("Occupied days per 0.25° cell · after = expected days · shared square-root color scale")
    header(fig, "The estimate follows nearby MODIS observations",
           "2023 development benchmark · same observations and mask as Experiment 01",
           f"The selected kernel has radius {model['radius']} source cells ({model['radius']*.25:g}°); gray cells are static-source exclusions.\n"
           "Spatial pooling smooths the estimate. This is not 375-m upscaling or recovered fire locations.")
    return export(fig, "04_spatial_before_after_2023", "2023 MODIS, kernel-response estimate, and observed VIIRS on a shared coarse grid.")


def regional_figure(metrics):
    fig, axes = plt.subplots(1,2,figsize=(12,5.5),sharey=True,layout="constrained")
    for ax, phase, title in zip(axes, ("validation","test"), ("Selection · 2020–2022", "Development benchmark · 2023–2025")):
        for j, name in enumerate((BEFORE,MODEL,POWER)):
            values = [next(r for r in metrics if r["phase"]==phase and r["region"]==reg and r["model"]==name)["weekly_mae"] for reg in SECTORS]
            bars = ax.bar(np.arange(3)+(j-1)*.24, values, .24, color=style.COLORS[name], label=style.NAMES[name])
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
        ax.set_xticks(range(3), ["West","Central","East"]); ax.set_ylim(0,240)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Weekly MAE · weighted occupied cell-days")
    axes[0].legend(loc="upper right",frameon=False,fontsize=9)
    header(fig, "Does the new response improve each region?",
           "Same y-axis in both panels · lower error is better · selection years chose the model settings",
           "Selected settings use regional response scales; unseen regions use a pooled scale. No coordinate regression is used.\n"
           "Development-year results are retrospective: these years were already inspected in the earlier experiment.")
    return export(fig,"05_regional_errors","Regional error before and after kernel-response calibration, compared with the power law.")


def rolling_figure(rows):
    fig,ax = plt.subplots(figsize=(12,5.5),layout="constrained")
    years=list(range(2020,2026)); x=np.arange(len(years))
    for j,name in enumerate((BEFORE,MODEL,POWER)):
        values=[]
        for year in years:
            r=[r for r in rows if r["phase"]=="rolling_test" and r["region"]=="canada_rectangle" and r["model"]==name and int(r["year"])==year]
            values.append(float(np.mean([abs(float(a["predicted_viirs"])-float(a["observed_viirs"])) for a in r])))
        bars=ax.bar(x+(j-1)*.24, values,.24,color=style.COLORS[name],label=style.NAMES[name])
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    ax.set_xticks(x,years);ax.set_ylabel("Weekly MAE · weighted occupied cell-days")
    ax.set_ylim(0, max(p.get_height() for p in ax.patches)*1.2)
    ax.grid(axis="y");ax.set_axisbelow(True);ax.legend(frameon=False,loc="upper left",fontsize=9)
    header(fig,"Rolling-year backtests check whether the improvement persists",
           "Each year uses only earlier years for fitting and inner selection · six chronological outer folds",
           "Models and simple baselines are refitted on the same prior history. Fold-specific masks and settings are recorded.\n"
           "This prevents within-run look-ahead, but does not erase prior researcher exposure to the dataset.")
    return export(fig,"06_rolling_backtests","Rolling-origin weekly error across 2020–2025 with matched-history baselines.")


def assessment(pairs, metric_rows):
    table={r["model"]:r for r in metric_rows if r["phase"]=="test" and r["region"]=="canada_rectangle"}
    result={"fixed_split":table,"peak_bins":[],"weekly_wins_vs_modis":0,"weekly_wins_vs_power":0}
    for row in pairs:
        error=lambda m:abs(row[m]-row[REF])
        result["weekly_wins_vs_modis"]+=error(MODEL)<error(BEFORE)
        result["weekly_wins_vs_power"]+=error(MODEL)<error(POWER)
    for year in YEARS:
        r=[r for r in pairs if r["year"]==year]
        peak=lambda m:max(r,key=lambda x:x[m]/x["valid_days"])["week"]
        result["peak_bins"].append({"year":year,"reference":peak(REF),"before":peak(BEFORE),"after":peak(MODEL),"power":peak(POWER)})
    return result


def readout(summary, metrics, selected):
    lookup=lambda phase,model,region="canada_rectangle": next(r for r in metrics if r["phase"]==phase and r["model"]==model and r["region"]==region)
    raw,new,power,old=(lookup("test",m) for m in (BEFORE,MODEL,POWER,OLD))
    cold=lookup("spatial_test",MODEL,"east")
    rolling=lookup("rolling_test",MODEL); rolling_power=lookup("rolling_test",POWER)
    support=[float(r['unsupported_share']) for r in read_csv('support_diagnostics.csv')
             if r['phase']=='test' and r['region']=='canada_rectangle']
    lines=["# Experiment 02: decision and visual evaluation","",
           "**The constrained model improves spatial transfer, but the power law remains the stronger overall calendar benchmark. Do not promote it.**","",
           "## Why this experiment","",
           "The first logistic model predicts a small nonzero background, extrapolates spatial coefficients, and uses past-day context. "
           "This experiment uses only the current day's local MODIS signal and a bounded, monotone Poisson-occupancy response. "
           "Training scales are fitted in the calendar's area-weighted cell-day units. These are established statistical building blocks; "
           "this particular adaptation was an experiment, not a known MODIS accuracy improvement.","",
           f"Validation selected radius **{selected['radius']} source cells ({selected['radius']*.25:g}°)**, "
           f"beta **{selected['beta']}**, and regional pooling weight **{selected['pooling']}** from the frozen 27-setting candidate set. "
           "Original radiances/footprints are still unavailable, so native-resolution upscaling was not attempted.","",
           "## Comparable 2023–2025 development benchmark","",
           "All methods use the same 2012–2019 training history, training-only static mask and matched dates. "
           "The new settings use 2020–2022 selection data. **2023–2025 was already inspected in Experiment 01; this is not a fresh blind test.**","",
           "| Method | Weekly MAE | Aggregate bias | Worst matched-summer error |",
           "|---|---:|---:|---:|"]
    for name,label in [(BEFORE,"Uncorrected MODIS"),(OLD,"Original logistic model"),(MODEL,"New kernel response"),(POWER,"Power law")]:
        r=lookup("test",name)
        lines.append(f"| {label} | {r['weekly_mae']:.1f} | {r['aggregate_bias']:+.1%} | {r['worst_season_abs_relative_error']:.1%} |")
    lines += ["",f"The new model reduces weekly MAE **{1-new['weekly_mae']/raw['weekly_mae']:.1%}** versus MODIS, "
              f"but remains **{new['weekly_mae']/power['weekly_mae']-1:.1%} worse** than the power law and "
              f"**{new['weekly_mae']/old['weekly_mae']-1:+.1%}** relative to the old logistic model. "
              f"It beats raw MODIS in {summary['weekly_wins_vs_modis']}/42 bins and the power law in {summary['weekly_wins_vs_power']}/42.","",
              "## What improved, and what did not","",
              f"- Cold-east weekly MAE is **{cold['weekly_mae']:.1f}**, versus **201.6** for the old spatially held-out logistic model. "
              f"Aggregate bias is **{cold['aggregate_bias']:+.1%}**, versus **+165.3%** previously. "
              "Both use west/central fitting and selection only. The new pooled fallback substantially reduces that failure.",
              f"- Cold-east worst matched-season relative error is still **{cold['worst_season_abs_relative_error']:.1%}**. "
              "An improved pooled average does not make quiet-region estimates reliable.",
              f"- Rolling-origin country weekly MAE is **{rolling['weekly_mae']:.1f}** for the new model and "
              f"**{rolling_power['weekly_mae']:.1f}** for the power law. The simpler baseline also wins across the six-fold pooled backtest.",
              "- Zero local MODIS signal now produces exactly zero prediction; the model cannot recover VIIRS-only activity outside that support. "
              "The support diagnostics report this limitation instead of filling isolated locations from a geographic prior.",
              f"- **{min(support):.1%}–{max(support):.1%}** of the benchmark VIIRS-reference activity lies outside that MODIS support. "
              "Regional count matching can compensate by assigning activity elsewhere. Better totals therefore do not establish better spatial localization.",
              "- The response has a probabilistic form, but fitting matches aggregate counts. Cell-level probability calibration was not established.","",
              "## Peak-bin timing","",
              "Peaks use mean activity per matched day, exactly as the calendar figure. A week number is the bin beginning June 1 plus seven days per bin.","",
              "| Summer | VIIRS reference | Before MODIS | After kernel model | Power law |","|---|---|---|---|---|"]
    for r in summary["peak_bins"]:
        d=lambda k:(date(r['year'],6,1)+timedelta(days=7*r[k])).isoformat()
        lines.append(f"| {r['year']} | {d('reference')} | {d('before')} | {d('after')} | {d('power')} |")
    lines += ["","## Figures","",
              "Five original figure families are repeated, with a sixth rolling-backtest comparison. "
              "After = expected VIIRS-equivalent coarse-grid occupancy. These are not new detections or finer-resolution measurements.",""]
    for fig in CREATED:
        lines += [f"### {fig['caption']}","",f"![{fig['caption']}](../visualizations/{fig['stem']}.png)","",
                  f"[PNG](../visualizations/{fig['stem']}.png) · [SVG](../visualizations/{fig['stem']}.svg)",""]
    lines += ["## Validation and limits","",
              "The rolling protocol has six outer test years (2020–2025). In each, select settings using two immediately preceding years, "
              "then refit on all years before the outer test year. Baselines use the same history and masks. "
              "All earlier data had been publicly available and some results previously inspected, so the rolling procedure is a retrospective backtest, not an independent prospective trial.","",
              "The current geography, June–August scope, heuristic-gap screening and absence of clear-sky observation masks remain unchanged. "
              "Different final training histories in rolling folds make their scores incomparable to fixed-split scores without that qualification. "
              "Neighborhoods are defined in degrees; the model is a coarse activity calibration, not physical fire spread or satellite super-resolution. "
              "No model was retuned after reading its benchmark results, no pre-2012 reconstruction was published, and no main-app integration was performed.","",
              "## Recommendation","",
              "Retain the power law as the leading calendar benchmark. The new experiment shows that constraining spatial support can improve "
              "transfer, but does not establish a new best harmonizer. Before a native-footprint or upscaling experiment, acquire the original "
              "NASA measurements, geometry and observation masks. Keep all present outputs in research.","",
              "[Frozen protocol](../PROTOCOL.md) · [All metrics](metrics.csv) · [Selection candidates](fixed_validation_candidates.csv) · "
              "[Rolling fold ledger](rolling_fold_ledger.json) · [Support diagnostics](support_diagnostics.csv) · [Validation checks](validation.json)",""]
    output_path("results/REPORT.md").write_text("\n".join(lines))
    output_path("README.md").write_text("# Experiment 02 — isolated research\n\n"
        "[Read the evaluation and all figures](results/REPORT.md).\n\n"
        "[Frozen rationale and protocol](PROTOCOL.md). The model is a bounded local-signal calibration, not a native-resolution upscaler.\n\n"
        "Run from the project root:\n\n```sh\n"
        "fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/kernel_experiment.py\n"
        "fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/plot_experiment.py\n"
        "fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-02/src/validate_experiment.py\n"
        "fire-harmonization-research/.venv/bin/python -m unittest discover -s fire-harmonization-research/experiment-02/tests -v\n```\n\n"
        "All output is confined to this folder. Previous experiment models, prediction tables and figures are preserved. Main app unchanged.\n")
    write_json("results/assessment.json",summary)
    write_json("results/promotion_decision.json",{
        "approved_for_app_integration":False,
        "weekly_mae_beats_power_law":new['weekly_mae']<power['weekly_mae'],
        "worst_season_error_not_worse_than_power_law":new['worst_season_abs_relative_error']<=power['worst_season_abs_relative_error'],
        "rolling_mae_beats_power_law":rolling['weekly_mae']<rolling_power['weekly_mae'],
        "peak_bins_match_reference":all(r['after']==r['reference'] for r in summary['peak_bins']),
        "reason":"Improved spatial transfer and peak timing versus the first model, but power-law calendar error remains lower; research-only scope persists."})


def main():
    configure_style()
    rows=read_csv("weekly_predictions.csv"); seasonal=read_csv("seasonal_predictions.csv")
    metrics=json.loads((ROOT/"results/metrics.json").read_text())
    pairs=paired(rows)
    style.season_figure(seasonal);style.weekly_figure(pairs);style.calendar_figure(pairs)
    map_figure(seasonal);regional_figure(metrics);rolling_figure(rows)
    selected=json.loads((ROOT/"models/fixed_selected.json").read_text())
    summary=assessment(pairs,metrics)
    readout(summary,metrics,selected)
    manifest={"figures":CREATED,"source_hashes":{}}
    for name in ["results/weekly_predictions.csv","results/metrics.json","models/fixed_selected.json"]:
        manifest["source_hashes"][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    write_json("results/figure_manifest.json",manifest)
    assert verify_isolation()["passed"]
    print(json.dumps({"figures":len(CREATED),"peak_bins":summary['peak_bins'],"app_unchanged":True},indent=2))


if __name__=="__main__":main()
