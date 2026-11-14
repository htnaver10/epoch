"""Scientific figures of the 11-year temporal benchmark, never final-fit holdout claims."""
import csv
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

from larger_model import (ROOT, Engine, HistogramBoost, MODEL, FEATURES, SHAPE, AREA_WEIGHT,
                          REGIONS, unpack, output, write_json, write_csv, verify_preserved)
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.labelcolor":"#374151","text.color":"#17212B",
                     "axes.edgecolor":"#D1D5DB","svg.fonttype":"none","savefig.facecolor":"white"})
BEFORE, POWER, REF, EIGHT = "uncorrected_modis", "power_law", "observed_viirs", "boosted_8year"
NAMES={BEFORE:"Observed MODIS",MODEL:"Boosted estimate · 11-year fit",POWER:"Power-law calibration",REF:"Observed VIIRS",EIGHT:"Boosted estimate · 8-year fit"}
COLORS={BEFORE:"#2878B5",MODEL:"#D55E00",POWER:"#8060A0",REF:"#17212B",EIGHT:"#009E73"}
YEARS=(2023,2024,2025)
CREATED=[]


def read(name):
    with (ROOT/"results"/name).open() as f:
        return list(csv.DictReader(f))


def heading(fig,title,subtitle,note):
    fig.get_layout_engine().set(rect=(0,.06,1,.84))
    fig.suptitle(title,x=.02,y=1.025,ha="left",fontsize=18,fontweight="bold")
    fig.text(.02,.975,subtitle,va="top",fontsize=10,color="#4B5563")
    fig.text(.02,.015,note+"\nNASA FIRMS-derived local archive; 0.25° grid. VIIRS is an observation reference, not ground-truth fire.",
             va="top",fontsize=9,color="#4B5563")


def export(fig,stem,caption):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    headers=[t for t in fig.texts if t.get_position()[1]>=.97]
    for ax in fig.axes:
        for title in (ax.title,ax._left_title,ax._right_title):
            if title.get_text():
                assert not any(title.get_window_extent(renderer).overlaps(t.get_window_extent(renderer)) for t in headers)
    for ext in ("png","svg"):
        fig.savefig(output(f"visualizations/{stem}.{ext}"),dpi=180,bbox_inches="tight",pad_inches=.22,
                    metadata={"Title":caption,"Description":"Experiment 04 retrospective benchmark: train through 2022, evaluate 2023–2025. Final 14-year model is not evaluated here."})
    plt.close(fig)
    CREATED.append({"stem":stem,"caption":caption})


def paired(rows):
    pairs=[]
    for y in YEARS:
        for w in range(14):
            group={r["model"]:r for r in rows if r["phase"]=="test" and r["region"]=="canada_rectangle" and int(r["year"])==y and int(r["week"])==w}
            models=(BEFORE,MODEL,POWER,EIGHT)
            reference=float(group[MODEL]["observed_viirs"])
            assert all(math.isclose(float(group[m]["observed_viirs"]),reference,rel_tol=1e-12) for m in models)
            pairs.append({"year":y,"week":w,"valid_days":int(group[MODEL]["valid_days"]),REF:reference,
                          **{m:float(group[m]["predicted_viirs"]) for m in models}})
    write_csv("visualizations/weekly_figure_data.csv",pairs)
    return pairs


def figures(rows,seasonal,metrics):
    pairs=paired(rows)
    fig,ax=plt.subplots(figsize=(12,5.5),layout="constrained")
    x=np.arange(3);width=.18
    for j,m in enumerate((BEFORE,MODEL,POWER,REF)):
        subset=[next(r for r in seasonal if r["phase"]=="test" and r["region"]=="canada_rectangle" and int(r["year"])==y and r["model"]==(MODEL if m==REF else m)) for y in YEARS]
        values=[float(r["observed_viirs"] if m==REF else r["predicted_viirs"]) for r in subset]
        bars=ax.bar(x+(j-1.5)*width,values,width,label=NAMES[m],color=COLORS[m])
        for b,r,v in zip(bars,subset,values):
            label=f"{v:,.0f}" if m==REF else f"{float(r['relative_error']):+.1%}"
            ax.annotate(label,(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,5),textcoords="offset points",ha="center",fontsize=9)
    ax.set_xticks(x,["2023\n91 matched days","2024\n88 matched days","2025\n89 matched days"])
    ax.set_ylabel("Area-weighted occupied cell-days")
    ax.set_ylim(0,max(b.get_height() for b in ax.patches)*1.3)
    ax.legend(ncol=2,frameon=False,loc="upper right",fontsize=9)
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    heading(fig,"Matched-summer activity, before and after","Retrospective 2023–2025 benchmark · model refit on 2012–2022 · boreal June–August",
            "Percent labels show error relative to VIIRS; black labels show reference totals. No missing dates are imputed.")
    export(fig,"01_summer_before_after","Matched-summer totals for the eleven-year benchmark fit.")

    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,sharey=True,layout="constrained")
    for ax,y in zip(axes,YEARS):
        group=[r for r in pairs if r["year"]==y]
        for m,ls in ((REF,"-"),(BEFORE,"--"),(POWER,":"),(EIGHT,"--"),(MODEL,"-")):
            ax.plot([r["week"] for r in group],[r[m]/r["valid_days"] for r in group],ls,color=COLORS[m],
                    marker="o" if m in (REF,MODEL) else None,markersize=3,linewidth=1.7,label=NAMES[m])
        ax.set_title(str(y),loc="left");ax.grid(axis="y",alpha=.2);ax.set_ylim(bottom=0)
    starts=[(date(2001,6,1)+timedelta(days=7*w)).strftime("%b %d") for w in range(14)]
    axes[-1].set_xticks(range(0,14,2),starts[::2]);axes[-1].set_xlabel("Calendar-bin start")
    fig.supylabel("Mean weighted occupied cells per matched day")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper left",bbox_to_anchor=(.01,.94),frameon=False,fontsize=8.7)
    heading(fig,"Weekly agreement and the effect of more training years","Same axes and matched dates · eight-year and eleven-year fits use the same selected tree count",
            "The final bin is August 31 only. All curves are recorded activity or estimates, not predictions of future fire.")
    export(fig,"02_weekly_before_after","Weekly activity includes eight-year and eleven-year boosted models on the same dates.")

    matrices=[np.array([[next(r for r in pairs if r["year"]==y and r["week"]==w)[m]/next(r for r in pairs if r["year"]==y and r["week"]==w)["valid_days"] for w in range(14)] for y in YEARS]) for m in (BEFORE,MODEL,REF)]
    norm=Normalize(0,max(float(a.max()) for a in matrices))
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout="constrained")
    for ax,m,a in zip(axes,(BEFORE,MODEL,REF),matrices):
        im=ax.imshow(a,cmap="magma",norm=norm,aspect="auto",interpolation="nearest")
        ax.set_yticks(range(3),YEARS);ax.set_title(NAMES[m],loc="left",fontsize=11)
        for iy in range(3):
            for ix in range(14):
                ax.text(ix,iy,f"{a[iy,ix]:.0f}",ha="center",va="center",fontsize=8,color="white" if norm(a[iy,ix])<.65 else "#17212B")
    axes[-1].set_xticks(range(14),starts,rotation=40,ha="right")
    fig.colorbar(im,ax=list(axes),shrink=.85,pad=.02,label="Mean weighted occupied cells / matched day")
    heading(fig,"The burning-activity calendar, before and after","Retrospective 2023–2025 · shared color scale and identical observed-date windows",
            "The middle panel is an estimate from MODIS inputs. Its values do not represent additional observed hotspots.")
    export(fig,"03_calendar_before_after","Before, after and reference calendars on a shared linear scale.")

    cfg=json.loads((ROOT/"results/config.json").read_text());static=np.zeros(SHAPE,bool)
    for iy,ix in cfg["static_cells"]:static[iy,ix]=True
    engine=Engine(static);model=HistogramBoost.load(ROOT/"models/boosted_11year_benchmark.json")
    d=engine.load(2023);keys,p=engine.prediction(2023,model)
    after=np.zeros(SHAPE)
    for day in np.flatnonzero(d["valid"]):
        after[~static]+=p[np.searchsorted(keys,d["codes"][day,~static])]
    before=d["modis"][d["valid"]].sum(axis=0).astype(float)
    reference=d["viirs"][d["valid"]].sum(axis=0).astype(float)
    for a in (before,after,reference):a[static]=0
    np.savez_compressed(output("visualizations/map_2023_arrays.npz"),before=before,after=after,reference=reference,excluded=static)
    row=next(r for r in seasonal if r["phase"]=="test" and r["model"]==MODEL and r["region"]=="canada_rectangle" and r["year"]=="2023")
    assert math.isclose(float(np.sum(after*AREA_WEIGHT)),float(row["predicted_viirs"]),rel_tol=1e-12)
    assert math.isclose(float(np.sum(reference*AREA_WEIGHT)),float(row["observed_viirs"]),rel_tol=1e-12)
    norm=PowerNorm(.5,vmin=0,vmax=max(float(a.max()) for a in (before,after,reference)))
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True,layout="constrained")
    cmap=plt.colormaps["magma"].with_extremes(bad="#B4BAC3")
    for ax,m,a in zip(axes,(BEFORE,MODEL,REF),(before,after,reference)):
        im=ax.imshow(np.ma.array(a,mask=static),origin="lower",extent=(-140,-60,50,70),aspect=2,norm=norm,cmap=cmap,interpolation="nearest")
        ax.set_title(NAMES[m],loc="left",fontsize=10.5);ax.set_xlabel("Longitude")
        ax.set_xticks([-140,-120,-100,-80,-60]);ax.set_yticks([50,60,70])
    axes[0].set_ylabel("Latitude")
    fig.colorbar(im,ax=list(axes),location="bottom",shrink=.7,pad=.1,aspect=45,label="Occupied days per 0.25° cell · after = expected days · shared square-root scale")
    heading(fig,"Spatial activity at the existing coarse resolution","2023 temporal benchmark · 91 matched dates · no native-resolution upscaling",
            "Gray cells are the same training-derived persistent-source exclusions in all panels.")
    export(fig,"04_spatial_before_after_2023","Shared-scale coarse maps reconcile exactly to the saved seasonal predictions.")

    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout="constrained")
    for ax,phase,regions,model_key,title in ((axes[0],"test",("west","central","east"),MODEL,"Temporal benchmark · all regions trained"),
                                            (axes[1],"spatial_test",("east",),"boosted_cold_east","East held out · two-degree buffer")):
        for j,m in enumerate((BEFORE,model_key,POWER)):
            vals=[float(next(r for r in metrics if r["phase"]==phase and r["region"]==reg and r["model"]==m)["weekly_mae"]) for reg in regions]
            key=MODEL if m==model_key else m
            bars=ax.bar(np.arange(len(regions))+(j-1)*.24,vals,.24,color=COLORS[key],label=NAMES[key])
            ax.bar_label(bars,fmt="%.1f",fontsize=9,padding=4)
        ax.set_xticks(range(len(regions)),[r.title() for r in regions]);ax.set_title(title,loc="left",fontsize=11)
        ax.set_ylim(0,max(b.get_height() for b in ax.patches)*1.3);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel("Weekly MAE · weighted occupied cell-days")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper left",bbox_to_anchor=(.01,.94),frameon=False,fontsize=9)
    heading(fig,"Temporal performance and geographic transfer","Lower error is better · all comparisons within each panel use identical labels and date windows",
            "The eastern transfer fit never uses eastern VIIRS labels. Its source mask and calibration are learned in the training area.")
    export(fig,"05_regional_errors","Regional temporal errors and a separately trained eastern transfer test.")

    selection=read("selection_curve.csv");chosen=json.loads((ROOT/"results/selection.json").read_text())["selected_trees"]
    fig,axes=plt.subplots(1,2,figsize=(11,5.3),layout="constrained")
    axes[0].plot([int(r["trees"]) for r in selection],[100*float(r["validation_score"]) for r in selection],"o-",color=COLORS[MODEL])
    axes[0].axvline(chosen,ls="--",color="#6B7280",label=f"Selected: {chosen} trees")
    axes[0].set_ylabel("Validation selection score (%)");axes[0].legend(frameon=False)
    axes[0].set_title("Validation · 2020–2022",loc="left")
    axes[1].plot([int(r["trees"]) for r in selection],[float(r["training_logloss"]) for r in selection],"o-",color=COLORS[EIGHT])
    axes[1].set_ylabel("Supported-cell binary log loss");axes[1].set_title("Training · 2012–2019",loc="left")
    for ax in axes:ax.set_xlabel("Trees");ax.grid(alpha=.2)
    heading(fig,"Model size chosen before benchmark scoring","Fixed depth 5 · no test-year tuning · 200-tree maximum",
            "The selection score weights national weekly WAPE and mean regional weekly WAPE equally. Lower is better.")
    export(fig,"06_training_selection","Validation checkpoint selection and supported-cell training loss.")
    write_json("results/visualization_manifest.json",CREATED)
    return pairs


def report(pairs,metric_rows):
    ledger=json.loads((ROOT/"results/training_ledgers.json").read_text())
    chosen=json.loads((ROOT/"results/selection.json").read_text())["selected_trees"]
    final=json.loads((ROOT/"models/boosted_14year_final.json").read_text())
    leaves=sum(sum(n["feature"]<0 for n in tree) for tree in final["trees"])
    national={r["model"]:r for r in metric_rows if r["phase"]=="test" and r["region"]=="canada_rectangle"}
    raw,after,power=[national[m] for m in (BEFORE,MODEL,POWER)]
    binary=json.loads((ROOT/"results/binary_metrics.json").read_text())
    wins={m:sum(abs(r[MODEL]-r[REF])<abs(r[m]-r[REF]) for r in pairs) for m in (BEFORE,POWER,EIGHT)}
    lines=["# Larger model: actual training and evaluation","",
           f"Trained and saved a **{chosen}-tree, maximum-depth-5 NumPy gradient-boosted model** on all 14 local overlapping summers (2012–2025). "
           f"The final ensemble has **{leaves:,} leaves** and uses ten MODIS-only features. "
           "A separately fitted 11-year model supplies the retrospective temporal benchmark below; the final 14-year fit has no new complete holdout.","",
           "| Fit | Years | Full eligible cell-days processed | Supported weighted cell-days | Distinct supported patterns |","|---|---:|---:|---:|---:|"]
    for key,label in (("selection_fit_8years","Selection fit"),("benchmark_fit_11years","Temporal benchmark fit"),("final_fit_14years","Final research fit")):
        r=ledger[key]
        lines.append(f"| {label} | {len(r['years'])} | {r['full_grid_cell_days']:,} | {r['supported_weighted_cell_days']:,.0f} | {r['supported_patterns']:,} |")
    lines += ["","Every eligible cell-day is accounted for, without background sampling. Identical binned input patterns are collapsed into exact weighted sufficient statistics. "
              "The full-grid total includes unsupported cells with forced-zero output; supported exposure is the portion used to learn the conditional response. "
              "Area-weighted exposures and distinct patterns are different units. The earlier kernel also used the full grid; the expansion relative to it is training years, features and model capacity.","",
              "## Retrospective temporal benchmark: 2023–2025","",
              "Train through 2022; tree count selected using only 2020–2022 after an initial 2012–2019 fit. The dates, masks and references match within each comparison. "
              "These years have been inspected in earlier research and are not a new blind test.","",
              "| Model | Weekly MAE | Weekly RMSE | Weekly WAPE | Aggregate bias | Worst summer error |","|---|---:|---:|---:|---:|---:|"]
    for key in (BEFORE,"numpy_daily_occupancy","kernel_occupancy",EIGHT,POWER,MODEL):
        r=national[key]
        lines.append(f"| {key} | {float(r['weekly_mae']):.2f} | {float(r['weekly_rmse']):.2f} | {float(r['weekly_wape']):.2%} | {float(r['aggregate_bias']):+.2%} | {float(r['worst_season_abs_relative_error']):.2%} |")
    reduction=1-float(after["weekly_mae"])/float(raw["weekly_mae"])
    change=float(after["weekly_mae"])/float(power["weekly_mae"])-1
    history_change=float(after["weekly_mae"])/float(national[EIGHT]["weekly_mae"])-1
    peaks=[]
    for year in YEARS:
        yr=[r for r in pairs if r["year"]==year]
        ref_week=max(yr,key=lambda r:r[REF]/r["valid_days"])["week"]
        for key in (BEFORE,EIGHT,MODEL,POWER):
            peak=max(yr,key=lambda r:r[key]/r["valid_days"])["week"]
            peaks.append({"year":year,"model":key,"reference_peak_week":ref_week,"predicted_peak_week":peak,
                          "peak_bin_start_error_days":7*(peak-ref_week),
                          "reference_peak_bin_start":(date(year,6,1)+timedelta(days=7*ref_week)).isoformat(),
                          "predicted_peak_bin_start":(date(year,6,1)+timedelta(days=7*peak)).isoformat()})
    write_csv("results/peak_timing.csv",peaks)
    peak_matches=sum(r["peak_bin_start_error_days"]==0 for r in peaks if r["model"]==MODEL)
    importance=[{"feature":f,"split_count":0,"total_training_gain":0.} for f in FEATURES]
    for tree in final["trees"]:
        for node in tree:
            if node["feature"]>=0:
                importance[node["feature"]]["split_count"]+=1
                importance[node["feature"]]["total_training_gain"]+=node["gain"]
    write_csv("results/final_model_feature_gain.csv",importance)
    lines += ["",f"The larger benchmark fit has **{reduction:.1%} lower weekly MAE than raw MODIS** and "
              f"**{change:+.1%} MAE change relative to matched-history power-law calibration**. "
              f"It improves {wins[BEFORE]}/42 bins over raw MODIS, {wins[POWER]}/42 over power law, and {wins[EIGHT]}/42 over the eight-year boosted fit. "
              "The old logistic and kernel models use eight-year fits; the displayed power-law baseline uses the same eleven-year history as the larger benchmark.","",
              f"**More years did not improve national weekly MAE in this comparison:** the same boosted architecture fitted on eight years has MAE "
              f"{float(national[EIGHT]['weekly_mae']):.2f}, compared with {float(after['weekly_mae']):.2f} after refitting eleven years "
              f"({history_change:+.1%}). Higher capacity improved on the old regression/kernel; adding the 2020–2022 years did not produce a monotonic improvement. "
              "Do not interpret the fourteen-year refit as the best-performing model without a new holdout.","",
              "### Peak timing","","Peak means the calendar bin with the highest mean activity per matched day, including the one-day final bin. "
              f"The eleven-year model matches **{peak_matches}/3** reference peak bins; details are in `peak_timing.csv`.","",
              "| Year | VIIRS peak-bin start | Larger-model peak-bin start | Timing error |","|---|---|---|---:|"]
    for r in peaks:
        if r["model"]==MODEL:
            lines.append(f"| {r['year']} | {r['reference_peak_bin_start']} | {r['predicted_peak_bin_start']} | {r['peak_bin_start_error_days']:+d} days |")
    lines += ["",
              "## Daily recorded-occupancy metrics","",
              "Area-weighted, full eligible grid, 2023–2025. These scores compare recorded VIIRS occupancy; they do not measure true-fire detection or predict future ignition.","",
              "| Metric | Raw MODIS | Larger benchmark |","|---|---:|---:|"]
    for key in ("brier","logloss","roc_auc","average_precision","precision_at_0_5","recall_at_0_5","f1_at_0_5"):
        lines.append(f"| {key} | {binary['raw_modis'][key]:.5f} | {binary['model'][key]:.5f} |")
    lines += ["","## Geographic transfer","","The east (longitude >= -90) was excluded from fitting and checkpoint selection, with training longitude < -92. "
              "Only training-region labels identify persistent sources. Eastern persistent sources remain in the reference because no eastern labels can be used to identify them.","",
              "| Model | East weekly MAE | WAPE | Bias |","|---|---:|---:|---:|"]
    for r in metric_rows:
        if r["phase"]=="spatial_test":
            lines.append(f"| {r['model']} | {float(r['weekly_mae']):.2f} | {float(r['weekly_wape']):.2%} | {float(r['aggregate_bias']):+.2%} |")
    lines += ["","## Limits and saved deliverables","",
              f"- The final fit processes **{ledger['final_fit_14years']['full_grid_cell_days']:,}** eligible cell-days and covers **75% more training years** than the original eight-year fit. No new NASA files were downloaded.",
              f"- Its MODIS-context support excludes **{ledger['final_fit_14years']['unsupported_positive_share']:.1%}** of recorded VIIRS activity during training; isolated detections cannot be recovered under this model.",
              "- The original archive has aggregated counts only: no native radiances, FRP, confidence, exact overpass times or clear-sky coverage. More model capacity cannot restore those fields or create native 375m resolution.",
              "- Geographic transfer is limited to parts of the same boreal rectangle. Global and year-round generalization remain untested. No main-app integration or promotion was performed.",
              "- `models/boosted_14year_final.json` is the final larger research model; `models/boosted_11year_benchmark.json` is the evaluated temporal model. `results/weekly_predictions.csv`, `binary_metrics.json`, `training_ledgers.json` and `selection_curve.csv` retain evidence.","",
              "## Figures",""]
    for f in CREATED:
        lines.append(f"- [{f['caption']}](../visualizations/{f['stem']}.png) · [SVG](../visualizations/{f['stem']}.svg)")
    output("results/REPORT.md").write_text("\n".join(lines)+"\n")
    write_json("results/readout.json",{"selected_trees":chosen,"final_leaves":leaves,"benchmark_mae_reduction_vs_raw":reduction,
               "benchmark_mae_change_vs_power":change,"benchmark_mae_change_from_8_to_11_years":history_change,
               "weekly_wins":wins,"benchmark_peak_bin_matches":peak_matches,
               "final_model_evaluated_on_fresh_holdout":False,"promotion_supported":False})
    gallery=["# Larger-model scientific figures","","All before/after figures use the retrospective eleven-year benchmark model, evaluated in 2023–2025. "
             "The final fourteen-year refit is not presented as held-out evidence.",""]
    for f in CREATED:
        gallery += [f"## {f['caption']}","",f"![{f['caption']}]({f['stem']}.png)","",f"[SVG]({f['stem']}.svg)",""]
    output("visualizations/README.md").write_text("\n".join(gallery))


if __name__=="__main__":
    rows=read("weekly_predictions.csv");seasonal=read("seasonal_predictions.csv");metric_rows=read("metrics.csv")
    pairs=figures(rows,seasonal,metric_rows);report(pairs,metric_rows)
    verify_preserved()
    print("Exported six figure families, report and larger-model readout.")
