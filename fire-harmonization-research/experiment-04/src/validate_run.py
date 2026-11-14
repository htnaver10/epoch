"""Independently reconcile saved model predictions, metrics, source totals and artifacts."""
import csv
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

from larger_model import (ROOT, Engine, HistogramBoost, MODEL, SHAPE, AREA_WEIGHT,
                          unpack, verify_preserved, write_json)


def close(actual,expected):
    assert math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-9),(actual,expected)


def read(name):
    with (ROOT/"results"/name).open() as f:
        return list(csv.DictReader(f))


def main():
    rows=read("weekly_predictions.csv");stored=read("metrics.csv");seasonal=read("seasonal_predictions.csv")
    groups=defaultdict(list);references={};seen=set()
    for row in rows:
        key=tuple(row[k] for k in ("phase","model","region","year","week"))
        assert key not in seen,key
        seen.add(key)
        groups[key[:3]].append(row)
        ref_key=tuple(row[k] for k in ("phase","region","year","week"))
        ref=(float(row["observed_viirs"]),float(row["observed_modis"]),int(row["valid_days"]))
        if ref_key in references:
            for a,b in zip(ref,references[ref_key]):close(a,b)
        references[ref_key]=ref
        assert math.isfinite(float(row["predicted_viirs"])) and float(row["predicted_viirs"])>=0
    for metric in stored:
        key=tuple(metric[k] for k in ("phase","model","region"));g=groups[key]
        pred=np.array([float(r["predicted_viirs"]) for r in g]);obs=np.array([float(r["observed_viirs"]) for r in g])
        close(metric["weekly_mae"],np.mean(np.abs(pred-obs)))
        close(metric["weekly_rmse"],np.sqrt(np.mean((pred-obs)**2)))
        close(metric["weekly_wape"],np.abs(pred-obs).sum()/obs.sum())
        close(metric["aggregate_bias"],(pred.sum()-obs.sum())/obs.sum())
        assert int(metric["weeks"])==len(g)
        errors=[]
        for year in {r["year"] for r in g}:
            yr=[r for r in g if r["year"]==year]
            yp=sum(float(r["predicted_viirs"]) for r in yr);yo=sum(float(r["observed_viirs"]) for r in yr)
            s=next(r for r in seasonal if tuple(r[k] for k in ("phase","model","region"))==key and r["year"]==year)
            close(s["predicted_viirs"],yp);close(s["observed_viirs"],yo);close(s["relative_error"],(yp-yo)/yo)
            errors.append(abs((yp-yo)/yo))
        close(metric["median_season_abs_relative_error"],np.median(errors))
        close(metric["worst_season_abs_relative_error"],max(errors))
    cfg=json.loads((ROOT/"results/config.json").read_text());static=np.zeros(SHAPE,bool)
    for y,x in cfg["static_cells"]:static[y,x]=True
    engine=Engine(static);model=HistogramBoost.load(ROOT/"models/boosted_11year_benchmark.json")
    # Rebuild probabilities on the full raw eligible grid, rather than the training/grouped metric path.
    total_w,total_brier,total_ll,tp,fp,fn=0.,0.,0.,0.,0.,0.
    for year in (2023,2024,2025):
        d=engine.load(year)
        code=d["codes"][d["valid"]][:,~static]
        keys,inv=np.unique(code.ravel(),return_inverse=True)
        p=model.predict(unpack(keys))[inv].reshape(code.shape)
        y=d["viirs"][d["valid"]][:,~static].astype(float)
        w=AREA_WEIGHT[~static][None,:]
        total_w+=float(np.sum(np.broadcast_to(w,p.shape)))
        total_brier+=float(np.sum(w*(p-y)**2))
        clip=np.clip(p,1e-12,1-1e-12)
        total_ll-=float(np.sum(w*(y*np.log(clip)+(1-y)*np.log1p(-clip))))
        tp+=float(np.sum(w*((p>=.5)&(y>0))));fp+=float(np.sum(w*((p>=.5)&(y==0))))
        fn+=float(np.sum(w*((p<.5)&(y>0))))
        country=[r for r in rows if r["phase"]=="test" and r["model"]==MODEL and r["region"]=="canada_rectangle" and int(r["year"])==year]
        close(np.sum(p*w),sum(float(r["predicted_viirs"]) for r in country))
        close(np.sum(y*w),sum(float(r["observed_viirs"]) for r in country))
    b=json.loads((ROOT/"results/binary_metrics.json").read_text())["model"]
    close(b["brier"],total_brier/total_w);close(b["logloss"],total_ll/total_w)
    close(b["precision_at_0_5"],tp/(tp+fp));close(b["recall_at_0_5"],tp/(tp+fn))
    close(b["f1_at_0_5"],2*tp/(2*tp+fp+fn))
    figure_rows=[]
    with (ROOT/"visualizations/weekly_figure_data.csv").open() as f:
        figure_rows=list(csv.DictReader(f))
    assert len(figure_rows)==42
    for r in figure_rows:
        source=next(x for x in rows if x["phase"]=="test" and x["region"]=="canada_rectangle" and x["year"]==r["year"] and x["week"]==r["week"] and x["model"]==MODEL)
        close(r[MODEL],source["predicted_viirs"]);close(r["observed_viirs"],source["observed_viirs"])
    with np.load(ROOT/"visualizations/map_2023_arrays.npz") as z:
        s=next(r for r in seasonal if r["phase"]=="test" and r["model"]==MODEL and r["region"]=="canada_rectangle" and r["year"]=="2023")
        close(np.sum(z["after"]*AREA_WEIGHT),s["predicted_viirs"])
        close(np.sum(z["reference"]*AREA_WEIGHT),s["observed_viirs"])
    for path,digest in json.loads((ROOT/"results/protocol_hashes.json").read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    final=json.loads((ROOT/"models/boosted_14year_final.json").read_text())
    assert final["metadata"]["training"]["years"]==list(range(2012,2026))
    assert len(final["trees"])==200
    manifest=json.loads((ROOT/"results/visualization_manifest.json").read_text())
    assert len(manifest)==6
    from PIL import Image
    sizes={}
    for item in manifest:
        for extension in ("png","svg"):
            p=ROOT/f"visualizations/{item['stem']}.{extension}"
            assert p.exists() and p.stat().st_size>2000
        with Image.open(ROOT/f"visualizations/{item['stem']}.png") as im:
            im.verify()
        with Image.open(ROOT/f"visualizations/{item['stem']}.png") as im:
            sizes[item["stem"]]=list(im.size)
    result={"passed":True,"prediction_rows_checked":len(rows),"metric_groups_checked":len(stored),
            "independent_full_grid_binary_metrics":True,"full_grid_probability_totals_reconciled":True,
            "figure_data_matches_predictions":True,"map_reconciled":True,"frozen_protocol_and_training_code_unchanged":True,
            "final_training_years":14,"final_trees":200,"figure_sizes":sizes,"isolation":verify_preserved()}
    write_json("results/validation.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
