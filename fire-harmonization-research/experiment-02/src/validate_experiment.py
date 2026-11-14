"""Independently reconcile Experiment 02 outputs and isolation guarantees."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from kernel_experiment import ROOT, RESEARCH, MODEL, write_json
from experiment import AREA_WEIGHT, verify_isolation


def csv_rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))


def main():
    rows=csv_rows("results/weekly_predictions.csv")
    groups=defaultdict(list); keys=set()
    for r in rows:
        key=(r['phase'],r['model'],r['region'],r['year'],r['week'])
        assert key not in keys; keys.add(key)
        for field in ['observed_viirs','observed_modis','predicted_viirs']:
            assert math.isfinite(float(r[field])) and float(r[field])>=0
        groups[key[:3]].append(r)
    for metric in csv_rows("results/metrics.csv"):
        group=groups[(metric['phase'],metric['model'],metric['region'])]
        errors=[float(r['predicted_viirs'])-float(r['observed_viirs']) for r in group]
        assert math.isclose(statistics.mean(abs(e) for e in errors),float(metric['weekly_mae']),rel_tol=1e-12)
        assert math.isclose(math.sqrt(statistics.mean(e*e for e in errors)),float(metric['weekly_rmse']),rel_tol=1e-12)
    for season in csv_rows("results/seasonal_predictions.csv"):
        group=[r for r in groups[(season['phase'],season['model'],season['region'])] if r['year']==season['year']]
        for field in ['predicted_viirs','observed_viirs']:
            assert math.isclose(sum(float(r[field]) for r in group),float(season[field]),rel_tol=1e-12)
        actual=float(season['observed_viirs'])
        assert math.isclose((float(season['predicted_viirs'])-actual)/actual,float(season['relative_error']),abs_tol=1e-12)
    for r in rows:
        if r['model']==MODEL and r['region']=='canada_rectangle':
            subset=[x for x in rows if x['phase']==r['phase'] and x['model']==MODEL and x['year']==r['year']
                    and x['week']==r['week'] and x['region'] in ['west','central','east']]
            assert math.isclose(sum(float(x['predicted_viirs']) for x in subset),float(r['predicted_viirs']),rel_tol=1e-11)
    # Same matched observations as the original experiment, not a different target.
    old={}
    with (RESEARCH/'results/numpy_weekly_predictions.csv').open() as f:
        for r in csv.DictReader(f):
            if r['model']=='uncorrected_modis':old[(r['phase'],r['region'],r['year'],r['week'])]=r
    for r in rows:
        if r['phase'] in ['validation','test'] and r['model']==MODEL:
            original=old[(r['phase'],r['region'],r['year'],r['week'])]
            assert original['valid_days']==r['valid_days']
            for field in ['observed_modis','observed_viirs']:
                assert math.isclose(float(original[field]),float(r[field]),rel_tol=1e-10)
    ledger=json.loads((ROOT/'results/rolling_fold_ledger.json').read_text())
    for fold in ledger:
        year=fold['test_year']
        assert max(fold['inner_training'])<min(fold['inner_validation'])
        assert max(fold['inner_validation'])<year
        assert max(fold['final_training'])<year
        assert not set(fold['inner_training']) & set(fold['inner_validation'])
        candidates=csv_rows(f"results/rolling_{year}_selection.csv")
        best=min(candidates,key=lambda r:(float(r['validation_score']),float(r['pooling']),int(r['radius']),abs(float(r['beta'])-1)))
        assert all(float(best[k])==fold[k] for k in ['radius','beta','pooling'])
        assert len(candidates)==27
    chosen=json.loads((ROOT/'models/fixed_selected.json').read_text())
    candidates=csv_rows('results/fixed_validation_candidates.csv')
    best=min(candidates,key=lambda r:(float(r['validation_score']),float(r['pooling']),int(r['radius']),abs(float(r['beta'])-1)))
    assert all(float(best[k])==chosen[k] for k in ['radius','beta','pooling'])
    assert len(candidates)==27
    data=np.load(ROOT/'visualizations/map_2023_arrays.npz')
    season=next(r for r in csv_rows('results/seasonal_predictions.csv') if r['phase']=='test' and r['model']==MODEL
                and r['year']=='2023' and r['region']=='canada_rectangle')
    assert math.isclose(float(np.sum(data['after']*AREA_WEIGHT)),float(season['predicted_viirs']),rel_tol=1e-10)
    figures=json.loads((ROOT/'results/figure_manifest.json').read_text())
    for fig in figures['figures']:
        png=ROOT/f"visualizations/{fig['stem']}.png";svg=png.with_suffix('.svg')
        with Image.open(png) as im:
            assert im.width>1000 and im.height>500;im.verify()
        assert ET.parse(svg).getroot().tag.endswith('svg')
    protected=json.loads((ROOT/'results/previous_experiment_hashes.json').read_text())
    for name,digest in protected.items():assert hashlib.sha256((RESEARCH/name).read_bytes()).hexdigest()==digest
    for name,digest in json.loads((ROOT/'results/protocol_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in json.loads((RESEARCH/'results/input_hashes.json').read_text()).items():
        assert hashlib.sha256((RESEARCH.parent/name).read_bytes()).hexdigest()==digest
    isolation=verify_isolation();assert isolation['passed']
    result={"passed":True,"prediction_rows_checked":len(rows),"chronological_outer_folds":len(ledger),
            "figures_checked":len(figures['figures']),"previous_artifacts_unchanged":len(protected),
            "main_app_files_unchanged":isolation['files_checked'],
            "checks":["metric recomputation","season/week reconciliation","kernel region additivity",
                      "same original matched observations","validation-only parameter selection",
                      "chronological inner/outer splits","map-to-saved-total reconciliation",
                      "PNG integrity and SVG parsing","frozen protocol/implementation hashes",
                      "input and prior artifact integrity","main app isolation"]}
    write_json('results/validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
