"""Describe dataset coverage and compare recorded validation scores with baselines.

This reports evidence, not a universal sample-count rule for a useful model.
The planning targets are taken from PLAN.md and do not guarantee generalization.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.metrics import BinaryMetrics


def assess(manifest,run,output):
    rows=core.validate_tiles(manifest)
    partitions={split:[r for r in rows if r['split']==split] for split in ('train','validation','test')}
    summaries={}
    for split,selected in partitions.items():
        summaries[split]={'tiles':len(selected),'acquisition_event_groups':len({r['event_group'] for r in selected}),
            'regions':sorted({r['region_id'] for r in selected}),
            'positive_reference_queries':sum(r.get('positive_queries',0) for r in selected),
            'known_reference_queries':sum(r.get('valid_queries',0) for r in selected),
            'fire_positive_tiles':sum(r.get('positive_queries',0)>0 for r in selected)}
    training=summaries['train'];prevalence=training['positive_reference_queries']/max(1,training['known_reference_queries'])
    baselines={}
    if partitions['validation']:
        modis=BinaryMetrics();constant=BinaryMetrics()
        for row in partitions['validation']:
            with np.load(core.path(row['tile']),allow_pickle=False) as tile:
                modis.add(tile['modis_baseline'],tile['target'],tile['weight'])
                constant.add(np.full(tile['target'].shape,prevalence),tile['target'],tile['weight'])
        baselines={'official_modis_on_same_reference_support':modis.result(),
            'constant_training_prevalence':constant.result()}
    recorded=[]
    log=core.path(f'logs/{run}/epochs.jsonl')
    if log.exists():recorded=core.read_jsonl(log)
    best=min(recorded,key=lambda r:r['validation']['logloss']) if recorded else None
    problems=[]
    if training['acquisition_event_groups']<100:problems.append('Below the project feasibility target of 100 independent reference-event groups; current acquisition groups are conservative mergers, not verified incidents.')
    if training['tiles']<10000:problems.append('Below the project feasibility target of 10,000 tiles; repeated/overlapping crops do not replace independent events.')
    if training['positive_reference_queries']<1000:problems.append('Very few positive training labels; increasing epochs cannot supply missing fire diversity. The 1,000-label flag is an investigation heuristic, not a sufficiency threshold.')
    if not partitions['test']:problems.append('No independent sealed test partition.')
    if not all(r.get('forest_only') for r in rows):problems.append('Dataset is not forest-only; prior land-cover context is required for that claim.')
    report={'manifest':manifest,'manifest_sha256':core.sha256(manifest),'partitions':summaries,
        'training_reference_prevalence':prevalence,'recorded_run':run,
        'best_recorded_validation':best['validation'] if best else None,
        'best_recorded_epoch':best['epoch'] if best else None,'validation_baselines':baselines,
        'issues':problems,'usefulness_established':False,
        'acceptance':'Use a frozen independent test to demonstrate better reference detection than MODIS at matched false-alarm rate, calibration and acceptable region/day-night behavior. Pixel counts alone do not establish usefulness.',
        'scope':'VIIRS-reference thermal/fire activity, not independently verified wildfire truth'}
    core.write_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/manifests/real_training_pilot_v1.jsonl')
    p.add_argument('--run',default='real_training_pilot_v1')
    p.add_argument('--output',default='results/useful_training_readiness.json')
    a=p.parse_args();print(json.dumps(assess(a.manifest,a.run,a.output),indent=2))
