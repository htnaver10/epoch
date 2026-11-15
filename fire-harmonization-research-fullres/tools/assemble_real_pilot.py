"""Assemble the downloaded real all-land pilot with separated native footprints.

Run under .data-env after both preparation manifests exist. This small pilot has
training and validation only; it is not a forest-only model or final test set.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core,native

TRAIN='data/manifests/real_pilot_all_land_qa.jsonl'
VALIDATION='data/manifests/real_pilot_validation_all_land.jsonl'
OUTPUT='data/manifests/real_training_pilot_v1.jsonl'
STATS='data/manifests/real_training_pilot_v1_stats.json'
REPORT='results/real_training_pilot_v1_audit.json'
VALIDATION_BBOX=(-80.,45.,-65.,55.)


def main():
    if core.path(OUTPUT).exists() or core.path(STATS).exists():
        raise ValueError('Pilot output exists; preserve it and choose a new dataset version')
    train=core.validate_tiles(TRAIN);validation=core.validate_tiles(VALIDATION)
    bundles=core.read_jsonl('data/manifests/real_pilot_all_land_qa_bundles.jsonl')+\
        core.read_jsonl('data/manifests/real_pilot_validation_all_land_bundles.jsonl')
    companions={b['modis_1km']:b['modis_geo'] for b in bundles}
    coordinates={}
    def bounds(row):
        source=next(s for s in row['source_ids'] if Path(s).name.startswith('MYD021KM.'))
        geo=companions[source]
        if geo not in coordinates:
            coordinates[geo]=(native.read_modis_grid(geo,'Latitude'),native.read_modis_grid(geo,'Longitude'))
        latitude,longitude=coordinates[geo];r,c,height,width=row['source_window']
        latitude=latitude[r:r+height,c:c+width];longitude=longitude[r:r+height,c:c+width]
        if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()
                and (np.abs(latitude)<=90).all() and (np.abs(longitude)<=180).all()):
            raise ValueError('Invalid native input footprint in selected pilot tile')
        return (float(longitude.min()),float(latitude.min()),float(longitude.max()),float(latitude.max()))
    training_boxes=[bounds(row) for row in train]
    training_extent=(min(b[0] for b in training_boxes),min(b[1] for b in training_boxes),
                     max(b[2] for b in training_boxes),max(b[3] for b in training_boxes))
    buffered=(training_extent[0]-2,training_extent[1]-2,training_extent[2]+2,training_extent[3]+2)
    kept=[];dropped=[];validation_boxes=[]
    for row in validation:
        box=bounds(row)
        inside=box[0]>=VALIDATION_BBOX[0] and box[1]>=VALIDATION_BBOX[1] and box[2]<=VALIDATION_BBOX[2] and box[3]<=VALIDATION_BBOX[3]
        overlaps=box[0]<=buffered[2] and box[2]>=buffered[0] and box[1]<=buffered[3] and box[3]>=buffered[1]
        if not inside or overlaps:
            dropped.append({'tile_id':row['tile_id'],'native_bbox':box,
                'reason':'outside validation block' if not inside else 'overlaps buffered training footprint'})
        else:kept.append(row);validation_boxes.append(box)
    if not train or not kept:raise ValueError('Both geographic partitions must contain tiles')
    if any(r['split']!='train' for r in train) or any(r['split']!='validation' for r in kept):
        raise ValueError('Unexpected existing partition; do not reassign tiles to manufacture a split')
    rows=train+kept
    if any(r['dataset_kind']!='nasa_native' or r['forest_only'] for r in rows):
        raise ValueError('This pilot requires actual all-land native tiles')
    # Checksums, arrays, shared source/event groups and duplicate IDs are audited
    # together; statistics then use only the frozen training partition.
    core.write_jsonl(OUTPUT,rows);core.validate_tiles(OUTPUT)
    stats=core.statistics(OUTPUT,STATS)
    report={'passed':True,'scope':'small real-NASA all-land training pilot; no final held-out test',
        'manifest':OUTPUT,'manifest_sha256':core.sha256(OUTPUT),'statistics':STATS,
        'input_manifests':{name:core.sha256(name) for name in (TRAIN,VALIDATION)},
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'positive_reference_queries':{split:sum(r['positive_queries'] for r in rows if r['split']==split)
            for split in ('train','validation')},
        'known_reference_queries':{split:sum(r['valid_queries'] for r in rows if r['split']==split)
            for split in ('train','validation')},
        'native_training_bbox':training_extent,'validation_block_bbox':VALIDATION_BBOX,
        'native_validation_bbox':[min(b[0] for b in validation_boxes),min(b[1] for b in validation_boxes),
            max(b[2] for b in validation_boxes),max(b[3] for b in validation_boxes)],
        'training_footprint_buffer_degrees':2,'excluded_validation_tiles':dropped,
        'validation_preview_index':len(train)+max(range(len(kept)),key=lambda i:kept[i]['positive_queries']),
        'normalization_fit_partition':stats['fit_partition'],'forest_only':False,
        'event_group_policy':'Conservatively merge all shared-source acquisitions within each dated regional pilot; no random crop split.',
        'independent_test_available':False,'wildfire_ground_truth_validated':False}
    core.write_json(REPORT,report)
    print(json.dumps({k:v for k,v in report.items() if k!='excluded_validation_tiles'},indent=2))


if __name__=='__main__':main()
