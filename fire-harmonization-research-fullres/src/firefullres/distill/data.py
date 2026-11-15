"""Exact native VIIRS thermal crops aligned to immutable MODIS query tiles."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import core
from ..native import h5_field,read_modis_grid
from ..prepare import query_crop
from .acquisition import acquisition_time,parent_rows

SCHEMA='viirs_thermal_teacher_v1'
BANDS=('I04','I05')


def text_attr(value):
    if isinstance(value,bytes):return value.decode('utf-8')
    if isinstance(value,np.ndarray) and value.size==1:return text_attr(value.item())
    return str(value)


def scalar(attrs,name):
    if name not in attrs:raise ValueError('Missing VIIRS calibration attribute: '+name)
    value=np.asarray(attrs[name])
    if value.size!=1 or not np.isfinite(value.astype(float)).all():
        raise ValueError('Invalid VIIRS calibration scalar: '+name)
    return float(value.item())


def calibrate(raw,quality,attrs,qa_attrs):
    """CF thermal radiance = SI*scale_factor + add_offset (not MODIS scaling)."""
    scale=scalar(attrs,'scale_factor');offset=scalar(attrs,'add_offset')
    lo=scalar(attrs,'valid_min');hi=scalar(attrs,'valid_max');fill=scalar(attrs,'_FillValue')
    units=text_attr(attrs.get('units','')).lower()
    if scale<=0 or lo<0 or hi>65527 or lo>hi or not ('watt' in units and 'steradian' in units):
        raise ValueError('Unsupported VIIRS thermal units, scale or valid range')
    if raw.shape!=quality.shape or raw.dtype.kind!='u' or quality.dtype.kind!='u':
        raise ValueError('VIIRS thermal values and quality must be paired unsigned arrays')
    if 'flag_masks' not in qa_attrs or 'flag_meanings' not in qa_attrs:
        raise ValueError('Missing VIIRS quality flag definitions')
    masks=np.asarray(qa_attrs['flag_masks'],dtype=np.uint32).reshape(-1)
    meanings=text_attr(qa_attrs['flag_meanings']).split()
    if len(masks)!=len(meanings) or not len(masks):raise ValueError('Inconsistent VIIRS quality flag definitions')
    # Gain states are descriptive; all other advertised quality anomalies are excluded.
    bad=0
    for mask,meaning in zip(masks,meanings):
        if meaning not in ('Low_Gain','Mixed_Gain'):bad|=int(mask)
    known_bits=int(np.bitwise_or.reduce(masks))
    quality_ok=((quality.astype(np.uint32)&np.uint32(bad))==0)&((quality.astype(np.uint32)&np.uint32(65535^known_bits))==0)
    valid=(raw>=lo)&(raw<=hi)&(raw!=fill)&quality_ok
    values=raw.astype(np.float64)*scale+offset
    valid &= np.isfinite(values)&(values>=0)
    return np.where(valid,values,0).astype(np.float32),valid


def read_radiances(filename):
    import h5py
    if not Path(filename).name.startswith('VNP02IMG.') or '.002.' not in Path(filename).name:
        raise ValueError('Teacher requires VNP02IMG collection 002 imagery')
    values=[];valid=[]
    with h5py.File(core.path(filename),'r') as stream:
        for name in BANDS:
            if f'observation_data/{name}' not in stream or f'observation_data/{name}_quality_flags' not in stream:
                raise ValueError('Missing mandatory VIIRS thermal band/quality: '+name)
            band=stream[f'observation_data/{name}'];quality=stream[f'observation_data/{name}_quality_flags']
            a,b=calibrate(band[...],quality[...],dict(band.attrs),dict(quality.attrs))
            values.append(a);valid.append(b)
    if values[0].shape!=values[1].shape:raise ValueError('VIIRS thermal band dimensions disagree')
    return np.stack(values),np.stack(valid)


def verify_crop(qlat,qlon,student_arrays):
    for value,key in ((qlat,'query_lat'),(qlon,'query_lon')):
        expected=np.where(np.isfinite(value),value,0).astype(np.float64)
        if expected.shape!=student_arrays[key].shape or not np.array_equal(expected,student_arrays[key]):
            raise ValueError('VIIRS imagery/query coordinates do not match the frozen student tile')


def validate_teacher_arrays(arrays,query_shape):
    if set(arrays)!= {'radiance','valid'}:raise ValueError('Teacher tiles contain imagery and validity only')
    if arrays['radiance'].shape!=(2,*query_shape) or arrays['valid'].shape!=(2,*query_shape):
        raise ValueError('Teacher imagery/query shapes disagree')
    if arrays['radiance'].dtype!=np.float32 or arrays['valid'].dtype!=np.bool_:
        raise ValueError('Teacher imagery must be float32 with Boolean validity')
    if not np.isfinite(arrays['radiance']).all() or (arrays['radiance']<0).any():raise ValueError('Invalid VIIRS thermal radiance')
    if np.any(arrays['radiance'][~arrays['valid']]!=0):raise ValueError('Invalid radiance must be zero with a false validity flag')


def validate_pairs(manifest,config,allow_synthetic=False,hashes=True):
    parents=parent_rows(config,allow_synthetic);pairs=core.read_jsonl(manifest)
    expected={r['tile_id']:r for r in parents};seen=set();source_splits={}
    for row in pairs:
        identity=row['tile_id']
        if identity in seen or identity not in expected:raise ValueError('Duplicate or foreign teacher/student tile')
        seen.add(identity);parent=expected[identity]
        for key in ('tile','sha256','split','event_group','region_id','source_ids','query_shape'):
            if row[key]!=parent[key]:raise ValueError('Paired metadata changed the frozen parent partition or tile')
        if row['teacher_schema']!=SCHEMA:raise ValueError('Unknown teacher tile schema')
        for source in (*row['source_ids'],row['viirs_l1b']):
            if source in source_splits and source_splits[source]!=row['split']:raise ValueError('VIIRS source leakage across partitions')
            source_splits[source]=row['split']
        if hashes and core.sha256(row['teacher_tile'])!=row['teacher_sha256']:raise ValueError('Teacher tile checksum changed')
    if seen!=set(expected):raise ValueError('Teacher/student manifest must retain every parent tile')
    return pairs


def prepare_pairs(config):
    name=config['name'];manifest=f'data/manifests/{name}_pairs.jsonl'
    if core.path(manifest).exists():
        validate_pairs(manifest,config)
        if not core.read_json(f'results/{name}/paired_data_audit.json')['ready_for_teacher_training']:
            raise ValueError('Teacher data admission has not passed')
        return manifest
    rows=parent_rows(config);downloads=core.read_json(f'results/{name}/l1b_downloads.json')
    by_geo={r['geo']:r for r in downloads.values()};units=defaultdict(list)
    for row in rows:
        geos=[p for p in row['source_ids'] if Path(p).name.startswith('VNP03IMG.')]
        if len(geos)!=1 or geos[0] not in by_geo:raise ValueError('Missing exact VIIRS radiance/geolocation companion')
        units[geos[0]].append(row)
    paired=[]
    for index,(geo,group) in enumerate(sorted(units.items())):
        raw=by_geo[geo]
        if core.sha256(raw['path'])!=raw['sha256']:raise ValueError('VIIRS radiance source checksum changed')
        if acquisition_time(geo)!=acquisition_time(raw['path']):raise ValueError('VIIRS image/geolocation acquisition mismatch')
        radiance,valid=read_radiances(raw['path'])
        lat=h5_field(geo,'latitude').astype(np.float64);lon=h5_field(geo,'longitude').astype(np.float64)
        if lat.shape!=lon.shape or radiance.shape[1:]!=lat.shape:raise ValueError('VIIRS image/geolocation dimensions disagree')
        modis_cache={}
        for row in group:
            mgeo=next(p for p in row['source_ids'] if Path(p).name.startswith('MYD03.'))
            if mgeo not in modis_cache:
                modis_cache[mgeo]=(read_modis_grid(mgeo,'Latitude').astype(np.float64),read_modis_grid(mgeo,'Longitude').astype(np.float64))
            mlat,mlon=modis_cache[mgeo];r,c,h,w=row['source_window']
            crop=query_crop(lat,lon,mlat[r:r+h,c:c+w],mlon[r:r+h,c:c+w],config['max_query_side'])
            if crop is None:raise ValueError('Frozen student tile lost its native VIIRS query crop')
            if core.sha256(row['tile'])!=row['sha256']:raise ValueError('Frozen student tile checksum changed')
            with np.load(core.path(row['tile']),allow_pickle=False) as student:
                verify_crop(lat[crop],lon[crop],student)
                arrays={'radiance':radiance[(slice(None),*crop)].copy(),'valid':valid[(slice(None),*crop)].copy()}
                validate_teacher_arrays(arrays,tuple(row['query_shape']))
                support=arrays['valid'].any(0);weight=student['weight']*support
                positive=float((weight*student['target']).sum(dtype=np.float64));known=float(weight.sum(dtype=np.float64))
            dest=f'data/tiles/{name}_teacher/{row["tile_id"]}.npz'
            if core.path(dest).exists():
                with np.load(core.path(dest),allow_pickle=False) as old:
                    if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) or old[k].dtype!=v.dtype for k,v in arrays.items()):
                        raise ValueError('Interrupted teacher crop differs; use a new dataset version')
            else:core.write_npz(dest,**arrays)
            paired.append({**row,'teacher_schema':SCHEMA,'teacher_tile':dest,'teacher_sha256':core.sha256(dest),
                'viirs_l1b':raw['path'],'viirs_l1b_sha256':raw['sha256'],
                'viirs_crop':[[s.start,s.stop] for s in crop],
                'teacher_known_weight':known,'teacher_positive_weight':positive})
        print(f'Prepared aligned VIIRS teacher source {index+1}/{len(units)} ({len(paired)} tiles)',flush=True)
    # Preserve the parent's exact row order for controlled student sampling.
    lookup={r['tile_id']:r for r in paired};paired=[lookup[r['tile_id']] for r in rows]
    counts={split:{'tiles':sum(r['split']==split for r in paired),
        'hard_known_weight':sum(r['valid_queries'] for r in paired if r['split']==split),
        'teacher_known_weight':sum(r['teacher_known_weight'] for r in paired if r['split']==split),
        'teacher_positive_weight':sum(r['teacher_positive_weight'] for r in paired if r['split']==split)}
        for split in ('train','validation','test')}
    for split in ('train','validation'):
        if counts[split]['teacher_known_weight']/max(counts[split]['hard_known_weight'],1)<config['minimum_teacher_support_fraction']:
            raise ValueError('Too little valid VIIRS thermal support; inspect teacher coverage')
        if counts[split]['teacher_positive_weight']<=0:raise ValueError('Teacher partition has no supported positive labels')
    core.write_jsonl(manifest,paired);validate_pairs(manifest,config)
    core.write_json(f'results/{name}/paired_data_audit.json',{'ready_for_teacher_training':True,
        'parent_manifest_sha256':config['parent_manifest_sha256'],
        'paired_manifest_sha256':core.sha256(manifest),'counts':counts,'bands':list(BANDS),
        'teacher_input_channels':['log1p_I04_radiance','log1p_I05_radiance','I04_valid','I05_valid'],
        'teacher_target':'existing confident VIIRS reference labels; no fire mask in inputs',
        'parent_partitions_and_tiles_preserved':True})
    return manifest


def fit_statistics(manifest,config,allow_synthetic=False):
    output=f'data/manifests/{config["name"]}_teacher_stats.json'
    if core.path(output).exists():
        stats=core.read_json(output)
        if stats['paired_manifest_sha256']!=core.sha256(manifest):raise ValueError('Teacher statistics belong to another manifest')
        return output
    rows=validate_pairs(manifest,config,allow_synthetic);sums=np.zeros(2);squares=sums.copy();counts=sums.copy()
    positive=known=0.
    for row in rows:
        if row['split']!='train':continue
        with np.load(core.path(row['teacher_tile']),allow_pickle=False) as z:
            values=np.log1p(z['radiance'].astype(np.float64));valid=z['valid']
            sums+=(values*valid).sum((1,2));squares+=(values**2*valid).sum((1,2));counts+=valid.sum((1,2))
        positive+=row['teacher_positive_weight'];known+=row['teacher_known_weight']
    if (counts<=0).any() or not 0<positive<known:raise ValueError('Teacher needs supported positive and negative training observations in both bands')
    mean=sums/counts;scale=np.sqrt(np.maximum(squares/counts-mean**2,1e-8))
    stats={'schema':SCHEMA,'fit_partition':'train','paired_manifest_sha256':core.sha256(manifest),
        'mean':mean.tolist(),'scale':scale.tolist(),'valid_pixels':counts.tolist(),
        'positive_prevalence':float(np.clip(positive/known,1e-6,1-1e-6)),
        'training_tile_ids':[r['tile_id'] for r in rows if r['split']=='train']}
    core.write_json(output,stats);return output


def normalize_teacher(radiance,valid,stats):
    values=(np.log1p(radiance)-np.array(stats['mean'])[:,None,None])/np.array(stats['scale'])[:,None,None]
    return np.concatenate([values*valid,valid]).astype(np.float32)
