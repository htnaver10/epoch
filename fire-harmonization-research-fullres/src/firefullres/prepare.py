from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import numpy as np

from .core import (ROOT, SCHEMA, path, relative, sha256, read_jsonl, write_jsonl, write_json,
                   write_npz, time_bound_seconds, utc, validate_arrays)
from .geometry import inverse_geolocation, reference_targets, geographic_queries, wrap_lon
from .native import check_bundle_sources, read_modis_grid, h5_field, native_patch, forest_at


def check_land_cover_metadata(bundle, required):
    files=bundle.get('land_cover') or []
    if required and not files:raise ValueError('Forest-only mode requires prior MCD12Q1 files')
    if not files:return
    year=bundle.get('land_cover_year');available=bundle.get('land_cover_available_utc')
    if not isinstance(year,int) or not isinstance(available,str):
        raise ValueError('Land-cover year and verified availability timestamp are required whenever used as input')
    acquisition=utc(bundle['modis_start_utc'])
    if year>=acquisition.year or utc(available)>acquisition:
        raise ValueError('Land-cover context leaks future information')
    for filename in files:
        m=re.match(r'MCD12Q1\.A(\d{4})\d{3}\..*\.061\.',path(filename).name)
        if not m or int(m.group(1))!=year:raise ValueError('Land-cover filename collection/year disagree with declared context')


def query_crop(vlat,vlon,lat,lon,max_side):
    """Coarse geographic index then exact inversion; no fine target is truncated."""
    step=16;center=float(np.median(lon));a=vlat[::step,::step];b=wrap_lon(vlon[::step,::step],center)
    l=wrap_lon(lon,center)
    hit=(a>=lat.min()-.02)&(a<=lat.max()+.02)&(b>=l.min()-.02)&(b<=l.max()+.02)&np.isfinite(a)&np.isfinite(b)
    if not hit.any():return None
    rows,cols=np.where(hit)
    r0=max(0,int(rows.min()*step-step));r1=min(vlat.shape[0],int((rows.max()+2)*step))
    c0=max(0,int(cols.min()*step-step));c1=min(vlat.shape[1],int((cols.max()+2)*step))
    if max(r1-r0,c1-c0)>max_side:raise ValueError('query_extent_over_cap')
    return slice(r0,r1),slice(c0,c1)


def prepare(bundles, output_manifest, tile_size=96, stride=64, halo=16, max_query_side=384,
            forest_only=True, inference=False):
    if tile_size%8 or 2*halo>=tile_size or stride!=tile_size-2*halo:
        raise ValueError('Use tile-size divisible by 8 and stride = tile-size - 2*halo to avoid duplicated/gapped evaluation cores')
    if path(output_manifest).exists():raise ValueError('Output manifest already exists; use a new dataset version')
    records=read_jsonl(bundles);rows=[];excluded=[];chosen=set();source_checksums={}
    # Closest interval is selected first for duplicate MODIS windows, independent of fire labels.
    if not inference:records.sort(key=lambda b:(time_bound_seconds(b),str(b.get('viirs_fire'))))
    for b in records:
        if not all(b.get(k) for k in ['event_group','region_id','split']):
            raise ValueError('Review event_group, region_id and split in every bundle before preparation')
        if b['split'] not in (('inference',) if inference else ('train','validation','test')):
            raise ValueError('Invalid split for preparation mode')
        check_bundle_sources(b,supervised=not inference)
        if not inference and time_bound_seconds(b)>900:raise ValueError('Pair interval worst-case time gap exceeds 15 minutes')
        check_land_cover_metadata(b,forest_only)
        sources=[b[k] for k in ['modis_1km','modis_geo','modis_fire','modis_500m','modis_250m','viirs_fire','viirs_geo'] if b.get(k)]
        for p in [*sources,*b.get('land_cover',[])]:
            source_checksums[relative(p)]=sha256(p)
        mlat=read_modis_grid(b['modis_geo'],'Latitude').astype(np.float64)
        mlon=read_modis_grid(b['modis_geo'],'Longitude').astype(np.float64)
        if not inference:
            vlat=h5_field(b['viirs_geo'],'latitude').astype(np.float64)
            vlon=h5_field(b['viirs_geo'],'longitude').astype(np.float64)
            classes=h5_field(b['viirs_fire'],'fire mask')
            qa=h5_field(b['viirs_fire'],'algorithm QA')
            if not(vlat.shape==vlon.shape==classes.shape==qa.shape):raise ValueError('VIIRS companion dimensions disagree')
        for r in range(0,mlat.shape[0]-tile_size+1,stride):
            for c in range(0,mlat.shape[1]-tile_size+1,stride):
                identity=(b['modis_1km'],r,c,tile_size)
                if identity in chosen:continue
                lat=mlat[r:r+tile_size,c:c+tile_size];lon=mlon[r:r+tile_size,c:c+tile_size]
                try:
                    if inference:
                        qlat,qlon=geographic_queries(lat,lon,max_side=max_query_side)
                    else:
                        crop=query_crop(vlat,vlon,lat,lon,max_query_side)
                        if crop is None:continue
                        qlat,qlon=vlat[crop],vlon[crop]
                    grid,inside,residual,ir,ic=inverse_geolocation(lat,lon,qlat,qlon)
                    if not inside.any():continue
                    arrays,usable,modis_fire=native_patch(b,r,c,tile_size,lat,lon)
                    iy=np.clip(np.floor(ir+.5).astype(int),0,tile_size-1)
                    ix=np.clip(np.floor(ic+.5).astype(int),0,tile_size-1)
                    inside &= usable[iy,ix]
                    forest=None
                    if forest_only:
                        f,fvalid=forest_at(b['land_cover'],qlat,qlon);forest=(f>.5)&fvalid
                    if inference:
                        target=np.zeros(qlat.shape,np.float32)
                        eligible=inside&(ir>=halo)&(ir<tile_size-halo)&(ic>=halo)&(ic<tile_size-halo)
                        if forest is not None:eligible &= forest
                        weight=np.zeros_like(target);nt=np.zeros((tile_size,tile_size),np.float32);nw=nt.copy()
                        inference_valid=eligible.astype(np.float32)
                    else:
                        target,weight,nt,nw=reference_targets(classes[crop],qa[crop],grid,inside,(tile_size,tile_size),halo,forest)
                        if weight.sum()==0:continue
                        inference_valid=(inside&(ir>=halo)&(ir<tile_size-halo)&(ic>=halo)&(ic<tile_size-halo)).astype(np.float32)
                        if forest is not None:inference_valid *= forest
                    if inference_valid.sum()==0:continue
                    arrays.update(query_grid=grid,target=target,weight=weight,native_target=nt,native_weight=nw,
                                  query_lat=np.where(np.isfinite(qlat),qlat,0).astype(np.float64),
                                  query_lon=np.where(np.isfinite(qlon),qlon,0).astype(np.float64),
                                  inference_valid=inference_valid,modis_baseline=modis_fire[iy,ix].astype(np.float32))
                    validate_arrays(arrays)
                except ValueError as e:
                    excluded.append({'source':relative(b['modis_1km']),'row':r,'col':c,'reason':str(e)})
                    continue
                identifier=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
                dest=f'data/tiles/{path(output_manifest).stem}/{identifier}.npz'
                if path(dest).exists():raise ValueError('Tile already exists; choose a new manifest/dataset name')
                write_npz(dest,**arrays)
                rows.append({'schema':SCHEMA,'dataset_kind':'nasa_native','tile_id':identifier,'tile':dest,'sha256':sha256(dest),
                    'split':b['split'],'event_group':b['event_group'],'region_id':b['region_id'],
                    'source_ids':[relative(p) for p in sources], 'source_window':[r,c,tile_size,tile_size],
                    'valid_queries':float(weight.sum()),'positive_queries':float((target*weight).sum()),
                    'forest_only':forest_only,'modis_start_utc':b['modis_start_utc'],
                    'interval_time_bound_seconds':None if inference else time_bound_seconds(b),
                    'max_inverse_residual_m':float(residual[inside].max()) if inside.any() else None})
                chosen.add(identity)
        print(f'Prepared {len(rows)} tiles through {path(b["modis_1km"]).name}',flush=True)
    if not rows:
        write_json(f'results/{path(output_manifest).stem}_failed_preparation.json',{'tiles':0,'excluded_windows':excluded})
        raise ValueError('No valid native tiles; inspect the failed_preparation audit for calibration/geolocation exclusions')
    write_jsonl(output_manifest,rows)
    audit={'manifest':relative(output_manifest),'tiles':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
           'tile_size':tile_size,'stride':stride,'halo':halo,'forest_only':forest_only,'inference':inference,
           'source_checksums':source_checksums,'excluded_windows':excluded,'bundle_manifest_sha256':sha256(bundles),
           'target':'native VIIRS nominal/high-confidence fire status vs observed land at query centers',
           'native_auxiliary_target':'mean known VIIRS fire status assigned to each native MODIS cell',
           'geolocation_method':'local bilinear inversion; residual threshold 20m is interpolation consistency, not absolute accuracy',
           'not_claimed':['exact pixel footprints','subpixel ground-truth fire locations','native reader scientific validation on this host']}
    write_json(f'results/{path(output_manifest).stem}_preparation.json',audit)
    return audit
