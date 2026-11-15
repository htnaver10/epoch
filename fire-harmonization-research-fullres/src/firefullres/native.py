"""Native NASA readers. Fail on missing calibration; never treat CSV counts as imagery."""
from __future__ import annotations

import re
from datetime import datetime,timezone
import numpy as np
from .core import path, THERMAL_BANDS


def gdal_module():
    try:
        from osgeo import gdal
    except ImportError as e:
        raise RuntimeError('Native MODIS requires GDAL with HDF4/HDF4Image support; see RUNNING.md') from e
    gdal.UseExceptions()
    if gdal.GetDriverByName('HDF4') is None or gdal.GetDriverByName('HDF4Image') is None:
        raise RuntimeError('This GDAL build lacks HDF4/HDF4Image; HDF5 support alone is insufficient')
    return gdal


def sds(filename, name):
    gdal=gdal_module();root=gdal.Open(str(path(filename)))
    # MODIS also exposes e.g. EV_1KM_Emissive_Uncert_Indexes. Substring
    # matching selected both arrays and refused every real L1B granule.
    # Match a complete SDS name, allowing GDAL's spaces/underscores in names
    # such as "fire mask", while keeping suffixes and prefixes distinct.
    tokens=re.split(r'[^a-zA-Z0-9]+',name.strip())
    pattern=r'(?<![a-zA-Z0-9_])'+r'[\s_]+'.join(map(re.escape,tokens))+r'(?![a-zA-Z0-9_])'
    candidates=[uri for uri,description in root.GetSubDatasets() if re.search(pattern,description,re.IGNORECASE)]
    if not candidates:
        # GDAL hides raw SDS when EOS swaths exist. MYD03 Latitude/Longitude
        # are geolocation fields and are absent from the default swath list.
        # Only retry a missing name and only inspect raw SDS, so the expanded
        # list cannot duplicate an already selected EOS science field.
        expanded=gdal.OpenEx(str(path(filename)),gdal.OF_RASTER,open_options=['LIST_SDS=YES'])
        candidates=[uri for uri,description in expanded.GetSubDatasets()
                    if uri.startswith('HDF4_SDS:') and re.search(pattern,description,re.IGNORECASE)]
    if len(candidates)!=1:
        raise ValueError(f'Expected one {name} subdataset in {path(filename).name}; found {len(candidates)}')
    return gdal.Open(candidates[0])


def numbers(value):
    if isinstance(value,str):
        return np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',value)])
    return np.asarray(value).reshape(-1).astype(float)


def calibrated_values(raw, metadata, indices, kind):
    """MODIS L1B documented convention: scale * (scaled integer - offset)."""
    for key in (kind+'_scales',kind+'_offsets','valid_range'):
        if key not in metadata:raise ValueError(f'Missing MODIS calibration attribute: {key}')
    scales=numbers(metadata[kind+'_scales'])[indices]
    offsets=numbers(metadata[kind+'_offsets'])[indices]
    bounds=numbers(metadata['valid_range'])
    if len(bounds)!=2:raise ValueError('Expected a two-value native valid_range')
    valid=(raw>=bounds[0])&(raw<=bounds[1])
    if '_FillValue' in metadata:valid &= raw!=numbers(metadata['_FillValue'])[0]
    values=scales[:,None,None]*(raw.astype(np.float64)-offsets[:,None,None])
    valid &= np.isfinite(values)
    return np.where(valid,values,0).astype(np.float32),valid.astype(np.float32)


def read_calibrated(filename, variable, bands, window, kind):
    ds=sds(filename,variable);meta=ds.GetMetadata()
    if 'band_names' not in meta:raise ValueError(f'Missing band_names in {variable}')
    names=[x.strip() for x in meta['band_names'].split(',')]
    indices=[names.index(str(b)) for b in bands]
    r,c,h,w=window
    raw=np.stack([ds.GetRasterBand(i+1).ReadAsArray(c,r,w,h) for i in indices])
    values,valid=calibrated_values(raw,meta,indices,kind)
    return np.concatenate([values,valid],axis=0)


def read_modis_grid(filename, variable, window=None, scaled=False):
    ds=sds(filename,variable)
    a=ds.ReadAsArray() if window is None else ds.ReadAsArray(window[1],window[0],window[3],window[2])
    if scaled:
        meta=ds.GetMetadata();valid=np.isfinite(a)
        if '_FillValue' in meta:valid &= a!=numbers(meta['_FillValue'])[0]
        if 'valid_range' in meta:
            lo,hi=numbers(meta['valid_range']);valid &= (a>=lo)&(a<=hi)
        scale=numbers(meta.get('scale_factor','1'))[0]
        offset=numbers(meta.get('add_offset','0'))[0]
        a=np.where(valid,a.astype(np.float64)*scale+offset,np.nan)
    return a


def h5_field(filename, name):
    try:import h5py
    except ImportError as e:raise RuntimeError('Install h5py to read VIIRS NetCDF4/HDF5') from e
    norm=lambda s:re.sub('[^a-z0-9]','',s.lower())
    with h5py.File(path(filename),'r') as f:
        names=[]
        f.visititems(lambda key,obj:names.append(key) if isinstance(obj,h5py.Dataset) and norm(key.rsplit('/',1)[-1])==norm(name) else None)
        if len(names)!=1:raise ValueError(f'Expected one VIIRS {name} array; found {len(names)}')
        a=f[names[0]][...]
    return a


def forest_at(files, lat, lon):
    """Strict MCD12Q1 C61: IGBP classes 1..5, QC 0 classified land only."""
    try:from pyproj import Transformer
    except ImportError as e:raise RuntimeError('Install pyproj for forest-map reprojection') from e
    gdal=gdal_module();forest=np.zeros(lat.shape,np.float32);known=np.zeros(lat.shape,bool)
    for filename in files:
        land=sds(filename,'LC_Type1');quality=sds(filename,'QC')
        transform=land.GetGeoTransform();inv=gdal.InvGeoTransform(transform)
        projection=land.GetProjection()
        if not projection or inv is None:raise ValueError('MCD12Q1 requires a valid georeferenced grid')
        x,y=Transformer.from_crs('EPSG:4326',projection,always_xy=True).transform(lon,lat)
        col=np.floor(inv[0]+inv[1]*x+inv[2]*y).astype(int)
        row=np.floor(inv[3]+inv[4]*x+inv[5]*y).astype(int)
        inside=(row>=0)&(row<land.RasterYSize)&(col>=0)&(col<land.RasterXSize)
        iy=np.clip(row,0,land.RasterYSize-1);ix=np.clip(col,0,land.RasterXSize-1)
        classes=land.ReadAsArray()[iy,ix];qc=quality.ReadAsArray()[iy,ix]
        ok=inside&(qc==0)&(classes>=1)&(classes<=17)
        values=np.isin(classes,[1,2,3,4,5]).astype(np.float32)
        if np.any(known&ok&(forest!=values)):raise ValueError('Conflicting forest-map coverage')
        forest[ok]=values[ok];known|=ok
    return forest,known


def check_bundle_sources(bundle, supervised=True):
    required={'modis_1km':'MYD021KM','modis_geo':'MYD03','modis_fire':'MYD14'}
    if supervised:required.update(viirs_fire='VNP14IMG',viirs_geo='VNP03IMG')
    optional={'modis_500m':'MYD02HKM','modis_250m':'MYD02QKM'}
    acquisitions={}
    for key,product in {**required,**optional}.items():
        filename=bundle.get(key)
        if not filename:
            if key in required:raise ValueError(f'Missing required bundle file: {key}')
            continue
        p=path(filename)
        if not p.is_file() or not p.name.startswith(product+'.'):
            raise ValueError(f'{key} must be a real {product} native granule')
        collection='002' if product.startswith('VNP') else '061'
        if f'.{collection}.' not in p.name:raise ValueError(f'Expected {collection} processing for {p.name}')
        match=re.search(r'\.A(\d{7})\.(\d{4})\.',p.name)
        if match is None:raise ValueError('Unsupported native granule name')
        sensor='viirs' if product.startswith('VNP') else 'modis'
        acq=match.group(0)
        if sensor in acquisitions and acquisitions[sensor]!=acq:raise ValueError('Companion files are from different acquisitions')
        acquisitions[sensor]=acq
        date=datetime.strptime(match.group(1)+match.group(2),'%Y%j%H%M').replace(tzinfo=timezone.utc)
        start=bundle.get(sensor+'_start_utc')
        if start:
            from .core import utc
            if abs((utc(start)-date).total_seconds())>=60:raise ValueError('Declared acquisition start does not match granule filename')


def native_patch(bundle, r, c, size, lat, lon):
    window=(r,c,size,size)
    thermal=read_calibrated(bundle['modis_1km'],'EV_1KM_Emissive',THERMAL_BANDS,window,'radiance')
    ref500=read_calibrated(bundle['modis_500m'],'EV_500_RefSB',[3,4,5,6,7],(2*r,2*c,2*size,2*size),'reflectance') if bundle.get('modis_500m') else np.zeros((10,2*size,2*size),np.float32)
    ref250=read_calibrated(bundle['modis_250m'],'EV_250_RefSB',[1,2],(4*r,4*c,4*size,4*size),'reflectance') if bundle.get('modis_250m') else np.zeros((4,4*size,4*size),np.float32)
    angles=[read_modis_grid(bundle['modis_geo'],n,window,True) for n in ['SensorZenith','SensorAzimuth','SolarZenith','SolarAzimuth']]
    for a in angles:
        if np.any(np.abs(a[np.isfinite(a)])>360):raise ValueError('Invalid angle scale/units in MOD03')
    geo=[];valid=[]
    for a in angles:
        ok=np.isfinite(a);rad=np.deg2rad(np.where(ok,a,0))
        geo.extend([np.sin(rad)*ok,np.cos(rad)*ok]);valid.append(ok)
    if bundle.get('land_cover'):
        forest,known=forest_at(bundle['land_cover'],lat,lon)
    else:forest,known=np.zeros(lat.shape),np.zeros(lat.shape,bool)
    geometry=np.stack([*geo,*valid,forest,known]).astype(np.float32)
    classes=read_modis_grid(bundle['modis_fire'],'fire mask',window)
    usable=np.isin(classes,[5,7,8,9])&(thermal[6:].sum(axis=0)>=2)&np.isfinite(angles[0])&(angles[0]<=40)&(angles[0]>=0)
    return {'thermal':thermal,'reflective_500':ref500,'reflective_250':ref250,'geometry':geometry},usable,np.isin(classes,[7,8,9])
