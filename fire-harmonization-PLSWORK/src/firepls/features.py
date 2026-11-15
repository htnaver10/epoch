"""Label-independent MODIS radiance, annular-context and finer reflectance features."""
import numpy as np

from .common import config

THERMAL=(20,21,22,23,31,32)


def box_sum(array,size):
    if size<1 or size%2!=1:raise ValueError('Neighborhood windows must be positive odd numbers')
    radius=size//2
    a=np.pad(array.astype(np.float64),radius,mode='constant')
    integral=np.pad(a,((1,0),(1,0))).cumsum(0).cumsum(1)
    return integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]


def annular_statistics(values,valid,outer,inner,minimum):
    if inner>=outer:raise ValueError('Background exclusion must be smaller than its window')
    count=np.maximum(box_sum(valid,outer)-box_sum(valid,inner),0)
    total=box_sum(values*valid,outer)-box_sum(values*valid,inner)
    square=box_sum(values**2*valid,outer)-box_sum(values**2*valid,inner)
    supported=count>=minimum
    mean=np.divide(total,count,out=np.zeros_like(total),where=supported)
    variance=np.divide(square,count,out=np.zeros_like(total),where=supported)-mean**2
    std=np.where(supported,np.sqrt(np.maximum(variance,0)),0)
    return mean,std,supported,count/(outer**2-inner**2)


def query_indices(grid,height,width,scale=1):
    if grid.ndim!=3 or grid.shape[-1]!=2 or not np.isfinite(grid).all():raise ValueError('Query grid must be finite HxWx2')
    # Native pixel centers use grid_sample align_corners=False convention.
    row=(grid[...,1]+1)*height/2-.5;col=(grid[...,0]+1)*width/2-.5
    iy=np.clip(np.floor((row+.5)*scale).astype(int),0,height*scale-1)
    ix=np.clip(np.floor((col+.5)*scale).astype(int),0,width*scale-1)
    return iy,ix,row,col


def extract(inputs,cfg=None):
    """No labels, fire mask, baseline detections or geographic coordinates allowed."""
    cfg=cfg or config()
    required={'thermal','reflective_500','reflective_250','geometry','query_grid'}
    if set(inputs)!=required:raise ValueError('Feature extractor accepts MODIS inputs and query_grid only')
    thermal=inputs['thermal'];h,w=thermal.shape[-2:];grid=inputs['query_grid']
    if thermal.shape!=(12,h,w) or inputs['geometry'].shape!=(14,h,w):raise ValueError('Invalid native MODIS shape')
    if not all(np.isfinite(v).all() for v in inputs.values()):raise ValueError('Nonfinite input imagery')
    iy,ix,row,col=query_indices(grid,h,w);features=[];names=[]
    def add(name,array):names.append(name);features.append(array.astype(np.float32))
    raw=thermal[:6];valid=thermal[6:]>0
    log=np.sign(raw)*np.log1p(np.abs(raw))
    for i,band in enumerate(THERMAL):
        add(f'log_radiance_{band}',log[i,iy,ix]*valid[i,iy,ix]);add(f'thermal_valid_{band}',valid[i,iy,ix])
    for size in cfg['context_windows']:
        for i in (0,2,4,5):
            mean,std,support,fraction=annular_statistics(log[i],valid[i],size,cfg['background_exclusion_window'],cfg['minimum_background_pixels'])
            delta=np.where(support&valid[i],log[i]-mean,0)
            z=np.clip(delta/np.maximum(std,.01),-cfg['contrast_z_clip'],cfg['contrast_z_clip'])
            for key,value in [('mean',mean),('std',std),('delta',delta),('z',z),('valid_fraction',fraction)]:
                add(f'band{THERMAL[i]}_ring{size}_{key}',value[iy,ix])
            add(f'band{THERMAL[i]}_ring{size}_supported',support[iy,ix])
    for a,b in ((0,4),(2,4),(4,5)):
        good=valid[a]&valid[b]
        add(f'log_contrast_{THERMAL[a]}_{THERMAL[b]}',((log[a]-log[b])*good)[iy,ix])
        add(f'contrast_valid_{THERMAL[a]}_{THERMAL[b]}',good[iy,ix])
    for key,scale,n,bands in [('reflective_500',2,5,(3,4,5,6,7)),('reflective_250',4,2,(1,2))]:
        arr=inputs[key]
        if arr.shape!=(2*n,scale*h,scale*w):raise ValueError('Invalid native reflective shape')
        ry,rx,_,_=query_indices(grid,h,w,scale)
        for i,band in enumerate(bands):
            ok=arr[n+i,ry,rx]>0
            add(f'reflectance_{band}',arr[i,ry,rx]*ok);add(f'reflectance_valid_{band}',ok)
    for i in range(14):add(f'geometry_{i}',inputs['geometry'][i,iy,ix])
    add('query_native_row_offset',row-iy);add('query_native_col_offset',col-ix)
    result=np.stack(features,-1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite contextual feature')
    return result,names
