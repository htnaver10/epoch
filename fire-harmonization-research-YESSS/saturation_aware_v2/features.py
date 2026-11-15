"""Native MODIS blind-spot features: central target measurements are excluded."""
import numpy as np
from common import config

INPUT_KEYS={'thermal','reflective_500','reflective_250','geometry'}

def check(inputs):
    if set(inputs)!=INPUT_KEYS:raise ValueError('Only native MODIS measurements are accepted; no fire labels or query grids')
    h,w=inputs['thermal'].shape[-2:]
    expected={'thermal':(12,h,w),'reflective_500':(10,2*h,2*w),'reflective_250':(4,4*h,4*w),'geometry':(14,h,w)}
    if any(inputs[k].shape!=v or not np.isfinite(inputs[k]).all() for k,v in expected.items()):raise ValueError('Invalid MODIS input arrays')
    return h,w

def signed_log(x):return np.sign(x)*np.log1p(np.abs(x))

def box(a,size):
    r=size//2;v=np.pad(np.asarray(a,np.float64),r);s=np.pad(v,((1,0),(1,0))).cumsum(0).cumsum(1)
    return s[size:,size:]-s[:-size,size:]-s[size:,:-size]+s[:-size,:-size]

def ring(values,valid,outer,inner=3):
    good=(valid>0)&np.isfinite(values);v=np.where(good,values,0)
    n=np.maximum(box(good,outer)-box(good,inner),0)
    total=box(v,outer)-box(v,inner);sq=box(v*v,outer)-box(v*v,inner)
    mean=np.divide(total,n,out=np.zeros_like(total),where=n>=8)
    variance=np.divide(sq,n,out=np.zeros_like(sq),where=n>=8)-mean*mean
    std=np.where(n>=8,np.sqrt(np.maximum(variance,0)),0)
    return mean,std,n/(outer*outer-inner*inner),(n>=8).astype(float)

def interior(shape,halo):
    out=np.zeros(shape,bool)
    if min(shape)>2*halo:out[halo:-halo,halo:-halo]=True
    return out

def geometric_valid(geo):
    return (geo[8]>.5)&(geo[10]>.5)&(geo[1]>=np.cos(np.deg2rad(40)))

def coarse(inputs):
    h,w=check(inputs);a=inputs['thermal'];geo=inputs['geometry'];v=a[6:]>0
    values=signed_log(a[:6]);f=[];names=[];support=[]
    for outer in (7,21):
        for band in range(6):
            parts=ring(values[band],v[band],outer)
            for key,p in zip(('mean','std','fraction','supported'),parts):
                f.append(p);names.append(f'band{band}_ring{outer}_{key}')
            if band==1:support.append(parts[3]>.5)
    for band in (4,5):
        f.extend([values[band]*v[band],v[band]])
        names.extend([f'central_longwave_{band}',f'central_longwave_{band}_valid'])
    for i in range(12):f.append(geo[i]);names.append(f'geometry_{i}')
    x=np.stack(f,-1).astype(np.float32)
    measured=np.stack([values[1],values[2]],-1).astype(np.float32)
    # Keep unsaturated band 21 when bands 20/22 are unavailable. No numeric code is imputed.
    target_valid=np.stack([v[1],v[2]],-1)
    valid=interior((h,w),config()['context_radius'])&geometric_valid(geo)&(v[1]|v[2])&v[4]&v[5]&np.logical_and.reduce(support)
    day=geo[5]>=config()['minimum_daylight_cosine']
    return x,measured,valid,day,names,target_valid

def pool2(value,valid):
    h,w=value.shape;good=valid>0
    count=good.reshape(h//2,2,w//2,2).sum((1,3))
    total=np.where(good,value,0).reshape(h//2,2,w//2,2).sum((1,3),dtype=np.float64)
    return np.divide(total,count,out=np.zeros_like(total),where=count>0),count/4.

def fine(inputs):
    h,w=check(inputs);a=inputs['reflective_500'];v=a[5:]>0;values=signed_log(a[:5])
    geo=np.repeat(np.repeat(inputs['geometry'],2,1),2,2);f=[];names=[];supports=[]
    # Central band 7 (index 4) is never in its predictor, nor any other central SWIR band.
    for outer in (7,15):
        for band in (0,1,4):
            parts=ring(values[band],v[band],outer)
            for key,p in zip(('mean','std','fraction','supported'),parts):
                f.append(p);names.append(f'reflective{band}_ring{outer}_{key}')
            if band==4:supports.append(parts[3]>.5)
    for band in (0,1):
        f.extend([values[band]*v[band],v[band]]);names.extend([f'central_visible_{band}',f'visible_valid_{band}'])
    for band in (0,1):
        mean,support=pool2(inputs['reflective_250'][band],inputs['reflective_250'][2+band])
        f.extend([signed_log(mean),support]);names.extend([f'pooled_250_{band}',f'pooled_250_valid_{band}'])
    for i in range(12):f.append(geo[i]);names.append(f'geometry_{i}')
    day=geo[5]>=config()['minimum_daylight_cosine']
    valid=interior((2*h,2*w),2*config()['context_radius'])&geometric_valid(geo)&day&v[4]&np.logical_and.reduce(supports)
    return np.stack(f,-1).astype(np.float32),values[4,...,None].astype(np.float32),valid,day,names,v[4,...,None]

def extract(inputs,stage):
    if stage=='coarse':return coarse(inputs)
    if stage=='fine':return fine(inputs)
    raise ValueError('Unknown stage')
