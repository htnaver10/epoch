"""Float64 geolocation inversion with explicit residual/coverage checks."""
import numpy as np


def wrap_lon(lon, center):
    return center + (np.asarray(lon, np.float64)-center+180)%360-180


def bilinear(field, row, col):
    h,w = field.shape
    r = np.clip(row,0,h-1.000001); c = np.clip(col,0,w-1.000001)
    i,j = np.floor(r).astype(int),np.floor(c).astype(int)
    u,v = r-i,c-j
    a,b,e,d = field[i,j],field[i+1,j],field[i,j+1],field[i+1,j+1]
    value=(1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*e+u*v*d
    dr=(1-v)*(b-a)+v*(d-e)
    dc=(1-u)*(e-a)+u*(d-b)
    return value,dr,dc


def inverse_geolocation(lat, lon, query_lat, query_lon, tolerance_m=20.):
    """Invert smooth local swath coordinates; reject folds, holes and extrapolation.

    Bilinear geolocation is an explicit approximation, not an exact sensor PSF.
    Caller must audit it on actual native files. Invalid geography is rejected.
    """
    lat=np.asarray(lat,np.float64);lon=np.asarray(lon,np.float64)
    qlat=np.asarray(query_lat,np.float64);qlon=np.asarray(query_lon,np.float64)
    if lat.ndim!=2 or min(lat.shape)<2 or lat.shape!=lon.shape:
        raise ValueError('A 2D native latitude/longitude grid is required')
    if not (np.isfinite(lat).all() and np.isfinite(lon).all() and np.all(np.abs(lat)<=90)):
        raise ValueError('Native geolocation has invalid samples; reject this source window')
    h,w=lat.shape; center=float(lon[h//2,w//2]);lon=wrap_lon(lon,center)
    valid_query=np.isfinite(qlat)&np.isfinite(qlon)&(np.abs(qlat)<=90)&(np.abs(qlon)<=180)
    qlat=np.where(valid_query,qlat,float(np.median(lat)))
    qlon=wrap_lon(np.where(valid_query,qlon,center),center)
    rr,cc=np.indices(lat.shape,dtype=float)
    design=np.column_stack([np.ones(lat.size),rr.ravel(),cc.ravel()])
    coefficients=np.linalg.lstsq(design,np.column_stack([lat.ravel(),lon.ravel()]),rcond=None)[0]
    jac=np.array([[coefficients[1,0],coefficients[2,0]],[coefficients[1,1],coefficients[2,1]]])
    if abs(np.linalg.det(jac))<1e-14:
        raise ValueError('Degenerate native geolocation')
    rc=np.linalg.solve(jac,np.stack([qlat.ravel()-coefficients[0,0],qlon.ravel()-coefficients[0,1]]))
    row,col=rc[0],rc[1]; q1,q2=qlat.ravel(),qlon.ravel()
    for _ in range(12):
        a,ar,ac=bilinear(lat,row,col);b,br,bc=bilinear(lon,row,col)
        det=ar*bc-ac*br
        good=np.isfinite(det)&(np.abs(det)>1e-14)
        dr=np.divide((a-q1)*bc-(b-q2)*ac,det,out=np.zeros_like(row),where=good)
        dc=np.divide((b-q2)*ar-(a-q1)*br,det,out=np.zeros_like(col),where=good)
        row-=np.clip(dr,-8,8);col-=np.clip(dc,-8,8)
    a,_,_=bilinear(lat,row,col);b,_,_=bilinear(lon,row,col)
    residual=111195*np.sqrt((a-q1)**2+((b-q2)*np.cos(np.deg2rad(q1)))**2)
    valid=(valid_query.ravel()&np.isfinite(row)&np.isfinite(col)&np.isfinite(residual)&(residual<=tolerance_m)
           &(row>=0)&(row<=h-1)&(col>=0)&(col<=w-1)&(np.abs(q1)<=90))
    shape=qlat.shape
    grid=np.stack([2*(col+.5)/w-1,2*(row+.5)/h-1],-1).reshape(*shape,2)
    grid[~valid.reshape(shape)]=2.
    return grid.astype(np.float32),valid.reshape(shape),residual.reshape(shape),row.reshape(shape),col.reshape(shape)


def reference_targets(classes, qa, grid, geometric_valid, native_shape, halo=16, forest=None):
    """Primary label: VIIRS classes 8/9 vs observed land 5; others unknown.

    Native auxiliary target is the mean reference-fire fraction among known
    query centers assigned to each native cell. It is not any-fire probability.
    """
    h,w=native_shape
    row=(grid[...,1]+1)*h/2-.5;col=(grid[...,0]+1)*w/2-.5
    valid=geometric_valid & np.isin(classes,[5,8,9]) & ((qa.astype(np.uint32)&(1<<22))==0)
    valid &= (row>=halo)&(row<h-halo)&(col>=halo)&(col<w-halo)
    if forest is not None:valid &= forest
    target=np.isin(classes,[8,9]).astype(np.float32)
    weights=valid.astype(np.float32)
    iy=np.clip(np.floor(row+.5).astype(int),0,h-1);ix=np.clip(np.floor(col+.5).astype(int),0,w-1)
    index=(iy*w+ix).ravel()
    n=np.bincount(index,weights=weights.ravel(),minlength=h*w)
    p=np.bincount(index,weights=(weights*target).ravel(),minlength=h*w)
    native=np.divide(p,n,out=np.zeros_like(p),where=n>0).reshape(h,w)
    return target,weights,native.astype(np.float32),n.reshape(h,w).astype(np.float32)


def geographic_queries(lat,lon,spacing_m=375.,max_side=384):
    try:
        from pyproj import CRS,Transformer
    except ImportError as e:
        raise RuntimeError('Install requirements-data.txt: pyproj is required for MODIS-only metric query grids') from e
    cy,cx=float(np.median(lat)),float(np.median(lon))
    crs=CRS.from_proj4(f'+proj=laea +lat_0={cy} +lon_0={cx} +datum=WGS84 +units=m')
    forward=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    inverse=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    x,y=forward.transform(lon,lat)
    xs=np.arange(np.nanmin(x),np.nanmax(x)+spacing_m*.5,spacing_m)
    ys=np.arange(np.nanmin(y),np.nanmax(y)+spacing_m*.5,spacing_m)
    if max(len(xs),len(ys))>max_side:
        raise ValueError('Query extent exceeds configured cap; reduce native crop extent or explicitly raise/profile cap')
    xx,yy=np.meshgrid(xs,ys);qlon,qlat=inverse.transform(xx,yy)
    return np.asarray(qlat,np.float64),np.asarray(qlon,np.float64)
