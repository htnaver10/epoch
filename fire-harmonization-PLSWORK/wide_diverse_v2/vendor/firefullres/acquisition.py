"""Bounded NASA inventory/download and proposed pairing; never invent event splits."""
import json
import re
from .core import path,relative,sha256,write_json,write_jsonl,read_jsonl,utc,time_bound_seconds

PRODUCTS=['MYD021KM','MYD03','MYD14','MYD02HKM','MYD02QKM','VNP14IMG','VNP03IMG']
# CMR version strings are provider metadata, not the filename collection code.
# LAADS uses 6.1/2; LP DAAC uses 061/002 for the same supported collections.
CMR_VERSIONS={'MYD021KM':'6.1','MYD03':'6.1','MYD14':'061','MYD02HKM':'6.1',
              'MYD02QKM':'6.1','VNP14IMG':'002','VNP03IMG':'2','MCD12Q1':'061'}


def bbox_from_umm(umm):
    geometry=umm.get('SpatialExtent',{}).get('HorizontalSpatialDomain',{}).get('Geometry',{})
    rectangles=geometry.get('BoundingRectangles',[])
    if rectangles:
        r=rectangles[0];return [r['WestBoundingCoordinate'],r['SouthBoundingCoordinate'],r['EastBoundingCoordinate'],r['NorthBoundingCoordinate']]
    points=[p for polygon in geometry.get('GPolygons',[]) for p in polygon.get('Boundary',{}).get('Points',[])]
    if points:
        x=[p['Longitude'] for p in points];y=[p['Latitude'] for p in points]
        if max(x)-min(x)>180:return None
        return [min(x),min(y),max(x),max(y)]
    return None


def acquire(start,end,bbox,output,download=False,products=None,max_per_product=50):
    try:import earthaccess
    except ImportError as e:raise RuntimeError('Install requirements-data.txt to use NASA acquisition') from e
    if path(output).exists():raise ValueError('Inventory already exists; use a new output filename')
    if len(bbox)!=4 or bbox[0]>=bbox[2] or bbox[1]>=bbox[3]:raise ValueError('Use a non-antimeridian west,south,east,north bounding box')
    if download:
        auth=earthaccess.login(strategy='environment',persist=False)
        if not auth.authenticated:raise ValueError('Earthdata authentication failed; configure environment credentials locally')
    rows=[]
    for product in products or PRODUCTS:
        if product not in [*PRODUCTS,'MCD12Q1']:raise ValueError('Unsupported product in this Aqua/S-NPP baseline')
        version=CMR_VERSIONS[product]
        results=earthaccess.search_data(short_name=product,version=version,bounding_box=tuple(bbox),
                                        temporal=(start,end),count=max_per_product+1)
        if len(results)>max_per_product:raise ValueError(f'{product} exceeds --max-per-product; narrow the pilot or explicitly raise the bound')
        for granule in results:
            umm=granule['umm'];meta=granule['meta'];timing=umm['TemporalExtent'].get('RangeDateTime',{})
            row={'product':product,'version':version,
                 'filename_collection':'002' if product.startswith('VNP') else '061',
                 'granule_id':meta.get('concept-id'),
                 'collection_id':meta.get('collection-concept-id'),'granule_ur':umm['GranuleUR'],
                 'start_utc':timing.get('BeginningDateTime'),'end_utc':timing.get('EndingDateTime'),
                 'bbox':bbox_from_umm(umm),'local_path':None,'sha256':None}
            if download:
                dest=path(f'data/raw/{product}');dest.mkdir(parents=True,exist_ok=True)
                files=earthaccess.download([granule],local_path=str(dest))
                files=[path(str(f)) for f in files if path(str(f)).name.startswith(product+'.') and path(str(f)).suffix in ('.hdf','.nc')]
                if len(files)!=1:raise ValueError('Expected exactly one downloaded science granule; inspect the inventory')
                row.update(local_path=relative(files[0]),sha256=sha256(files[0]))
            rows.append(row)
        print(f'{product}: {len(results)} granules'+(' downloaded' if download else ' inventoried'),flush=True)
    write_jsonl(output,rows)
    write_json(f'results/{path(output).stem}_query.json',{'start':start,'end':end,'bbox':bbox,'downloaded':download,
               'products':products or PRODUCTS,'max_per_product':max_per_product,'granules':len(rows)})
    return {'inventory':relative(output),'granules':len(rows),'downloaded':download}


def propose_pairs(inventory,output):
    if path(output).exists():raise ValueError('Pair file exists; use a new version')
    records=read_jsonl(inventory);groups={};viirs=[]
    def acquisition_key(name):
        m=re.search(r'\.A\d{7}\.\d{4}\.',name)
        return m.group(0) if m else None
    for row in records:
        if not row.get('local_path'):continue
        if sha256(row['local_path'])!=row['sha256']:raise ValueError('Downloaded source checksum mismatch')
        key=acquisition_key(path(row['local_path']).name)
        if key:
            lookup=groups.setdefault(key,{})
            if row['product'] in lookup:raise ValueError('Multiple granules for the same product/acquisition; resolve reprocessing duplicates')
            lookup[row['product']]=row
    for key,g in groups.items():
        if all(p in g for p in ('VNP14IMG','VNP03IMG')):viirs.append(g)
    proposals=[]
    for key,g in groups.items():
        if not all(p in g for p in ['MYD021KM','MYD03','MYD14']):continue
        m=g['MYD021KM']
        for v in viirs:
            t=v['VNP14IMG'];a,b=m.get('bbox'),t.get('bbox')
            if a is None or b is None or a[0]>a[2] or b[0]>b[2]:continue
            if a[2]<b[0] or b[2]<a[0] or a[3]<b[1] or b[3]<a[1]:continue
            bundle={'event_group':None,'region_id':None,'split':None,
                    'modis_start_utc':m['start_utc'],'modis_end_utc':m['end_utc'],
                    'viirs_start_utc':t['start_utc'],'viirs_end_utc':t['end_utc'],
                    'land_cover':[],'land_cover_year':None,'land_cover_available_utc':None}
            if not all(bundle[k] for k in ['modis_start_utc','modis_end_utc','viirs_start_utc','viirs_end_utc']):continue
            if time_bound_seconds(bundle)>900:continue
            for product,field in [('MYD021KM','modis_1km'),('MYD03','modis_geo'),('MYD14','modis_fire'),
                                  ('MYD02HKM','modis_500m'),('MYD02QKM','modis_250m')]:
                bundle[field]=g[product]['local_path'] if product in g else None
            bundle.update(viirs_fire=v['VNP14IMG']['local_path'],viirs_geo=v['VNP03IMG']['local_path'])
            proposals.append(bundle)
    write_jsonl(output,proposals)
    return {'proposed_pairs':len(proposals),'review_required':['event_group','region_id','split','forest context and availability'],
            'note':'No automatic random tile split. Candidate bounds/intervals still need geolocation and QA review.'}
