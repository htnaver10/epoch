"""Bounded, journaled L1B companions for already-frozen native reference tiles."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil

from .. import core


def acquisition_time(filename):
    match=re.search(r'\.A(\d{7})\.(\d{4})\.',Path(filename).name)
    if not match:raise ValueError('Unsupported VIIRS acquisition filename')
    return datetime.strptime(''.join(match.groups()),'%Y%j%H%M').replace(tzinfo=timezone.utc)


def parent_rows(config,allow_synthetic=False):
    if core.sha256(config['parent_manifest'])!=config['parent_manifest_sha256']:
        raise ValueError('Parent tile manifest changed')
    return core.validate_tiles(config['parent_manifest'],allow_synthetic,False,False)


def inventory(config):
    import earthaccess
    output=f'results/{config["name"]}/l1b_inventory.json'
    if core.path(output).exists():return core.read_json(output)
    geos=sorted({p for row in parent_rows(config) for p in row['source_ids'] if Path(p).name.startswith('VNP03IMG.')})
    result=[]
    for index,geo in enumerate(geos):
        start=acquisition_time(geo)
        found=earthaccess.search_data(short_name=config['product'],version=config['cmr_version'],
            temporal=(start.isoformat(),(start+timedelta(minutes=6)).isoformat()),count=20)
        exact=[g for g in found if core.utc(g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])==start]
        if len(exact)!=1:raise ValueError('Expected exactly one VNP02IMG companion for '+Path(geo).name)
        granule=exact[0];umm=granule['umm'];data=umm['DataGranule']
        estimates=sum(float(a.get('SizeInBytes',0)) for a in data.get('ArchiveAndDistributionInformation',[]))
        estimate=int(estimates or float(granule.size)*1024**2)
        if estimate<=0:raise ValueError('Unknown VIIRS image size; refusing unbounded download')
        names=[item['Identifier'] for item in data.get('Identifiers',[]) if item.get('IdentifierType')=='ProducerGranuleId']
        if len(names)!=1 or not names[0].startswith('VNP02IMG.'):
            raise ValueError('Expected one native VIIRS image filename in provider metadata')
        result.append({'geo':geo,'start_utc':start.isoformat(),'granule_id':granule['meta']['concept-id'],
            'filename':Path(names[0]).name,'estimated_bytes':estimate})
        print(f'VIIRS image inventory {index+1}/{len(geos)}',flush=True)
    if sum(r['estimated_bytes'] for r in result)*1.1>config['download_budget_gib']*1024**3:
        raise ValueError('VIIRS companion inventory exceeds the download budget')
    core.write_json(output,result);return result


def download(config):
    import earthaccess
    rows=inventory(config)
    output=f'results/{config["name"]}/l1b_downloads.json'
    journal=core.read_json(output) if core.path(output).exists() else {}
    needed=[row for row in rows if row['granule_id'] not in journal or not core.path(journal[row['granule_id']]['path']).is_file()]
    if needed:
        auth=earthaccess.login(strategy='environment',persist=False)
        if not auth.authenticated:raise ValueError('Earthdata authentication failed')
    for index,row in enumerate(rows):
        old=journal.get(row['granule_id'])
        if old and core.path(old['path']).is_file():
            if core.sha256(old['path'])!=old['sha256']:raise ValueError('Cached VIIRS radiance checksum changed')
            continue
        used=sum(value.get('charged_bytes',0) for value in journal.values())
        if used+1.1*row['estimated_bytes']>config['download_budget_gib']*1024**3:
            raise ValueError('VIIRS companion transfer budget reached')
        if shutil.disk_usage(core.ROOT).free<config['reserve_free_gib']*1024**3+1.1*row['estimated_bytes']:
            raise ValueError('VIIRS image transfer would cross the disk reserve')
        start=core.utc(row['start_utc'])
        found=earthaccess.search_data(short_name=config['product'],version=config['cmr_version'],
            temporal=(start.isoformat(),(start+timedelta(minutes=6)).isoformat()),count=20)
        selected=[g for g in found if g['meta']['concept-id']==row['granule_id']]
        if len(selected)!=1:raise ValueError('Locked VIIRS image disappeared; preserve the inventory')
        destination=core.path('data/raw/VNP02IMG');destination.mkdir(parents=True,exist_ok=True)
        existed=(destination/row['filename']).is_file()
        print(f'Download VIIRS thermal imagery {index+1}/{len(rows)}',flush=True)
        files=earthaccess.download(selected,local_path=str(destination),threads=2)
        files=[core.path(str(p)) for p in files if Path(p).name==row['filename']]
        if len(files)!=1:raise ValueError('Expected one exact VIIRS L1B companion file')
        file=files[0];size=file.stat().st_size
        if size<.8*row['estimated_bytes']:raise ValueError('VIIRS image is substantially smaller than its metadata estimate')
        journal[row['granule_id']]={**row,'path':core.relative(file),'sha256':core.sha256(file),
            'bytes':size,'charged_bytes':0 if existed else size}
        core.write_json(output,journal)
        if sum(v.get('charged_bytes',0) for v in journal.values())>config['download_budget_gib']*1024**3:
            raise ValueError('Actual VIIRS transfer budget exceeded; no further files requested')
    return journal
