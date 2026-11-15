"""Bounded native-data campaign with frozen temporal partitions and resume journals."""
from collections import defaultdict,Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from . import core
from .acquisition import CMR_VERSIONS,bbox_from_umm

FIELDS={'MYD021KM':'modis_1km','MYD03':'modis_geo','MYD14':'modis_fire',
        'MYD02HKM':'modis_500m','MYD02QKM':'modis_250m','VNP14IMG':'viirs_fire','VNP03IMG':'viirs_geo'}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def overlap(a,b):
    return a is not None and b is not None and a[0]<=a[2] and b[0]<=b[2] and a[0]<=b[2] and a[2]>=b[0] and a[1]<=b[3] and a[3]>=b[1]


def evenly_spaced(rows,count):
    if len(rows)<=count:return list(rows)
    return [rows[i] for i in np.linspace(0,len(rows)-1,count,dtype=int)]


def candidate_pairs(modis,viirs,bbox):
    pairs=[]
    for m in modis:
        for v in viirs:
            times={'modis_start_utc':m['start_utc'],'modis_end_utc':m['end_utc'],
                   'viirs_start_utc':v['start_utc'],'viirs_end_utc':v['end_utc']}
            if overlap(m['bbox'],v['bbox']) and overlap(m['bbox'],bbox) and overlap(v['bbox'],bbox):
                gap=core.time_bound_seconds(times)
                if gap<=900:pairs.append({'modis':m,'viirs':v,'interval_bound_seconds':gap})
    # One metadata-selected acquisition per UTC day, before any label screening.
    days={}
    for pair in sorted(pairs,key=lambda p:(p['interval_bound_seconds'],p['modis']['granule_id'],p['viirs']['granule_id'])):
        days.setdefault(pair['modis']['start_utc'][:10],pair)
    return [days[day] for day in sorted(days)]


def select_training_candidates(scored,count,controls):
    rich=sorted(scored,key=lambda p:(-p['candidate_nominal_high_fire_pixels'],p['modis']['start_utc']))
    keep=[];rich_count=count-controls
    for kind in ('Day','Night'):
        match=next((p for p in rich if p['modis'].get('day_night')==kind),None)
        if match is not None and len(keep)<rich_count:keep.append(match)
    ids={p['modis']['granule_id'] for p in keep}
    for pair in rich:
        if len(keep)>=rich_count:break
        if pair['modis']['granule_id'] not in ids:keep.append(pair);ids.add(pair['modis']['granule_id'])
    return keep+evenly_spaced([p for p in scored if p['modis']['granule_id'] not in ids],controls)


class Campaign:
    def __init__(self,config_path):
        self.config_path=config_path;self.config=core.read_json(config_path);self.name=self.config['name']
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.name):
            raise ValueError('Use an alphanumeric campaign name')
        self.folder=f'results/{self.name}';self.live={};self.verified={}
        lock=f'{self.folder}/config.json'
        if core.path(lock).exists() and core.read_json(lock)!=self.config:
            raise ValueError('Campaign configuration changed; use a fresh name/version')
        if not core.path(lock).exists():core.write_json(lock,self.config)
        self.journal_path=f'{self.folder}/downloads.json'
        self.journal=core.read_json(self.journal_path) if core.path(self.journal_path).exists() else {}

    @staticmethod
    def entry(product,granule):
        umm=granule['umm'];meta=granule['meta'];data=umm.get('DataGranule',{})
        interval=umm['TemporalExtent']['RangeDateTime']
        archive=data.get('ArchiveAndDistributionInformation',[])
        exact=sum(float(a.get('SizeInBytes',0)) for a in archive)
        estimated=exact or float(granule.size)*1024**2
        return {'product':product,'version':CMR_VERSIONS[product],
            'granule_id':meta['concept-id'],'collection_id':meta.get('collection-concept-id'),
            'start_utc':interval['BeginningDateTime'],'end_utc':interval['EndingDateTime'],
            'bbox':bbox_from_umm(umm),'estimated_bytes':int(math.ceil(estimated)),
            'day_night':data.get('DayNightFlag','unknown'),
            'producer_filename':Path(data.get('ProducerGranuleId','')).name}

    def search(self,product,start,end,bbox):
        import earthaccess
        bound=self.config['max_granules_per_query']
        found=earthaccess.search_data(short_name=product,version=CMR_VERSIONS[product],
            temporal=(start,end),bounding_box=tuple(bbox),count=bound+1)
        if len(found)>bound:raise ValueError(f'{product} metadata query exceeds its explicit bound')
        rows=[]
        for g in found:
            row=self.entry(product,g);self.live[row['granule_id']]=g;rows.append(row)
        return rows

    def inventory(self):
        target=f'{self.folder}/inventory.json'
        if core.path(target).exists():return core.read_json(target)
        inventories={}
        for group in self.config['groups']:
            if group.get('existing_bundles'):
                inventories[group['id']]={'existing_bundles':group['existing_bundles']};continue
            print(f'Inventory: {group["id"]}',flush=True)
            m=self.search('MYD14',group['start'],group['end'],group['bbox'])
            v=self.search('VNP14IMG',group['start'],group['end'],group['bbox'])
            candidates=candidate_pairs(m,v,group['bbox'])
            selected=evenly_spaced(candidates,group['max_candidates'])
            if not selected:raise ValueError(f'No supported close-time metadata pairs for {group["id"]}; revise a NEW campaign version')
            inventories[group['id']]={'available_days':len(candidates),'candidates':selected,
                'selection':'daily closest-interval candidate, then evenly spaced calendar days; no labels inspected'}
            print(f'  {len(candidates)} matched days; {len(selected)} bounded candidates',flush=True)
        core.write_json(target,inventories);return inventories

    def resolve(self,row,bbox):
        key=row['granule_id']
        if key not in self.live:
            found=self.search(row['product'],row['start_utc'],row['end_utc'],bbox)
            if not any(g['granule_id']==key for g in found):
                raise ValueError('Locked CMR granule changed or disappeared; do not silently change the inventory')
        return self.live[key]

    def download(self,row,bbox):
        import earthaccess
        key=row['granule_id'];old=self.journal.get(key)
        if old and core.path(old['local_path']).is_file():
            p=core.path(old['local_path']);signature=(p.stat().st_size,p.stat().st_mtime_ns)
            if self.verified.get(key)!=signature:
                if core.sha256(old['local_path'])!=old['sha256']:raise ValueError('Cached campaign source checksum changed')
                self.verified[key]=signature
            return old['local_path']
        estimate=row['estimated_bytes']
        if estimate<=0:raise ValueError('Unknown granule size; refusing an unbounded transfer')
        used=sum(r.get('charged_bytes',0) for r in self.journal.values())
        if used+1.1*estimate>self.config['download_budget_gib']*1024**3:
            raise ValueError('Campaign transfer budget reached; journal saved. Plan a fresh bounded expansion.')
        if shutil.disk_usage(core.ROOT).free<1.1*estimate+self.config['reserve_free_gib']*1024**3:
            raise ValueError('Reserved free disk space would be crossed')
        dest=core.path(f'data/raw/{row["product"]}');dest.mkdir(parents=True,exist_ok=True)
        existing={p.name for p in dest.iterdir() if p.is_file()}
        files=earthaccess.download([self.resolve(row,bbox)],local_path=str(dest),threads=2)
        files=[core.path(str(p)) for p in files if Path(p).name.startswith(row['product']+'.') and Path(p).suffix in ('.hdf','.nc')]
        if len(files)!=1:raise ValueError('Expected one native science file; inspect local provider output')
        p=files[0];size=p.stat().st_size
        if size<0.8*estimate:raise ValueError('Downloaded file is substantially smaller than the metadata estimate')
        self.journal[key]={'local_path':core.relative(p),'sha256':core.sha256(p),'bytes':size,
            'charged_bytes':0 if p.name in existing else size,'product':row['product']}
        core.write_json(self.journal_path,self.journal)
        if sum(r.get('charged_bytes',0) for r in self.journal.values())>self.config['download_budget_gib']*1024**3:
            raise ValueError('Actual transfers exceeded the metadata-based budget; no further files will be requested')
        return core.relative(p)

    def companion(self,product,base,bbox):
        found=self.search(product,base['start_utc'],base['end_utc'],bbox)
        exact=[r for r in found if core.utc(r['start_utc'])==core.utc(base['start_utc'])]
        if not exact and product in ('MYD02HKM','MYD02QKM'):return None
        if len(exact)!=1:raise ValueError(f'Expected exactly one locked {product} companion acquisition')
        return exact[0]

    def fire_score(self,pair,group):
        from .native import h5_field
        filename=self.download(pair['viirs'],group['bbox'])
        try:
            lat=h5_field(filename,'FP_latitude');lon=h5_field(filename,'FP_longitude')
            confidence=h5_field(filename,'FP_confidence')
        except ValueError as error:
            if 'found 0' not in str(error):raise
            if not np.isin(h5_field(filename,'fire mask'),[8,9]).any():return 0
            raise ValueError('Sparse fire arrays missing despite positive reference mask') from error
        if not np.isin(confidence,[7,8,9]).all():raise ValueError('Unexpected C002 FP_confidence encoding; inspect the native product')
        a=pair['modis']['bbox'];b=group['bbox'];west=max(a[0],b[0]);east=min(a[2],b[2]);south=max(a[1],b[1]);north=min(a[3],b[3])
        # A candidate-ranking count only; final labels still use complete masks,
        # QA, geometry, exposure and halo exclusions during native preparation.
        return int(((lat>=south)&(lat<=north)&(lon>=west)&(lon<=east)&np.isin(confidence,[8,9])).sum())

    def acquire(self):
        import earthaccess
        inventories=self.inventory();target=f'data/manifests/{self.name}_bundles.jsonl'
        if core.path(target).exists():return core.read_jsonl(target)
        auth=earthaccess.login(strategy='environment',persist=False)
        if not auth.authenticated:raise ValueError('Earthdata authentication failed')
        bundles=[];selection={}
        for group in self.config['groups']:
            if group.get('existing_bundles'):
                selected=core.read_jsonl(group['existing_bundles'])
                for b in selected:
                    b={**b,'event_group':group['id'],'region_id':group['id'],'split':group['split'],
                        'campaign_group':group['id']}
                    bundles.append(b)
                continue
            candidates=inventories[group['id']]['candidates']
            if group['split']=='train':
                scored=[{**p,'candidate_nominal_high_fire_pixels':self.fire_score(p,group)} for p in candidates]
                chosen=select_training_candidates(scored,group['max_pairs'],group.get('control_pairs',0))
            else:chosen=evenly_spaced(candidates,group['max_pairs'])
            selection[group['id']]={'split':group['split'],'selected':chosen,
                'rule':'training-only fire-rich ranking, one day and night candidate when available, plus calendar controls' if group['split']=='train' else 'metadata/calendar only; no fire-label selection'}
            core.write_json(f'{self.folder}/selection.json',selection)
            for i,pair in enumerate(chosen):
                print(f'Download {group["id"]}: acquisition {i+1}/{len(chosen)}',flush=True)
                m=pair['modis'];v=pair['viirs']
                sources={'MYD14':m,'VNP14IMG':v}
                for product in ('MYD021KM','MYD03','MYD02HKM','MYD02QKM'):
                    sources[product]=self.companion(product,m,group['bbox'])
                sources['VNP03IMG']=self.companion('VNP03IMG',v,group['bbox'])
                bundle={'event_group':group['id'],'region_id':group['id'],'split':group['split'],
                    'campaign_group':group['id'],'modis_start_utc':m['start_utc'],'modis_end_utc':m['end_utc'],
                    'viirs_start_utc':v['start_utc'],'viirs_end_utc':v['end_utc'],
                    'land_cover':[],'land_cover_year':None,'land_cover_available_utc':None}
                for product,row in sources.items():bundle[FIELDS[product]]=self.download(row,group['bbox']) if row is not None else None
                bundles.append(bundle)
        core.write_jsonl(target,bundles);return bundles

    def prepare(self):
        from .prepare import prepare
        manifest=f'data/manifests/{self.name}.jsonl';stats_path=f'data/manifests/{self.name}_stats.json'
        if core.path(manifest).exists() and core.path(stats_path).exists():
            core.validate_tiles(manifest)
            if core.read_json(stats_path)['manifest_sha256']!=core.sha256(manifest):raise ValueError('Campaign statistics mismatch')
            return manifest,stats_path
        bundles=core.read_jsonl(f'data/manifests/{self.name}_bundles.jsonl')
        groups={g['id']:g for g in self.config['groups']};units=defaultdict(list)
        for b in bundles:units[(b['campaign_group'],b['modis_1km'])].append(b)
        rows=[];skipped=[]
        for i,((group_id,source),unit) in enumerate(sorted(units.items())):
            unit_id=self.name+'_'+digest([group_id,source])[:12]
            source_manifest=f'data/manifests/{self.name}_units/{unit_id}_bundles.jsonl'
            tile_manifest=f'data/manifests/{self.name}_units/{unit_id}.jsonl'
            print(f'Prepare source unit {i+1}/{len(units)}: {group_id}',flush=True)
            if not core.path(tile_manifest).exists():
                core.write_jsonl(source_manifest,unit)
                try:prepare(source_manifest,tile_manifest,forest_only=False,bbox=groups[group_id]['bbox'],reuse_tiles=True)
                except ValueError as error:
                    if not str(error).startswith('No valid native tiles'):raise
                    skipped.append({'source':source,'reason':'no eligible tiles after fixed QA/geography'});continue
            rows+=core.validate_tiles(tile_manifest)
        # Conservatively merge same-split cohort labels connected by shared raw sources.
        parent={r['event_group']:r['event_group'] for r in rows}
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        owners={}
        for row in rows:
            for source in row['source_ids']:
                if source in owners:
                    other=owners[source]
                    if other['split']!=row['split']:raise ValueError('Cross-partition source reuse in campaign')
                    a,b=find(row['event_group']),find(other['event_group']);parent[max(a,b)]=min(a,b)
                else:owners[source]=row
        for row in rows:row['event_group']=find(row['event_group'])
        counts={split:{'tiles':sum(r['split']==split for r in rows),
            'positive_queries':sum(r['positive_queries'] for r in rows if r['split']==split),
            'regions':len({r['region_id'] for r in rows if r['split']==split}),
            'conservative_groups':len({r['event_group'] for r in rows if r['split']==split})}
            for split in ('train','validation','test')}
        ready=(counts['train']['tiles']>=self.config['min_training_tiles'] and
            counts['train']['positive_queries']>=self.config['min_training_positive_queries'] and
            counts['train']['regions']>=self.config['min_training_regions'] and
            counts['validation']['tiles']>=10 and counts['validation']['positive_queries']>0 and counts['test']['tiles']>=10)
        report={'ready_for_expanded_experiment':ready,'counts':counts,'skipped_sources':skipped,
            'scope':self.config['scope'],'config_sha256':core.sha256(self.config_path),
            'note':'Acquisition-cohort groups are conservative, not verified incident IDs. Test scenes were not selected by fire labels.'}
        core.write_json(f'{self.folder}/dataset_audit.json',report)
        if not ready:raise ValueError('Expanded-round data targets not met; inspect dataset_audit.json. No long training was started.')
        core.write_jsonl(manifest,rows);core.validate_tiles(manifest)
        stats=core.statistics(manifest,stats_path)
        totals=np.zeros(4,np.float64)
        for row in rows:
            if row['split']!='train':continue
            with np.load(core.path(row['tile']),allow_pickle=False) as z:
                totals+=np.array([(z['target']*z['weight']).sum(dtype=np.float64),z['weight'].sum(dtype=np.float64),
                    (z['native_target']*z['native_weight']).sum(dtype=np.float64),z['native_weight'].sum(dtype=np.float64)])
        stats['training_target_prevalence']={'fine':float(np.clip(totals[0]/totals[1],1e-6,1-1e-6)),
            'native':float(np.clip(totals[2]/totals[3],1e-6,1-1e-6))}
        core.write_json(stats_path,stats);return manifest,stats_path
