"""Read-only catalogue; no PyTorch or research-folder dependency at request time."""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

LAYERS={'all','mlp','modis','viirs','difference'}

class CatalogueUnavailable(RuntimeError):pass

class MLCatalogue:
    def __init__(self,root):
        self.root=Path(root).resolve();self.catalogue=None;self.error=None
        try:
            value=json.loads((self.root/'catalog.json').read_text(encoding='utf-8'))
            if value['schema']!='epoch_mlp_cache_v1':raise ValueError('Unsupported cache schema')
            self.scenes={r['id']:r for r in value['scenes']}
            if len(self.scenes)!=value['scene_count']:raise ValueError('Duplicate scene ids')
            for relative,expected in value['files'].items():
                f=(self.root/relative).resolve()
                if not f.is_relative_to(self.root):raise ValueError('Cache path escapes root')
                with f.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=expected:raise ValueError('Cached predictions failed checksum verification')
            self.catalogue=value
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.error=f'MLP cache unavailable: {type(exc).__name__}. Export the frozen predictions before starting the service.'

    def ready(self):
        if self.catalogue is None:raise CatalogueUnavailable(self.error)

    def model(self):self.ready();return self.catalogue['model']

    def list_scenes(self,date=None):
        self.ready()
        return [s for s in self.catalogue['scenes'] if date is None or s['date']==date]

    def scene(self,scene_id):
        self.ready()
        if not re.fullmatch(r'[0-9a-f]{20}',scene_id) or scene_id not in self.scenes:raise KeyError('Unknown scene')
        return self.scenes[scene_id]

    @lru_cache(maxsize=12)
    def _load(self,scene_id,kind):
        self.scene(scene_id)
        return json.loads((self.root/kind/f'{scene_id}.json').read_text(encoding='utf-8'))

    def predictions(self,scene_id,layer='all'):
        if layer not in LAYERS:raise ValueError('Unknown layer')
        data=self._load(scene_id,'scenes');features=data['features']
        def keep(f):
            p=f['properties']
            return layer=='all' or (p['mlp_detected'] if layer=='mlp' else p['modis_detected'] if layer=='modis' else p['viirs_reference'] if layer=='viirs' else p['mlp_detected']!=p['modis_detected'])
        return {**data,'layer':layer,'features':[f for f in features if keep(f)]}

    def coverage(self,scene_id):return self._load(scene_id,'coverage')
