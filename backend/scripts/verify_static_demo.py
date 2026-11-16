"""Verify the complete static demo, including data that must survive a fresh clone."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(site):
    site = site.resolve()
    html = (site / 'index.html').read_text(encoding='utf-8')
    if any(marker in html for marker in ('<<<<<<< ', '>>>>>>> ')):
        raise ValueError('Conflict markers in built entry point')
    for reference in re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html):
        if not (site / reference.lstrip('/')).is_file():
            raise ValueError('Missing built asset: ' + reference)
    checked = 0
    for folder in ('ml', 'ml-presentation'):
        base = site / 'data' / folder
        catalogue = json.loads((base / 'catalog.json').read_text(encoding='utf-8'))
        for relative, expected in catalogue['files'].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError('Cache path outside its root')
            if digest(path) != expected:
                raise ValueError('Cache file missing or corrupted: ' + str(path))
            checked += 1
    presentation = site / 'data/ml-presentation/catalog.json'
    catalogue = json.loads(presentation.read_text(encoding='utf-8'))
    overview = json.loads((site / 'data/ml-overview.json').read_text(encoding='utf-8'))
    tiles = json.loads((site / 'data/ml-tiles.json').read_text(encoding='utf-8'))
    for value in (overview, tiles):
        if value['source_catalogue_sha256'] != digest(presentation):
            raise ValueError('Overview/tile provenance differs from the frozen catalogue')
    if tiles['model_version'] != catalogue['model']['version'] or overview['model']['version'] != catalogue['model']['version']:
        raise ValueError('Mixed model versions')
    if len(tiles['tiles']) != catalogue['scene_count']:
        raise ValueError('Tile identities are incomplete')
    for year in range(2003, 2012):
        json.loads((site / f'data/previirs-mlp/{year}.json').read_text(encoding='utf-8'))
    for relative in ('data/blue-marble.jpg', 'data/seasons.json', 'data/observations.json', 'data/ops/index.json', 'START-HERE.txt'):
        if not (site / relative).is_file():
            raise ValueError('Missing bundled demo input: ' + relative)
    return {'passed': True, 'cache_files_checked': checked, 'scenes': catalogue['scene_count'],
            'tiles': len(tiles['tiles']), 'overview_queries': overview['source_totals']['queries'],
            'model_version': catalogue['model']['version'], 'backend_required': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[2] / 'ready-to-run')
    args = parser.parse_args()
    print(json.dumps(verify(args.site), indent=2))
