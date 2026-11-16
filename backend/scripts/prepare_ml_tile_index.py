"""Bundle scene identities and bounds for the static globe without new inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'frontend/public/data/ml-presentation/catalog.json'
catalogue = json.loads(source.read_text(encoding='utf-8'))
tiles = [{key: scene[key] for key in ('id', 'date', 'observed_at', 'split', 'group', 'bounds', 'counts')}
         for scene in catalogue['scenes']]
assert len({tile['id'][:8] for tile in tiles}) == len(tiles), 'Short tile IDs must be unambiguous'
for tile in tiles:
    west, south, east, north = tile['bounds']
    assert -180 <= west <= east <= 180 and -90 <= south <= north <= 90
value = {'schema': 'epoch_ml_tiles_v1', 'model_version': catalogue['model']['version'],
         'source_catalogue_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'tiles': tiles,
         'bounds_note': 'Rectangles bound sampled queries; they do not imply that every location inside was observed.'}
destination = ROOT / 'frontend/public/data/ml-tiles.json'
destination.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'tiles': len(tiles), 'bytes': destination.stat().st_size, 'new_inference': False}))
