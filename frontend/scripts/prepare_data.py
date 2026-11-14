"""Build reproducible cell-day data from the repository's FIRMS CSVs (stdlib only)."""
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def aggregate(path):
    cells = {}
    total = 0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            lat, lon = float(row['latitude']), float(row['longitude'])
            sensor = row['instrument'].upper()
            if not (-90 <= lat <= 90 and -180 <= lon <= 180) or sensor not in ('MODIS', 'VIIRS'):
                continue
            key = (row['acq_date'], math.floor(lat * 4), math.floor(lon * 4))
            cell = cells.setdefault(key, [0, 0])
            cell[sensor == 'VIIRS'] += 1
            total += 1
    records = [[date, y / 4 + .125, x / 4 + .125, *counts] for (date, y, x), counts in sorted(cells.items())]
    assert sum(row[3] + row[4] for row in records) == total
    return {'records': records, 'total': total}

if __name__ == '__main__':
    data = {'grid_degrees': .25, 'columns': ['date', 'latitude', 'longitude', 'modis', 'viirs'],
            'history': aggregate(ROOT / 'data/historical/2025_06/canada_boreal_combined.csv'),
            'recent': aggregate(ROOT / 'data/raw/canada_boreal_combined.csv')}
    manifest = list(csv.DictReader((ROOT / 'data/historical/2025_06/download_manifest.csv').open()))
    assert all(row['status'] == 'validated' for row in manifest)
    assert sum(int(row['rows']) for row in manifest) == data['history']['total']
    target = ROOT / 'frontend/public/data/observations.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, separators=(',', ':')))
    print(f"Wrote {target.stat().st_size:,} bytes; history={data['history']['total']:,}, recent={data['recent']['total']:,}")
