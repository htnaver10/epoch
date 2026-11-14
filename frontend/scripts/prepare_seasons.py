"""Build the explorer's multi-year files. Standard library only.

    python3 frontend/scripts/prepare_seasons.py              # from data/seasons/*.json
    python3 frontend/scripts/prepare_seasons.py --rebuild    # from the published cell files

The first form reads the raw downloads made by backend/download_seasons.py. The second
recomputes seasons.json from frontend/public/data/seasons/<YEAR>.json, so the daily series
can be re-derived (for example after changing the weighting) without a FIRMS key.

Writes:
  seasons.json        per-region, per-year daily series: occupied cells, area-weighted
                      cell-days, raw detections and same-day sensor overlap, with static
                      heat sources left out and archive gaps filled (both listed in the file)
  seasons/<YEAR>.json cell-day records [date, lat, lon, modis, viirs] for the map
"""
import argparse, json, math, statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENSORS = {"MODIS_SP": "modis", "VIIRS_SNPP_SP": "viirs"}
REGIONS = {  # longitude sectors of the 50-70N download rectangle
    "canada": (-140, -60), "west": (-140, -110), "central": (-110, -90), "east": (-90, -60),
}
SOURCES = {"modis": "MODIS_SP (Terra + Aqua, standard processing)",
           "viirs": "VIIRS_SNPP_SP (Suomi NPP, standard processing)"}
REFERENCE_LATITUDE = 60.0  # mid-latitude of the region: a cell here weighs exactly 1

# Static heat sources: oil sands upgraders, refineries, gas plants, a cement kiln, pulp mills.
# They burn every summer whatever the fire season does, and VIIRS sees them far more often
# than MODIS, so left in they bend the comparison between the two sensors. A cell is static
# when VIIRS saw it in at least STATIC_SHARE of its full seasons, on a median of at least
# STATIC_DAYS days of each. A wildfire cell burns in one or a few seasons; a farm burn on a
# day or two a year; neither passes both tests.
STATIC_SHARE = 0.8
STATIC_DAYS = 5

# Archive gaps: a day on which one sensor's archive is all but empty (under GAP_SHARE of the
# median of the three days either side, when that median is at least GAP_FLOOR cells) while
# the other sensor, that same day, sees at least GAP_CONFIRM of its own usual. A quiet or
# cloudy day lowers both sensors; a missing orbit lowers one.
GAP_SHARE = 0.15
GAP_FLOOR = 20
GAP_CONFIRM = 0.5
GAP_REACH = 3


def weight(lat):
    """Ground area of a grid cell centred at `lat`, relative to one at 60N.

    Cells are equal in degrees, not in kilometres: one at 70N covers about half the ground
    of one at 50N, so unweighted counts overstate northern fires.
    """
    return math.cos(math.radians(lat)) / math.cos(math.radians(REFERENCE_LATITUDE))


def load(input_dir):
    years = {}
    for path in sorted(input_dir.glob("*_[0-9][0-9][0-9][0-9].json")):
        d = json.loads(path.read_text())
        sensor = SENSORS.get(d["source"])
        if not sensor:
            continue
        assert sum(r[3] for r in d["records"]) == d["total_detections"], path
        years.setdefault(d["year"], {})[sensor] = d
    return years


def merge(years):
    """Raw per-sensor downloads -> (base metadata, {year: merged cell-day records})."""
    if not years:
        raise SystemExit("No season files found. Run backend/download_seasons.py first.")
    any_file = next(iter(next(iter(years.values())).values()))
    grid = any_file["grid_degrees"]
    coverage = {"modis": {}, "viirs": {}}
    cell_files = {}
    for year, by_sensor in sorted(years.items()):
        merged = {}
        for sensor, d in by_sensor.items():
            coverage[sensor][str(year)] = {"days": d["days_covered"], "detections": d["total_detections"]}
            col = 3 if sensor == "modis" else 4
            for iso, iy, ix, n, _night in d["records"]:
                row = merged.setdefault((iso, iy, ix), [iso, (iy + 0.5) * grid, (ix + 0.5) * grid, 0, 0])
                row[col] += n
        cell_files[year] = sorted(merged.values())
    base = {"bbox": any_file["bbox"], "grid_degrees": grid, "season": list(any_file["season"]),
            "coverage": coverage}
    return base, cell_files


def season_length(start_mmdd, end_mmdd, reference=2001):  # a non-leap year
    return (date.fromisoformat(f"{reference}-{end_mmdd}") - date.fromisoformat(f"{reference}-{start_mmdd}")).days + 1


def static_sources(cell_files, coverage, n_days):
    """Cells VIIRS sees most days of nearly every season: industrial heat, not fire."""
    full = [y for y in cell_files if coverage["viirs"].get(str(y), {}).get("days", 0) >= n_days]
    if len(full) < 5:
        return []
    days = {}
    for year in full:
        for _iso, lat, lon, _m, v in cell_files[year]:
            if v:
                per = days.setdefault((lat, lon), {})
                per[year] = per.get(year, 0) + 1
    need = math.ceil(STATIC_SHARE * len(full))
    found = []
    for cell, per in days.items():
        counts = sorted(per.values())
        mid = len(counts) // 2
        median = counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2
        if len(counts) >= need and median >= STATIC_DAYS:
            found.append(list(cell))
    return sorted(found)


def archive_gaps(cell_files, coverage, start_mmdd, n_days, static):
    """{sensor: {year: [day of season, ...]}}: days missing from one sensor's archive."""
    gaps = {"modis": {}, "viirs": {}}
    for year, records in sorted(cell_files.items()):
        first = date.fromisoformat(f"{year}-{start_mmdd}")
        count = {"modis": [0] * n_days, "viirs": [0] * n_days}
        for iso, lat, lon, m, v in records:
            i = (date.fromisoformat(iso) - first).days
            if 0 <= i < n_days and (lat, lon) not in static:
                count["modis"][i] += bool(m)
                count["viirs"][i] += bool(v)
        covered = {s: coverage[s].get(str(year), {}).get("days", 0) for s in count}
        for sensor, other in (("modis", "viirs"), ("viirs", "modis")):
            if not covered[sensor] or not covered[other]:
                continue  # one sensor alone cannot tell a missing day from a quiet one
            limit = min(covered[sensor], covered[other])
            for i in range(limit):
                near = [j for j in range(i - GAP_REACH, i + GAP_REACH + 1) if j != i and 0 <= j < limit]
                usual = statistics.median(count[sensor][j] for j in near)
                usual_other = statistics.median(count[other][j] for j in near)
                if (usual >= GAP_FLOOR and count[sensor][i] < GAP_SHARE * usual
                        and count[other][i] >= GAP_CONFIRM * usual_other):
                    gaps[sensor].setdefault(str(year), []).append(i)
    return gaps


def fill_gaps(row, gaps, year):
    """Replace each gap day with what the other sensor saw that day, scaled by the ratio
    between the two over the neighbouring days. The same rule runs in the browser for any
    area of interest (src/lib/harmonize.js, fillGaps)."""
    for sensor, other in (("modis", "viirs"), ("viirs", "modis")):
        days = gaps[sensor].get(str(year), [])
        if not days or sensor not in row or other not in row:
            continue
        skip = set(days) | set(gaps[other].get(str(year), []))
        n = len(row[sensor]["cells"])
        for i in days:
            near = [j for j in range(i - GAP_REACH, i + GAP_REACH + 1) if j not in skip and 0 <= j < n]
            for k in ("cells", "w", "raw"):
                mine = sum(row[sensor][k][j] for j in near)
                theirs = sum(row[other][k][j] for j in near)
                estimate = row[other][k][i] * mine / theirs if theirs else (mine / len(near) if near else 0)
                row[sensor][k][i] = max(row[sensor][k][i], round(estimate, 2))


def summarise(base, cell_files):
    """Daily series per region, year and sensor from merged cell-day records."""
    start_mmdd, end_mmdd = base["season"]
    coverage = base["coverage"]
    n_days = season_length(start_mmdd, end_mmdd)
    static_list = static_sources(cell_files, coverage, n_days)
    static = {tuple(c) for c in static_list}
    gaps = archive_gaps(cell_files, coverage, start_mmdd, n_days, static)
    series = {r: {} for r in REGIONS}
    for year, records in sorted(cell_files.items()):
        sensors = [s for s in ("modis", "viirs") if str(year) in coverage[s]]
        first = date.fromisoformat(f"{year}-{start_mmdd}")
        for region, (lo, hi) in REGIONS.items():
            out = {s: {"cells": [0] * n_days, "w": [0.0] * n_days, "raw": [0] * n_days,
                       "days": coverage[s][str(year)]["days"]} for s in sensors}
            if len(sensors) == 2:
                out["shared"] = [0] * n_days
            for iso, lat, lon, m, v in records:
                if not lo <= lon < hi or (lat, lon) in static:
                    continue
                i = (date.fromisoformat(iso) - first).days
                if not 0 <= i < n_days:
                    continue
                w = weight(lat)
                for sensor, n in (("modis", m), ("viirs", v)):
                    if n and sensor in out:
                        out[sensor]["cells"][i] += 1
                        out[sensor]["w"][i] += w
                        out[sensor]["raw"][i] += n
                if m and v and "shared" in out:
                    out["shared"][i] += 1
            fill_gaps(out, gaps, year)
            for sensor in sensors:
                out[sensor]["w"] = [round(x, 2) for x in out[sensor]["w"]]
            series[region][str(year)] = out
    return {
        "bbox": base["bbox"], "grid_degrees": base["grid_degrees"], "season": [start_mmdd, end_mmdd],
        "season_days": n_days,
        "weighting": {"kind": "cos(latitude)", "reference_latitude": REFERENCE_LATITUDE},
        "sources": SOURCES,
        "coverage": coverage,
        "regions": {k: {"lon": list(v), "lat": [50, 70]} for k, v in REGIONS.items()},
        "static_sources": static_list,
        "static_rule": {"share_of_full_viirs_seasons": STATIC_SHARE, "median_days_per_season": STATIC_DAYS},
        "gaps": gaps,
        "gap_rule": {"share_of_neighbours": GAP_SHARE, "floor_cells": GAP_FLOOR,
                     "other_sensor_share": GAP_CONFIRM, "reach_days": GAP_REACH},
        "series": series,
    }


def build(years):
    base, cell_files = merge(years)
    return summarise(base, cell_files), cell_files


def rebuild(output):
    """Recompute seasons.json from the cell files already published next to it."""
    old = json.loads((output / "seasons.json").read_text())
    base = {k: old[k] for k in ("bbox", "grid_degrees", "season", "coverage")}
    cell_files = {}
    for path in sorted((output / "seasons").glob("[0-9][0-9][0-9][0-9].json")):
        d = json.loads(path.read_text())
        cell_files[d["year"]] = d["records"]
    if not cell_files:
        raise SystemExit(f"No cell files in {output / 'seasons'}.")
    return summarise(base, cell_files), cell_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "data" / "seasons")
    p.add_argument("--output", type=Path, default=ROOT / "frontend" / "public" / "data")
    p.add_argument("--rebuild", action="store_true",
                   help="recompute seasons.json from the published cell files in --output")
    a = p.parse_args()
    meta, cells = rebuild(a.output) if a.rebuild else build(load(a.input))
    (a.output / "seasons").mkdir(parents=True, exist_ok=True)
    (a.output / "seasons.json").write_text(json.dumps(meta, separators=(",", ":")))
    if not a.rebuild:
        has = meta["coverage"]
        for year, records in cells.items():
            (a.output / "seasons" / f"{year}.json").write_text(json.dumps({
                "year": year, "viirs_available": str(year) in has["viirs"],
                "modis_available": str(year) in has["modis"], "records": records,
            }, separators=(",", ":")))
    print(f"Wrote seasons.json ({(a.output / 'seasons.json').stat().st_size:,} bytes) "
          f"for {len(cells)} seasons: {min(cells)}-{max(cells)}")


if __name__ == "__main__":
    main()
