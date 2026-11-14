"""Download multi-year fire-season archives from NASA FIRMS and store cell-day aggregates.

Standard library only (Python 3.8+). Reads FIRMS_MAP_KEY from backend/.env or the environment.

    python3 backend/download_seasons.py              # all years, resumable
    python3 backend/download_seasons.py --years 2020-2025

Output: data/seasons/<SOURCE>_<YEAR>.json, one file per sensor-year, containing
0.25-degree cell-day aggregates [date, lat_index, lon_index, detections, night_detections]
plus a per-request manifest. Raw hotspot rows are not kept (they would be several GB).
"""
import argparse, csv, io, json, math, os, ssl, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "seasons"
API = "https://firms.modaps.eosdis.nasa.gov/api"
BBOX = "-140,50,-60,70"  # same rectangle as the June 2025 archive
GRID = 4  # cells per degree (0.25 degrees)
SEASON = ((6, 1), (8, 31))  # June 1 - August 31, the core boreal fire season
SOURCES = {"MODIS_SP": (2003, 2026), "VIIRS_SNPP_SP": (2012, 2026)}


MAP_KEY_CACHE = ["\x00"]


def map_key():
    key = os.environ.get("FIRMS_MAP_KEY")
    env = ROOT / "backend" / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            name, sep, value = line.partition("=")
            if sep and name.strip() in ("FIRMS_MAP_KEY", "MAP_KEY", "FIRMS_KEY"):
                key = value.split(" #")[0].strip().strip("'\"")
    if not key:
        sys.exit("FIRMS_MAP_KEY not found. Put FIRMS_MAP_KEY=... in backend/.env")
    MAP_KEY_CACHE[0] = key
    return key


try:  # macOS python.org builds ship without root certificates; certifi fixes that
    import certifi
    SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL = ssl.create_default_context()


def get(url, tries=30):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180, context=SSL) as r:
                text = r.read().decode("utf-8", "replace")
            if "transaction" in text.lower() and "limit" in text.lower():
                raise RuntimeError("rate limited")
            return text
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace").strip().replace(MAP_KEY_CACHE[0], "<key>")
            if "limit" in body.lower():  # FIRMS caps transactions per 10 minutes; wait for the window
                print("   FIRMS 10-minute request limit reached; pausing 3 min (safe to leave running)", flush=True)
                time.sleep(180)
                continue
            if e.code == 400:  # a bad request will not succeed on retry: report FIRMS's own message
                raise SystemExit(f"FIRMS rejected the request (HTTP 400): {body[:200]}")
            wait = 90 if e.code == 429 else 10 * (attempt + 1)
            print(f"   retrying in {wait}s (HTTP {e.code})", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            wait = 90 if "limit" in str(e) or "429" in str(e) else 10 * (attempt + 1)
            reason = str(getattr(e, "reason", e))
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                sys.exit("SSL certificates missing for this Python. Run with the project venv:\n"
                         "  backend/.venv/bin/python backend/download_seasons.py\n"
                         "or: pip3 install certifi   (or open 'Install Certificates.command' in /Applications/Python 3.x)")
            print(f"   retrying in {wait}s ({type(e).__name__}: {reason[:80]})", flush=True)
            time.sleep(wait)
    raise RuntimeError("FIRMS request failed repeatedly (URL withheld: it contains your key)")


def availability(key):
    text = get(f"{API}/data_availability/csv/{key}/ALL")
    return {r["data_id"]: (r["min_date"], r["max_date"]) for r in csv.DictReader(io.StringIO(text))}


def chunks(year, lo, hi, step=5):  # FIRMS area API accepts at most 5 days per request
    start, end = date(year, *SEASON[0]), date(year, *SEASON[1])
    start, end = max(start, lo), min(end, hi)
    while start <= end:
        days = min(step, (end - start).days + 1)
        yield start, days
        start += timedelta(days=days)


def fetch_chunk(key, source, start, days):
    text = get(f"{API}/area/csv/{key}/{source}/{BBOX}/{days}/{start.isoformat()}")
    if not text.lstrip().startswith("latitude"):
        if days > 5:  # some FIRMS sources cap the day range at 5: split and retry
            a, ca = fetch_chunk(key, source, start, 5)
            b, cb = fetch_chunk(key, source, start + timedelta(days=5), days - 5)
            for k, v in cb.items():
                c = ca.setdefault(k, [0, 0])
                c[0] += v[0]
                c[1] += v[1]
            return {"start": start.isoformat(), "days": days, "rows": a["rows"] + b["rows"]}, ca
        raise RuntimeError(f"{source} {start}: unexpected response: {text[:120]!r}")
    cells, rows = {}, 0
    for row in csv.DictReader(io.StringIO(text)):
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        k = (row["acq_date"], math.floor(lat * GRID), math.floor(lon * GRID))
        c = cells.setdefault(k, [0, 0])
        c[0] += 1
        c[1] += row.get("daynight") == "N"
        rows += 1
    return {"start": start.isoformat(), "days": days, "rows": rows}, cells


def download_year(key, source, year, lo, hi):
    target = OUT / f"{source}_{year}.json"
    if target.exists():
        print(f"✓ {source} {year} already downloaded")
        return
    plan = list(chunks(year, lo, hi))
    if not plan:
        print(f"- {source} {year}: outside FIRMS availability, skipped")
        return
    print(f"↓ {source} {year}: {len(plan)} requests", flush=True)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda c: fetch_chunk(key, source, *c), plan))
    merged, manifest = {}, []
    for info, cells in results:
        manifest.append(info)
        for k, (n, night) in cells.items():
            m = merged.setdefault(k, [0, 0])
            m[0] += n
            m[1] += night
    records = [[d, y, x, n, night] for (d, y, x), (n, night) in sorted(merged.items())]
    total = sum(i["rows"] for i in manifest)
    assert sum(r[3] for r in records) == total
    covered = sum(i["days"] for i in manifest)
    data = {"source": source, "year": year, "bbox": BBOX, "grid_degrees": 1 / GRID,
            "season": [f"{SEASON[0][0]:02d}-{SEASON[0][1]:02d}", f"{SEASON[1][0]:02d}-{SEASON[1][1]:02d}"],
            "days_covered": covered, "total_detections": total, "requests": manifest,
            "columns": ["date", "lat_index", "lon_index", "detections", "night_detections"],
            "records": records}
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(target)
    print(f"  saved {total:,} detections in {len(records):,} cell-days ({covered} days)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", help="e.g. 2015-2025")
    args = p.parse_args()
    key = map_key()
    OUT.mkdir(parents=True, exist_ok=True)
    avail = availability(key)
    for source, (first, last) in SOURCES.items():
        if args.years:
            a, b = (int(v) for v in args.years.split("-"))
            first, last = max(first, a), min(last, b)
        lo, hi = (date.fromisoformat(v) for v in avail[source])
        for year in range(first, last + 1):
            download_year(key, source, year, lo, hi)
    print("\nDone. Next: python3 frontend/scripts/prepare_seasons.py")


if __name__ == "__main__":
    main()
