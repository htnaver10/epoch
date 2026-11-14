"""Bounded, read-only FIRMS access check. Never records keys or authenticated URLs."""
import csv
import io
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
API = "https://firms.modaps.eosdis.nasa.gov/api"
SOURCES = ["MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"]
REQUIRED = {"latitude", "longitude", "acq_date", "acq_time", "satellite",
            "instrument", "confidence", "frp", "scan", "track", "daynight"}


def find_key():
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    for path in (ROOT / ".env", PROJECT / "backend/.env", PROJECT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            name, sep, value = line.removeprefix("export ").partition("=")
            if sep and name.strip() in {"FIRMS_MAP_KEY", "MAP_KEY", "FIRMS_KEY"}:
                value = value.split(" #")[0].strip().strip("'\"")
                if value:
                    return value
    return None


def summarize_csv(text):
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if not REQUIRED.issubset(fields):
        raise ValueError("Missing required hotspot fields")
    rows = list(reader)
    for row in rows:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid location")
    return {"fields": fields, "rows": len(rows),
            "dates": sorted({r["acq_date"] for r in rows}),
            "satellites": sorted({r["satellite"] for r in rows})}


def get_text(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8-sig")


def main():
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "public_docs_reviewed": "2026-11-14", "authenticated": False}
    key = find_key()
    if not key:
        result.update(status="not_run", reason="No FIRMS_MAP_KEY configured; no authenticated request was attempted.")
    else:
        result["authenticated"] = True
        checks = []
        try:
            text = get_text(f"{API}/data_availability/csv/{key}/ALL")
            availability = list(csv.DictReader(io.StringIO(text)))
            if not availability or not {"data_id", "min_date", "max_date"}.issubset(availability[0]):
                raise ValueError("Unexpected availability schema")
            result["availability"] = [r for r in availability if r["data_id"] in SOURCES]
            for source in SOURCES:
                try:
                    text = get_text(f"{API}/area/csv/{key}/{source}/-125,54,-120,58/1/2023-07-01")
                    checks.append({"source": source, "status": "validated", **summarize_csv(text)})
                except Exception as error:
                    checks.append({"source": source, "status": "failed", "error_type": type(error).__name__})
            result.update(checks=checks, status="passed" if all(c["status"] == "validated" for c in checks) else "incomplete")
        except Exception as error:
            # Exception messages can contain the full URL and its secret key.
            result.update(status="failed", error_type=type(error).__name__)
    path = ROOT / "results/nasa_access_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"] + ": " + result.get("reason", "See results/nasa_access_check.json"))


if __name__ == "__main__":
    main()
