"""Plan public monthly downloads and audit real inputs; standard library only.

No model is trained here. All writes stay inside experiment-03. Never infer an
observed non-fire from the absence of a row in a fire-location-only product.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import socket
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODIS_DIR = "data/MODIS/C6/MCD14ML"  # documented directory; current 061 inventory unverified
VIIRS_DIR = "data/VIIRS/C2/VNP14IMGML"
SOURCES = {
    "modis": {"product": "MCD14ML", "collection": "061", "directory": MODIS_DIR},
    "viirs": {"product": "VNP14IMGML", "collection": "C2", "directory": VIIRS_DIR},
}
MODIS_FIELDS = "YYYYMMDD HHMM sat lat lon T21 T31 sample FRP conf type dn".split()
VIIRS_FIELDS = ["YYYYMMDD", "HHMM", "Lat", "Lon", "FRP", "Confidence", "DNFlag", "Type", "VZA", "PixArea"]
PATTERN = re.compile(r"(MCD14ML|VNP14IMGML)\.(\d{6})\.(061|006|C2|02)\.(\d{2})\.(txt|csv)\.gz")


def output(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Writes must stay in experiment-03")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(relative, value):
    output(relative).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def months(stage):
    years = (2019, 2023) if stage == "pilot" else range(2013, 2025)
    return [f"{year}{month:02}" for year in years for month in range(1, 13)]


def prepare():
    for sensor in SOURCES:
        output(f"data/raw/{sensor}/.keep").touch()
    inventory = []
    for stage in ("pilot", "full"):
        commands = []
        for sensor, source in SOURCES.items():
            commands.extend([f'lcd "{ROOT / "data/raw" / sensor}"', f'cd /{source["directory"]}'])
            for month in months(stage):
                # Accept only .061 for MODIS; VIIRS collection is validated after download.
                token, suffix = ("061", "txt") if sensor == "modis" else ("*", "csv")
                pattern = f'{source["product"]}.{month}.{token}.03.{suffix}.gz'
                commands.append(f"get {pattern}")
                inventory.append({"stage": stage, "sensor": sensor, "month": month,
                                  "remote_directory": source["directory"], "requested_pattern": pattern,
                                  "remote_file_verified": False})
        commands.append("bye")
        output(f"acquisition/{stage}.sftp").write_text("\n".join(commands) + "\n")
    output("acquisition/inspect.sftp").write_text(
        "pwd\nls data/MODIS\nls data/MODIS/C6/MCD14ML/MCD14ML.2023*\n"
        "ls data/VIIRS/C2/VNP14IMGML/VNP14IMGML.2023*\n")
    with output("acquisition/planned_months.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)
    return {"pilot_sensor_months": 48, "full_sensor_months": 288, "downloaded_by_prepare": 0}


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")
    return result


def inspect_file(path):
    name = PATTERN.fullmatch(path.name)
    if not name:
        raise ValueError(f"Unexpected filename: {path.name}")
    product, month, collection, version, suffix = name.groups()
    sensor = "modis" if product == "MCD14ML" else "viirs"
    if version != "03" or (sensor == "modis" and (collection != "061" or suffix != "txt")) or (
            sensor == "viirs" and (collection not in ("C2", "02") or suffix != "csv")):
        raise ValueError(f"Wrong collection/version/format: {path.name}; no silent fallback")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    counts, daynight, types = Counter(), Counter(), Counter()
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        fields = header.split() if sensor == "modis" else [s.strip() for s in next(csv.reader([header]))]
        required = MODIS_FIELDS if sensor == "modis" else VIIRS_FIELDS
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"Missing/duplicate schema fields: {path.name}: {fields}")
        parsed = (line.split() for line in f) if sensor == "modis" else csv.reader(f)
        for line_number, values in enumerate(parsed, 2):
            if not values:
                continue
            if len(values) != len(fields):
                raise ValueError(f"Row width changed: {path.name}:{line_number}")
            row = dict(zip(fields, (v.strip() for v in values)))
            dt = datetime.strptime(row["YYYYMMDD"], "%Y%m%d").date()
            if dt.strftime("%Y%m") != month:
                raise ValueError(f"Date outside filename month: {path.name}:{line_number}")
            clock = row["HHMM"].zfill(4)
            if len(clock) != 4 or not clock.isdigit() or int(clock[:2]) > 23 or int(clock[2:]) > 59:
                raise ValueError("Invalid UTC time")
            lat, lon = (number(row[k]) for k in (("lat", "lon") if sensor == "modis" else ("Lat", "Lon")))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180) or number(row["FRP"]) < 0:
                raise ValueError("Invalid coordinate or FRP")
            dn = row["dn" if sensor == "modis" else "DNFlag"].upper()
            kind = int(row["type" if sensor == "modis" else "Type"])
            if dn not in ("D", "N") or kind not in range(4):
                raise ValueError("Invalid day/night or hotspot type")
            if sensor == "modis":
                if row["sat"] not in ("A", "T") or not 0 <= number(row["conf"]) <= 100:
                    raise ValueError("Invalid MODIS platform or confidence")
            elif row["Confidence"].upper() not in ("L", "N", "H") or not 0 <= number(row["VZA"]) <= 90 or number(row["PixArea"]) <= 0:
                raise ValueError("Invalid VIIRS confidence, viewing angle or area")
            counts[dt.isoformat()] += 1; daynight[dn] += 1; types[kind] += 1
    if not counts:
        raise ValueError(f"Empty monthly global file: {path.name}")
    return {"file": path.name, "sensor": sensor, "month": month, "collection": collection,
            "version": version, "compressed_bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "rows": sum(counts.values()), "first_detection": min(counts), "last_detection": max(counts),
            "daily_rows": dict(sorted(counts.items())), "daynight_rows": dict(daynight),
            "type_rows": dict(types), "fields": fields}


def audit(stage):
    required = {(sensor, month) for sensor in SOURCES for month in months(stage)}
    found, errors = {}, []
    for sensor in SOURCES:
        for path in sorted((ROOT / "data/raw" / sensor).glob("*.gz")):
            match = PATTERN.fullmatch(path.name)
            if match and match.group(2) not in months(stage):
                continue
            try:
                info = inspect_file(path)
                key = info["sensor"], info["month"]
                if info["sensor"] != sensor or key in found:
                    raise ValueError(f"Wrong directory or duplicate sensor-month: {path.name}")
                found[key] = info
            except (ValueError, OSError, EOFError) as e:
                errors.append(str(e))
    missing = sorted(required - found.keys())
    result = {"stage": stage, "required_sensor_months": len(required), "valid_files": len(found),
              "validated_rows": sum(r["rows"] for r in found.values()), "missing_sensor_months": missing,
              "errors": errors, "ingestion_passed": not errors and not missing,
              "training_ready": False, "model_trained": False,
              "remaining_checks": ["deduplication", "upstream completeness and observation coverage", "paired features and blocked splits"],
              "files": list(found.values())}
    write_json(f"results/{stage}_audit.json", result)
    return {k: v for k, v in result.items() if k != "files"}


def probe():
    hosts = {}
    for host in ("fuoco.geog.umd.edu", "firms.modaps.eosdis.nasa.gov", "ladsweb.modaps.eosdis.nasa.gov"):
        try:
            hosts[host] = {"dns": "resolved", "addresses": sorted({x[4][0] for x in socket.getaddrinfo(host, None)})}
        except OSError as e:
            hosts[host] = {"dns": "failed", "error": str(e)}
    command = ["sftp", "-oBatchMode=yes", "-oConnectTimeout=10", "fire@fuoco.geog.umd.edu"]
    try:
        p = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        connection = {"command": command, "exit_code": p.returncode, "output": (p.stdout+p.stderr).strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        connection = {"command": command, "error": str(e)}
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "hosts": hosts, "sftp_probe": connection,
              "probe_scope": "Noninteractive transport probe; does not authenticate with public password or establish file availability.",
              "remote_inventory_verified": False, "new_nasa_files_downloaded": 0,
              "larger_data_training_status": "not_run_pending_acquisition",
              "sources": ["https://firms.modaps.eosdis.nasa.gov/download/",
                          "https://lpdaac.usgs.gov/documents/1005/MOD14_User_Guide_V61.pdf",
                          "https://ladsweb.modaps.eosdis.nasa.gov/archive/Document%20Archive/Science%20Data%20Product%20Documentation/VIIRS_C2_AF-375m_User_Guide_1.2.pdf"]}
    write_json("results/access_status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit", "probe"))
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    value = audit(args.stage) if args.action == "audit" else prepare() if args.action == "prepare" else probe()
    display = dict(value)
    if args.action == "audit":
        display["missing_sensor_months"] = len(value["missing_sensor_months"])
    print(json.dumps(display, indent=2))
    if args.action == "audit" and not value["ingestion_passed"]:
        raise SystemExit(2)
