from __future__ import annotations

"""Build bounded, training-only native fire tiles from local MODIS/VIIRS granules."""

import hashlib
import json
import math
import os
import argparse
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
GDAL_PLUGINS = ROOT.parents[1] / "fire-harmonization-research-fullres" / ".data-env" / "Library" / "lib" / "gdalplugins"
GDAL_DLLS = ROOT.parents[1] / "fire-harmonization-research-fullres" / ".data-env" / "Library" / "bin"
if GDAL_PLUGINS.is_dir():
    os.environ.setdefault("GDAL_DRIVER_PATH", str(GDAL_PLUGINS))
_GDAL_DLL_HANDLE = os.add_dll_directory(str(GDAL_DLLS)) if GDAL_DLLS.is_dir() else None
if GDAL_DLLS.is_dir():
    os.environ["PATH"] = str(GDAL_DLLS) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(VENDOR))

from firefullres import core  # noqa: E402

# The vendored scientific readers resolve relative paths through this root.
core.ROOT = ROOT
from firefullres.core import (SCHEMA, path, relative, sha256, write_json, write_jsonl,
                              time_bound_seconds)
from firefullres.geometry import inverse_geolocation, reference_targets
from firefullres.native import (check_bundle_sources, read_modis_grid, h5_field,
                                native_patch)
from firefullres.prepare import query_crop, store_tile

PROJECT = ROOT.parent
WORKSPACE = PROJECT.parent
FULLRES = WORKSPACE / "fire-harmonization-research-fullres"
SOURCE_ROOT = PROJECT / "wide_diverse_v1"
PARENT_MANIFEST = PROJECT / "data" / "manifests" / "source_v3.jsonl"
V1_MANIFEST = SOURCE_ROOT / "data" / "manifests" / "extra_train.jsonl"
V1_PREPARATION = SOURCE_ROOT / "results" / "expansion_preparation.json"
CONFIG_PATH = ROOT / "expansion_config.json"
OUT_MANIFEST = "data/manifests/extra_train.jsonl"
PARTIAL_MANIFEST = "data/manifests/extra_train.partial.jsonl"
JOURNAL_MANIFEST = "data/manifests/candidate_windows.jsonl"
EXCLUSION_JOURNAL = "results/exclusions.jsonl"
PROGRESS_PATH = "results/progress.json"
OUT_RECEIPT = "results/expansion_receipt.json"
LOG_PATH = ROOT / "results" / "execution.log"
SOURCE_KEYS = ("modis_1km", "modis_geo", "modis_fire", "modis_500m",
               "modis_250m", "viirs_fire", "viirs_geo")


class Logger:
    def __init__(self, filename: Path):
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.file = filename.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self.file.write(line + "\n")

    def close(self):
        self.file.close()


def read_jsonl(filename: Path):
    return [json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def parent_rows(filename: Path):
    return [json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def basename_set(rows):
    return {Path(source).name for row in rows for source in row.get("source_ids", [])}


def source_basename_set(bundle):
    return {Path(bundle[key]).name for key in SOURCE_KEYS if bundle.get(key)}


def file_sha256(filename: Path):
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_to_local_copy(bundle, logger: Logger):
    result = dict(bundle)
    for key in SOURCE_KEYS:
        original = bundle.get(key)
        if not original:
            result[key] = None
            continue
        source = (SOURCE_ROOT / original).resolve()
        expected_root = SOURCE_ROOT.resolve()
        if not source.is_relative_to(expected_root) or not source.is_file():
            raise FileNotFoundError(f"Missing/out-of-root acquisition source: {original}")
        dest = (ROOT / original).resolve()
        if not dest.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Output source path escapes the new experiment: {original}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256(dest) != file_sha256(source):
                raise ValueError(f"Existing copied source differs from source: {dest.name}")
        else:
            shutil.copy2(source, dest)
            if sha256(dest) != file_sha256(source):
                raise IOError(f"Copied source checksum mismatch: {dest.name}")
        result[key] = dest.relative_to(ROOT).as_posix()
    return result


def identity_for(bundle, r, c, tile_size):
    return (Path(bundle["modis_1km"]).name, int(r), int(c), int(tile_size))


def parent_identities(rows, tile_size):
    result = set()
    for row in rows:
        if row.get("split") != "train":
            continue
        r, c, size, *_ = row.get("source_window", [0, 0, tile_size])
        source = next((s for s in row.get("source_ids", [])
                       if Path(s).name.startswith("MYD021KM.")), None)
        if source:
            result.add((Path(source).name, int(r), int(c), int(size)))
    return result


def prior_v1_identities(tile_size):
    """Include accepted and rejected v1 candidates so v2 never repeats attempted windows."""
    result = set()
    for row in parent_rows(V1_MANIFEST):
        if row.get("split") != "train":
            continue
        source = next((s for s in row.get("source_ids", [])
                       if Path(s).name.startswith("MYD021KM.")), None)
        if source:
            r, c, size, *_ = row["source_window"]
            result.add((Path(source).name, int(r), int(c), int(size)))
    prep = json.loads(V1_PREPARATION.read_text(encoding="utf-8"))
    for row in prep.get("excluded_candidates", []):
        result.add((Path(row["source"]).name, int(row["r"]), int(row["c"]), int(tile_size)))
    return result


def progress_snapshot(processed, total, rows, exclusions):
    reason_counts = Counter(item["reason"] for item in exclusions)
    return {"status": "preparing", "processed": processed, "selected_candidates": total,
            "admitted": len(rows), "excluded": len(exclusions),
            "excluded_reason_counts": dict(reason_counts),
            "positive_tiles": sum(r["positive_queries"] > 0 for r in rows),
            "positive_queries": int(sum(r["positive_queries"] for r in rows)),
            "groups": dict(Counter(r["event_group"] for r in rows))}


def eligible_fire_points(bundle, bbox):
    """Nominal/high-confidence VIIRS fire detections used for train-only sampling."""
    lat = h5_field(bundle["viirs_geo"], "latitude").astype(np.float64)
    lon = h5_field(bundle["viirs_geo"], "longitude").astype(np.float64)
    classes = h5_field(bundle["viirs_fire"], "fire mask")
    qa = h5_field(bundle["viirs_fire"], "algorithm QA").astype(np.uint32)
    if not (lat.shape == lon.shape == classes.shape == qa.shape):
        raise ValueError("VIIRS geolocation, fire and QA shapes disagree")
    west, south, east, north = bbox
    valid = (np.isin(classes, [8, 9]) & ((qa & (1 << 22)) == 0) &
             np.isfinite(lat) & np.isfinite(lon) & (lat >= south) & (lat <= north) &
             (lon >= west) & (lon <= east))
    return lat[valid], lon[valid]


def tile_candidates(bundle, bbox, tile_size, stride, logger):
    mlat = read_modis_grid(bundle["modis_geo"], "Latitude").astype(np.float64)
    mlon = read_modis_grid(bundle["modis_geo"], "Longitude").astype(np.float64)
    if mlat.shape != mlon.shape:
        raise ValueError("MODIS latitude/longitude shapes disagree")
    fire_lat, fire_lon = eligible_fire_points(bundle, bbox)
    west, south, east, north = bbox
    positive = []
    controls = []
    for r in range(0, mlat.shape[0] - tile_size + 1, stride):
        for c in range(0, mlat.shape[1] - tile_size + 1, stride):
            lat = mlat[r:r + tile_size, c:c + tile_size]
            lon = mlon[r:r + tile_size, c:c + tile_size]
            if not np.isfinite(lat).all() or not np.isfinite(lon).all():
                continue
            if (lat.min() < south or lat.max() > north or lon.min() < west or lon.max() > east):
                continue
            bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
            if fire_lat.size:
                hit = ((fire_lat >= bounds[1]) & (fire_lat <= bounds[3]) &
                       (fire_lon >= bounds[0]) & (fire_lon <= bounds[2]))
                count = int(hit.sum())
            else:
                count = 0
            item = {"r": r, "c": c, "positive_count": count,
                    "center_lat": float(np.mean(lat)), "center_lon": float(np.mean(lon))}
            if count:
                positive.append(item)
            else:
                controls.append(item)
    logger(f"  {Path(bundle['modis_1km']).name}: {len(positive)} positive-candidate windows, "
            f"{len(controls)} fire-clear windows; VIIRS detections={fire_lat.size}")
    return positive, controls


def greedy_spread(items, limit, min_distance, score_key="positive_count"):
    ordered = sorted(items, key=lambda x: (-x.get(score_key, 0), x["center_lat"], x["center_lon"]))
    chosen = []
    for item in ordered:
        if all((item["center_lat"] - old["center_lat"]) ** 2 +
               (item["center_lon"] - old["center_lon"]) ** 2 >= min_distance ** 2
               for old in chosen):
            chosen.append(item)
            if len(chosen) >= limit:
                break
    if len(chosen) < limit:
        chosen_keys = {(x["r"], x["c"]) for x in chosen}
        for item in ordered:
            if (item["r"], item["c"]) not in chosen_keys:
                chosen.append(item)
                if len(chosen) >= limit:
                    break
    return chosen


def select_windows(bundles, config, parent_ids, logger):
    per_source = {}
    seen_source_windows = set(parent_ids)
    total_positive = 0
    total_controls = 0
    for bundle in sorted(bundles, key=lambda b: (time_bound_seconds(b), b["event_group"], b["modis_start_utc"])):
        bbox = config["regions"][bundle["event_group"]]
        candidates, controls = tile_candidates(bundle, bbox, config["tile_size"], config["stride"], logger)
        source_key = Path(bundle["modis_1km"]).name
        bucket = per_source.setdefault(source_key, {"bundle": bundle, "positive": [], "control": []})
        for item in candidates:
            key = identity_for(bundle, item["r"], item["c"], config["tile_size"])
            if key not in seen_source_windows:
                bucket["positive"].append(item)
        for item in controls:
            key = identity_for(bundle, item["r"], item["c"], config["tile_size"])
            if key not in seen_source_windows:
                bucket["control"].append(item)

    selected = []
    max_total = config["max_new_tiles"]
    for source_key, bucket in sorted(per_source.items()):
        bundle = bucket["bundle"]
        # Reserve room for both positive-bearing regions and non-fire controls.
        positives = greedy_spread(bucket["positive"], config["positive_candidates_per_bundle"],
                                  config["positive_cell_spacing_degrees"])
        controls = greedy_spread(bucket["control"], config["control_candidates_per_bundle"],
                                 config["control_fire_clearance_degrees"])
        for kind, items in (("positive_candidate", positives), ("control_candidate", controls)):
            for item in items:
                selected.append({"bundle": bundle, "r": item["r"], "c": item["c"],
                                 "sampling_stratum": kind,
                                 "viirs_fire_detections_in_tile_bbox": item["positive_count"],
                                 "center_lat": item["center_lat"], "center_lon": item["center_lon"]})
        total_positive += len(positives)
        total_controls += len(controls)

    if len(selected) > max_total:
        positives = [x for x in selected if x["sampling_stratum"] == "positive_candidate"]
        controls = [x for x in selected if x["sampling_stratum"] == "control_candidate"]
        # Preserve the positive-stratum ratio and deterministic coverage when the cap is hit.
        keep_pos = min(len(positives), int(max_total * 0.72))
        keep_ctl = max_total - keep_pos
        positives = sorted(positives, key=lambda x: (-x["viirs_fire_detections_in_tile_bbox"],
                                                      x["bundle"]["event_group"], x["center_lat"], x["center_lon"]))[:keep_pos]
        controls = sorted(controls, key=lambda x: (x["bundle"]["event_group"], x["center_lat"], x["center_lon"]))[:keep_ctl]
        selected = positives + controls
        total_positive, total_controls = len(positives), len(controls)

    # Keep the fire candidate first at a shared native window, if bundles reuse a source.
    deduped = {}
    for item in selected:
        key = identity_for(item["bundle"], item["r"], item["c"], config["tile_size"])
        deduped.setdefault(key, item)
    selected = list(deduped.values())
    logger(f"Selected {len(selected)} candidate native windows before full QA "
           f"({total_positive} positive-stratum, {total_controls} fire-clear controls)")
    return selected


def prepare_selected(selected, config, logger):
    partial_path = path(PARTIAL_MANIFEST, mkdir=True)
    exclusions_path = path(EXCLUSION_JOURNAL, mkdir=True)
    rows = read_jsonl(partial_path) if partial_path.is_file() else []
    excluded = read_jsonl(exclusions_path) if exclusions_path.is_file() else []
    processed = set()
    for row in rows:
        source = next((s for s in row.get("source_ids", []) if Path(s).name.startswith("MYD021KM.")), None)
        if source:
            r, c, size, *_ = row["source_window"]
            processed.add((Path(source).name, int(r), int(c), int(size)))
    processed.update((Path(row["source"]).name, int(row["r"]), int(row["c"]), int(row["size"]))
                     for row in excluded)
    source_checksums = {}
    seen = set()
    cached_viirs_key = None
    cached_viirs = None
    for n, item in enumerate(selected, 1):
        bundle = item["bundle"]
        r, c, size = item["r"], item["c"], config["tile_size"]
        identity = identity_for(bundle, r, c, size)
        if identity in seen:
            continue
        seen.add(identity)
        if identity in processed:
            continue
        if bundle["split"] != "train":
            raise ValueError("Expansion bundles must remain train-only")
        if time_bound_seconds(bundle) > 900:
            raise ValueError("MODIS/VIIRS interval exceeds the established 900-second cap")
        check_bundle_sources(bundle, supervised=True)
        sources = [bundle[k] for k in SOURCE_KEYS if bundle.get(k)]
        for source in sources:
            rel = relative(source)
            if rel not in source_checksums:
                source_checksums[rel] = sha256(source)
        try:
            lat = read_modis_grid(bundle["modis_geo"], "Latitude", (r, c, size, size)).astype(np.float64)
            lon = read_modis_grid(bundle["modis_geo"], "Longitude", (r, c, size, size)).astype(np.float64)
            viirs_key = (bundle["viirs_geo"], bundle["viirs_fire"])
            if viirs_key != cached_viirs_key:
                cached_viirs = (h5_field(bundle["viirs_geo"], "latitude").astype(np.float64),
                                h5_field(bundle["viirs_geo"], "longitude").astype(np.float64),
                                h5_field(bundle["viirs_fire"], "fire mask"),
                                h5_field(bundle["viirs_fire"], "algorithm QA"))
                cached_viirs_key = viirs_key
            vlat, vlon, classes, qa = cached_viirs
            if not (vlat.shape == vlon.shape == classes.shape == qa.shape):
                raise ValueError("VIIRS companion dimensions disagree")
            crop = query_crop(vlat, vlon, lat, lon, config["max_query_side"])
            if crop is None:
                raise ValueError("no_viirs_geolocation_overlap")
            qlat, qlon = vlat[crop], vlon[crop]
            grid, inside, residual, ir, ic = inverse_geolocation(lat, lon, qlat, qlon)
            if not inside.any():
                raise ValueError("no_invertible_geolocation")
            arrays, usable, modis_fire = native_patch(bundle, r, c, size, lat, lon)
            iy = np.clip(np.floor(ir + 0.5).astype(int), 0, size - 1)
            ix = np.clip(np.floor(ic + 0.5).astype(int), 0, size - 1)
            inside &= usable[iy, ix]
            target, weight, nt, nw = reference_targets(classes[crop], qa[crop], grid, inside,
                                                        (size, size), config["halo"])
            if weight.sum() == 0:
                raise ValueError("no_valid_reference_queries")
            inference_valid = (inside & (ir >= config["halo"]) & (ir < size - config["halo"]) &
                               (ic >= config["halo"]) & (ic < size - config["halo"])).astype(np.float32)
            if inference_valid.sum() == 0:
                raise ValueError("no_valid_inference_queries")
            arrays.update(query_grid=grid, target=target, weight=weight, native_target=nt,
                          native_weight=nw, query_lat=np.where(np.isfinite(qlat), qlat, 0).astype(np.float64),
                          query_lon=np.where(np.isfinite(qlon), qlon, 0).astype(np.float64),
                          inference_valid=inference_valid,
                          modis_baseline=modis_fire[iy, ix].astype(np.float32))
            core.validate_arrays(arrays)
            tile_id = hashlib.sha256(json.dumps((bundle["modis_1km"], r, c, size)).encode()).hexdigest()[:20]
            destination = f"data/tiles/extra_train/{tile_id}.npz"
            store_tile(destination, arrays, reuse=True)
            row = {
                "schema": SCHEMA, "dataset_kind": "nasa_native", "tile_id": tile_id,
                "tile": destination, "sha256": sha256(destination), "split": "train",
                "event_group": bundle["event_group"], "region_id": bundle["region_id"],
                "source_ids": [relative(source) for source in sources],
                "source_window": [r, c, size, size], "native_shape": [size, size],
                "query_shape": list(target.shape), "valid_queries": float(weight.sum()),
                "positive_queries": float((target * weight).sum()), "forest_only": False,
                "modis_start_utc": bundle["modis_start_utc"],
                "interval_time_bound_seconds": time_bound_seconds(bundle),
                "max_inverse_residual_m": float(residual[inside].max()) if inside.any() else None,
                "sampling_stratum": item["sampling_stratum"],
                "selection_fire_detections": int(item["viirs_fire_detections_in_tile_bbox"]),
                "selection_center_lat": item["center_lat"], "selection_center_lon": item["center_lon"]
            }
            rows.append(row)
            with partial_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            processed.add(identity)
        except Exception as exc:
            rejection = {"source": Path(bundle["modis_1km"]).name, "r": r, "c": c,
                         "size": size, "event_group": bundle["event_group"],
                         "reason": f"{type(exc).__name__}: {exc}"}
            excluded.append(rejection)
            with exclusions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rejection, allow_nan=False) + "\n")
                stream.flush()
            processed.add(identity)
        if n % 10 == 0 or n == len(selected):
            write_json(PROGRESS_PATH, progress_snapshot(len(processed), len(selected), rows, excluded))
            logger(f"Prepared/checked {n}/{len(selected)} candidates; admitted {len(rows)} tiles")
    if not rows:
        raise RuntimeError("No candidate passed native source, calibration, geometry and QA checks")
    write_jsonl(OUT_MANIFEST, rows)
    write_json("results/expansion_preparation.json", {
        "manifest": OUT_MANIFEST, "tiles": len(rows), "split_counts": {"train": len(rows)},
        "tile_size": config["tile_size"], "stride": config["stride"], "halo": config["halo"],
        "max_query_side": config["max_query_side"], "forest_only": False,
        "source_checksums": source_checksums, "excluded_candidates": excluded,
        "target": "native VIIRS nominal/high-confidence fire status vs observed land at query centers",
        "group_policy": "Conservative existing regional-season train groups plus the distinct July 2023 western-Canada train cohort",
        "sampling_policy": "Training-only candidates selected near nominal/high-confidence VIIRS fire detections plus deterministic windows containing no such detections; all admitted tiles pass full QA"
    })
    complete = progress_snapshot(len(selected), len(selected), rows, excluded)
    complete["status"] = "prepared"
    complete["manifest"] = OUT_MANIFEST
    write_json(PROGRESS_PATH, complete)
    return rows, excluded, source_checksums


def positive_cells(rows, manifest_root: Path):
    cells = set()
    for row in rows:
        filename = manifest_root / row["tile"]
        if not filename.is_file():
            continue
        with np.load(filename, allow_pickle=False) as z:
            valid = (z["weight"] > 0) & (z["target"] > 0.5)
            if not valid.any():
                continue
            lat = z["query_lat"][valid]
            lon = z["query_lon"][valid]
            cells.update((int(math.floor(float(a))), int(math.floor(float(b))))
                         for a, b in zip(lat, lon))
    counts = Counter(cells)
    return {"unique_1deg_positive_cells": len(cells),
            "by_latitude_degree": {str(k): sum(1 for lat, _ in cells if lat == k)
                                   for k in sorted({lat for lat, _ in cells})},
            "cell_count": len(counts),
            "cells_lat_lon": [list(cell) for cell in sorted(cells)]}


def merge_cell_audits(*audits):
    cells = set()
    by_latitude = Counter()
    for audit in audits:
        current = {tuple(pair) for pair in audit["cells_lat_lon"]}
        cells.update(current)
    for lat, _ in cells:
        by_latitude[str(lat)] += 1
    return {"unique_1deg_positive_cells": len(cells),
            "by_latitude_degree": dict(sorted(by_latitude.items(), key=lambda x: int(x[0]))),
            "cell_count": len(cells), "cells_lat_lon": [list(cell) for cell in sorted(cells)]}


def audit_reference_qa(rows, logger):
    """Count original reference classes and QA bits at the exact stored known queries."""
    by_group = defaultdict(lambda: {
        "known_queries": 0, "positive_queries": 0, "negative_queries": 0,
        "known_class_counts": Counter(), "positive_class_counts": Counter(),
        "known_qa_bit_set_counts": Counter(), "positive_qa_bit_set_counts": Counter(),
        "positive_tiles": 0, "negative_tiles": 0,
    })
    cache_key = None
    cache = None
    bit_values = [0, 1, 2, 3, 4, 5, 6, 22]
    for index, row in enumerate(rows, 1):
        vnp_fire = next(s for s in row["source_ids"] if Path(s).name.startswith("VNP14IMG."))
        vnp_geo = next(s for s in row["source_ids"] if Path(s).name.startswith("VNP03IMG."))
        key = (vnp_geo, vnp_fire)
        if key != cache_key:
            cache = (h5_field(vnp_geo, "latitude").astype(np.float64),
                     h5_field(vnp_geo, "longitude").astype(np.float64),
                     h5_field(vnp_fire, "fire mask"),
                     h5_field(vnp_fire, "algorithm QA"))
            cache_key = key
        vlat, vlon, classes, qa = cache
        r, c, h, w = row["source_window"]
        mlat = read_modis_grid(next(s for s in row["source_ids"] if Path(s).name.startswith("MYD03.")),
                               "Latitude", (r, c, h, w)).astype(np.float64)
        mlon = read_modis_grid(next(s for s in row["source_ids"] if Path(s).name.startswith("MYD03.")),
                               "Longitude", (r, c, h, w)).astype(np.float64)
        crop = query_crop(vlat, vlon, mlat, mlon, 768)
        if crop is None:
            raise ValueError(f"QA audit could not recrop VIIRS query for tile {row['tile_id']}")
        cclasses = classes[crop]
        cqa = qa[crop].astype(np.uint32)
        with np.load(path(row["tile"]), allow_pickle=False) as tile:
            known = tile["weight"] > 0
            positive = known & (tile["target"] > 0.5)
        if cclasses.shape != known.shape or cqa.shape != known.shape:
            raise ValueError(f"QA source crop shape mismatch for tile {row['tile_id']}")
        group = by_group[row["event_group"]]
        n_known = int(known.sum())
        n_positive = int(positive.sum())
        group["known_queries"] += n_known
        group["positive_queries"] += n_positive
        group["negative_queries"] += n_known - n_positive
        group["positive_tiles"] += int(n_positive > 0)
        group["negative_tiles"] += int(n_known > n_positive)
        for class_value in (5, 8, 9):
            group["known_class_counts"][str(class_value)] += int(np.count_nonzero(known & (cclasses == class_value)))
            group["positive_class_counts"][str(class_value)] += int(np.count_nonzero(positive & (cclasses == class_value)))
        for bit in bit_values:
            mask = (cqa & (1 << bit)) != 0
            group["known_qa_bit_set_counts"][str(bit)] += int(np.count_nonzero(known & mask))
            group["positive_qa_bit_set_counts"][str(bit)] += int(np.count_nonzero(positive & mask))
        if index % 50 == 0 or index == len(rows):
            logger(f"QA provenance audit {index}/{len(rows)} tiles")
    return {name: {k: (dict(v) if isinstance(v, Counter) else v)
                   for k, v in stats.items()} for name, stats in by_group.items()}


def main():
    logger = Logger(LOG_PATH)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--audit-existing", action="store_true",
                            help="Add source class/QA provenance counts to an existing receipt without rebuilding tiles")
        parser.add_argument("--finalize-existing", action="store_true",
                            help="Reconstruct a complete receipt/progress file from a prepared manifest")
        args = parser.parse_args()
        if args.finalize_existing:
            manifest_path = path(OUT_MANIFEST)
            if not manifest_path.is_file():
                raise FileNotFoundError("Prepared expansion manifest is required for --finalize-existing")
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            parent = parent_rows(PARENT_MANIFEST)
            parent_train = [r for r in parent if r["split"] == "train"]
            parent_heldout = [r for r in parent if r["split"] in ("validation", "test")]
            v1_rows = parent_rows(V1_MANIFEST)
            prior_train = [r for r in parent_train if r.get("split") == "train"] + v1_rows
            rows = read_jsonl(manifest_path)
            heldout_names = basename_set(parent_heldout)
            heldout_groups = {r["event_group"] for r in parent_heldout}
            for row in v1_rows:
                if row.get("split") != "train" or row["event_group"] in heldout_groups:
                    raise ValueError("Existing v1 expansion is not strictly train-only")
                overlap = {Path(s).name for s in row["source_ids"]} & heldout_names
                if overlap:
                    raise ValueError(f"V1 expansion has heldout source overlap: {sorted(overlap)}")
            for row in rows:
                if row.get("split") != "train" or row.get("dataset_kind") != "nasa_native":
                    raise ValueError("Prepared expansion contains non-native/non-training rows")
                if row["event_group"] in heldout_groups:
                    raise ValueError(f"Heldout group overlap: {row['event_group']}")
                overlap = {Path(s).name for s in row["source_ids"]} & heldout_names
                if overlap:
                    raise ValueError(f"Heldout source overlap: {sorted(overlap)}")
            prior_ids = parent_identities(parent_train, config["tile_size"]) | prior_v1_identities(config["tile_size"])
            new_ids = [identity_for({"modis_1km": next(s for s in row["source_ids"]
                         if Path(s).name.startswith("MYD021KM."))}, row["source_window"][0],
                         row["source_window"][1], config["tile_size"]) for row in rows]
            if len(new_ids) != len(set(new_ids)) or set(new_ids) & prior_ids:
                raise ValueError("Prepared manifest contains a parent/v1/duplicate native window")
            parent_cells = positive_cells(parent_train, PROJECT)
            v1_cells = positive_cells(v1_rows, SOURCE_ROOT)
            old_cells = merge_cell_audits(parent_cells, v1_cells)
            new_cells = positive_cells(rows, ROOT)
            old_set = {tuple(item) for item in old_cells["cells_lat_lon"]}
            new_set = {tuple(item) for item in new_cells["cells_lat_lon"]}
            union_set = old_set | new_set
            by_group = defaultdict(lambda: {"tiles": 0, "positive_tiles": 0,
                                             "negative_tiles": 0, "positive_queries": 0,
                                             "valid_queries": 0, "dates": set()})
            for row in rows:
                summary = by_group[row["event_group"]]
                summary["tiles"] += 1
                summary["positive_tiles"] += int(row["positive_queries"] > 0)
                summary["negative_tiles"] += int(row["valid_queries"] > row["positive_queries"])
                summary["positive_queries"] += row["positive_queries"]
                summary["valid_queries"] += row["valid_queries"]
                summary["dates"].add(row["modis_start_utc"][:10])
            by_group = {name: {**summary, "dates": sorted(summary["dates"]),
                               "positive_queries": int(summary["positive_queries"]),
                               "valid_queries": int(summary["valid_queries"]),
                               "acquisition_date_count": len(summary["dates"])}
                        for name, summary in by_group.items()}
            bundle_path = path("data/manifests/expansion_bundles.jsonl")
            bundles = read_jsonl(bundle_path) if bundle_path.is_file() else []
            raw_paths = sorted({bundle[k] for bundle in bundles for k in SOURCE_KEYS if bundle.get(k)})
            raw_bytes = sum(path(item).stat().st_size for item in raw_paths)
            source_checksums = {item: sha256(item) for item in raw_paths}
            candidate_path = path(JOURNAL_MANIFEST)
            selected_candidates = read_jsonl(candidate_path) if candidate_path.is_file() else []
            exclusion_path = path(EXCLUSION_JOURNAL)
            exclusions = read_jsonl(exclusion_path) if exclusion_path.is_file() else []
            excluded_v1_count = len(json.loads(V1_PREPARATION.read_text(encoding="utf-8")).get("excluded_candidates", []))
            prior_dates = {r["modis_start_utc"][:10] for r in prior_train}
            new_dates = {r["modis_start_utc"][:10] for r in rows}
            receipt = {
                "schema": "wide_diverse_train_expansion_v1", "manifest": OUT_MANIFEST,
                "tiles": len(rows), "split": "train", "dataset_kinds": sorted({r["dataset_kind"] for r in rows}),
                "parent_manifest": str(PARENT_MANIFEST.relative_to(WORKSPACE)),
                "d1_baseline_parent_plus_v1": {"tiles": len(prior_train),
                                                "groups": sorted({r["event_group"] for r in prior_train}),
                                                "acquisition_date_count": len(prior_dates),
                                                "positive_queries": int(sum(r["positive_queries"] for r in prior_train)),
                                                "valid_queries": int(sum(r["valid_queries"] for r in prior_train)),
                                                "positive_1deg_cells": old_cells},
                "parent_training": {"tiles": len(parent_train),
                                    "groups": sorted({r["event_group"] for r in parent_train}),
                                    "acquisition_dates": sorted({r["modis_start_utc"][:10] for r in parent_train}),
                                    "acquisition_date_count": len({r["modis_start_utc"][:10] for r in parent_train}),
                                    "positive_1deg_cells": old_cells},
                "prior_wide_diverse_v1": {"tiles": len(v1_rows), "accepted_plus_rejected_windows": len(prior_v1_identities(config["tile_size"])),
                                           "groups": sorted({r["event_group"] for r in v1_rows}),
                                           "positive_1deg_cells": v1_cells},
                "expanded_training": {"tiles": len(rows), "groups": sorted(by_group),
                                      "added_groups": sorted(set(by_group) - {r["event_group"] for r in prior_train}),
                                      "same_existing_cohorts": sorted(set(by_group) & {r["event_group"] for r in prior_train}),
                                      "acquisition_dates": sorted({r["modis_start_utc"][:10] for r in rows}),
                                      "acquisition_date_count": len({r["modis_start_utc"][:10] for r in rows}),
                                      "new_acquisition_dates_vs_parent_and_v1": sorted(new_dates - prior_dates),
                                      "new_acquisition_date_count_vs_parent_and_v1": len(new_dates - prior_dates),
                                      "positive_1deg_cells": new_cells,
                                      "positive_tiles": sum(r["positive_queries"] > 0 for r in rows),
                                      "negative_tiles": sum(r["valid_queries"] > r["positive_queries"] for r in rows),
                                      "positive_queries": int(sum(r["positive_queries"] for r in rows)),
                                      "valid_queries": int(sum(r["valid_queries"] for r in rows)),
                                      "by_event_group": by_group},
                "combined_positive_1deg_cells": {"unique_cells": len(union_set),
                                                  "new_cells_outside_parent_and_v1": len(new_set - old_set),
                                                  "cells_lat_lon": [list(cell) for cell in sorted(union_set)]},
                "combined_training_group_count": len({r["event_group"] for r in prior_train} | set(by_group)),
                "heldout_group_source_overlap": False, "heldout_groups": sorted(heldout_groups),
                "selected_candidates": len(selected_candidates), "excluded_candidates": len(exclusions),
                "excluded_reason_counts": dict(Counter(item["reason"] for item in exclusions)),
                "skipped_prior_v1_window_count": excluded_v1_count + len(v1_rows),
                "raw_source_files": len(raw_paths), "raw_source_bytes": raw_bytes,
                "raw_source_gib": round(raw_bytes / (1024 ** 3), 3),
                "raw_source_checksums": source_checksums,
                "tile_policy": {"size": config["tile_size"], "stride": config["stride"],
                                "halo": config["halo"], "bbox_by_group": config["regions"],
                                "selection": "Training-only VIIRS fire-near windows plus spatially spread detection-free controls; full native preparation and QA required"},
                "limitations": ["regional-season groups are conservative cohorts, not individually verified incidents",
                                "one-degree cells measure spatial support, not independent fires",
                                "reference labels are nominal/high-confidence VIIRS activity, not ground truth"]
            }
            receipt["reference_qa_audit_by_group"] = audit_reference_qa(rows, logger)
            july = receipt["reference_qa_audit_by_group"].get("pilot_20230701_all_shared_sources")
            if july:
                receipt["july_2023_label_support"] = {
                    "positive_queries": july["positive_queries"], "negative_queries": july["negative_queries"],
                    "positive_tiles": july["positive_tiles"], "negative_tiles": july["negative_tiles"],
                    "both_classes_present": bool(july["positive_queries"] > 0 and july["negative_queries"] > 0)}
            write_json(OUT_RECEIPT, receipt)
            progress = {"status": "complete", "prepared": len(rows), "expected_cap": config["max_new_tiles"],
                        "selected_candidates": len(selected_candidates), "positive_queries": receipt["expanded_training"]["positive_queries"],
                        "positive_tiles": receipt["expanded_training"]["positive_tiles"],
                        "group_counts": {k: v["tiles"] for k, v in by_group.items()},
                        "excluded_reason_counts": dict(Counter(item["reason"] for item in exclusions)), "receipt": OUT_RECEIPT}
            write_json("results/progress.json", progress)
            logger(f"Finalized {len(rows)} tiles; {receipt['expanded_training']['positive_queries']} "
                   f"positive queries; new one-degree cells={len(new_set-old_set)}")
            return
        if args.audit_existing:
            manifest_path = path(OUT_MANIFEST)
            receipt_path = path(OUT_RECEIPT)
            if not manifest_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError("Existing expansion manifest and receipt are required for --audit-existing")
            rows = read_jsonl(manifest_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["reference_qa_audit_by_group"] = audit_reference_qa(rows, logger)
            july = receipt["reference_qa_audit_by_group"].get("pilot_20230701_all_shared_sources")
            if july:
                receipt["july_2023_label_support"] = {
                    "positive_queries": july["positive_queries"], "negative_queries": july["negative_queries"],
                    "positive_tiles": july["positive_tiles"], "negative_tiles": july["negative_tiles"],
                    "both_classes_present": bool(july["positive_queries"] > 0 and july["negative_queries"] > 0)
                }
            write_json(OUT_RECEIPT, receipt)
            logger(f"QA audit added to {OUT_RECEIPT}")
            return
        if path(OUT_MANIFEST).exists() or path(OUT_RECEIPT).exists():
            raise FileExistsError("Expansion manifest/receipt already exists; use --finalize-existing or a new dataset version")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        parent = parent_rows(PARENT_MANIFEST)
        parent_train = [r for r in parent if r["split"] == "train"]
        parent_heldout = [r for r in parent if r["split"] in ("validation", "test")]
        heldout_basenames = basename_set(parent_heldout)
        heldout_groups = {r["event_group"] for r in parent_heldout}
        parent_train_ids = parent_identities(parent_train, config["tile_size"])
        prior_ids = parent_train_ids | prior_v1_identities(config["tile_size"])
        logger(f"Prior-window exclusion: parent={len(parent_train_ids)}; "
               f"v1 accepted+rejected={len(prior_v1_identities(config['tile_size']))}")
        logger(f"Parent manifest: {len(parent_train)} train tiles / "
               f"{len({r['event_group'] for r in parent_train})} groups; "
               f"heldout groups={sorted(heldout_groups)}")

        bundles = []
        for value in config["source_bundle_manifests"]:
            file = WORKSPACE / value
            bundles.extend(read_jsonl(file))
        candidates = [b for b in bundles if b.get("split") == "train" and
                      b.get("event_group") in config["regions"]]
        accepted, rejected = [], []
        used_acquisitions = set()
        for bundle in candidates:
            overlap = source_basename_set(bundle) & heldout_basenames
            if overlap or bundle.get("event_group") in heldout_groups:
                rejected.append({"event_group": bundle.get("event_group"),
                                 "start": bundle.get("modis_start_utc"),
                                 "reason": "heldout_group_or_source_overlap", "sources": sorted(overlap)})
                continue
            key = (Path(bundle["modis_1km"]).name, Path(bundle["viirs_fire"]).name)
            if key in used_acquisitions:
                continue
            used_acquisitions.add(key)
            accepted.append(bundle)
        if not accepted:
            raise RuntimeError("No train-only bundles remain after heldout source filtering")
        logger(f"Source bundles: {len(candidates)} train candidates, {len(accepted)} distinct pairs admitted, "
               f"{len(rejected)} rejected by heldout identity checks")

        # Bound the copy and storage footprint before touching any raw outputs.
        unique_source_paths = sorted({b[k] for b in accepted for k in SOURCE_KEYS if b.get(k)})
        source_bytes = sum((SOURCE_ROOT / rel).stat().st_size for rel in unique_source_paths)
        if source_bytes > config["copy_raw_max_gib"] * (1024 ** 3):
            raise RuntimeError(f"Raw input copy budget exceeded: {source_bytes / (1024 ** 3):.2f} GiB")
        disk_free = shutil.disk_usage(ROOT).free
        output_reserve = 5 * (1024 ** 3)
        minimum_free = config["reserve_free_gib"] * (1024 ** 3) + source_bytes + output_reserve
        if disk_free < minimum_free:
            raise RuntimeError(f"Insufficient free space: {disk_free / (1024 ** 3):.2f} GiB free; "
                               f"need {minimum_free / (1024 ** 3):.2f} GiB to retain 40 GiB reserve "
                               f"plus {source_bytes / (1024 ** 3):.2f} GiB sources and 5 GiB output")
        logger(f"Copying {len(unique_source_paths)} native files ({source_bytes / (1024 ** 3):.2f} GiB); "
               f"free space before copy={disk_free / (1024 ** 3):.2f} GiB")
        local_bundles = [move_to_local_copy(bundle, logger) for bundle in accepted]
        write_jsonl("data/manifests/expansion_bundles.jsonl", local_bundles)

        candidate_journal = path(JOURNAL_MANIFEST)
        if candidate_journal.is_file():
            selected = read_jsonl(candidate_journal)
            logger(f"Resuming from persisted candidate journal with {len(selected)} windows")
        else:
            selected = select_windows(local_bundles, config, prior_ids, logger)
            if len(selected) > config["max_new_tiles"]:
                raise RuntimeError("Candidate selection exceeded configured tile cap")
            write_jsonl(JOURNAL_MANIFEST, selected)
            write_json(PROGRESS_PATH, {"status": "candidates_selected", "selected_candidates": len(selected),
                                      "expected_cap": config["max_new_tiles"], "admitted": 0, "excluded": 0})
            logger(f"Persisted {len(selected)} candidate windows before native preparation")
        rows, excluded, source_checksums = prepare_selected(selected, config, logger)

        # Guard again at the output-manifest boundary, including basenames and native windows.
        heldout_source_names = {Path(s).name for r in parent_heldout for s in r["source_ids"]}
        parent_group_names = {r["event_group"] for r in parent_heldout}
        for row in rows:
            if row["split"] != "train" or row["dataset_kind"] != "nasa_native":
                raise RuntimeError("Expansion emitted a non-native or non-training row")
            if row["event_group"] in parent_group_names:
                raise RuntimeError(f"Heldout event-group overlap: {row['event_group']}")
            overlap = {Path(s).name for s in row["source_ids"]} & heldout_source_names
            if overlap:
                raise RuntimeError(f"Heldout source overlap: {sorted(overlap)}")
        keys = [identity_for({"modis_1km": next(s for s in r["source_ids"]
                     if Path(s).name.startswith("MYD021KM."))},
                     r["source_window"][0], r["source_window"][1], config["tile_size"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) & prior_ids:
            raise RuntimeError("Duplicate native window identity against parent or v1 windows")

        old_cells = positive_cells(parent_train, PROJECT)
        new_cells = positive_cells(rows, ROOT)
        parent_groups = {r["event_group"] for r in parent_train}
        new_groups = {r["event_group"] for r in rows}
        per_group = defaultdict(lambda: {"tiles": 0, "positive_tiles": 0, "positive_queries": 0,
                                         "valid_queries": 0, "dates": set(), "positive_cells": set()})
        for row in rows:
            g = per_group[row["event_group"]]
            g["tiles"] += 1
            g["positive_tiles"] += int(row["positive_queries"] > 0)
            g["positive_queries"] += row["positive_queries"]
            g["valid_queries"] += row["valid_queries"]
            g["dates"].add(row["modis_start_utc"][:10])
        for group in per_group.values():
            group["dates"] = sorted(group["dates"])
            group["positive_queries"] = int(group["positive_queries"])
            group["valid_queries"] = int(group["valid_queries"])
            group.pop("positive_cells", None)
        receipt = {
            "schema": "wide_diverse_train_expansion_v1", "manifest": OUT_MANIFEST,
            "tiles": len(rows), "split": "train", "dataset_kinds": sorted({r["dataset_kind"] for r in rows}),
            "parent_manifest": str(PARENT_MANIFEST.relative_to(WORKSPACE)),
            "parent_training": {"tiles": len(parent_train), "groups": sorted(parent_groups),
                                "positive_1deg_cells": old_cells},
            "expanded_training": {"tiles": len(rows), "groups": sorted(new_groups),
                                   "added_groups": sorted(new_groups - parent_groups),
                                   "positive_1deg_cells": new_cells,
                                   "positive_tiles": sum(r["positive_queries"] > 0 for r in rows),
                                   "positive_queries": int(sum(r["positive_queries"] for r in rows)),
                                   "valid_queries": int(sum(r["valid_queries"] for r in rows)),
                                   "by_event_group": dict(per_group)},
            "combined_training_group_count": len(parent_groups | new_groups),
            "heldout_group_source_overlap": False,
            "heldout_groups": sorted(heldout_groups),
            "selected_candidates": len(selected), "excluded_candidates": len(excluded),
            "rejected_source_bundles": rejected,
            "raw_source_files": len(unique_source_paths), "raw_source_bytes": source_bytes,
            "raw_source_gib": round(source_bytes / (1024 ** 3), 3),
            "raw_source_checksums": {rel: file_sha256(SOURCE_ROOT / rel) for rel in unique_source_paths},
            "prepared_source_checksums": source_checksums,
            "tile_policy": {"size": config["tile_size"], "stride": config["stride"],
                            "halo": config["halo"], "bbox_by_group": config["regions"],
                            "positive_candidate_cap_per_source": config["positive_candidates_per_bundle"],
                            "control_candidate_cap_per_source": config["control_candidates_per_bundle"],
                            "selection": "training-only reliable VIIRS fire detections plus spatially spread detection-free controls; full native calibration/geolocation/QA required"},
            "excluded_candidates": excluded,
            "limitations": ["regional-season groups remain conservative cohorts, not individually verified fire incidents",
                            "one-degree cell counts measure spatial support, not distinct fires",
                            "VIIRS nominal/high-confidence labels are reference activity labels, not ground truth"]
        }
        write_json(OUT_RECEIPT, receipt)
        logger(f"DONE: {len(rows)} new native training tiles, {receipt['expanded_training']['positive_tiles']} "
               f"positive; new groups={sorted(new_groups - parent_groups)}; "
               f"positive 1-degree cells old={old_cells['unique_1deg_positive_cells']} "
               f"new={new_cells['unique_1deg_positive_cells']}; wrote {OUT_MANIFEST} and {OUT_RECEIPT}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
