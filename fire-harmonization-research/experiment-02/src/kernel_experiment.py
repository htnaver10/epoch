"""Independent, bounded MODIS -> VIIRS calibration; NumPy only, no app writes."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
sys.path.insert(0, str(RESEARCH / "src"))
from experiment import (Archive, AREA_WEIGHT, LON, REGIONS, SHAPE, TRAIN,
                        VALIDATION, TEST, fit_baselines, baseline_rows,
                        metrics, verify_isolation)

RADII = (1, 2, 4)
BETAS = (.5, .75, 1.)
POOLING = (0., .5, 1.)
SECTORS = ("west", "central", "east")
MODEL = "kernel_occupancy"


def output_path(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Experiment-02 writes must stay in its own folder")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(name, value):
    output_path(name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    import csv
    if rows:
        with output_path(name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def box_sum(array, radius):
    """Exact spatial box sums, with no temporal smoothing or wraparound."""
    pad = [(0, 0)] * array.ndim
    pad[-2:] = [(radius, radius), (radius, radius)]
    a = np.pad(array.astype(np.int16), pad)
    a = np.pad(a, [(0, 0)] * (array.ndim - 2) + [(1, 0), (1, 0)])
    integral = a.cumsum(axis=-2, dtype=np.int32).cumsum(axis=-1, dtype=np.int32)
    k = 2 * radius + 1
    return (integral[..., k:, k:] - integral[..., :-k, k:]
            - integral[..., k:, :-k] + integral[..., :-k, :-k]).astype(np.uint8)


def fraction_table(radius):
    nmax = (2 * radius + 1) ** 2
    pairs = [(k, n) for n in range(1, nmax + 1) for k in range(n + 1)]
    fractions = np.unique(np.round([k/n for k, n in pairs], 12))
    lookup = np.zeros((nmax + 1, nmax + 1), dtype=np.int32)
    for k, n in pairs:
        lookup[k, n] = np.searchsorted(fractions, round(k/n, 12))
    return fractions, lookup


def response(signal, alpha, beta):
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    return -np.expm1(-alpha * np.asarray(signal) ** beta)


def fit_alpha(hist, target, fractions, beta):
    """Moment calibration using exact weighted support histograms, not test data."""
    maximum = float(hist[fractions > 0].sum())
    if not 0 < target < maximum:
        raise ValueError("Observed activity cannot be matched within the model support")
    def expected(alpha):
        return float(hist @ response(fractions, alpha, beta))
    lo, hi = 1e-8, 1.
    while expected(hi) < target and hi < 1e10:
        hi *= 2
    if expected(hi) < target:
        raise ValueError("Moment fit failed to bracket target")
    for _ in range(65):
        mid = (lo+hi)/2
        if expected(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


class Engine:
    def __init__(self):
        self.archive = Archive()
        self.data = {}
        self.base = {}
        self.tables = {r: fraction_table(r) for r in RADII}
        self.hist_cache = {}

    def load(self, year):
        if year not in self.data:
            m, v = self.archive.arrays(year)
            self.data[year] = ((m > 0), (v > 0), self.archive.valid_dates(year))
        return self.data[year]

    def mask(self, years, geographic=None):
        days = np.stack([self.load(y)[1].sum(axis=0) for y in years])
        mask = ((days > 0).mean(axis=0) >= .8) & (np.median(days, axis=0) >= 5)
        if geographic is not None:
            mask &= geographic
        return mask

    def signals(self, year, radius, static):
        m, v, valid = self.load(year)
        key = year, radius
        if key not in self.base:
            self.base[key] = box_sum(m, radius)
        counts = self.base[key].copy()
        # Subtract static-source contributions without recomputing the full integral.
        for iy, ix in np.argwhere(static):
            a, b = max(0, iy-radius), min(SHAPE[0], iy+radius+1)
            c, d = max(0, ix-radius), min(SHAPE[1], ix+radius+1)
            counts[:, a:b, c:d] -= m[:, iy, ix, None, None].astype(np.uint8)
        denominator = box_sum(~static, radius)
        return counts, denominator

    def hist(self, year, radius, static, geographic=None):
        mask_hash = hashlib.sha256(static.tobytes()).hexdigest()
        geo_hash = "all" if geographic is None else hashlib.sha256(geographic.tobytes()).hexdigest()
        key = year, radius, mask_hash, geo_hash
        if key in self.hist_cache:
            return self.hist_cache[key]
        m, v, valid = self.load(year)
        counts, denominator = self.signals(year, radius, static)
        fractions, lookup = self.tables[radius]
        level = lookup[counts, denominator[None, :, :]]
        eligible = ~static if geographic is None else (~static & geographic)
        w = AREA_WEIGHT * eligible
        h = np.zeros((92, 3, len(fractions)))
        target = np.zeros((92, 3)); modis = np.zeros_like(target); unsupported = np.zeros_like(target)
        for j, region in enumerate(SECTORS):
            weights = w * REGIONS[region]
            cells = weights.ravel() > 0
            if not cells.any():
                continue
            flat_weights = weights.ravel()[cells]
            for day in np.flatnonzero(valid):
                h[day, j] = np.bincount(level[day].ravel()[cells], weights=flat_weights, minlength=len(fractions))
                target[day, j] = np.sum(v[day] * weights)
                modis[day, j] = np.sum(m[day] * weights)
                unsupported[day, j] = np.sum(v[day] * (counts[day] == 0) * weights)
        result = {"hist": h, "target": target, "modis": modis, "valid": valid,
                  "unsupported": unsupported, "fractions": fractions}
        self.hist_cache[key] = result
        return result

    def clear_hist(self):
        self.hist_cache.clear()


def fit(engine, years, radius, beta, static, geographic=None):
    summaries = [engine.hist(y, radius, static, geographic) for y in years]
    h = np.sum([s["hist"].sum(axis=0) for s in summaries], axis=0)
    targets = np.sum([s["target"].sum(axis=0) for s in summaries], axis=0)
    fractions = summaries[0]["fractions"]
    pooled = fit_alpha(h.sum(axis=0), targets.sum(), fractions, beta)
    regional = {}
    for j, region in enumerate(SECTORS):
        if targets[j] > 0:
            regional[region] = fit_alpha(h[j], targets[j], fractions, beta)
    return {"radius": radius, "beta": beta, "pooled_alpha": pooled,
            "regional_alpha": regional, "training_years": list(years)}


def alpha_for(model, region, pooling):
    pooled = model["pooled_alpha"]
    local = model["regional_alpha"].get(region, pooled)
    return math.exp((1-pooling)*math.log(pooled)+pooling*math.log(local))


def rows_for(engine, years, model, pooling, static, phase, geographic=None, regions=None):
    output = []; support = []
    for year in years:
        data = engine.hist(year, model["radius"], static, geographic)
        predictions = np.zeros((92, 3))
        for j, region in enumerate(SECTORS):
            p = response(data["fractions"], alpha_for(model, region, pooling), model["beta"])
            predictions[:, j] = data["hist"][:, j] @ p
        region_names = list(regions or (*SECTORS, "canada_rectangle"))
        for region in region_names:
            column = SECTORS.index(region) if region in SECTORS else None
            series = lambda key: data[key][:, column] if column is not None else data[key].sum(axis=1)
            predicted = predictions[:, column] if column is not None else predictions.sum(axis=1)
            for week in range(14):
                start, stop = week*7, min(92, week*7+7)
                days = int(data["valid"][start:stop].sum())
                if not days:
                    continue
                output.append({"phase": phase, "year": year, "week": week,
                    "region": region, "valid_days": days,
                    "observed_viirs": float(series("target")[start:stop].sum()),
                    "observed_modis": float(series("modis")[start:stop].sum()),
                    "model": MODEL, "predicted_viirs": float(predicted[start:stop].sum())})
            target = float(series("target").sum())
            unsupported = float(series("unsupported").sum())
            support.append({"phase": phase, "year": year, "region": region,
                "unsupported_viirs_activity": unsupported, "observed_viirs": target,
                "unsupported_share": unsupported/target if target else 0.})
    return output, support


def selection_score(rows, regions=SECTORS):
    wapes = {}
    for region in (*regions, "canada_rectangle"):
        r = [x for x in rows if x["region"] == region]
        denominator = sum(x["observed_viirs"] for x in r)
        if denominator > 0:
            wapes[region] = sum(abs(x["predicted_viirs"]-x["observed_viirs"]) for x in r)/denominator
    return .5*wapes["canada_rectangle"] + .5*np.mean([wapes[r] for r in regions if r in wapes])


def select(engine, train, validation, static, geographic=None):
    candidates = []
    region_names = SECTORS if geographic is None else ("west", "central")
    for radius in RADII:
        for beta in BETAS:
            model = fit(engine, train, radius, beta, static, geographic)
            for pooling in POOLING:
                rows, _ = rows_for(engine, validation, model, pooling, static, "selection", geographic)
                score = float(selection_score(rows, region_names))
                candidates.append({"radius": radius, "beta": beta, "pooling": pooling,
                                   "validation_score": score, "model": model})
    chosen = min(candidates, key=lambda r: (r["validation_score"], r["pooling"], r["radius"], abs(r["beta"]-1)))
    return chosen, [{k: v for k, v in r.items() if k != "model"} for r in candidates]


def observation_rows(engine, years, model, static, phase, geographic=None):
    rows, _ = rows_for(engine, years, model, 0, static, phase, geographic)
    return [{k: v for k, v in r.items() if k not in ("model", "predicted_viirs")} for r in rows]


def run_fixed(engine):
    static = engine.mask(TRAIN)
    selected, candidates = select(engine, TRAIN, VALIDATION, static)
    write_json("models/fixed_selected.json", {**selected, "static_cells": np.argwhere(static).tolist()})
    write_csv("results/fixed_validation_candidates.csv", candidates)
    print("Frozen fixed-split selection:", {k:v for k,v in selected.items() if k != "model"}, flush=True)
    model = selected["model"]
    baseline = fit_baselines(observation_rows(engine, TRAIN, model, static, "train"))
    rows, support = [], []
    for phase, years in (("validation", VALIDATION), ("test", TEST)):
        prediction, s = rows_for(engine, years, model, selected["pooling"], static, phase)
        rows.extend(prediction); support.extend(s)
        rows.extend(baseline_rows(observation_rows(engine, years, model, static, phase), baseline))
    # Frozen old model is a comparator only; it never influences fitting/selection.
    import csv
    with (RESEARCH / "results/numpy_weekly_predictions.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["phase"] in ("validation", "test") and r["model"] == "numpy_daily_occupancy":
                rows.append({**r, "year": int(r["year"]), "week": int(r["week"]), "valid_days": int(r["valid_days"]),
                    **{k: float(r[k]) for k in ("observed_viirs", "observed_modis", "predicted_viirs")}})
    return rows, support, selected


def run_spatial(engine):
    eligible = LON < -92
    static = engine.mask(TRAIN, eligible)
    selected, candidates = select(engine, TRAIN, VALIDATION, static, eligible)
    write_json("models/cold_east_selected.json", selected)
    write_csv("results/cold_east_validation_candidates.csv", candidates)
    rows, support = rows_for(engine, TEST, selected["model"], selected["pooling"], static,
                             "spatial_test", regions=("east",))
    train = observation_rows(engine, TRAIN, selected["model"], static, "train", eligible)
    # Pool matching-size weekly regional rows; do not fit east-region labels.
    pooled = [r for r in train if r["region"] in ("west", "central")]
    x = np.array([r["observed_modis"] for r in pooled]); y = np.array([r["observed_viirs"] for r in pooled])
    b, a = np.polyfit(np.log1p(x), np.log1p(y), 1)
    baseline = {"east": {"ratio": float(y.sum()/x.sum()), "a": float(a), "b": float(b)}}
    actual = observation_rows(engine, TEST, selected["model"], static, "spatial_test")
    rows.extend(baseline_rows([r for r in actual if r["region"] == "east"], baseline))
    return rows, support


def run_rolling(engine):
    all_rows, ledger = [], []
    for year in range(2020, 2026):
        train = tuple(range(2012, year-2)); val = (year-2, year-1)
        print("Rolling fold:", year, "inner train", train, "selection", val, flush=True)
        static_inner = engine.mask(train)
        selected, candidates = select(engine, train, val, static_inner)
        history = tuple(range(2012, year))
        static = engine.mask(history)
        fitted = fit(engine, history, selected["radius"], selected["beta"], static)
        final = {"radius": selected["radius"], "beta": selected["beta"], "pooling": selected["pooling"],
                 "selection_score": selected["validation_score"], "model": fitted}
        write_json(f"models/rolling_{year}.json", final)
        write_csv(f"results/rolling_{year}_selection.csv", candidates)
        rows, _ = rows_for(engine, [year], fitted, selected["pooling"], static, "rolling_test")
        baseline = fit_baselines(observation_rows(engine, history, fitted, static, "train"))
        rows.extend(baseline_rows(observation_rows(engine, [year], fitted, static, "rolling_test"), baseline))
        all_rows.extend(rows)
        ledger.append({"test_year": year, "inner_training": train, "inner_validation": val,
                       "final_training": history, "radius": selected["radius"], "beta": selected["beta"],
                       "pooling": selected["pooling"], "training_static_cells": int(static.sum())})
        engine.clear_hist()
    write_json("results/rolling_fold_ledger.json", ledger)
    return all_rows


def main():
    # Freeze protocol and implementation before any fitting/selection/test score.
    write_json("results/protocol_hashes.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"PROTOCOL.md", Path(__file__)]})
    engine = Engine()
    rows, support, _ = run_fixed(engine)
    engine.clear_hist()
    spatial, s = run_spatial(engine); rows.extend(spatial); support.extend(s)
    engine.clear_hist()
    rows.extend(run_rolling(engine))
    summary, seasonal = metrics(rows)
    write_csv("results/weekly_predictions.csv", rows)
    write_csv("results/seasonal_predictions.csv", seasonal)
    write_csv("results/metrics.csv", summary)
    write_json("results/metrics.json", summary)
    write_csv("results/support_diagnostics.csv", support)
    protected = json.loads((ROOT / "results/previous_experiment_hashes.json").read_text())
    assert all(hashlib.sha256((RESEARCH/name).read_bytes()).hexdigest() == digest for name,digest in protected.items())
    isolation = verify_isolation(); assert isolation["passed"]
    write_json("results/isolation.json", {"main_app": isolation, "prior_experiment_artifacts_unchanged": len(protected)})
    print("Experiment 02 completed; main app and previous experiment artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
