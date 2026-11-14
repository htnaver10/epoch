"""Isolated MODIS -> VIIRS experiments. Reads app JSON; writes only this research folder.

Targets are recorded VIIRS occupancy, never true-fire labels. Empty grid cells are
recorded-zero proxies conditional on the bundled download coverage metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
PROJECT = RESEARCH.parent
DATA = PROJECT / "frontend/public/data"
TRAIN = tuple(range(2012, 2020))
VALIDATION = (2020, 2021, 2022)
TEST = (2023, 2024, 2025)
SEED = 42
SHAPE = (80, 320)
FEATURES = [
    "modis_log_detections", "modis_occupied_days", "modis_log_max_daily_detections",
    "modis_neighbors_log_detections", "modis_neighbors_occupied_cell_days",
    "modis_previous_6_log_detections", "modis_previous_6_occupied_days",
    "previous_valid_days", "window_valid_days", "latitude", "longitude",
    "day_of_year_sin", "day_of_year_cos",
]
LAT, LON = np.meshgrid(50.125 + np.arange(80) * .25,
                      -139.875 + np.arange(320) * .25, indexing="ij")
AREA_WEIGHT = np.cos(np.deg2rad(LAT)) / np.cos(np.deg2rad(60))
REGIONS = {
    "west": LON < -110,
    "central": (LON >= -110) & (LON < -90),
    "east": LON >= -90,
    "canada_rectangle": np.ones(SHAPE, dtype=bool),
}


def output_path(relative):
    path = (RESEARCH / relative).resolve()
    if not path.is_relative_to(RESEARCH):
        raise ValueError("All experiment output must remain inside the research folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(relative, value):
    output_path(relative).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(relative, rows):
    if not rows:
        return
    with output_path(relative).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protected_files():
    """Fingerprint existing project files, excluding research and dependency caches."""
    result = {}
    excluded = {RESEARCH.name, ".git", "node_modules", ".venv", "__pycache__"}
    for folder, dirs, files in os.walk(PROJECT):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                result[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify_isolation():
    before = json.loads((RESEARCH / "results/main_app_before.json").read_text())
    after = protected_files()
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    result = {"files_checked": len(before), "changed": changed,
              "added_outside_research": sorted(after.keys() - before.keys()),
              "removed": sorted(before.keys() - after.keys())}
    result["passed"] = not any(result[k] for k in ("changed", "added_outside_research", "removed"))
    write_json("results/isolation_check.json", result)
    return result


@dataclass
class Archive:
    root: Path = DATA

    def __post_init__(self):
        self.meta = json.loads((self.root / "seasons.json").read_text())
        if self.meta["bbox"] != "-140,50,-60,70" or self.meta["grid_degrees"] != .25:
            raise ValueError("Experiment assumes the existing 0.25-degree boreal rectangle")
        if self.meta["season"] != ["06-01", "08-31"]:
            raise ValueError("Experiment currently supports June-August only")

    def arrays(self, year):
        content = json.loads((self.root / f"seasons/{year}.json").read_text())
        m = np.zeros((92, *SHAPE), dtype=np.int32)
        v = np.zeros_like(m)
        first = date(year, 6, 1)
        seen = set()
        for iso, lat, lon, mc, vc in content["records"]:
            i = (date.fromisoformat(iso) - first).days
            iy, ix = round((lat - 50.125) * 4), round((lon + 139.875) * 4)
            if not (0 <= i < 92 and 0 <= iy < 80 and 0 <= ix < 320):
                raise ValueError(f"Out-of-bounds record: {year}")
            if abs(LAT[iy, ix] - lat) > 1e-7 or abs(LON[iy, ix] - lon) > 1e-7:
                raise ValueError("Coordinates are not grid centers")
            if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in (mc, vc)):
                raise ValueError("Invalid detection count")
            key = (i, iy, ix)
            if key in seen:
                raise ValueError("Duplicate cell-day record")
            seen.add(key)
            m[key], v[key] = mc, vc
        for source, array in (("modis", m), ("viirs", v)):
            expected = self.meta["coverage"][source].get(str(year), {}).get("detections", 0)
            if int(array.sum()) != expected:
                raise ValueError(f"Detection total mismatch: {year} {source}")
        return m, v

    def valid_dates(self, year):
        m = self.meta["coverage"]["modis"].get(str(year), {}).get("days", 0)
        v = self.meta["coverage"]["viirs"].get(str(year), {}).get("days", 0)
        if min(m, v) < 92:
            raise ValueError(f"{year}: partial overlap excluded from this benchmark")
        valid = np.ones(92, dtype=bool)
        for sensor in ("modis", "viirs"):
            for day in self.meta.get("gaps", {}).get(sensor, {}).get(str(year), []):
                valid[day] = False
        return valid

    def static_mask(self, years, geographic_training_mask=None):
        """Fit persistent-source exclusions on training years only; never test labels."""
        days = np.stack([(self.arrays(y)[1] > 0).sum(axis=0) for y in years])
        mask = ((days > 0).mean(axis=0) >= .8) & (np.median(days, axis=0) >= 5)
        if geographic_training_mask is not None:
            mask &= geographic_training_mask
        return mask

    def audit(self):
        rows, sources = [], {}
        for path in sorted((self.root / "seasons").glob("*.json")):
            year = int(path.stem)
            m, v = self.arrays(year)
            sources[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            md = self.meta["coverage"]["modis"].get(str(year), {}).get("days", 0)
            vd = self.meta["coverage"]["viirs"].get(str(year), {}).get("days", 0)
            rows.append({"year": year, "modis_detections": int(m.sum()),
                         "viirs_detections": int(v.sum()), "modis_download_days": md,
                         "viirs_download_days": vd,
                         "modis_occupied_cell_days": int((m > 0).sum()),
                         "viirs_occupied_cell_days": int((v > 0).sum()),
                         "complete_overlap": md == vd == 92})
        sources["frontend/public/data/seasons.json"] = hashlib.sha256((self.root / "seasons.json").read_bytes()).hexdigest()
        write_csv("results/archive_audit.csv", rows)
        write_json("results/input_hashes.json", sources)
        result = {"years": rows, "complete_overlap_years": [r["year"] for r in rows if r["complete_overlap"]],
                  "retained_fields": ["date", "latitude", "longitude", "modis_count", "viirs_count"],
                  "missing_fields": ["FRP", "brightness", "confidence", "acquisition_time", "satellite_within_MODIS", "nighttime_count", "scan", "track", "clear_sky_coverage"],
                  "coverage_caveat": "Downloaded-day metadata does not establish clear-sky observations or complete satellite swath coverage.",
                  "gap_policy": "Exclude existing metadata's suspected-gap dates from BOTH sensors, never impute labels. These flags are heuristic and need upstream verification."}
        write_json("results/archive_audit.json", result)
        return result


def neighbors(array):
    padded = np.pad(array, 1)
    out = np.zeros_like(array)
    for y in range(3):
        for x in range(3):
            if (y, x) != (1, 1):
                out += padded[y:y + array.shape[0], x:x + array.shape[1]]
    return out


def make_features(m, start, stop, valid):
    """Every predictor depends only on MODIS, coordinates and calendar metadata."""
    window = m[start:stop][valid[start:stop]]
    prior = m[max(0, start - 6):start][valid[max(0, start - 6):start]]
    count = window.sum(axis=0)
    active = (window > 0).sum(axis=0)
    pc = prior.sum(axis=0) if len(prior) else np.zeros(SHAPE)
    pa = (prior > 0).sum(axis=0) if len(prior) else np.zeros(SHAPE)
    angle = 2 * math.pi * (151 + (start + stop - 1) / 2) / 365.25
    scalar = lambda x: np.full(SHAPE, x)
    columns = [np.log1p(count), active, np.log1p(window.max(axis=0)),
               np.log1p(neighbors(count)), neighbors(active), np.log1p(pc), pa,
               scalar(len(prior)), scalar(len(window)), LAT, LON,
               scalar(math.sin(angle)), scalar(math.cos(angle))]
    return np.column_stack([x.ravel() for x in columns]).astype(np.float32)


def windows(archive, year, granularity, static):
    m, v = archive.arrays(year)
    valid = archive.valid_dates(year)
    m[:, static] = 0
    v[:, static] = 0
    for start in range(0, 92, granularity):
        stop = min(start + granularity, 92)
        good = valid[start:stop]
        if not good.any():
            continue
        # Excluded dates are removed from both predictor and target window totals.
        x = make_features(m, start, stop, valid)
        y = (v[start:stop][good] > 0).sum(axis=0).ravel().astype(float)
        modis = (m[start:stop][good] > 0).sum(axis=0).ravel().astype(float)
        yield start, int(good.sum()), x, y, modis


def sample_mask(x, y, eligible, rng, background_rate):
    if not 0 < background_rate <= 1:
        raise ValueError("background_rate must be in (0, 1]")
    # Outcome-based oversampling is corrected with inverse inclusion weights.
    active = (x[:, 0] > 0) | (y > 0)
    selected = eligible & (active | (rng.random(len(y)) < background_rate))
    weights = np.where(active[selected], 1., 1. / background_rate)
    return selected, weights


def samples(archive, years, granularity, static, region_mask=None, rate=.01):
    eligible = ~static.ravel()
    if region_mask is not None:
        eligible &= region_mask.ravel()
    xs, ys, ws = [], [], []
    for year in years:
        rng = np.random.default_rng(SEED + year + granularity)
        for _, _, x, y, _ in windows(archive, year, granularity, static):
            selected, weight = sample_mask(x, y, eligible, rng, rate)
            xs.append(x[selected]); ys.append(y[selected]); ws.append(weight)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


class NumpyGLM:
    """Weighted logistic / Poisson regression; an executable ML baseline, not LightGBM."""
    def __init__(self, objective, l2=1e-4):
        if objective not in ("binary", "poisson"):
            raise ValueError(objective)
        self.objective, self.l2 = objective, l2

    def mean_response(self, eta):
        if self.objective == "binary":
            return 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        return np.exp(np.clip(eta, -30, 30))

    def loss(self, z, y, w, beta):
        eta = z @ beta
        terms = np.logaddexp(0, eta) - y * eta if self.objective == "binary" else self.mean_response(eta) - y * eta
        return float(np.dot(w, terms) / w.sum() + .5 * self.l2 * np.dot(beta[1:], beta[1:]))

    def fit(self, x, y, w):
        x = x.astype(float)
        self.center = np.average(x, axis=0, weights=w)
        self.scale = np.sqrt(np.average((x - self.center) ** 2, axis=0, weights=w))
        self.scale[self.scale < 1e-8] = 1
        z = np.column_stack([np.ones(len(x)), (x - self.center) / self.scale])
        del x
        beta = np.zeros(z.shape[1])
        mean = np.clip(np.average(y, weights=w), 1e-8, 1 - 1e-8) if self.objective == "binary" else max(np.average(y, weights=w), 1e-8)
        beta[0] = math.log(mean / (1 - mean)) if self.objective == "binary" else math.log(mean)
        normalized_w = w / w.sum()
        penalty = np.full(len(beta), self.l2); penalty[0] = 0
        self.trace = []
        for iteration in range(50):
            p = self.mean_response(z @ beta)
            curvature = np.maximum(p * (1 - p) if self.objective == "binary" else p, 1e-10)
            gradient = z.T @ (normalized_w * (p - y)) + penalty * beta
            hessian = z.T @ (z * (normalized_w * curvature)[:, None]) + np.diag(penalty + 1e-10)
            step = np.linalg.solve(hessian, gradient)
            old_loss = self.loss(z, y, w, beta)
            fraction = 1.
            while fraction > 1e-7:
                candidate = beta - fraction * step
                new_loss = self.loss(z, y, w, candidate)
                if new_loss <= old_loss + 1e-12:
                    break
                fraction *= .5
            self.trace.append({"iteration": iteration + 1, "loss": new_loss, "step_fraction": fraction})
            beta = candidate
            if abs(old_loss - new_loss) < 1e-9:
                break
        self.beta = beta
        return self

    def predict(self, x):
        z = (x - self.center) / self.scale
        return self.mean_response(self.beta[0] + z @ self.beta[1:])

    def save(self, name):
        write_json(f"models/{name}.json", {"engine": "numpy_glm", "objective": self.objective,
                   "features": FEATURES, "l2": self.l2, "center": self.center.tolist(),
                   "scale": self.scale.tolist(), "beta": self.beta.tolist(), "trace": self.trace})


def fit_lightgbm(x, y, w, validation, objective, name):
    import lightgbm as lgb
    vx, vy, vw = validation
    params = {"objective": objective, "learning_rate": .05, "num_leaves": 15,
              "max_depth": 5, "min_data_in_leaf": 100, "lambda_l2": 1.,
              "num_threads": 2, "seed": SEED, "deterministic": True,
              "force_col_wise": True, "verbosity": -1}
    model = lgb.train(params, lgb.Dataset(x, label=y, weight=w, feature_name=FEATURES),
                      num_boost_round=400,
                      valid_sets=[lgb.Dataset(vx, label=vy, weight=vw, feature_name=FEATURES)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    model.save_model(str(output_path(f"models/{name}.txt")))
    return model


def region_totals(values, static, region):
    mask = REGIONS[region] & ~static
    return float(np.sum(values.reshape(SHAPE)[mask] * AREA_WEIGHT[mask]))


def observed_rows(archive, years, static, phase):
    result = []
    for year in years:
        for start, days, _, y, m in windows(archive, year, 7, static):
            for region in REGIONS:
                result.append({"phase": phase, "year": year, "week": start // 7,
                    "region": region, "valid_days": days,
                    "observed_viirs": region_totals(y, static, region),
                    "observed_modis": region_totals(m, static, region)})
    return result


def fit_baselines(rows):
    result = {}
    for region in REGIONS:
        r = [x for x in rows if x["region"] == region]
        x = np.array([x["observed_modis"] for x in r])
        y = np.array([x["observed_viirs"] for x in r])
        b, a = np.polyfit(np.log1p(x), np.log1p(y), 1)
        result[region] = {"ratio": float(y.sum() / max(x.sum(), 1e-12)), "a": float(a), "b": float(b)}
    return result


def baseline_rows(rows, models):
    result = []
    for row in rows:
        m = row["observed_modis"]; p = models[row["region"]]
        for name, prediction in (("uncorrected_modis", m),
                                 ("constant_ratio", m * p["ratio"]),
                                 ("power_law", max(0, math.expm1(p["a"] + p["b"] * math.log1p(m))))):
            result.append({**row, "model": name, "predicted_viirs": prediction})
    return result


def score(archive, years, static, model, granularity, name, phase, regions=None):
    rows = {}
    diagnostics = {"brier_sum": 0., "logloss_sum": 0., "cell_days": 0,
                   "clipped_predictions": 0, "predictions": 0}
    region_names = list(regions or REGIONS)
    for year in years:
        print(f"Scoring {name}: {phase} {year}", flush=True)
        for start, days, x, y, m in windows(archive, year, granularity, static):
            raw = np.asarray(model.predict(x))
            pred = np.clip(raw, 0, days)
            eligible = ~static.ravel()
            if regions:
                eligible &= np.logical_or.reduce([REGIONS[r] for r in regions]).ravel()
            diagnostics["clipped_predictions"] += int(np.sum((raw != pred) & eligible))
            diagnostics["predictions"] += int(eligible.sum())
            if granularity == 1:
                p, target = np.clip(pred[eligible], 1e-12, 1 - 1e-12), y[eligible]
                diagnostics["brier_sum"] += float(np.sum((p - target) ** 2))
                diagnostics["logloss_sum"] += float(np.sum(-target * np.log(p) - (1 - target) * np.log1p(-p)))
                diagnostics["cell_days"] += len(p)
            for region in region_names:
                key = year, start // 7, region
                row = rows.setdefault(key, {"phase": phase, "year": year, "week": start // 7,
                    "region": region, "valid_days": 0, "observed_viirs": 0.,
                    "observed_modis": 0., "model": name, "predicted_viirs": 0.})
                row["valid_days"] += days
                row["observed_viirs"] += region_totals(y, static, region)
                row["observed_modis"] += region_totals(m, static, region)
                row["predicted_viirs"] += region_totals(pred, static, region)
    if diagnostics["cell_days"]:
        diagnostics["brier"] = diagnostics["brier_sum"] / diagnostics["cell_days"]
        diagnostics["logloss"] = diagnostics["logloss_sum"] / diagnostics["cell_days"]
    return list(rows.values()), diagnostics


def metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["model"], row["region"])].append(row)
    result = []
    seasonal = []
    for (phase, model, region), group in sorted(groups.items()):
        obs = np.array([x["observed_viirs"] for x in group])
        pred = np.array([x["predicted_viirs"] for x in group])
        errors = []
        for year in sorted({r["year"] for r in group}):
            annual = [r for r in group if r["year"] == year]
            actual = sum(r["observed_viirs"] for r in annual)
            estimate = sum(r["predicted_viirs"] for r in annual)
            error = (estimate - actual) / actual if actual else None
            if error is not None:
                errors.append(abs(error))
            seasonal.append({"phase": phase, "model": model, "region": region, "year": year,
                             "matched_days": sum(r["valid_days"] for r in annual),
                             "observed_viirs": actual, "predicted_viirs": estimate,
                             "relative_error": error})
        result.append({"phase": phase, "model": model, "region": region,
                       "weekly_mae": float(np.mean(np.abs(pred - obs))),
                       "weekly_rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
                       "aggregate_bias": float((pred.sum() - obs.sum()) / obs.sum()) if obs.sum() else None,
                       "median_season_abs_relative_error": float(np.median(errors)) if errors else None,
                       "worst_season_abs_relative_error": max(errors) if errors else None,
                       "weeks": len(group)})
    return result, seasonal


def run(engine):
    if engine == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError) as error:
            write_json("results/lightgbm_status.json", {"status": "blocked", "reason": str(error),
                "installation_attempt": "PyPI DNS resolution failed; no sandbox escalation attempted.",
                "numpy_results_are_lightgbm_results": False})
            print("LightGBM unavailable. See results/lightgbm_status.json", flush=True)
            return 2
    archive = Archive()
    archive.audit()
    static = archive.static_mask(TRAIN)
    config = {"direction": "MODIS -> VIIRS Suomi NPP reference", "engine": engine,
              "train_years": TRAIN, "validation_years": VALIDATION, "test_years": TEST,
              "seed": SEED, "background_inclusion_probability": .01,
              "training_static_cells": int(static.sum()), "features": FEATURES,
              "target_daily": "Recorded VIIRS grid-cell occupancy (0/1)",
              "target_weekly": "Recorded VIIRS occupied days (0..valid window days)",
              "metric_units": "cos(latitude)/cos(60N) weighted occupied cell-days",
              "full_grid_evaluation": True, "true_fire_ground_truth": False,
              "daily_model_is_forecast": False, "test_set_used_for_tuning": False,
              "platform": "VIIRS_SNPP_SP", "scope": "50-70N, 140-60W, June-August",
              "python": platform.python_version(), "numpy": np.__version__}
    write_json(f"results/{engine}_config.json", config)
    print(f"Training-only static sources excluded: {int(static.sum())}", flush=True)
    train_obs = observed_rows(archive, TRAIN, static, "train")
    baselines = fit_baselines(train_obs)
    write_json(f"models/{engine}_reference_baselines.json", baselines)
    all_rows, diagnostics = [], {}
    for phase, years in (("validation", VALIDATION), ("test", TEST)):
        all_rows.extend(baseline_rows(observed_rows(archive, years, static, phase), baselines))
    for granularity, objective in ((1, "binary"), (7, "poisson")):
        name = f"{engine}_{'daily_occupancy' if granularity == 1 else 'weekly_activity'}"
        print(f"Building weighted samples for {name}", flush=True)
        x, y, w = samples(archive, TRAIN, granularity, static)
        vx, vy, vw = samples(archive, VALIDATION, granularity, static)
        diagnostics[name] = {"training_rows": len(y), "validation_rows": len(vy),
                             "weighted_training_target_mean": float(np.average(y, weights=w))}
        if engine == "numpy":
            model = NumpyGLM(objective).fit(x, y, w)
            model.save(name)
        else:
            model = fit_lightgbm(x, y, w, (vx, vy, vw),
                                 "binary" if granularity == 1 else "tweedie", name)
        del x, y, w, vx, vy, vw
        for phase, years in (("validation", VALIDATION), ("test", TEST)):
            rows, diag = score(archive, years, static, model, granularity, name, phase)
            all_rows.extend(rows); diagnostics[name][phase] = diag
    # Cold geographic transfer: east excluded from training AND validation;
    # 2-degree longitude buffer reduces adjacent-cell leakage.
    west_training = LON < -92
    spatial_static = archive.static_mask(TRAIN, west_training)
    x, y, w = samples(archive, TRAIN, 1, spatial_static, west_training)
    vx, vy, vw = samples(archive, VALIDATION, 1, spatial_static, west_training)
    name = f"{engine}_daily_east_holdout"
    if engine == "numpy":
        model = NumpyGLM("binary").fit(x, y, w); model.save(name)
    else:
        model = fit_lightgbm(x, y, w, (vx, vy, vw), "binary", name)
    del x, y, w, vx, vy, vw
    rows, diag = score(archive, TEST, spatial_static, model, 1, name, "spatial_test", ["east"])
    all_rows.extend(rows); diagnostics[name] = diag
    # Transfer baselines use west+central training data, never east labels.
    # Match the model's exact 2-degree training buffer in baseline fitting.
    buffer_mask = spatial_static | ~west_training
    pooled_exact = observed_rows(archive, TRAIN, buffer_mask, "train")
    pooled_exact = [r for r in pooled_exact if r["region"] in ("west", "central")]
    sx = np.array([r["observed_modis"] for r in pooled_exact]); sy = np.array([r["observed_viirs"] for r in pooled_exact])
    b, a = np.polyfit(np.log1p(sx), np.log1p(sy), 1)
    spatial_baselines = {"east": {"ratio": float(sy.sum()/sx.sum()), "a": float(a), "b": float(b)}}
    test_east = [r for r in observed_rows(archive, TEST, spatial_static, "spatial_test") if r["region"] == "east"]
    all_rows.extend(baseline_rows(test_east, spatial_baselines))
    summary, seasonal = metrics(all_rows)
    write_csv(f"results/{engine}_weekly_predictions.csv", all_rows)
    write_csv(f"results/{engine}_metrics.csv", summary)
    write_csv(f"results/{engine}_seasonal_predictions.csv", seasonal)
    write_json(f"results/{engine}_metrics.json", summary)
    write_json(f"results/{engine}_diagnostics.json", diagnostics)
    result = verify_isolation()
    print(f"Completed {engine} experiments; main app unchanged: {result['passed']}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["numpy", "lightgbm"], default="numpy")
    args = parser.parse_args()
    raise SystemExit(run(args.engine))
