"""Exact binned-population logistic gradient boosting; no app mutations."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
sys.path.insert(0, str(RESEARCH / "src"))
from experiment import Archive, AREA_WEIGHT, LON, REGIONS, SHAPE, metrics, fit_baselines, baseline_rows, protected_files

FEATURES = ["current_count", "occupied_3x3_neighbors", "occupied_5x5_outer_ring",
            "occupied_9x9_outer_ring", "count_5x5", "prior6_count", "prior6_occupied_days",
            "prior6_3x3_occupied_cell_days", "prior_available_days", "season_phase"]
THRESHOLDS = {
    0: [1,2,3,4,8,16,32,64], 1: [1,2,3,4,6,8], 2: [1,2,3,4,6,8,12,16],
    3: [1,2,3,4,8,16,32,56], 4: [1,2,4,8,16,32,64,128,256],
    5: [1,2,4,8,16,32,64,128], 7: [1,2,3,4,8,16,32,54],
}
CHECKPOINTS = (25, 50, 75, 100, 150, 200)
SECTORS = ("west", "central", "east")
MODEL = "boosted_11year"


def output(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Writes must stay in experiment-04")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(name, value):
    output(name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    if rows:
        with output(name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def box_sum(a, radius):
    pad = [(0, 0)] * (a.ndim-2) + [(radius, radius), (radius, radius)]
    b = np.pad(a.astype(np.int32), pad)
    b = np.pad(b, [(0, 0)] * (a.ndim-2) + [(1, 0), (1, 0)])
    z = b.cumsum(-2, dtype=np.int64).cumsum(-1, dtype=np.int64)
    k = 2*radius+1
    return (z[..., k:, k:] - z[..., :-k, k:] - z[..., k:, :-k] + z[..., :-k, :-k]).astype(np.int32)


def prior6(a, valid):
    """Strictly preceding six calendar days, never current/future or invalid days."""
    x = a * valid.reshape((-1,) + (1,)*(a.ndim-1))
    c = np.concatenate([np.zeros_like(x[:1]), np.cumsum(x, axis=0, dtype=np.int32)])
    day = np.arange(len(a))
    return c[day] - c[np.maximum(0, day-6)]


def pack(x):
    if x.shape[-1] != len(FEATURES) or np.any(x < 0) or np.any(x >= 16):
        raise ValueError("Feature digits must fit 4 bits")
    result = np.zeros(x.shape[:-1], np.uint64)
    for j in range(len(FEATURES)):
        result |= x[..., j].astype(np.uint64) << np.uint64(4*j)
    return result


def unpack(keys):
    return np.column_stack([((keys >> np.uint64(4*j)) & np.uint64(15)).astype(np.uint8)
                            for j in range(len(FEATURES))])


def supported(x):
    return (x[:, 4] > 0) | (x[:, 3] > 0) | (x[:, 7] > 0)


def feature_codes(m, valid):
    m = m.copy(); m[~valid] = 0
    occ = (m > 0).astype(np.int32)
    b3, b5, b9 = (box_sum(occ, r) for r in (1,2,4))
    values = [m, b3-occ, b5-b3, b9-b5, box_sum(m, 2), prior6(m, valid),
              prior6(occ, valid), prior6(b3, valid), prior6(valid.astype(np.int32), np.ones(len(valid), bool)),
              np.arange(len(m))//31]
    codes = np.zeros(m.shape, np.uint64)
    for j, a in enumerate(values):
        b = np.searchsorted(THRESHOLDS[j], a, side="right") if j in THRESHOLDS else a
        if np.max(b) >= 16 or np.min(b) < 0:
            raise ValueError("Binning overflow")
        if np.ndim(b) == 1:
            b = b[:, None, None]
        codes |= np.asarray(b, np.uint64) << np.uint64(4*j)
    return codes


def collapse(keys, exposure, positive):
    unique, inverse = np.unique(keys, return_inverse=True)
    return {"keys": unique,
            "weight": np.bincount(inverse, weights=exposure, minlength=len(unique)),
            "positive": np.bincount(inverse, weights=positive, minlength=len(unique))}


class Engine:
    def __init__(self, static, tag="temporal"):
        self.static = static
        self.tag = tag
        self.archive = Archive()
        self.loaded = {}
        self.groups = {}

    def load(self, year):
        if year not in self.loaded:
            signature = hashlib.sha256(Path(__file__).read_bytes() + self.static.tobytes()
                + (self.archive.root/f"seasons/{year}.json").read_bytes()
                + (self.archive.root/"seasons.json").read_bytes()).hexdigest()[:16]
            p = output(f".cache/{self.tag}_{year}_{signature}.npz")
            if p.exists():
                with np.load(p) as z:
                    self.loaded[year] = {k:z[k] for k in z.files}
            else:
                m, v = self.archive.arrays(year)
                valid = self.archive.valid_dates(year)
                m[:, self.static] = 0; v[:, self.static] = 0
                data = {"codes": feature_codes(m, valid), "viirs": v > 0,
                        "modis": m > 0, "valid": valid}
                np.savez_compressed(p, **data)
                self.loaded[year] = data
            print(f"Loaded {self.tag} {year}", flush=True)
        return self.loaded[year]

    def group(self, year, geographic=None):
        geo_key = "all" if geographic is None else hashlib.sha256(geographic.tobytes()).hexdigest()
        key = year, geo_key
        if key not in self.groups:
            d = self.load(year)
            eligible = ~self.static if geographic is None else ~self.static & geographic
            code = d["codes"][d["valid"]][:, eligible].ravel()
            target = d["viirs"][d["valid"]][:, eligible].ravel()
            w = np.tile(AREA_WEIGHT[eligible], int(d["valid"].sum()))
            g = collapse(code, w, w*target)
            g["cell_days"] = len(code)
            self.groups[key] = g
        return self.groups[key]

    def training(self, years, geographic=None):
        groups = [self.group(y, geographic) for y in years]
        g = collapse(np.concatenate([d["keys"] for d in groups]),
                     np.concatenate([d["weight"] for d in groups]),
                     np.concatenate([d["positive"] for d in groups]))
        x = unpack(g["keys"]); support = supported(x)
        ledger = {"years": list(years), "full_grid_cell_days": sum(d["cell_days"] for d in groups),
                  "weighted_cell_days": float(g["weight"].sum()), "distinct_patterns": len(x),
                  "supported_patterns": int(support.sum()),
                  "supported_weighted_cell_days": float(g["weight"][support].sum()),
                  "recorded_positive_weight": float(g["positive"].sum()),
                  "unsupported_positive_share": float(g["positive"][~support].sum()/g["positive"].sum()),
                  "background_sampling": False}
        return x[support], g["weight"][support], g["positive"][support], ledger

    def prediction(self, year, model, geographic=None):
        g = self.group(year, geographic)
        return g["keys"], model.predict(unpack(g["keys"]))

    def rows(self, years, model, name, phase, geographic=None, regions=None):
        rows = []
        for year in years:
            d = self.load(year)
            keys, pred = self.prediction(year, model, geographic)
            # Geographic mask here controls evaluation only; predictors remain MODIS-only.
            for week in range(14):
                start, stop = 7*week, min(92, 7*week+7)
                valid = d["valid"][start:stop]
                if not valid.any():
                    continue
                code = d["codes"][start:stop][valid]
                eligible_codes = ~self.static if geographic is None else ~self.static & geographic
                p = np.zeros(code.shape)
                p[:, eligible_codes] = pred[np.searchsorted(keys, code[:, eligible_codes])]
                for region in regions or REGIONS:
                    eligible = REGIONS[region] & ~self.static
                    if geographic is not None:
                        eligible &= geographic
                    w = AREA_WEIGHT*eligible
                    rows.append({"phase": phase, "year": year, "week": week, "region": region,
                        "valid_days": int(valid.sum()),
                        "observed_viirs": float(np.sum(d["viirs"][start:stop][valid]*w)),
                        "observed_modis": float(np.sum(d["modis"][start:stop][valid]*w)),
                        "model": name, "predicted_viirs": float(np.sum(p*w))})
        return rows


def sigmoid(a):
    return 1/(1+np.exp(-np.clip(a, -35, 35)))


def logistic_loss(logit, weight, positive):
    return float(np.sum(weight*np.logaddexp(0, logit)-positive*logit)/np.sum(weight))


class HistogramBoost:
    def __init__(self, depth=5, learning_rate=.075, l2=10., min_exposure=500.):
        self.depth, self.learning_rate, self.l2, self.min_exposure = depth, learning_rate, l2, min_exposure
        self.trees = []
        self.trace = []

    def fit_tree(self, x, gradient, hessian, weight):
        nodes = []
        update = np.zeros(len(x))
        def build(idx, depth):
            position = len(nodes)
            g, h = float(gradient[idx].sum()), float(hessian[idx].sum())
            value = float(np.clip(-g/(h+self.l2), -2, 2))
            nodes.append({"feature": -1, "value": value})
            if depth >= self.depth or len(idx) < 2 or weight[idx].sum() < 2*self.min_exposure:
                update[idx] = value
                return position
            best = None
            parent = g*g/(h+self.l2)
            for feature in range(x.shape[1]):
                bins = x[idx, feature]
                gs = np.cumsum(np.bincount(bins, weights=gradient[idx], minlength=16))[:-1]
                hs = np.cumsum(np.bincount(bins, weights=hessian[idx], minlength=16))[:-1]
                ws = np.cumsum(np.bincount(bins, weights=weight[idx], minlength=16))[:-1]
                valid = (ws >= self.min_exposure) & (weight[idx].sum()-ws >= self.min_exposure)
                gain = gs*gs/(hs+self.l2) + (g-gs)**2/(np.maximum(h-hs, 0)+self.l2)-parent
                gain[~valid] = -np.inf
                threshold = int(np.argmax(gain))
                candidate = float(gain[threshold]), feature, threshold
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None or best[0] <= 1e-9:
                update[idx] = value
                return position
            _, feature, threshold = best
            left = x[idx, feature] <= threshold
            l = build(idx[left], depth+1); r = build(idx[~left], depth+1)
            nodes[position] = {"feature": feature, "threshold": threshold, "left": l, "right": r, "gain": best[0]}
            return position
        build(np.arange(len(x)), 0)
        return nodes, update

    @staticmethod
    def tree_predict(x, nodes):
        output = np.empty(len(x))
        stack = [(0, np.arange(len(x)))]
        while stack:
            node, idx = stack.pop()
            spec = nodes[node]
            if spec["feature"] < 0:
                output[idx] = spec["value"]
            else:
                left = x[idx, spec["feature"]] <= spec["threshold"]
                stack.extend([(spec["left"], idx[left]), (spec["right"], idx[~left])])
        return output

    def fit(self, x, weight, positive, rounds=200, callback=None):
        mean = np.clip(positive.sum()/weight.sum(), 1e-8, 1-1e-8)
        self.intercept = float(np.log(mean/(1-mean)))
        logits = np.full(len(x), self.intercept)
        for iteration in range(1, rounds+1):
            p = sigmoid(logits)
            gradient = weight*p-positive
            hessian = weight*np.maximum(p*(1-p), 1e-8)
            nodes, update = self.fit_tree(x, gradient, hessian, weight)
            self.trees.append(nodes)
            logits += self.learning_rate*update
            self.trace.append({"trees": iteration, "training_logloss": logistic_loss(logits, weight, positive),
                               "nodes": len(nodes)})
            if iteration % 25 == 0 or iteration == rounds:
                print(f"Trees {iteration}/{rounds}; training logloss {self.trace[-1]['training_logloss']:.6f}", flush=True)
            if callback is not None and iteration in CHECKPOINTS:
                callback(self, iteration)
        return self

    def predict(self, x):
        support = supported(x)
        result = np.zeros(len(x))
        if support.any():
            z = x[support]
            logits = np.full(len(z), self.intercept)
            for tree in self.trees:
                logits += self.learning_rate*self.tree_predict(z, tree)
            result[support] = sigmoid(logits)
        return result

    def save(self, name, metadata):
        write_json(name, {"engine": "numpy_histogram_gradient_boosting", "objective": "recorded_viirs_binary_occupancy",
                   "features": FEATURES, "thresholds": THRESHOLDS, "depth": self.depth,
                   "learning_rate": self.learning_rate, "l2": self.l2, "min_exposure": self.min_exposure,
                   "intercept": self.intercept, "trees": self.trees, "trace": self.trace, "metadata": metadata})

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        assert obj["features"] == FEATURES
        assert {int(k):v for k,v in obj["thresholds"].items()} == THRESHOLDS
        model = cls(obj["depth"], obj["learning_rate"], obj["l2"], obj["min_exposure"])
        model.intercept, model.trees, model.trace = obj["intercept"], obj["trees"], obj["trace"]
        return model


def selection_score(rows, sectors=SECTORS):
    def wape(region):
        r = [x for x in rows if x["region"] == region]
        return sum(abs(x["predicted_viirs"]-x["observed_viirs"]) for x in r)/sum(x["observed_viirs"] for x in r)
    return .5*wape("canada_rectangle") + .5*float(np.mean([wape(s) for s in sectors]))


def binary_metrics(engine, years, model, geographic=None):
    groups = [engine.group(y, geographic) for y in years]
    g = collapse(np.concatenate([r["keys"] for r in groups]), np.concatenate([r["weight"] for r in groups]),
                 np.concatenate([r["positive"] for r in groups]))
    x = unpack(g["keys"]); w = g["weight"]; t = g["positive"]
    def evaluate(p):
        clipped = np.clip(p, 1e-12, 1-1e-12)
        pred = p >= .5
        tp, fp, fn = float(t[pred].sum()), float((w-t)[pred].sum()), float(t[~pred].sum())
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        # Merge ties before ranking so repeated identical probabilities are not arbitrarily ordered.
        levels, inv = np.unique(p, return_inverse=True)
        pos = np.bincount(inv, weights=t); neg = np.bincount(inv, weights=w-t)
        auc = float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
        rev_pos = pos[::-1]; rev_all = (pos+neg)[::-1]
        ap = float(np.sum(rev_pos/pos.sum()*np.cumsum(rev_pos)/np.cumsum(rev_all)))
        return {"brier": float(np.sum(w*p*p-2*t*p+t)/w.sum()),
                "logloss": float(-np.sum(t*np.log(clipped)+(w-t)*np.log1p(-clipped))/w.sum()),
                "precision_at_0_5": precision, "recall_at_0_5": recall,
                "f1_at_0_5": 2*precision*recall/(precision+recall) if precision+recall else 0,
                "roc_auc": auc, "average_precision": ap}
    return {"model": evaluate(model.predict(x)), "raw_modis": evaluate((x[:,0]>0).astype(float)),
            "cell_days": sum(r["cell_days"] for r in groups), "scope": "area-weighted recorded VIIRS occupancy, not ground-truth fire"}


def add_wape(summary, rows):
    for r in summary:
        g = [x for x in rows if all(x[k] == r[k] for k in ("phase", "model", "region"))]
        r["weekly_wape"] = sum(abs(x["predicted_viirs"]-x["observed_viirs"]) for x in g)/sum(x["observed_viirs"] for x in g)


def preserve():
    main = protected_files()
    old = {}
    for p in RESEARCH.rglob("*"):
        if not p.is_file() or p.is_symlink() or p.is_relative_to(ROOT):
            continue
        if any(part in (".cache", ".venv", "__pycache__") for part in p.relative_to(RESEARCH).parts):
            continue
        old[p.relative_to(RESEARCH).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    write_json("results/protected_hashes.json", {"main":main,"prior_research":old,
               "main_baseline_scope":"snapshot_at_run_start"})


def verify_preserved():
    before = json.loads((ROOT/"results/protected_hashes.json").read_text())
    now = protected_files()
    assert now == before["main"], "Main application changed"
    changed = [p for p,h in before["prior_research"].items() if hashlib.sha256((RESEARCH/p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    result = {"main_files_checked":len(now), "main_unchanged":True,
              "prior_research_files_checked":len(before["prior_research"]), "prior_research_unchanged":True}
    write_json("results/isolation.json", result)
    return result


def run():
    started = time.monotonic()
    preserve()
    write_json("results/protocol_hashes.json", {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/"PROTOCOL.md", Path(__file__))})
    fixed = json.loads((RESEARCH/"experiment-02/models/fixed_selected.json").read_text())
    static = np.zeros(SHAPE, bool)
    for y,x in fixed["static_cells"]:
        static[y,x] = True
    engine = Engine(static)
    write_json("results/config.json", {"features": FEATURES,"thresholds":THRESHOLDS,"static_cells":fixed["static_cells"],
               "scope":"0.25 degree, 50-70N,140-60W,June-August", "new_nasa_data_downloaded":False,
               "final_14year_model_has_fresh_holdout":False, "numpy":np.__version__, "python":sys.version,
               "depth":5,"max_trees":200,"learning_rate":.075,"l2":10,"min_exposure":500})
    ledgers, selection, rows = {}, [], []
    x,w,t,ledger = engine.training(range(2012,2020)); ledgers["selection_fit_8years"] = ledger
    print("Selection training", ledger, flush=True)
    checkpoints = {}
    def checkpoint(model, trees):
        val = engine.rows((2020,2021,2022), model, "boosted_8year", "validation")
        score = selection_score(val)
        item = {"trees":trees,"validation_score":score,"training_logloss":model.trace[-1]["training_logloss"]}
        selection.append(item); print("Checkpoint",item,flush=True)
        model.save(f".cache/selected_{trees}.json", {"training":ledger})
        checkpoints[trees] = val
    model = HistogramBoost().fit(x,w,t,200,checkpoint)
    chosen = min(selection, key=lambda r:(r["validation_score"],r["trees"]))["trees"]
    model = HistogramBoost.load(ROOT/f".cache/selected_{chosen}.json")
    model.save("models/boosted_8year_selected.json", {"training":ledger,"selected_trees":chosen})
    write_csv("results/selection_curve.csv",selection)
    write_json("results/selection.json", {"selected_trees":chosen,"validation_years":[2020,2021,2022],"candidates":selection})
    rows.extend(checkpoints[chosen])
    rows.extend(engine.rows((2023,2024,2025),model,"boosted_8year","test"))
    for phase, years, history in (("validation",(2020,2021,2022),range(2012,2020)),("test",(2023,2024,2025),range(2012,2023))):
        training = engine.rows(history,model,"ignored","train")
        base = fit_baselines(training)
        obs = engine.rows(years,model,"ignored",phase)
        for r in obs:
            r.pop("model");r.pop("predicted_viirs")
        rows.extend(baseline_rows(obs,base))
        write_json(f"models/{phase}_reference_baselines.json",base)
    x,w,t,ledger = engine.training(range(2012,2023));ledgers["benchmark_fit_11years"] = ledger
    print("Benchmark training",ledger,flush=True)
    benchmark = HistogramBoost().fit(x,w,t,chosen)
    benchmark.save("models/boosted_11year_benchmark.json",{"training":ledger,"evaluation_years":[2023,2024,2025],"selected_trees":chosen})
    rows.extend(engine.rows((2023,2024,2025),benchmark,MODEL,"test"))
    write_json("results/binary_metrics.json", binary_metrics(engine,(2023,2024,2025),benchmark))
    # Old model rows are comparators only; they cannot affect checkpoint selection.
    for path, wanted in ((RESEARCH/"results/numpy_weekly_predictions.csv",("numpy_daily_occupancy",)),
                         (RESEARCH/"experiment-02/results/weekly_predictions.csv",("kernel_occupancy",))):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["phase"] in ("test","validation") and r["model"] in wanted:
                    rows.append({**r,**{k:int(r[k]) for k in ("year","week","valid_days")},
                                 **{k:float(r[k]) for k in ("observed_viirs","observed_modis","predicted_viirs")}})
    # Release dense year caches when not needed. Sufficient statistics stay small.
    engine.loaded.clear()
    x,w,t,ledger = engine.training(range(2012,2026));ledgers["final_fit_14years"] = ledger
    print("Final expanded training",ledger,flush=True)
    final = HistogramBoost().fit(x,w,t,chosen)
    final.save("models/boosted_14year_final.json",{"training":ledger,"selected_trees":chosen,
               "fresh_holdout_evaluation":False,"promotion_supported":False})
    del x,w,t,final
    engine.loaded.clear()
    # Fully label-separated eastern transfer test, with a two-degree train/test buffer.
    training_geo, test_geo = LON < -92, LON >= -90
    cold = Engine(static & training_geo,"cold_east")
    x,w,t,ledger = cold.training(range(2012,2020),training_geo)
    ledgers["cold_selection_8years"] = ledger
    cold_selection = []
    def cold_checkpoint(model, trees):
        val = cold.rows((2020,2021,2022),model,"boosted_cold_east","selection",training_geo)
        cold_selection.append({"trees":trees,"validation_score":selection_score(val,("west","central"))})
    HistogramBoost().fit(x,w,t,200,cold_checkpoint)
    cold_trees = min(cold_selection,key=lambda r:(r["validation_score"],r["trees"]))["trees"]
    write_csv("results/cold_east_selection.csv",cold_selection)
    x,w,t,ledger = cold.training(range(2012,2023),training_geo);ledgers["cold_benchmark_11years"] = ledger
    cold_model = HistogramBoost().fit(x,w,t,cold_trees)
    cold_model.save("models/boosted_cold_east.json",{"training":ledger,"test_longitude_min":-90,"training_longitude_max":-92})
    cold_rows = cold.rows((2023,2024,2025),cold_model,"boosted_cold_east","spatial_test",test_geo,("east",))
    rows.extend(cold_rows)
    hist = cold.rows(range(2012,2023),cold_model,"ignored","train",training_geo)
    hist = [r for r in hist if r["region"] in ("west","central")]
    mx = np.array([r["observed_modis"] for r in hist]); vy = np.array([r["observed_viirs"] for r in hist])
    b,a = np.polyfit(np.log1p(mx),np.log1p(vy),1)
    cold_base={"east":{"ratio":float(vy.sum()/mx.sum()),"a":float(a),"b":float(b)}}
    obs=[{k:v for k,v in r.items() if k not in ("model","predicted_viirs")} for r in cold_rows]
    rows.extend(baseline_rows(obs,cold_base))
    write_json("results/cold_binary_metrics.json",binary_metrics(cold,(2023,2024,2025),cold_model,test_geo))
    summary,seasonal=metrics(rows);add_wape(summary,rows)
    write_csv("results/weekly_predictions.csv",rows);write_csv("results/seasonal_predictions.csv",seasonal)
    write_csv("results/metrics.csv",summary);write_json("results/metrics.json",summary)
    write_json("results/training_ledgers.json",ledgers)
    write_json("results/runtime.json",{"elapsed_seconds":time.monotonic()-started,"selected_trees":chosen,"cold_trees":cold_trees})
    verify_preserved()
    print("Larger model training completed; all prior files unchanged.",flush=True)


if __name__ == "__main__":
    run()
