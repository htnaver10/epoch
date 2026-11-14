"""Independent reconciliation of saved metrics, split boundaries and app hashes."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiment import RESEARCH, PROJECT, verify_isolation, write_json


def main():
    folder = RESEARCH / "results"
    config = json.loads((folder / "numpy_config.json").read_text())
    sets = [set(config[k]) for k in ("train_years", "validation_years", "test_years")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert max(sets[0]) < min(sets[1]) and max(sets[1]) < min(sets[2])
    rows = list(csv.DictReader((folder / "numpy_weekly_predictions.csv").open()))
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["phase"], row["model"], row["region"], int(row["year"]), int(row["week"])
        assert key not in seen
        seen.add(key)
        assert math.isfinite(float(row["predicted_viirs"])) and float(row["predicted_viirs"]) >= 0
        assert int(row["year"]) in (sets[1] if row["phase"] == "validation" else sets[2])
        grouped[key[:3]].append(row)
    checked = 0
    for metric in csv.DictReader((folder / "numpy_metrics.csv").open()):
        group = grouped[(metric["phase"], metric["model"], metric["region"])]
        errors = [float(r["predicted_viirs"]) - float(r["observed_viirs"]) for r in group]
        actual_mae = statistics.mean(abs(e) for e in errors)
        assert math.isclose(actual_mae, float(metric["weekly_mae"]), rel_tol=1e-12)
        checked += 1
    for row in csv.DictReader((folder / "numpy_seasonal_predictions.csv").open()):
        group = [r for r in grouped[(row["phase"], row["model"], row["region"])] if r["year"] == row["year"]]
        for field in ("predicted_viirs", "observed_viirs"):
            assert math.isclose(sum(float(r[field]) for r in group), float(row[field]), rel_tol=1e-12)
        assert sum(int(r["valid_days"]) for r in group) == int(row["matched_days"])
    hashes = json.loads((folder / "input_hashes.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == expected
    isolation = verify_isolation()
    assert isolation["passed"]
    result = {"passed": True, "weekly_prediction_rows_checked": len(rows),
              "metric_groups_recomputed": checked, "source_files_hash_checked": len(hashes),
              "checks": ["nonoverlapping chronological splits", "unique saved prediction keys",
                         "finite nonnegative predictions", "full held-out-year membership",
                         "independent weekly MAE recalculation", "season/week total reconciliation",
                         "source file integrity", "main app isolation"]}
    write_json("results/result_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
