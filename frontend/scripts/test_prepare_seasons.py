import json
import math
import tempfile
import unittest
from pathlib import Path
from prepare_seasons import build, load, rebuild, weight, static_sources, archive_gaps, fill_gaps


def season_file(folder, source, year, records, days=92):
    data = {"source": source, "year": year, "bbox": "-140,50,-60,70", "grid_degrees": 0.25,
            "season": ["06-01", "08-31"], "days_covered": days,
            "total_detections": sum(r[3] for r in records), "records": records}
    (folder / f"{source}_{year}.json").write_text(json.dumps(data))


def sample(folder):
    # cell (200, -400) -> 50.125N, -99.875E (central); (200, -500) -> -124.875 (west)
    season_file(folder, "MODIS_SP", 2015, [["2015-06-01", 200, -400, 3, 1],
                                           ["2015-06-02", 200, -500, 1, 0]])
    season_file(folder, "VIIRS_SNPP_SP", 2015, [["2015-06-01", 200, -400, 9, 4],
                                                ["2015-06-03", 200, -400, 2, 0]], days=61)
    season_file(folder, "MODIS_SP", 2005, [["2005-08-31", 200, -400, 5, 0]])


class SeasonTests(unittest.TestCase):
    def test_merges_sensors_by_cell_day_and_splits_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sample(folder)
            meta, cells = build(load(folder))
            self.assertIn(["2015-06-01", 50.125, -99.875, 3, 9], cells[2015])
            self.assertEqual(len(cells[2015]), 3)
            c = meta["series"]["central"]["2015"]
            self.assertEqual(c["modis"]["cells"][:3], [1, 0, 0])
            self.assertEqual(c["viirs"]["cells"][:3], [1, 0, 1])
            self.assertEqual(c["viirs"]["raw"][:3], [9, 0, 2])
            self.assertEqual(c["viirs"]["days"], 61)
            self.assertEqual(c["shared"][:3], [1, 0, 0])
            self.assertEqual(meta["series"]["west"]["2015"]["modis"]["cells"][1], 1)
            self.assertEqual(meta["series"]["canada"]["2015"]["modis"]["raw"][:2], [3, 1])
            # pre-VIIRS year has no VIIRS series rather than zeros
            self.assertNotIn("viirs", meta["series"]["canada"]["2005"])
            self.assertEqual(meta["series"]["canada"]["2005"]["modis"]["cells"][91], 1)

    def test_cells_are_weighted_by_ground_area(self):
        self.assertAlmostEqual(weight(60), 1.0)
        self.assertAlmostEqual(weight(70) / weight(50), math.cos(math.radians(70)) / math.cos(math.radians(50)))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sample(folder)
            meta, _ = build(load(folder))
            w = meta["series"]["central"]["2015"]["modis"]["w"]
            self.assertEqual(w[0], round(weight(50.125), 2))
            self.assertGreater(w[0], 1)  # south of 60N a cell covers more ground
            self.assertEqual(w[1], 0)

    def test_rebuild_from_published_cells_matches_the_first_build(self):
        with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as out:
            sample(Path(raw))
            meta, cells = build(load(Path(raw)))
            out = Path(out)
            (out / "seasons").mkdir()
            (out / "seasons.json").write_text(json.dumps(meta))
            for year, records in cells.items():
                (out / "seasons" / f"{year}.json").write_text(json.dumps({"year": year, "records": records}))
            again, _ = rebuild(out)
            self.assertEqual(again, json.loads(json.dumps(meta)))


class CorrectionTests(unittest.TestCase):
    def test_a_cell_hot_every_summer_is_static_and_a_wildfire_cell_is_not(self):
        years = range(2012, 2022)
        coverage = {"viirs": {str(y): {"days": 92} for y in years}, "modis": {}}
        cells = {}
        for y in years:
            plant = [[f"{y}-06-{d:02d}", 57.125, -111.625, 1, 2] for d in range(1, 8)]  # 7 days, every year
            farm = [[f"{y}-07-01", 51.875, -102.875, 1, 1]]  # one day a year
            fire = [[f"{y}-07-{d:02d}", 60.125, -120.125, 3, 9] for d in range(1, 30)] if y == 2015 else []
            cells[y] = plant + farm + fire
        self.assertEqual(static_sources(cells, coverage, 92), [[57.125, -111.625]])

    def test_a_day_missing_from_one_archive_is_found_and_filled(self):
        days = 15
        coverage = {s: {"2024": {"days": days}} for s in ("modis", "viirs")}
        records = []
        for i in range(days):
            iso = f"2024-06-{i + 1:02d}"
            for c in range(30):  # 30 cells burn every day; VIIRS loses day 7
                records.append([iso, 55 + c * 0.25, -110.125, 1, 0 if i == 7 else 2])
        gaps = archive_gaps({2024: records}, coverage, "06-01", days, set())
        self.assertEqual(gaps, {"modis": {}, "viirs": {"2024": [7]}})
        row = {"modis": {k: [30.0] * days for k in ("cells", "w", "raw")},
               "viirs": {k: [30.0] * days for k in ("cells", "w", "raw")}}
        for k in row["viirs"]:
            row["viirs"][k][7] = 0
        row["modis"]["cells"][7] = 45  # a busy day for MODIS: the fill follows it
        fill_gaps(row, gaps, 2024)
        self.assertEqual(row["viirs"]["cells"][7], 45)
        self.assertEqual(row["viirs"]["w"][7], 30)

    def test_one_sensor_alone_never_has_gaps(self):
        coverage = {"modis": {"2005": {"days": 10}}, "viirs": {}}
        records = [[f"2005-06-{i + 1:02d}", 55.125, -110.125, 0 if i == 5 else 40, 0] for i in range(10)]
        self.assertEqual(archive_gaps({2005: records}, coverage, "06-01", 10, set()), {"modis": {}, "viirs": {}})


if __name__ == "__main__":
    unittest.main()
