"""Synthetic format/integrity tests, not evidence about remote NASA files."""
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import archive_tools as archive


class MonthlyArchiveTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.root / name
        with gzip.open(path, "wt") as f:
            f.write(text)
        return path

    def modis(self, name="MCD14ML.201901.061.03.txt.gz", row=None):
        return self.write(name, " ".join(archive.MODIS_FIELDS) + "\n" +
                          (row or "20190101 0051 A -12.1 143.2 321.6 289.6 681 11.5 53 0 D") + "\n")

    def test_modis_retains_counts_and_provenance(self):
        info = archive.inspect_file(self.modis())
        self.assertEqual(info["rows"], 1)
        self.assertEqual(info["daynight_rows"], {"D": 1})
        self.assertEqual(info["type_rows"], {0: 1})
        self.assertEqual(len(info["sha256"]), 64)

    def test_viirs_collection_aliases_and_csv_extra_fields(self):
        for token in ("C2", "02"):
            path = self.write(f"VNP14IMGML.201901.{token}.03.csv.gz",
                ",".join(archive.VIIRS_FIELDS + ["T4"]) + "\n" +
                "20190101,51,-12.1,143.2,11.5,N,N,0,23.4,0.14,321.6\n")
            info = archive.inspect_file(path)
            self.assertEqual(info["rows"], 1)
            self.assertEqual(info["sensor"], "viirs")
            self.assertIn("T4", info["fields"])

    def test_rejects_collection_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "Wrong collection"):
            archive.inspect_file(self.modis("MCD14ML.201901.006.03.txt.gz"))

    def test_rejects_wrong_month_bad_time_and_nonfinite(self):
        for row in ("20190201 0051 A -12.1 143.2 321.6 289.6 681 11.5 53 0 D",
                    "20190101 2460 A -12.1 143.2 321.6 289.6 681 11.5 53 0 D",
                    "20190101 0051 A nan 143.2 321.6 289.6 681 11.5 53 0 D"):
            with self.assertRaises(ValueError):
                archive.inspect_file(self.modis(row=row))

    def test_rejects_truncated_gzip(self):
        path = self.modis()
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaises((EOFError, OSError)):
            archive.inspect_file(path)

    def test_duplicate_month_blocks_ingestion_and_never_claims_training(self):
        old_root = archive.ROOT
        archive.ROOT = self.root
        try:
            raw = self.root / "data/raw/viirs"
            raw.mkdir(parents=True)
            for token in ("C2", "02"):
                path = self.write(f"VNP14IMGML.201901.{token}.03.csv.gz",
                    ",".join(archive.VIIRS_FIELDS) + "\n20190101,0051,-12,143,11,N,N,0,20,0.15\n")
                path.rename(raw / path.name)
            result = archive.audit("pilot")
            self.assertFalse(result["ingestion_passed"])
            self.assertFalse(result["model_trained"])
            self.assertTrue(any("duplicate" in e for e in result["errors"]))
            self.assertEqual(result["required_sensor_months"], 48)
        finally:
            archive.ROOT = old_root

    def test_plan_has_all_months_and_is_not_a_download(self):
        old_root = archive.ROOT
        archive.ROOT = self.root
        try:
            result = archive.prepare()
            self.assertEqual(result["full_sensor_months"], 288)
            self.assertEqual(result["downloaded_by_prepare"], 0)
            commands = (self.root / "acquisition/full.sftp").read_text()
            self.assertEqual(sum(s.startswith("get ") for s in commands.splitlines()), 288)
            self.assertIn("MCD14ML.201301.061.03.txt.gz", commands)
        finally:
            archive.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
