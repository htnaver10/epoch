"""The one-server entry point must expose the built site and real cache together."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from viewer import app


class LocalViewerTests(unittest.TestCase):
    def test_compiled_site_and_api_share_an_origin(self):
        with TestClient(app) as client:
            health = client.get("/viewer-health").json()
            self.assertTrue(health["cache_ready"])
            self.assertEqual(health["scene_count"], 84)
            page = client.get("/?mode=ml")
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="root"', page.text)
            self.assertIn("text/html", page.headers["content-type"])
            model = client.get("/api/ml/model").json()
            self.assertEqual(model["version"], health["model_version"])
            snapshot = client.get("/data/ml/catalog.json").json()
            self.assertEqual(snapshot["model"], model)
            self.assertEqual(client.get("/api/").json()["message"], "EPOCH API is running")


if __name__ == "__main__":
    unittest.main()
