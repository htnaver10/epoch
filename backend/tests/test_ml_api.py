"""Exercise the real exported predictions and the API's failure boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

BACKEND=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BACKEND))
from fastapi.testclient import TestClient
from main import app
from services.ml_catalogue import MLCatalogue,CatalogueUnavailable

class MLAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=TestClient(app);cls.client=cls.context.__enter__()
    @classmethod
    def tearDownClass(cls):cls.context.__exit__(None,None,None)

    def test_catalogue_and_model(self):
        model=self.client.get('/ml/model');self.assertEqual(model.status_code,200)
        self.assertEqual(model.json()['metrics']['true_positives'],70)
        scenes=self.client.get('/ml/scenes').json();self.assertEqual(scenes['count'],84)
        self.assertEqual(sum(s['counts']['queries'] for s in scenes['scenes']),585464)
        self.assertEqual({s['model_version'] for s in scenes['scenes']},{model.json()['version']})

    def test_real_coordinates_and_policy_counts_reconcile(self):
        scenes=self.client.get('/ml/scenes').json()['scenes'];tp=fp=fn=0
        for scene in scenes:
            response=self.client.get(f'/ml/scenes/{scene["id"]}/predictions');self.assertEqual(response.status_code,200)
            all_features=response.json()['features'];self.assertEqual(response.json()['scene'],scene)
            west,south,east,north=scene['bounds']
            for f in all_features:
                lon,lat=f['geometry']['coordinates'];p=f['properties']
                self.assertTrue(west-1e-6<=lon<=east+1e-6 and south-1e-6<=lat<=north+1e-6)
                tp+=int(p['mlp_detected'] and p['viirs_reference']);fp+=int(p['mlp_detected'] and not p['viirs_reference']);fn+=int(not p['mlp_detected'] and p['viirs_reference'])
            coverage=self.client.get(f'/ml/scenes/{scene["id"]}/coverage').json()
            self.assertEqual(len(coverage['geometry']['coordinates']),scene['counts']['queries'])
        self.assertEqual((tp,fp,fn),(70,72,25))

    def test_layer_filters_and_unavailable_dates(self):
        scenes=self.client.get('/ml/scenes').json()['scenes'];scene=max(scenes,key=lambda s:s['counts']['mlp'])
        for layer,field in [('mlp','mlp_detected'),('modis','modis_detected'),('viirs','viirs_reference')]:
            r=self.client.get(f'/ml/scenes/{scene["id"]}/predictions?layer={layer}');self.assertEqual(r.status_code,200)
            self.assertTrue(all(f['properties'][field] for f in r.json()['features']))
        self.assertEqual(self.client.get('/ml/scenes?date=2003-01-01').json()['count'],0)
        self.assertEqual(self.client.get('/ml/scenes?date=not-a-date').status_code,422)
        self.assertEqual(self.client.get(f'/ml/scenes/{scene["id"]}/predictions?layer=unknown').status_code,422)
        self.assertEqual(self.client.get('/ml/scenes/not-a-scene/predictions').status_code,404)
        self.assertEqual(self.client.get('/ml/scenes/'+'f'*20+'/coverage').status_code,404)

    def test_missing_cache_does_not_break_existing_routes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'EPOCH_ML_CACHE_DIR':tmp}):
            with TestClient(app) as client:
                self.assertEqual(client.get('/').status_code,200);self.assertEqual(client.get('/ml/model').status_code,503)
        # Restore the real lifespan catalogue for the shared client after the nested context.
        app.state.ml_catalogue=MLCatalogue(BACKEND/'data/ml_cache')

    def test_checksum_failure_is_not_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data.json').write_text('{}')
            (root/'catalog.json').write_text(json.dumps({'schema':'epoch_mlp_cache_v1','scenes':[],'scene_count':0,'files':{'data.json':'0'*64}}))
            with self.assertRaises(CatalogueUnavailable):MLCatalogue(root).ready()

if __name__=='__main__':unittest.main()
