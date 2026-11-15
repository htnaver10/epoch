"""Scientific stage boundaries, native-field encoding, resume integrity and calibration."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from firefullres import core
from firefullres.campaign import Campaign,candidate_pairs,evenly_spaced,select_training_candidates
from firefullres.calibration import sigmoid,fit_intercept,matched_false_alarm_cutoff,exact_ranking,metrics
from firefullres.prepare import inside_bbox,store_tile


class RoundTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.cache').mkdir(exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.cache')
        self.root_patch=patch.object(core,'ROOT',Path(self.temp.name));self.root_patch.start()
        core.write_json('config.json',{'name':'test_round','groups':[],'max_granules_per_query':100,
            'download_budget_gib':1,'reserve_free_gib':0})

    def tearDown(self):self.root_patch.stop();self.temp.cleanup()

    def test_complete_native_context_and_resumed_arrays_are_verified(self):
        lat=np.array([[50.,50.5],[51.,51.5]]);lon=np.array([[-120.,-119.],[-120.,-119.]])
        self.assertTrue(inside_bbox(lat,lon,[-121,49,-118,52]))
        self.assertFalse(inside_bbox(lat,lon,[-121,49,-118,51]))
        lat[0,0]=np.nan;self.assertFalse(inside_bbox(lat,lon,[-121,49,-118,52]))
        arrays={'a':np.ones((2,2),np.float32)};store_tile('tile.npz',arrays)
        store_tile('tile.npz',arrays,True)
        with self.assertRaisesRegex(ValueError,'already exists'):store_tile('tile.npz',arrays)
        with self.assertRaisesRegex(ValueError,'differs'):store_tile('tile.npz',{'a':np.zeros((2,2),np.float32)},True)
        with self.assertRaisesRegex(ValueError,'differs'):store_tile('tile.npz',{'a':np.ones((2,2),np.float64)},True)

    def test_budget_stops_before_any_network_transfer(self):
        campaign=Campaign('config.json');download=Mock()
        row={'granule_id':'G1','estimated_bytes':2*1024**3}
        with patch.dict(sys.modules,earthaccess=SimpleNamespace(download=download)):
            with self.assertRaisesRegex(ValueError,'budget'):campaign.download(row,[-125,50,-110,65])
        download.assert_not_called()

    def test_modified_cached_granule_is_rejected(self):
        campaign=Campaign('config.json');p=core.path('data/raw/file.nc',True);p.write_bytes(b'original')
        campaign.journal['G1']={'local_path':'data/raw/file.nc','sha256':core.sha256(p)}
        p.write_bytes(b'modified-longer')
        with self.assertRaisesRegex(ValueError,'checksum'):campaign.download({'granule_id':'G1'},[-125,50,-110,65])

    def test_training_prefilter_uses_actual_c002_codes_8_and_9(self):
        campaign=Campaign('config.json')
        fields={'FP_latitude':np.array([51.,51.,51.,51.]),'FP_longitude':np.array([-120.,-120.,-120.,-100.]),
                'FP_confidence':np.array([7,8,9,9],np.uint8)}
        pair={'modis':{'bbox':[-125,50,-110,65]},'viirs':{}}
        with patch.object(campaign,'download',return_value='file.nc'),\
             patch('firefullres.native.h5_field',side_effect=lambda file,key:fields[key]):
            self.assertEqual(campaign.fire_score(pair,{'bbox':[-125,50,-110,65]}),2)

    def test_missing_night_reflective_companions_remain_optional(self):
        campaign=Campaign('config.json');base={'start_utc':'2021-07-01T06:00:00Z','end_utc':'2021-07-01T06:05:00Z'}
        with patch.object(campaign,'search',return_value=[]):
            self.assertIsNone(campaign.companion('MYD02HKM',base,[-125,50,-110,65]))
            with self.assertRaisesRegex(ValueError,'MYD03'):campaign.companion('MYD03',base,[-125,50,-110,65])

    def test_absent_sparse_arrays_are_zero_only_when_full_mask_has_no_positive(self):
        campaign=Campaign('config.json');pair={'modis':{'bbox':[-125,50,-110,65]},'viirs':{}}
        def read(file,key):
            if key=='fire mask':return np.full((2,2),5,np.uint8)
            raise ValueError('Expected one field; found 0')
        with patch.object(campaign,'download',return_value='file.nc'),patch('firefullres.native.h5_field',side_effect=read):
            self.assertEqual(campaign.fire_score(pair,{'bbox':[-125,50,-110,65]}),0)

    def test_inventory_pairing_respects_full_intervals_and_one_day_selection(self):
        def row(key,start,end):return {'granule_id':key,'start_utc':start,'end_utc':end,'bbox':[-125,50,-110,65]}
        modis=[row('m','2023-07-01T21:25:00Z','2023-07-01T21:30:00Z')]
        viirs=[row('v','2023-07-01T21:30:00Z','2023-07-01T21:36:00Z'),
               row('late','2023-07-01T21:36:00Z','2023-07-01T21:42:00Z')]
        pairs=candidate_pairs(modis,viirs,[-125,50,-110,65])
        self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]['viirs']['granule_id'],'v')
        self.assertEqual(evenly_spaced(list(range(10)),3),[0,4,9])

    def test_intercept_matches_only_supplied_validation_exposure(self):
        z=np.array([90.,91.,92.,93.]);y=np.array([0.,0.,0.,1.]);w=np.array([1.,2.,3.,4.])
        offset=fit_intercept(z,y,w)
        self.assertAlmostEqual(float(np.sum(w*sigmoid(z+offset))),float(np.sum(w*y)),places=6)
        with self.assertRaises(ValueError):fit_intercept(z,np.zeros(4),w)

    def test_fire_rich_training_selection_retains_night_and_controls(self):
        scored=[{'modis':{'granule_id':str(i),'start_utc':str(i),'day_night':'Night' if i==9 else 'Day'},
            'candidate_nominal_high_fire_pixels':100-i} for i in range(10)]
        selected=select_training_candidates(scored,8,2)
        self.assertEqual(len(selected),8)
        self.assertEqual(len({p['modis']['granule_id'] for p in selected}),8)
        self.assertTrue(any(p['modis']['day_night']=='Night' for p in selected))

    def test_tied_float32_logits_do_not_exceed_modis_false_alarm_budget(self):
        z=np.array([0.,0.,0.,1.],np.float32);y=np.array([0,0,0,1]);w=np.ones(4);b=np.zeros(4)
        cut,budget=matched_false_alarm_cutoff(z,y,w,b)
        pred=z.astype(np.float64)>=cut
        self.assertEqual(budget,0);self.assertEqual(int((pred&(y==0)).sum()),0)
        self.assertTrue(pred[-1])
        ranking=exact_ranking(np.array([.1,.9,.5,.5]),np.array([0,1,0,1]),np.ones(4))
        self.assertAlmostEqual(ranking['roc_auc_exact'],.875);self.assertAlmostEqual(ranking['average_precision_exact'],5/6)

    def test_calibration_is_frozen_before_test_collection(self):
        import evaluate_substantial_round as evaluation
        core.write_json('evaluation_config.json',{'name':'sealed','scope':'test fixture'})
        core.path('models/sealed/best.pt',True).write_bytes(b'checkpoint')
        core.path('data/manifests/sealed.jsonl',True).write_text('frozen manifest')
        values={'logits':np.array([-2.,-1.,1.,2.],np.float32),'target':np.array([0.,0.,1.,1.]),
            'weight':np.ones(4),'modis':np.array([0.,1.,0.,1.]),'tile_index':np.zeros(4,np.int32)}
        def collect(manifest,checkpoint,split):
            if split=='test':
                frozen=core.read_json('results/sealed/calibration.json')
                self.assertEqual(frozen['fit_split'],'validation');self.assertFalse(frozen['test_used_for_fit'])
            return values,[]
        with patch.object(evaluation,'collect',side_effect=collect),contextlib.redirect_stdout(io.StringIO()):
            report=evaluation.evaluate('evaluation_config.json')
        self.assertIn('test',report)
        core.path('models/sealed/best.pt').write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(ValueError,'Sealed evaluation'):evaluation.evaluate('evaluation_config.json')


if __name__=='__main__':unittest.main()
