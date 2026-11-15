"""Context mathematics, label isolation, train-only scaling and export verification."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from firepls import common,features,prepare
from bootstrap_copy import contained
from firefullres import core
from firefullres.calibration import matched_false_alarm_cutoff
HAS_TORCH=importlib.util.find_spec('torch') is not None
HAS_LGBM=importlib.util.find_spec('lightgbm') is not None
if HAS_TORCH:
    import torch
    from firepls import mlp


def fixture_inputs(h=32,q=8):
    grid=np.zeros((q,q,2),np.float32)
    coords=np.linspace(-.4,.4,q,dtype=np.float32);grid[...,0]=coords[None,:];grid[...,1]=coords[:,None]
    return {'thermal':np.concatenate([np.ones((6,h,h),np.float32),np.ones((6,h,h),np.float32)]),
        'reflective_500':np.concatenate([np.full((5,h*2,h*2),.2,np.float32),np.ones((5,h*2,h*2),np.float32)]),
        'reflective_250':np.concatenate([np.full((2,h*4,h*4),.3,np.float32),np.ones((2,h*4,h*4),np.float32)]),
        'geometry':np.zeros((14,h,h),np.float32),'query_grid':grid}


class FeatureTests(unittest.TestCase):
    def setUp(self):self.cfg=common.config()
    def test_integral_box_sum_matches_direct_window_including_boundaries(self):
        a=np.arange(30).reshape(5,6);expected=np.zeros_like(a,dtype=float)
        for r in range(5):
            for c in range(6):expected[r,c]=a[max(0,r-1):min(5,r+2),max(0,c-1):min(6,c+2)].sum()
        np.testing.assert_array_equal(features.box_sum(a,3),expected)
    def test_annular_background_excludes_bright_center_and_tracks_support(self):
        a=np.ones((25,25));a[11:14,11:14]=1000;valid=np.ones_like(a,bool)
        mean,std,supported,fraction=features.annular_statistics(a,valid,21,3,8)
        self.assertAlmostEqual(mean[12,12],1.);self.assertAlmostEqual(std[12,12],0.)
        self.assertTrue(supported[12,12]);self.assertAlmostEqual(fraction[12,12],1.)
        mean,_,supported,_=features.annular_statistics(a,np.zeros_like(valid),21,3,8)
        self.assertFalse(supported.any());self.assertTrue((mean==0).all())
    def test_features_are_finite_and_accept_neither_fire_masks_nor_targets(self):
        inputs=fixture_inputs();values,names=features.extract(inputs,self.cfg)
        self.assertEqual(values.shape,(8,8,96));self.assertEqual(len(set(names)),96)
        self.assertTrue(np.isfinite(values).all())
        for key in ('target','modis_baseline','viirs_fire','query_lat'):
            with self.assertRaisesRegex(ValueError,'inputs'):features.extract({**inputs,key:np.zeros((8,8))},self.cfg)
    def test_invalid_thermal_values_do_not_leak_into_context(self):
        a=fixture_inputs();a['thermal'][6:]=0;b={k:v.copy() for k,v in a.items()};b['thermal'][:6]=10000
        x,names=features.extract(a,self.cfg);y,_=features.extract(b,self.cfg)
        np.testing.assert_array_equal(x,y)
    def test_fine_reflectance_samples_are_preserved_at_query_locations(self):
        a=fixture_inputs();iy,ix,_,_=features.query_indices(a['query_grid'],32,32,4)
        a['reflective_250'][0]=np.arange(128*128).reshape(128,128)
        x,names=features.extract(a,self.cfg)
        np.testing.assert_array_equal(x[...,names.index('reflectance_1')],a['reflective_250'][0,iy,ix])
    def test_path_guard_refuses_parent_escape(self):
        with self.assertRaisesRegex(ValueError,'escapes'):contained(ROOT,'../escaped')
    def test_tied_threshold_keeps_false_positives_within_validation_budget(self):
        z=np.array([0,0,0,1],np.float32);y=np.array([0,0,0,1]);w=np.ones(4);base=np.array([1,0,0,1])
        cut,budget=matched_false_alarm_cutoff(z,y,w,base)
        self.assertLessEqual(float(w[(z.astype(np.float64)>=cut)&(y==0)].sum()),budget)


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.cache').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.cache')
        self.patch=patch.multiple(common,ROOT=Path(self.tmp.name));self.patch.start()
        self.old_core=core.ROOT;core.ROOT=common.ROOT
        self.cfg=json.loads((ROOT/'configs/experiment.json').read_text());self.cfg['mlp'].update(device='cpu',precision='float32',epochs=2,batch_size=32,patience=3)
        rows=[]
        for split in ('train','validation','test'):
            a=fixture_inputs();n=a['query_grid'].shape[0];target=np.zeros((n,n),np.float32);target[2:4,2:4]=1.
            # Fixture-only label-dependent signal is deliberately created in the
            # test, not in production feature extraction or source preparation.
            a['thermal'][0,12:16,12:16]=5
            arrays={**a,'target':target,'weight':np.ones_like(target),'native_target':np.zeros((32,32),np.float32),
                'native_weight':np.ones((32,32),np.float32),'query_lat':np.zeros_like(target,dtype=np.float64),
                'query_lon':np.zeros_like(target,dtype=np.float64),'modis_baseline':target.copy(),'inference_valid':np.ones_like(target)}
            tile=f'data/tiles/{split}.npz';core.write_npz(tile,**arrays)
            rows.append({'schema':core.SCHEMA,'dataset_kind':'nasa_native','tile_id':'fixture_'+split,'tile':tile,
                'sha256':common.sha(tile),'split':split,'event_group':split,'region_id':split,'source_ids':['fixture_'+split],
                'query_shape':[n,n],'native_shape':[32,32],'valid_queries':float(n*n),'positive_queries':float(target.sum())})
        core.write_jsonl('data/manifests/source_v3.jsonl',rows);self.cfg['source_manifest_sha256']=common.sha('data/manifests/source_v3.jsonl')
        common.write('configs/experiment.json',self.cfg)
    def tearDown(self):core.ROOT=self.old_core;self.patch.stop();self.tmp.cleanup()
    def test_normalization_uses_training_only_and_preserves_every_known_query(self):
        validation=common.rows('data/manifests/source_v3.jsonl')[1]
        with np.load(common.path(validation['tile'])) as z:arrays={k:z[k] for k in z.files}
        arrays['thermal'][:6]=999.
        core.write_npz(validation['tile'],**arrays);rows=common.rows('data/manifests/source_v3.jsonl');rows[1]['sha256']=common.sha(validation['tile'])
        core.write_jsonl('data/manifests/source_v3.jsonl',rows);self.cfg['source_manifest_sha256']=common.sha('data/manifests/source_v3.jsonl');common.write('configs/experiment.json',self.cfg)
        receipt=prepare.prepare();stats=common.read('data/features/statistics.json');x=prepare.load_split('train')['X']
        np.testing.assert_allclose(stats['mean'],x.mean(0,dtype=np.float64),atol=1e-8)
        self.assertEqual(receipt['counts'],{'train':64,'validation':64,'test':64});self.assertEqual(stats['fit_partition'],'train')
    def test_preparation_receipt_rejects_tampered_feature_matrices(self):
        prepare.prepare();p=common.path('data/features/train_y.npy');a=np.load(p);a[0]=1-a[0];np.save(p,a)
        with self.assertRaisesRegex(ValueError,'matrix|normalizer'):prepare.prepare()
    def test_calibration_is_frozen_before_any_test_predictions(self):
        from firepls import evaluation
        prepare.prepare();common.path('models/mlp').mkdir(parents=True);common.path('models/tree').mkdir(parents=True)
        common.path('models/mlp/inference.pt').write_bytes(b'fixture_mlp');common.path('models/tree/model.txt').write_bytes(b'fixture_tree')
        def predict(method,split):
            self.assertEqual(split,'validation')
            y=prepare.load_split(split)['y']
            return np.where(y>0,1.,-1.) if method=='mlp' else np.where(y>0,.5,-.5)
        with patch.object(evaluation,'predict_method',side_effect=predict):policy=evaluation.freeze_policies()
        self.assertFalse(policy['test_used_for_fit']);self.assertEqual(policy['fit_split'],'validation')
        self.assertTrue(common.path('results/operating_policy.json').exists())
        common.path('models/tree/model.txt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'another model'):evaluation.freeze_policies()
    @unittest.skipUnless(HAS_LGBM,'LightGBM runs in its existing separate environment')
    def test_tree_fit_native_reload_and_label_free_inference(self):
        from firepls import tree
        from firepls.inference import infer
        self.cfg['lightgbm'].update(num_boost_round=5,stopping_rounds=2,min_data_in_leaf=2,num_threads=2)
        common.write('configs/experiment.json',self.cfg);prepare.prepare()
        with contextlib.redirect_stdout(io.StringIO()):result=tree.train();again=tree.train()
        self.assertEqual(result,again);self.assertTrue(result['native_reload_passed'])
        self.assertEqual(tree.predict('test').shape,(64,))
        core.write_npz('inference_only.npz',**fixture_inputs());inferred=infer('inference_only.npz','results/inference.npz','tree')
        self.assertFalse(inferred['requires_viirs'])
    @unittest.skipUnless(HAS_TORCH,'PyTorch unavailable')
    def test_mlp_train_reload_and_label_free_inference(self):
        from firepls.inference import infer
        prepare.prepare()
        with contextlib.redirect_stdout(io.StringIO()):result=mlp.train();again=mlp.train()
        self.assertEqual(result,again);model,saved=mlp.load_model()
        self.assertFalse(saved['requires_viirs_at_inference']);self.assertNotIn('optimizer',saved)
        fixture=fixture_inputs();core.write_npz('inference_only.npz',**fixture)
        inferred=infer('inference_only.npz','results/inference.npz')
        self.assertFalse(inferred['requires_viirs'])
        with np.load(common.path('results/inference.npz')) as z:self.assertEqual(z['raw_probability'].shape,(8,8));self.assertTrue(z['valid'].all())
    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_bfloat16_mlp_update_is_finite(self):
        torch.manual_seed(2);model=mlp.ContextMLP(96).cuda();x=torch.randn(32,96,device='cuda');y=torch.zeros(32,device='cuda');y[0]=1
        optimizer=torch.optim.AdamW(model.parameters());before=model.layers[-1].weight.detach().clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y);loss.backward();optimizer.step()
        self.assertTrue(torch.isfinite(loss));self.assertFalse(torch.equal(before,model.layers[-1].weight))


if __name__=='__main__':unittest.main()
