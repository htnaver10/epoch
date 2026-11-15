"""Focused tests for new scientific preprocessing and residual inference semantics."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from common import path,config
from preprocessing import pool_valid,features,aggregate_cells
from firefullres.calibration import matched_false_alarm_cutoff,exact_ranking

class HybridTests(unittest.TestCase):
    def test_valid_pool_excludes_missing_values(self):
        a=np.array([[1.,1000.],[3.,5.]]);valid=np.array([[1,0],[1,1]])
        mean,fraction=pool_valid(a,valid,2)
        self.assertEqual(mean[0,0],3.);self.assertEqual(fraction[0,0],.75)
        empty,fraction=pool_valid(a,valid*0,2)
        self.assertEqual(empty[0,0],0.);self.assertEqual(fraction[0,0],0.)

    def test_pool_aligned_cell_blocks(self):
        a=np.arange(16).reshape(4,4)
        mean,f=pool_valid(a,np.ones_like(a),2)
        np.testing.assert_array_equal(mean,[[2.5,4.5],[10.5,12.5]])
        np.testing.assert_array_equal(f,np.ones((2,2)))

    def test_features_reject_label_inputs(self):
        with self.assertRaises(ValueError):features({'target':np.ones((2,2))},np.zeros((2,2)))

    def test_native_cells_use_same_support(self):
        z=np.array([-5.,2.,1.,-2.]);y=np.array([0,1,0,0]);b=np.array([1,1,0,0]);cell=np.array([0,0,1,1])
        cz,cy,cw,cb=aggregate_cells(z,y,b,cell)
        np.testing.assert_array_equal(cz,[2,1]);np.testing.assert_array_equal(cy,[1,0])
        np.testing.assert_array_equal(cb,[1,0]);np.testing.assert_array_equal(cw,[1,1])

    def test_empty_cell_rejected(self):
        with self.assertRaises(ValueError):aggregate_cells(np.ones(2),np.zeros(2),np.zeros(2),np.array([0,2]))

    def test_tied_policy_keeps_false_positive_budget(self):
        z=np.array([1.,1.,1.,0.]);y=np.array([1,0,0,0]);b=np.array([0,1,0,0])
        cut,budget=matched_false_alarm_cutoff(z,y,np.ones(4),b)
        self.assertLessEqual(((z>=cut)&(y==0)).sum(),budget)

    def test_path_cannot_escape(self):
        with self.assertRaises(ValueError):path('../other_experiment.json')

    def test_test_predict_requires_frozen_policy(self):
        from model import predict
        with patch('model.path') as mocked:
            mocked.return_value.exists.return_value=False
            with self.assertRaises(ValueError):predict('test')

    def test_init_score_reload_requires_neural_component(self):
        import lightgbm as lgb
        from model import fit_residual
        rng=np.random.default_rng(42);x=rng.normal(size=(600,3)).astype(np.float32)
        prior=2*x[:,0]-1.;y=(prior+x[:,1]>.2).astype(np.float32);w=np.ones(600)
        cfg=config()['residual'].copy();cfg.update(num_boost_round=12,stopping_rounds=5,min_data_in_leaf=10,num_threads=2)
        model,history,z=fit_residual(x[:400],y[:400],w[:400],prior[:400],x[400:],y[400:],w[400:],prior[400:],cfg,['a','b','c'])
        with tempfile.TemporaryDirectory() as tmp:
            filename=tmp+'/residual.txt';model.save_model(filename,num_iteration=model.best_iteration)
            reload=lgb.Booster(model_file=filename)
            residual=reload.predict(x[400:],raw_score=True,num_threads=2)
        np.testing.assert_allclose(prior[400:]+residual,z,rtol=1e-12,atol=1e-12)
        self.assertGreater(np.max(np.abs(residual-z)),.1)
        ap=exact_ranking(z,y[400:],w[400:])['average_precision_exact']
        self.assertAlmostEqual(ap,model.best_score['validation']['average_precision'],places=10)

if __name__=='__main__':unittest.main()
