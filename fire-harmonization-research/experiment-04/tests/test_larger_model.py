import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import larger_model as lm


class LargerModelTests(unittest.TestCase):
    def test_spatial_sum_no_wrap_and_preserves_large_counts(self):
        a=np.zeros((2,4,5),int);a[0,0,0]=500
        b=lm.box_sum(a,1)
        self.assertEqual(b[0,:2,:2].sum(),2000)
        self.assertEqual(b[0,-1,-1],0)
        self.assertEqual(b[1].sum(),0)

    def test_prior6_uses_neither_current_future_nor_invalid(self):
        a=np.arange(10)
        valid=np.ones(10,bool);valid[2]=False
        b=lm.prior6(a,valid)
        self.assertEqual(b[0],0)
        self.assertEqual(b[3],1)
        self.assertEqual(b[8],sum(a[3:8]))
        c=a.copy();c[8:]=9999
        np.testing.assert_array_equal(lm.prior6(c,valid)[:9],b[:9])

    def test_feature_pack_roundtrip_and_future_invariance(self):
        rng=np.random.default_rng(3)
        x=rng.integers(0,16,(20,len(lm.FEATURES)),dtype=np.uint8)
        np.testing.assert_array_equal(lm.unpack(lm.pack(x)),x)
        m=np.zeros((10,4,5),int);m[1,1,1]=1
        valid=np.ones(10,bool)
        c=lm.feature_codes(m,valid)
        m[8,1,1]=1000
        np.testing.assert_array_equal(c[:8],lm.feature_codes(m,valid)[:8])

    def test_sufficient_statistics_preserve_binary_loss_and_gradients(self):
        keys=np.array([5,5,2,2,5],np.uint64)
        w=np.array([.7,1.2,.8,1.5,.9]);y=np.array([0,1,1,0,1])
        g=lm.collapse(keys,w,w*y)
        logits=np.array([-.8,.3]);idx=np.searchsorted(g["keys"],keys)
        self.assertAlmostEqual(lm.logistic_loss(logits,g["weight"],g["positive"]),
                               lm.logistic_loss(logits[idx],w,w*y),places=13)
        grad=w*(lm.sigmoid(logits[idx])-y)
        np.testing.assert_allclose(np.bincount(idx,weights=grad),g["weight"]*lm.sigmoid(logits)-g["positive"])
        p=lm.sigmoid(logits)
        np.testing.assert_allclose(np.bincount(idx,weights=w*lm.sigmoid(logits[idx])*(1-lm.sigmoid(logits[idx]))),g["weight"]*p*(1-p))

    def test_newton_split_reduces_loss_and_predict_matches_training_updates(self):
        x=np.zeros((6,len(lm.FEATURES)),np.uint8);x[:,0]=np.arange(6);x[:,4]=1
        w=np.full(6,100.);t=np.array([0,0,10,70,100,100.])
        model=lm.HistogramBoost(depth=2,min_exposure=50.)
        p=np.full(6,.5)
        tree,update=model.fit_tree(x,w*p-t,w*p*(1-p),w)
        np.testing.assert_allclose(model.tree_predict(x,tree),update)
        self.assertGreater(len(tree),1)
        self.assertLess(lm.logistic_loss(.075*update,w,t),lm.logistic_loss(np.zeros(6),w,t))
        model.fit(x,w,t,10)
        self.assertLess(model.trace[-1]["training_logloss"],model.trace[0]["training_logloss"])
        self.assertGreater(model.predict(x)[-1],model.predict(x)[0])

    def test_saved_model_roundtrip_and_zero_support(self):
        x=np.zeros((3,len(lm.FEATURES)),np.uint8);x[1:,4]=1;x[2,0]=1
        model=lm.HistogramBoost(depth=1,min_exposure=1).fit(x[1:],np.array([10.,10.]),np.array([2.,8.]),3)
        p=model.predict(x)
        self.assertEqual(p[0],0.)
        self.assertTrue(np.all((p>=0)&(p<=1)))
        old=lm.ROOT
        (ROOT/".cache").mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/".cache") as d:
                lm.ROOT=Path(d)
                model.save("model.json",{})
                restored=lm.HistogramBoost.load(Path(d)/"model.json")
                np.testing.assert_array_equal(restored.predict(x),p)
        finally:
            lm.ROOT=old

    def test_geographic_predictions_exclude_static_cells_and_reconcile(self):
        static=np.zeros(lm.SHAPE,bool);static[0,0]=True
        e=lm.Engine(static,"synthetic")
        codes=np.zeros((92,*lm.SHAPE),np.uint64)
        features=np.zeros((1,len(lm.FEATURES)),np.uint8);features[:,4]=1
        codes[0,0,:2]=lm.pack(features)[0]
        valid=np.zeros(92,bool);valid[0]=True
        viirs=np.zeros(codes.shape,bool);viirs[0,0,:2]=True
        modis=viirs.copy()
        e.loaded[2025]={"codes":codes,"viirs":viirs,"modis":modis,"valid":valid}
        model=lm.HistogramBoost(min_exposure=1).fit(features,np.array([10.]),np.array([5.]),1)
        geo=np.zeros(lm.SHAPE,bool);geo[0,:2]=True
        rows=e.rows([2025],model,"x","test",geo,("canada_rectangle",))
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]["observed_viirs"],lm.AREA_WEIGHT[0,1])
        self.assertAlmostEqual(rows[0]["predicted_viirs"],.5*lm.AREA_WEIGHT[0,1])

    def test_recorded_occupancy_metrics_handle_probability_ties(self):
        x=np.zeros((4,len(lm.FEATURES)),np.uint8);x[:,0]=np.arange(4);x[:,4]=1
        class FakeEngine:
            def group(self,year,geographic=None):
                return {"keys":lm.pack(x),"weight":np.ones(4),"positive":np.array([0.,1.,0.,1.]),"cell_days":4}
        class FakeModel:
            def predict(self,x):
                return np.array([.1,.9,.5,.5])
        r=lm.binary_metrics(FakeEngine(),[2025],FakeModel())["model"]
        self.assertAlmostEqual(r["roc_auc"],.875)
        self.assertAlmostEqual(r["average_precision"],5/6)
        self.assertAlmostEqual(r["brier"],.13)


if __name__=="__main__":
    unittest.main()
