"""Runs on CPU after PyTorch installation; explicitly skipped when unavailable."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
HAS_TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'PyTorch not installed on this review host')
class TorchPipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(3)
    def test_model_native_shapes_masked_loss_and_backward(self):
        import torch
        from firefullres.network import NativeFireNet
        from firefullres.engine import loss_sums
        model=NativeFireNet(base=8,branch=4)
        inputs={'thermal':torch.randn(1,12,32,32),'reflective_500':torch.randn(1,10,64,64),
                'reflective_250':torch.randn(1,4,128,128),'geometry':torch.randn(1,14,32,32),
                'query_grid':torch.zeros(1,48,64,2,dtype=torch.float32)}
        out=model(**inputs)
        self.assertEqual(out['logits'].shape,(1,1,48,64));self.assertEqual(out['native_logits'].shape,(1,1,32,32))
        batch={'target':torch.zeros_like(out['logits']),'weight':torch.ones_like(out['logits']),
               'native_target':torch.zeros_like(out['native_logits']),'native_weight':torch.ones_like(out['native_logits'])}
        a,b=loss_sums(out,batch);loss=a/batch['weight'].sum()+.2*b/batch['native_weight'].sum();loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_accumulation_uses_total_exposure_not_mean_of_means(self):
        import torch
        from torch.nn import functional as F
        initial=torch.tensor(.2)
        whole=initial.clone().requires_grad_()
        y=torch.tensor([0.,1.,1.]);weight=torch.tensor([1.,2.,7.])
        (F.binary_cross_entropy_with_logits(whole.expand(3),y,reduction='none')*weight).sum().div(weight.sum()).backward()
        accumulated=initial.clone().requires_grad_()
        for sl in (slice(0,1),slice(1,3)):
            (F.binary_cross_entropy_with_logits(accumulated.expand(len(y[sl])),y[sl],reduction='none')*weight[sl]).sum().div(weight.sum()).backward()
        torch.testing.assert_close(whole.grad,accumulated.grad)
    def test_query_padding_does_not_change_unpadded_predictions(self):
        import torch
        from firefullres.network import NativeFireNet
        model=NativeFireNet(base=8,branch=4).eval()
        source={'thermal':torch.randn(1,12,32,32),'reflective_500':torch.randn(1,10,64,64),
                'reflective_250':torch.randn(1,4,128,128),'geometry':torch.randn(1,14,32,32)}
        small=torch.zeros(1,16,16,2);large=torch.full((1,24,24,2),2.);large[:,:16,:16]=small
        with torch.no_grad():
            a=model(**source,query_grid=small)['logits'];b=model(**source,query_grid=large)['logits'][:,:,:16,:16]
        torch.testing.assert_close(a,b)

    def test_train_resume_evaluate_and_infer_roundtrip(self):
        import json
        from firefullres import core,engine
        from firefullres.demo import make_demo
        root=Path(__file__).resolve().parents[1];(root/'.cache').mkdir(exist_ok=True)
        config=json.loads((root/'configs/smoke_cpu.json').read_text());config['epochs']=1
        old=core.ROOT
        try:
            with tempfile.TemporaryDirectory(dir=root/'.cache') as temp:
                core.ROOT=Path(temp)
                make_demo('manifest.jsonl');core.statistics('manifest.jsonl','stats.json',True)
                core.write_json('config.json',config)
                engine.train('manifest.jsonl','stats.json','config.json','unit',True)
                engine.train('manifest.jsonl','stats.json','config.json','unit',True,True,epochs_override=2)
                result=engine.predict('manifest.jsonl','models/unit/best.pt','results/test','test','cpu',True,True)
                self.assertTrue(result['synthetic_smoke_only'])
                self.assertTrue(0<=result['model_metrics']['brier']<=1)
                result=engine.predict('manifest.jsonl','models/unit/best.pt','results/infer','test','cpu',True,False)
                self.assertEqual(len(result['predictions']),2)
        finally:core.ROOT=old


if __name__=='__main__':unittest.main()
