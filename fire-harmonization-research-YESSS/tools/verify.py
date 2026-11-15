"""End-to-end provenance, invariance and portable inference checks."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fireyes.common import path,read,write,sha,freeze_fit

def verify():
    from fireyes.policy import load_policy
    from fireyes.inference import infer_file
    from fireyes.features import INPUT_KEYS
    from fireyes.model import load_model
    freeze_fit();p=load_policy();r=read('results/evaluation.json');copy=read('results/copy_receipt.json')
    for f,d in copy['files'].items():
        if sha(path(f))!=d:raise ValueError('Copied local file checksum mismatch')
    for f,d in read('data/features/receipt.json')['files'].items():
        if sha(path(f))!=d:raise ValueError('Prepared input checksum mismatch')
    assert not p['uses_reference_labels'] and not p['threshold_uses_modis_fire_flags']
    for stage in ('coarse','fine'):
        model,saved=load_model(stage);assert not saved['uses_fire_labels'];assert 'optimizer' not in saved
    assert r['test']['query']['standard_MODIS']['TP']==76 and r['test']['query']['standard_MODIS']['FP']==93
    assert r['test']['query']['standard_MODIS']['reference_positives']==95
    assert r['test']['query']['standard_MODIS']['samples']==585464
    # Recompute the confusion counts with an independent encoding/bincount operation.
    for split in ('validation','test'):
        with np.load(path(f'results/{split}_predictions.npz')) as z:
            for method,key in [('standard_MODIS','modis'),('unsupervised_coarse','coarse'),('unsupervised_two_stage','fine')]:
                encoded=2*z['y'].astype(np.int64)+z[key].astype(np.int64)
                tn,fp,fn,tp=np.bincount(encoded,minlength=4).tolist()
                metric=r[split]['query'][method]
                assert [metric[k] for k in ('TN','FP','FN','TP')]==[tn,fp,fn,tp]
                assert np.isclose(metric['f1'],2*tp/max(2*tp+fp+fn,1))
            assert not np.any(z['fine']&~z['coarse'])
    for stage,groups in p['stages'].items():
        for group in groups.values():
            assert group['realized_alert_fraction']<=group['requested_alert_fraction']
    row=next(q for q in read('data/modis_manifest.json') if q['split']=='test')
    with np.load(path(row['input'])) as z:assert set(z.files)==INPUT_KEYS
    infer_file(row['input'],'results/inference_example.npz')
    with np.load(path('results/inference_example.npz')) as a, np.load(path(f'results/predictions/{row["tile_id"]}.npz')) as b:
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        h,w=a['coarse_detection'].shape;weights=a['localization_weights'].reshape(h,2,w,2).sum((1,3))
        np.testing.assert_allclose(weights,1.,atol=1e-7)
        any_fine=a['fine_detection'].reshape(h,2,w,2).any((1,3))
        np.testing.assert_array_equal(any_fine,a['coarse_detection'])
    result={'passed':True,'physical_copy_files_verified':len(copy['files']),'satellite_downloads':0,
            'fire_labels_used_for_training_or_selection':False,'fire_labels_used_for_thresholds':False,
            'test_inputs_used_for_fitting':False,'source_and_feature_hashes_verified':True,
            'input_only_inference_matches_evaluation':True,'localization_preserves_parent_alerts':True,
            'original_reference_exposure_unchanged':True,'independent_confusion_counts_reconcile':True,
            'unlabeled_alert_budgets_respected':True,'new_blind_test':False}
    write('results/completion_verification.json',result);return result
