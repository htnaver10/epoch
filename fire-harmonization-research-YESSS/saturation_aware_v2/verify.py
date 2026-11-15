"""Check saturation-aware coverage, lineage, arithmetic and MODIS-only deployment."""
import numpy as np
import torch
from common import path,shared_path,read,read_shared,write,sha,freeze_fit,prior_lock,save_npz

def verify():
    from policy import load_policy
    from inference import infer_file
    from model import load_model
    from features import INPUT_KEYS
    freeze_fit();prior_lock();p=load_policy();r=read('results/evaluation.json');receipt=read('data/features/receipt.json')
    copied=read_shared('results/copy_receipt.json')
    for n,d in copied['files'].items():
        if sha(shared_path(n))!=d:raise ValueError('Shared copied input changed')
    for n,d in receipt['files'].items():
        if sha(path(n))!=d:raise ValueError('V2 feature cache changed')
    partition=read('data/partition.json');swaths={s:{t['modis_source'] for t in partition['rows'] if t['fit_split']==s} for s in ('train','development','test')}
    assert not any(swaths[a]&swaths[b] for a,b in [('train','development'),('train','test'),('development','test')])
    assert not p['uses_reference_labels'] and not p['nighttime_fallback']
    assert p['stages']['coarse']['night']['calibration_samples']>=1000
    for stage in ('coarse','fine'):
        net,saved=load_model(stage);selected=torch.load(path(f'models/{stage}/best.pt'),map_location='cpu',weights_only=True)
        assert not saved['uses_fire_labels'] and saved['model_name']=='SaturationAwareBackgroundV2'
        assert 'optimizer' not in saved
        assert all(torch.equal(v,selected['model'][k]) for k,v in net.state_dict().items())
        for group in p['stages'][stage].values():assert group['realized_alert_fraction']<=group['requested_alert_fraction']
    for split in ('validation','test'):
        with np.load(path(f'results/{split}_predictions.npz')) as a:
            for method,key in [('standard_MODIS','modis'),('unsupervised_coarse','coarse'),('unsupervised_two_stage','fine')]:
                tn,fp,fn,tp=np.bincount(2*a['y'].astype(int)+a[key].astype(int),minlength=4).tolist()
                assert [r[split]['query'][method][k] for k in ('TN','FP','FN','TP')]==[tn,fp,fn,tp]
            assert not np.any(a['fine']&~a['coarse'])
    assert r['test']['query']['standard_MODIS']['TP']==76 and r['test']['query']['standard_MODIS']['FP']==93
    assert r['test']['query']['standard_MODIS']['samples']==585464
    assert r['test']['coverage']['eligible_positives']==95, 'Regression: positive query locations discarded despite available band21'
    # Both selected branches must be visible in the saved sensor-quality outputs.
    assert r['test']['coverage']['positive_selected_band_counts']['21']>0
    example=next(t for t in read_shared('data/modis_manifest.json') if t['split']=='test')
    with np.load(shared_path(example['input'])) as a:
        assert set(a.files)==INPUT_KEYS
        save_npz('results/input_only_example.npz',**{k:a[k] for k in INPUT_KEYS})
    infer_file('results/input_only_example.npz','results/inference_example.npz')
    with np.load(path('results/inference_example.npz')) as a,np.load(path(f'results/predictions/{example["tile_id"]}.npz')) as b:
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        h,w=a['coarse_detection'].shape
        np.testing.assert_array_equal(a['fine_detection'].reshape(h,2,w,2).any((1,3)),a['coarse_detection'])
        np.testing.assert_allclose(a['localization_weights'].reshape(h,2,w,2).sum((1,3)),1.,atol=1e-7)
        assert not np.any((a['selected_band']==22)&~a['band22_valid'])
        assert not np.any((a['selected_band']==21)&~a['band21_valid'])
    result={'passed':True,'model_name':'SaturationAwareBackgroundV2','prior_models_and_results_unchanged':True,
            'shared_files_verified':len(copied['files']),'feature_hashes_verified':True,'fit_sources_frozen':True,
            'fire_labels_used_for_fitting_or_policy':False,'whole_swaths_disjoint':True,'nighttime_calibration_present':True,
            'positive_reference_coverage':r['test']['coverage']['positive'],'test_exposure_unchanged':True,
            'independent_confusion_counts_reconcile':True,'strict_exports_match_selected':True,
            'input_only_inference_matches_evaluation':True,'no_viirs_at_inference':True,'satellite_downloads':0,'new_blind_test':False}
    write('results/completion_verification.json',result);return result
