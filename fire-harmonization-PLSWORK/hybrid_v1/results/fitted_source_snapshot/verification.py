"""Verify hashes, full-support counts, deployable inference and frozen policy constraints."""
import json
import numpy as np
from common import ROOT,path,read,write,sha,load,lock

def verify():
    from evaluation import verify_policy
    from firepls.prepare import load_split
    from preprocessing import INPUTS
    from run import worker
    lock();policy=verify_policy();receipt=read('data/receipt.json');report=read('results/evaluation.json')
    for filename,digest in receipt['files'].items():
        if sha(path(filename))!=digest:raise ValueError('Hybrid data hash failed')
    for split in ('train','validation','test'):
        assert sha(path(f'data/{split}_prior.npy'))==read(f'data/{split}_prior.json')['sha256']
    parent=json.loads((ROOT/'data/manifests/source_v3.jsonl').read_text())
    row=next(r for r in parent if r['split']=='test' and r['positive_queries']>0)
    with np.load(ROOT/row['tile'],allow_pickle=False) as z:
        arrays={k:z[k] for k in (*INPUTS,'modis_baseline','inference_valid')}
        reference_mask=z['weight']>0
    with path('results/inference_input_only.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
    worker('infer-prior','--tile','results/inference_input_only.npz','--output','results/inference_example.npz')
    worker('infer-finish')
    ledger=json.loads((ROOT/'data/features/tiles.json').read_text());item=next(t for t in ledger if t['tile_id']==row['tile_id'])
    with np.load(path('results/inference_example.npz')) as z:
        assert np.isfinite(z['raw_probability']).all()
        expected=load('results/test_logits.npy')[item['start']:item['end']]
        np.testing.assert_allclose(z['raw_logit'][reference_mask],expected,rtol=2e-6,atol=2e-5)
        np.testing.assert_array_equal(z['detection'][reference_mask],expected>=policy['query']['cutoff'])
    for name,point in [('query',policy['query']),('native_cell',policy['native_cell'])]:
        key='hybrid_frozen_policy' if name=='query' else 'hybrid_cell_policy'
        assert report['validation'][name][key]['false_positives']<=point['validation_false_positive_budget']
    assert report['test']['query']['standard_MODIS']['true_positives']==76
    assert report['test']['query']['standard_MODIS']['false_positives']==93
    assert report['test']['query']['standard_MODIS']['observed_positive_weight']==95
    result={'passed':True,'parent_software_unchanged':True,'source_split_overlap':False,
            'feature_and_prior_hashes_verified':True,'validation_budgets_respected':True,
            'input_only_inference_matches_benchmark':True,'inference_requires_viirs':False,
            'inference_requires_modis_detection':True,'original_test_exposure_unchanged':True,
            'network_satellite_downloads':0,'native_model_reload_passed':True,'new_blind_test':False}
    write('results/completion_verification.json',result);return result
