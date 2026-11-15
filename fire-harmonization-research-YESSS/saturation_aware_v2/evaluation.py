"""Reference labels are opened only after model and unlabeled policy are frozen."""
import importlib.util
import numpy as np
from common import ROOT,SHARED,path,shared_path,read,read_shared,write,sha,save_npz,config
from policy import load_policy
from data import load_inputs
from inference import infer_arrays

def summarize(y,pred,score=None):
    y=np.asarray(y,bool);pred=np.asarray(pred,bool)
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    out={'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':p,'recall':r,'f1':2*tp/max(2*tp+fp+fn,1),
         'reference_positives':int(y.sum()),'samples':len(y),'average_precision':None}
    if score is not None:
        # Load the frozen metric helper without changing module lookup for V2 report/verification.
        spec=importlib.util.spec_from_file_location('v2_reference_metrics',SHARED/'tools/reference_metrics.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        out['average_precision']=helper.exact_ranking(score,y.astype(float),np.ones(len(y)))['average_precision_exact']
    return out

def native_summary(y,base,coarse,score,cell):
    n=int(cell.max())+1;cy=np.zeros(n,bool);cb=cy.copy();cp=cy.copy();cz=np.full(n,-1.)
    for out,values in [(cy,y),(cb,base),(cp,coarse),(cz,score)]:np.maximum.at(out,cell,values)
    return {'VIIRS_reference_self_comparison':summarize(cy,cy,cy.astype(float)),
            'standard_MODIS':summarize(cy,cb,cb.astype(float)),
            'unsupervised_coarse':summarize(cy,cp,cz)}

def evaluate():
    policy=load_policy();dest='results/evaluation.json'
    if path(dest).exists():
        report=read(dest)
        if report['policy_sha256']!=sha(path('models/policy.json')):raise ValueError('Sealed policy changed')
        return report
    # This is the first pipeline stage allowed to read the physically separate reference manifest.
    refs=read_shared('data/reference_manifest.json');inputs={r['tile_id']:r for r in read_shared('data/modis_manifest.json')}
    report={'model_name':config()['name'],'new_blind_test':False,'uses_fire_labels_for_fit':False,'policy_sha256':sha(path('models/policy.json')),
            'training':{s:read(f'models/{s}/completed.json') for s in ('coarse','fine')},
            'policy':policy,'support':'VIIRS query centers assigned to native MODIS cells; approximate footprint support',
            'notes':'Both stages are anomaly detectors. Localization weights and reconstruction residuals are not fire probabilities.'}
    for split in ('validation','test'):
        chunks={k:[] for k in ('y','modis','coarse','fine','score','valid','cell','selected_band')};cell_ids={};next_cell=0;tile_metrics=[]
        native_alerts=0;localized=0
        for index,row in enumerate(r for r in refs if r['split']==split):
            item=inputs[row['tile_id']];out=infer_arrays(load_inputs(item),policy)
            # Save all native outputs for inspection; no VIIRS-dependent input is used for prediction.
            if split=='test':save_npz(f'results/predictions/{row["tile_id"]}.npz',**out)
            with np.load(shared_path(row['reference']),allow_pickle=False) as z:
                known=z['weight']>0;y=z['target'][known]>.5;grid=z['query_grid'][known];base=z['modis_baseline'][known]>.5
                if not np.all(z['weight'][known]==1):raise ValueError('Expected unit reference exposure')
            h,w=out['coarse_score'].shape
            fy=np.clip(np.floor((grid[:,1]+1)*h).astype(int),0,2*h-1)
            fx=np.clip(np.floor((grid[:,0]+1)*w).astype(int),0,2*w-1)
            iy,ix=fy//2,fx//2
            coarse=out['coarse_detection'][iy,ix];fine=out['fine_detection'][fy,fx];score=out['coarse_score'][iy,ix]
            r0,c0,_,_=item['source_window'];local=iy*w+ix;unique,inverse=np.unique(local,return_inverse=True);mapped=[]
            for value in unique:
                key=(item['modis_source'],r0+int(value)//w,c0+int(value)%w)
                if key not in cell_ids:cell_ids[key]=next_cell;next_cell+=1
                mapped.append(cell_ids[key])
            cell=np.asarray(mapped,np.int32)[inverse]
            for k,v in [('y',y),('modis',base),('coarse',coarse),('fine',fine),('score',score),('valid',out['coarse_valid'][iy,ix]),('cell',cell),('selected_band',out['selected_band'][iy,ix])]:chunks[k].append(v)
            native_alerts+=int(out['coarse_detection'].sum());localized+=int(out['localization_supported'].sum())
            tile_metrics.append({'tile_id':row['tile_id'],'positive_queries':int(y.sum()),'known_queries':len(y)})
            if (index+1)%25==0:print(f'Frozen {split} evaluation: {index+1} tiles',flush=True)
        a={k:np.concatenate(v) for k,v in chunks.items()};save_npz(f'results/{split}_predictions.npz',**a)
        q={'VIIRS_reference_self_comparison':summarize(a['y'],a['y'],a['y'].astype(float)),
           'standard_MODIS':summarize(a['y'],a['modis'],a['modis'].astype(float)),
           'unsupervised_coarse':summarize(a['y'],a['coarse'],a['score']),
           'unsupervised_two_stage':summarize(a['y'],a['fine'])}
        old=read_shared('results/evaluation.json')[split]
        q['V1_coarse']=old['query']['unsupervised_coarse'];q['V1_two_stage']=old['query']['unsupervised_two_stage']
        coverage={'positive':float(a['valid'][a['y']].mean()),'negative':float(a['valid'][~a['y']].mean()),
                  'eligible_positives':int((a['valid']&a['y']).sum()),'positive_count':int(a['y'].sum()),
                  'positive_selected_band_counts':{str(b):int((a['y']&(a['selected_band']==b)).sum()) for b in (0,21,22)}}
        report[split]={'query':q,'native_cell':native_summary(a['y'],a['modis'],a['coarse'],a['score'],a['cell']),
                       'coverage':coverage,
                       'known_query_model_coverage':float(a['valid'].mean()),'all_native_alerts':native_alerts,
                       'alerts_with_supported_localization':localized,'tile_metrics':tile_metrics,
                       'localization_removed_true_positive_queries':int((a['y']&a['coarse']&~a['fine']).sum()),
                       'localization_removed_false_positive_queries':int((~a['y']&a['coarse']&~a['fine']).sum())}
    write(dest,report);return report
