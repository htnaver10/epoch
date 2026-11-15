"""Rebuild local scientific comparisons from immutable, saved predictions.

No fitting, threshold search, inference, imagery download or source mutation.
Run with fire-harmonization-research-fullres/.venv/Scripts/python.exe.
"""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PLS=ROOT/'fire-harmonization-PLSWORK'
YES=ROOT/'fire-harmonization-research-YESSS'
V2=YES/'saturation_aware_v2'
SOURCES={}
NAMES={'viirs':'VIIRS reference','modis':'Standard MODIS','mlp':'PLSWORK contextual MLP',
       'tree':'PLSWORK tree baseline','hybrid':'PLSWORK neural/tree hybrid',
       'v2':'SaturationAwareBackgroundV2','v2_localized':'V2 with localization'}
SHORT={'viirs':'VIIRS*','modis':'MODIS','mlp':'PLS MLP','tree':'PLS tree','hybrid':'PLS hybrid',
       'v2':'V2 coarse','v2_localized':'V2 localized'}
COLORS={'viirs':'#334155','modis':'#64748b','mlp':'#2563eb','tree':'#7c3aed','hybrid':'#0891b2',
        'v2':'#d97706','v2_localized':'#be185d'}
METHODS=['modis','mlp','tree','hybrid','v2','v2_localized']
FOOT='Same 585,464 known query centers; 95 VIIRS-positive labels. Previously examined benchmark, not a new blind test.'
FILES=[]

def register(p):
    p=Path(p)
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    SOURCES[p.relative_to(ROOT).as_posix()]=d
    return p

def read(p):return json.loads(register(p).read_text(encoding='utf-8'))
def array(p):return np.load(register(p),allow_pickle=False)

def curve(score,y):
    # Group tied scores before counting. No threshold is selected from this curve.
    order=np.argsort(-score,kind='stable');s=score[order];yy=y[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(yy,dtype=np.int64)[ends];total=ends+1
    recall=tp/y.sum();precision=tp/total
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    return recall,precision,ap

def metric(y,pred,score=None):
    # Independent from the training/evaluation helpers.
    tn,fp,fn,tp=np.bincount(2*y.astype(np.int64)+pred.astype(np.int64),minlength=4).tolist()
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'F1':2*tp/max(2*tp+fp+fn,1),'AP':None if score is None else curve(score,y)[2]}

def compare_metrics(actual,saved,style):
    for a,b in [('TP','true_positives'),('FP','false_positives'),('FN','false_negatives'),('TN','true_negatives')]:
        key=b if style=='pls' else a
        assert actual[a]==saved[key],(a,actual[a],saved[key])
    for a,b in [('precision','precision'),('recall','recall'),('F1','f1')]:np.testing.assert_allclose(actual[a],saved[b],rtol=1e-12)
    apkey='average_precision_exact' if style=='pls' else 'average_precision'
    if actual['AP'] is not None:np.testing.assert_allclose(actual['AP'],saved[apkey],rtol=1e-10,atol=1e-12)

def load_evidence():
    saved_pls=read(PLS/'results/evaluation.json');pls_policy=read(PLS/'results/operating_policy.json')
    saved_hybrid=read(PLS/'hybrid_v1/results/evaluation.json');hybrid_policy=read(PLS/'hybrid_v1/results/operating_policy.json')
    saved_v2=read(V2/'results/evaluation.json');v2_policy=read(V2/'models/policy.json')
    v=array(V2/'results/test_predictions.npz');m=array(PLS/'results/test_mlp_predictions.npz');t=array(PLS/'results/test_tree_predictions.npz')
    assert saved_v2['policy_sha256']==hashlib.sha256((V2/'models/policy.json').read_bytes()).hexdigest()
    y=v['y'];baseline=v['modis'];n=len(y)
    np.testing.assert_array_equal(m['target'],y);np.testing.assert_array_equal(t['target'],y)
    np.testing.assert_array_equal(m['modis']>.5,baseline);np.testing.assert_array_equal(t['modis']>.5,baseline)
    np.testing.assert_array_equal(m['weight'],np.ones(n));np.testing.assert_array_equal(t['weight'],np.ones(n))
    np.testing.assert_array_equal(t['tile_index'],m['tile_index']);np.testing.assert_array_equal(t['flat_index'],m['flat_index'])
    np.testing.assert_array_equal(array(PLS/'hybrid_v1/data/test_cell.npy'),v['cell'])
    assert n==585464 and int(y.sum())==95
    scores={'viirs':y.astype(float),'modis':baseline.astype(float),'mlp':m['logits'],
            'tree':t['logits'],'hybrid':array(PLS/'hybrid_v1/results/test_logits.npy'),'v2':v['score'],'v2_localized':None}
    predictions={'viirs':y.copy(),'modis':baseline,
                 'mlp':scores['mlp']>=pls_policy['policies']['mlp']['raw_logit_cutoff'],
                 'tree':scores['tree']>=pls_policy['policies']['tree']['raw_logit_cutoff'],
                 'hybrid':scores['hybrid']>=hybrid_policy['query']['cutoff'],
                 'v2':v['coarse'],'v2_localized':v['fine']}
    metrics={key:metric(y,predictions[key],score) for key,score in scores.items()}
    compare_metrics(metrics['modis'],saved_pls['test']['regular_modis'],'pls')
    compare_metrics(metrics['mlp'],saved_pls['test']['mlp_frozen_policy'],'pls')
    compare_metrics(metrics['tree'],saved_pls['test']['tree_frozen_policy'],'pls')
    compare_metrics(metrics['hybrid'],saved_hybrid['test']['query']['hybrid_frozen_policy'],'pls')
    compare_metrics(metrics['v2'],saved_v2['test']['query']['unsupervised_coarse'],'v2')
    compare_metrics(metrics['v2_localized'],saved_v2['test']['query']['unsupervised_two_stage'],'v2')
    # Verify checkpoint provenance without loading the training runtime or altering a source file.
    for name,file in [('mlp',PLS/'models/mlp/inference.pt'),('tree',PLS/'models/tree/model.txt')]:
        assert hashlib.sha256(register(file).read_bytes()).hexdigest()==pls_policy['model_sha256'][name]
    for file in ['models/base_mlp.pt','models/residual.txt']:
        assert hashlib.sha256(register(PLS/'hybrid_v1'/file).read_bytes()).hexdigest()==hybrid_policy['artifact_sha256'][file]
    for file in ['models/coarse/inference.pt','models/fine/inference.pt']:
        assert hashlib.sha256(register(V2/file).read_bytes()).hexdigest()==v2_policy['artifacts'][file]
    refs=read(YES/'data/reference_manifest.json');modis_rows={r['tile_id']:r for r in read(YES/'data/modis_manifest.json')}
    ledger=read(PLS/'data/features/tiles.json');by_id={r['tile_id']:r for r in ledger}
    source_rows=[json.loads(line) for line in register(PLS/'data/manifests/source_v3.jsonl').read_text().splitlines() if line.strip()]
    source_index={r['tile_id']:i for i,r in enumerate(source_rows)}
    tiles=[];offset=0;swath=np.empty(n,dtype=np.int16);swath_labels=[]
    for row in refs:
        if row['split']!='test':continue
        item=by_id[row['tile_id']];start,end=item['start'],item['end'];assert start==offset
        z=array(YES/row['reference']);known=z['weight']>0;flat=np.flatnonzero(known)
        np.testing.assert_array_equal(flat,m['flat_index'][start:end])
        np.testing.assert_array_equal(np.full(len(flat),source_index[row['tile_id']]),m['tile_index'][start:end])
        np.testing.assert_array_equal(z['target'][known]>.5,y[start:end])
        np.testing.assert_array_equal(z['modis_baseline'][known]>.5,baseline[start:end])
        assert end-start==len(flat)
        src=modis_rows[row['tile_id']]['modis_source']
        if src not in swath_labels:swath_labels.append(src)
        swath[start:end]=swath_labels.index(src)
        tiles.append({'tile_id':row['tile_id'],'start':start,'end':end,'shape':list(known.shape),'flat':flat,
                      'positive_count':int(y[start:end].sum()),'source_input':modis_rows[row['tile_id']]['input']})
        offset=end
    assert offset==n and len(tiles)==84
    # Native-cell MLP uses the separately calibrated float32 prior from the hybrid experiment.
    cell=v['cell'];nc=int(cell.max())+1;cy=np.zeros(nc,bool);cb=cy.copy()
    np.maximum.at(cy,cell,y);np.maximum.at(cb,cell,baseline)
    native={'viirs':metric(cy,cy,cy.astype(float)),'modis':metric(cy,cb,cb.astype(float))}
    for key,score,p in [('mlp',array(PLS/'hybrid_v1/data/test_prior.npy'),hybrid_policy['base_mlp_cell']),
                        ('hybrid',scores['hybrid'],hybrid_policy['native_cell'])]:
        agg=np.full(nc,-np.inf);np.maximum.at(agg,cell,score)
        native[key]=metric(cy,agg>=p['cutoff'],agg)
    agg=np.full(nc,-np.inf);np.maximum.at(agg,cell,scores['v2'])
    cp=np.zeros(nc,bool);np.maximum.at(cp,cell,predictions['v2']);native['v2']=metric(cy,cp,agg)
    for key,oldkey in [('modis','standard_MODIS'),('mlp','base_MLP_cell_policy'),('hybrid','hybrid_cell_policy')]:
        compare_metrics(native[key],saved_hybrid['test']['native_cell'][oldkey],'pls')
    compare_metrics(native['v2'],saved_v2['test']['native_cell']['unsupervised_coarse'],'v2')
    assert int(cy.sum())==39 and nc==103069
    return {'y':y,'scores':scores,'predictions':predictions,'metrics':metrics,'native_metrics':native,
            'tiles':tiles,'swath':swath,'swath_sources':swath_labels,'policies':{'pls':pls_policy,'hybrid':hybrid_policy,'v2':v2_policy}}

def export(fig,name,title,caption):
    dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest.with_suffix('.png'),dpi=190,bbox_inches='tight')
    fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    FILES.append({'file':dest.with_suffix('.png').relative_to(OUT).as_posix(),'pdf':dest.with_suffix('.pdf').relative_to(OUT).as_posix(),
                  'title':title,'caption':caption})

def overview(e):
    m=e['metrics'];x=np.arange(len(METHODS));labels=[SHORT[k] for k in METHODS]
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for shift,key,label,color in [(-.24,'precision','Precision','#2563eb'),(0,'recall','Recall','#d97706'),(.24,'F1','F1','#0f766e')]:
        bars=axes[0].bar(x+shift,[100*m[k][key] for k in METHODS],width=.24,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    axes[0].set_xticks(x,labels,rotation=20,ha='right');axes[0].set(ylim=(0,110),ylabel='Percent / F1 × 100',title='Saved operating policies');axes[0].legend(ncol=3,loc='upper left')
    width=.24
    for shift,key,label,color in [(-width,'TP','Matched positives','#059669'),(0,'FP','False positives','#be185d'),(width,'FN','Missed positives','#d97706')]:
        bars=axes[1].bar(x+shift,[m[k][key] for k in METHODS],width=width,label=label,color=color)
        axes[1].bar_label(bars,fontsize=7,padding=2)
    axes[1].set_xticks(x,labels,rotation=20,ha='right');axes[1].set(ylabel='Query centers',ylim=(0,170),title='Matched, false and missed detections');axes[1].legend(ncol=2,loc='upper left')
    fig.suptitle('PLSWORK versus SaturationAwareBackgroundV2\n'+FOOT,fontsize=12)
    export(fig,'comparison/01_overview','Overall comparison','Each model uses its existing frozen policy. Policies were selected differently; these are not equal-alert-budget comparisons. VIIRS defines the reference labels.')

def precision_recall(e):
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained');y=e['y'];m=e['metrics']
    for key in ['mlp','tree','hybrid','v2']:
        recall,precision,ap=curve(e['scores'][key],y)
        # Preserve every positive group and its preceding negative endpoint.
        # Uniform sampling alone can skip nearly all fire detections in this imbalanced corpus.
        positive_groups=np.flatnonzero(np.diff(np.r_[0.,recall])>0)
        take=np.unique(np.r_[0,positive_groups,np.maximum(0,positive_groups-1),
                             np.arange(0,len(recall),max(1,len(recall)//1500)),len(recall)-1])
        assert set(positive_groups).issubset(set(take))
        for ax in axes:ax.plot(recall[take],precision[take],color=COLORS[key],lw=1.5,label=f'{SHORT[key]} (AP {ap:.3f})')
    for key in METHODS:
        for ax in axes:ax.scatter(m[key]['recall'],m[key]['precision'],s=60,color=COLORS[key],edgecolor='white',zorder=5,
                                 marker='s' if key in ('modis','v2_localized') else 'o')
    axes[0].set(xlim=(0,1.01),ylim=(0,1.02),title='Complete score ranking',xlabel='Recall',ylabel='Precision');axes[0].legend(fontsize=8,loc='upper right')
    axes[1].set(xlim=(.5,1),ylim=(0,.8),title='Higher-recall region',xlabel='Recall',ylabel='Precision')
    for key in METHODS:
        offsets={'modis':(6,-12),'mlp':(6,10),'tree':(6,-12),'hybrid':(6,-18),'v2':(6,8),'v2_localized':(6,-16)}
        axes[1].annotate(SHORT[key],(m[key]['recall'],m[key]['precision']),xytext=offsets[key],textcoords='offset points',fontsize=8)
    fig.suptitle('Precision–recall curves and saved operating points\n'+FOOT,fontsize=12)
    export(fig,'comparison/02_precision_recall','Precision–recall ranking','Dots/squares show frozen operating points. Curves are diagnostic sweeps of saved scores; no new threshold was selected. V2 localization has a binary policy, so it is shown as a point only. VIIRS self-comparison is omitted from ranking curves.')

def native_cells(e):
    m=e['native_metrics'];keys=['modis','mlp','hybrid','v2'];labels=[SHORT[k] for k in keys];x=np.arange(len(keys))
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for shift,key,label,c in [(-.24,'precision','Precision','#2563eb'),(0,'recall','Recall','#d97706'),(.24,'F1','F1','#0f766e')]:
        bars=axes[0].bar(x+shift,[100*m[k][key] for k in keys],width=.24,label=label,color=c);axes[0].bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    axes[0].set_xticks(x,labels);axes[0].set(ylim=(0,110),ylabel='Percent / F1 × 100',title='Native-cell policies');axes[0].legend(ncol=3)
    for shift,key,label,c in [(-.24,'TP','Matched','#059669'),(0,'FP','False','#be185d'),(.24,'FN','Missed','#d97706')]:
        bars=axes[1].bar(x+shift,[m[k][key] for k in keys],width=.24,label=label,color=c);axes[1].bar_label(bars,padding=3)
    axes[1].set_xticks(x,labels);axes[1].set(ylim=(0,43),ylabel='Observed native cells',title='39 reference-positive native cells');axes[1].legend(ncol=3)
    fig.suptitle('Fire presence within native MODIS cells\n103,069 observed cells. Approximate shared support; separate evaluation target.',fontsize=12)
    export(fig,'comparison/03_native_cells','Comparison at native MODIS-cell scale','PLS MLP and hybrid use separately frozen validation cell policies; V2 preserves its native alert policy. The MLP cell comparator uses CPU float32 scores; its original query comparator used BF16. No saved tree cell policy exists, so none was fitted for this chart. Partial observed support is not a full physical footprint.')

def spatial_setup(e):
    # Display-only scene selection by reference count, independent of model correctness.
    tile=max(e['tiles'],key=lambda t:t['positive_count']);s=slice(tile['start'],tile['end']);shape=tile['shape'];flat=tile['flat']
    include=e['y'][s].copy()
    for pred in e['predictions'].values():include|=pred[s]
    rr,cc=np.unravel_index(flat[include],shape)
    bounds=(max(0,int(rr.min())-8),min(shape[0],int(rr.max())+9),max(0,int(cc.min())-8),min(shape[1],int(cc.max())+9))
    maps={}
    for key in ['viirs',*METHODS]:
        grid=np.full(shape,np.nan);y=e['y'][s];pred=e['predictions'][key][s]
        grid.ravel()[flat]=np.where(y&pred,1,np.where(~y&pred,2,np.where(y&~pred,3,0)))
        maps[key]=grid
    return tile,bounds,maps

CMAP=ListedColormap(['#e2e8f0','#059669','#be185d','#d97706'])
CMAP.set_bad('#ffffff')
LEGEND=[Patch(facecolor=c,edgecolor='#94a3b8' if c=='#ffffff' else c,label=t) for c,t in [('#e2e8f0','Reference-clear, no alert'),('#059669','Matched positive'),('#be185d','False positive'),('#d97706','Missed positive'),('#ffffff','Unknown / excluded')]]

def show_map(ax,grid,bounds,title):
    r0,r1,c0,c1=bounds
    ax.imshow(grid[r0:r1,c0:c1],cmap=CMAP,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],CMAP.N),interpolation='nearest',extent=(c0,c1,r1,r0))
    ax.set(title=title,xlabel='VIIRS query column',ylabel='VIIRS query row')

def spatial(e,tile,bounds,maps):
    fig,axes=plt.subplots(2,4,figsize=(15,9),layout='constrained')
    for ax,key in zip(axes.ravel(),['viirs',*METHODS]):show_map(ax,maps[key],bounds,SHORT[key])
    axes.ravel()[-1].axis('off');axes.ravel()[-1].legend(handles=LEGEND,loc='center',frameon=False,fontsize=10)
    fig.suptitle(f'Shared spatial comparison — tile {tile["tile_id"]}\nChosen by most positive reference queries; identical crop for every model. VIIRS is a self-comparison.',fontsize=12)
    export(fig,'spatial/01_shared_scene','Shared scene: all models','This display uses the test tile with the most positive labels; it is not a typical-scene claim. The crop includes the union of reference positives and all model alerts in that tile, with an eight-query margin. Maps use query indices, not a claim of finer measured thermal resolution.')

def model_sheets(e,tile,bounds,maps):
    for key in ['viirs',*METHODS]:
        m=e['metrics'][key];fig,axes=plt.subplots(2,2,figsize=(11,8.5),layout='constrained')
        metric_keys=['precision','recall','F1']+(['AP'] if m['AP'] is not None else [])
        vals=[100*m[k] for k in metric_keys]
        bars=axes[0,0].bar(['Precision','Recall','F1']+(['AP'] if m['AP'] is not None else []),vals,color=COLORS[key]);axes[0,0].bar_label(bars,fmt='%.1f',padding=3)
        axes[0,0].set(ylim=(0,112),ylabel='Percent / score × 100',title='Full benchmark metrics')
        bars=axes[0,1].bar(['Matched (TP)','False (FP)','Missed (FN)'],[m[k] for k in ['TP','FP','FN']],color=['#059669','#be185d','#d97706'])
        axes[0,1].bar_label(bars,padding=3);axes[0,1].set(ylim=(0,max(1,m['TP'],m['FP'],m['FN'])*1.2),title='Full benchmark detection counts',ylabel='Query centers')
        show_map(axes[1,0],maps['viirs'],bounds,'VIIRS reference: matched color denotes label')
        show_map(axes[1,1],maps[key],bounds,SHORT[key]+': agreement and errors')
        fig.legend(handles=LEGEND,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
        extra=' Self-comparison; not independent accuracy.' if key=='viirs' else ''
        fig.suptitle(NAMES[key]+'\n'+FOOT+extra,fontsize=11)
        export(fig,f'models/{key}',NAMES[key],f'Top panels cover the complete benchmark; bottom panels show the same selected tile and crop ({tile["tile_id"]}).'+extra)

def by_overpass(e):
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');metrics={}
    for i,src in enumerate(e['swath_sources']):
        match=re.search(r'\.A(\d{7})\.(\d{4})\.',src);date=datetime.strptime(match[1]+match[2],'%Y%j%H%M').strftime('%Y-%m-%d %H:%M UTC')
        mask=e['swath']==i;values={k:metric(e['y'][mask],e['predictions'][k][mask]) for k in METHODS};metrics[date]=values
        x=np.arange(len(METHODS))
        for shift,name,label,color in [(-.24,'TP','Matched','#059669'),(0,'FP','False','#be185d'),(.24,'FN','Missed','#d97706')]:
            bars=axes[i].bar(x+shift,[values[k][name] for k in METHODS],width=.24,label=label,color=color);axes[i].bar_label(bars,fontsize=6,padding=2)
        axes[i].set_xticks(x,[SHORT[k] for k in METHODS],rotation=35,ha='right');axes[i].set(title=f'{date}\n{int(e["y"][mask].sum())} positive labels',ylabel='Query centers',ylim=(0,165))
    axes[0].legend(fontsize=8,ncol=3)
    fig.suptitle('Results by saved test overpass — fixed count scales\nThree overpasses from one regional cohort; these are not three independent fire events.',fontsize=12)
    export(fig,'comparison/04_by_overpass','Results by overpass','All benchmark locations are retained, including overpasses with few or no positive labels. Compare using the same count scale across panels.')
    return metrics

def training():
    mlp=read(PLS/'results/mlp_history.json');tree=read(PLS/'results/tree_history.json');hybrid=read(PLS/'hybrid_v1/results/training_history.json')
    coarse=read(V2/'results/coarse_history.json');fine=read(V2/'results/fine_history.json')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    axes[0,0].plot([r['epoch']+1 for r in mlp],[r['validation_average_precision'] for r in mlp],color=COLORS['mlp'])
    axes[0,0].set(title='PLS MLP',xlabel='Epoch',ylabel='Validation average precision')
    axes[0,1].plot(np.arange(len(tree['validation']['average_precision']))+1,tree['validation']['average_precision'],color=COLORS['tree'])
    axes[0,1].set(title='PLS tree',xlabel='Boosting round',ylabel='Validation average precision')
    axes[0,2].plot(np.arange(len(hybrid['validation']['average_precision']))+1,hybrid['validation']['average_precision'],color=COLORS['hybrid'])
    axes[0,2].set(title='PLS hybrid residual training',xlabel='Boosting round',ylabel='Validation average precision')
    for axis,rows,title in [(axes[1,0],coarse,'V2 coarse'),(axes[1,1],fine,'V2 localization')]:
        axis.plot([r['epoch'] for r in rows],[r['training_reconstruction_loss'] for r in rows],label='Training')
        axis.plot([r['epoch'] for r in rows],[r['validation_reconstruction_loss'] for r in rows],label='Unlabeled development')
        axis.set(title=title,xlabel='Epoch',ylabel='Observed-target reconstruction loss');axis.legend(fontsize=8)
    axes[1,2].axis('off');axes[1,2].text(0,.85,'Different learning objectives',weight='bold',fontsize=12)
    axes[1,2].text(0,.72,'PLSWORK selects checkpoints using\nVIIRS-label validation ranking.\n\nV2 selects checkpoints using unlabeled\nMODIS reconstruction error.\n\nLoss values are not comparable\nbetween these objectives.',va='top',fontsize=11)
    fig.suptitle('Saved training histories — separate objectives and scales',fontsize=13)
    export(fig,'training/01_learning_curves','Saved learning curves','These curves describe their original experiments; no model was retrained. Different objectives, development composition and precision prevent a direct training-loss comparison.')

def write_outputs(e,tile,bounds,overpass):
    metrics=e['metrics'];native=e['native_metrics']
    receipt={'passed':True,'benchmark_queries':len(e['y']),'positive_labels':int(e['y'].sum()),'native_cells':103069,'positive_native_cells':39,
             'all_query_labels_equal':True,'all_modis_baselines_equal':True,'tile_and_flat_indices_verified':True,
             'reference_masks_verified_across_84_tiles':True,'native_cell_ids_equal':True,'saved_operating_policies_used':True,
             'all_displayed_metrics_independently_recomputed':True,'model_hashes_match_policies':True,
             'no_training_or_threshold_changes':True,'new_blind_test':False,
             'display_tile':{k:v for k,v in tile.items() if k!='flat'},'display_crop_row0_row1_col0_col1':bounds,
             'spatial_selection':'Most positive reference labels; crop encloses reference/any-model alert union plus eight-query margin',
             'source_sha256':SOURCES,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result={'query':metrics,'native_cell':native,'by_overpass':overpass,'validation':receipt}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scale','method','TP','FP','FN','TN','precision','recall','F1','AP']);writer.writeheader()
        for scale,values in [('query',metrics),('native_cell',native)]:
            for k,m in values.items():writer.writerow({'scale':scale,'method':NAMES[k],**m})
    lines=['# PLSWORK versus SaturationAwareBackgroundV2','',
           '**The PLSWORK MLP has higher precision and F1; V2 coarse has higher recall and more false alarms.** '
           'The hybrid is close to the MLP. V2 localization reduces false alarms at a substantial cost in missed positive queries.','',
           'All rows use the same 585,464 known query centers, including 95 positive VIIRS labels. Saved policies are applied unchanged. '
           'This is a previously examined regional benchmark, not a new blind test.','',
           '| Method | Precision | Recall | TP | FP | FN | F1 | AP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in ['viirs',*METHODS]:
        m=metrics[key];ap='—' if m['AP'] is None else f'{m["AP"]:.4f}'
        lines.append(f'| {NAMES[key]} | {100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {m["TP"]} | {m["FP"]} | {m["FN"]} | {m["F1"]:.4f} | {ap} |')
    lines+=['','VIIRS defines the reference labels, so its perfect row is a self-comparison. It does not establish perfect real-world sensor accuracy. '
            'The 95 query labels are correlated and are not 95 independent wildfire incidents.','',
            f'Compared with the contextual MLP, V2 coarse detects {metrics["v2"]["TP"]-metrics["mlp"]["TP"]} more positive labels in total '
            f'and produces {metrics["v2"]["FP"]-metrics["mlp"]["FP"]} more false positives. '
            'Neither dominates the other across precision and recall. The MLP has the higher F1 at these saved operating points.','',
            '## How to read these figures','',
            '- `comparison/01_overview`: frozen-policy precision, recall, F1 and detection counts.',
            '- `comparison/02_precision_recall`: ranking curves and the saved operating points. Sweeps are diagnostic; no new threshold was chosen.',
            '- `comparison/03_native_cells`: approximate shared native-cell support with 39 positive cells. This is a distinct target. '
            'PLS methods use separately frozen native-cell policies; V2 retains its native detector policy.',
            '- `comparison/04_by_overpass`: all results separated into the three existing test overpasses, with the same count scale.',
            '- `models/`: separate sheets for the MLP, hybrid, tree, V2 coarse, V2 localization, MODIS and the VIIRS reference.',
            '- `spatial/01_shared_scene`: identical scene/crop and common error colors for all methods. White is unknown/excluded, not fire-free.',
            '- `training/01_learning_curves`: saved original training histories. Reconstruction losses and supervised ranking objectives are different.','',
            'Every figure has both a PNG and a PDF. Open `index.html` for the local gallery.','',
            '## Comparability and limitations','',
            'PLSWORK models used VIIRS supervision. Their operating policies use validation labels and a MODIS false-positive budget. '
            'V2 used self-supervised MODIS reconstruction with unlabeled day/night alert-volume thresholds, and held out additional training swaths. '
            'These are achieved operating points from different procedures, not equal-budget or causal architecture comparisons. '
            'The PLS query MLP used its saved BF16 scores; its separately calibrated native-cell comparator uses saved float32 prior scores. '
            'The hybrid includes the MODIS fire flag as an input. V2 and the contextual MLP do not.','',
            'Model scores have different meanings and units. Raw reconstruction anomalies must not be interpreted as fire probabilities. '
            'No Brier/calibration comparison or cross-model raw-score color scale is presented. '
            'V2 localization is a binary gated policy and therefore appears as an operating point rather than a separate ranking curve.','',
            'The selected spatial tile has the largest positive reference count; it is a display example rather than a typical or independently selected scene. '
            'The crop retains every alert from every displayed model within that tile. Full-benchmark statistics include all 84 tiles.','',
            'All model weights and source results were read without modification. Exact reference-mask, query-index, label, baseline and cell-ID checks passed. '
            'Metrics were independently recomputed and reconciled to the saved evaluations. `metrics.json` contains inputs, hashes and checks; '
            '`metrics.csv` contains the comparison values.','',
            '![Overall comparison](comparison/01_overview.png)','']
    (OUT/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
    links=[]
    for item in FILES:
        links.append(f'<article><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["caption"])}</p>'
                     f'<a href="{item["file"]}"><img loading="lazy" src="{item["file"]}" alt="{html.escape(item["title"])}"></a>'
                     f'<p><a href="{item["file"]}">PNG</a> · <a href="{item["pdf"]}">PDF</a></p></article>')
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Fire model comparisons</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#172033;background:#f1f5f9;margin:0}main{max-width:1150px;margin:auto;padding:32px}article{background:white;padding:24px;border-radius:12px;margin:24px 0}img{width:100%;height:auto}h1,h2{line-height:1.2}a{color:#1d4ed8}.note{border-left:4px solid #d97706;padding:12px 18px;background:#fffbeb}</style><main>'
        '<h1>PLSWORK and SaturationAwareBackgroundV2</h1><p><strong>PLSWORK MLP: higher precision and F1. V2 coarse: higher recall and more false alarms.</strong></p>'
        '<p class="note">Same 585,464 known query centers and 95 VIIRS-positive labels. Previously examined benchmark; no new training or threshold tuning. VIIRS defines the reference, so its perfect row is a self-comparison.</p>'
        '<p><a href="COMPARISON.md">Comparison and methods</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="metrics.json">Metrics and source verification</a></p>'
        +''.join(links)+'</main></html>',encoding='utf-8')
    (OUT/'README.md').write_text('# Fire-model visualization collection\n\nOpen [index.html](index.html) for the gallery or [COMPARISON.md](COMPARISON.md) for the results. '
        'All visualizations are exported as PNG and PDF. Source files and models remain unchanged.\n\nRebuild from the workspace root:\n\n```powershell\n'
        '& .\\fire-harmonization-research-fullres\\.venv\\Scripts\\python.exe .\\data-visualization\\build_visualizations.py\n```\n',encoding='utf-8')
    (OUT/'figure_manifest.json').write_text(json.dumps(FILES,indent=2)+'\n')
    # Validate every emitted resource and all source hashes again after rendering.
    for item in FILES:
        for key in ('file','pdf'):assert (OUT/item[key]).is_file() and (OUT/item[key]).stat().st_size>1000
    for rel,digest in SOURCES.items():
        with (ROOT/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
    check={'passed':True,'png_figures':len(FILES),'pdf_figures':len(FILES),'source_files_unchanged':len(SOURCES),
           'same_query_locations_verified':True,'metrics_reconciled':True,'native_cell_comparison_verified':True,
           'gallery_resources_exist':True,'new_blind_test':False,'no_model_or_policy_mutations':True}
    (OUT/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({'verification':check,'metrics':metrics},indent=2))

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    e=load_evidence();overview(e);precision_recall(e);native_cells(e)
    tile,bounds,maps=spatial_setup(e);spatial(e,tile,bounds,maps);model_sheets(e,tile,bounds,maps)
    overpass=by_overpass(e);training();write_outputs(e,tile,bounds,overpass)

if __name__=='__main__':main()
