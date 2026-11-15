"""Reproducible scientific report and figures; no training decisions here."""
import json
import numpy as np
from common import ROOT,path,read,load

LABELS={'VIIRS_reference_self_comparison':'VIIRS reference (self-comparison)',
        'standard_MODIS':'Standard MODIS','previous_MLP_frozen_policy':'Previous contextual MLP',
        'hybrid_raw_0.5':'Hybrid at raw 0.5','hybrid_frozen_policy':'Hybrid, frozen policy',
        'base_MLP_cell_policy':'MLP, native-cell policy','hybrid_cell_policy':'Hybrid, native-cell policy'}

def table(values):
    rows=['| Method | Precision | Recall | TP | FP | FN | F1 | AP |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in values.items():
        rows.append(f'| {LABELS[name]} | {m["precision"]*100:.2f}% | {m["recall"]*100:.2f}% | '
                    f'{m["true_positives"]:,.0f} | {m["false_positives"]:,.0f} | {m["false_negatives"]:,.0f} | '
                    f'{m["f1"]:.4f} | {m["average_precision_exact"]:.4f} |')
    return rows

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from firepls.prepare import load_split
    r=read('results/evaluation.json');history=read('results/training_history.json')['validation']['average_precision']
    m=r['test']['query']['hybrid_frozen_policy'];b=r['test']['query']['standard_MODIS']
    text=['# Contextual neural / residual-tree hybrid','',
          '**Completed local training. Previously examined benchmark; not a new blind test.**','',
          'The hybrid adds a trained tree correction to the existing frozen neural network logit. '
          'It uses MODIS observations and MODIS fire detections at inference; VIIRS provides training/evaluation labels only.','',
          f'Trained {r["training"]["trained_iterations"]} boosting rounds, selected round '
          f'{r["training"]["selected_iteration"]} by validation average precision. '
          f'All {r["training"]["training_queries"]:,} known training queries and '
          f'{r["training"]["positive_training_queries"]:,} positive labels were retained.','',
          f'On this benchmark, the hybrid changes precision from {100*b["precision"]:.2f}% to {100*m["precision"]:.2f}%, '
          f'recall from {100*b["recall"]:.2f}% to {100*m["recall"]:.2f}%, '
          f'false positives from {b["false_positives"]:.0f} to {m["false_positives"]:.0f}, '
          f'and F1 from {b["f1"]:.4f} to {m["f1"]:.4f}.','',
          '## Original query-center comparison','',
          'Same 585,464 known query centers and 95 positive VIIRS reference labels as the previous experiment. '
          'Model thresholds are frozen using validation false-positive budgets before computing this run\'s test predictions.','']
    text+=table(r['test']['query'])
    text+=['','VIIRS is perfect here only because it defines the labels. This is not measured VIIRS accuracy '
           'against real fires, and the 95 labels are not 95 independent fire incidents.','',
           '## Common native-cell comparison','',
           'Each observed native MODIS cell has a positive reference if any eligible VIIRS center assigned '
           'to it is positive. The model score is the maximum among the same eligible centers. '
           'Thresholds for this scale were fitted separately on validation against the native-cell MODIS FP budget. '
           'These are distinct any-fire-in-cell metrics, not improved fine-resolution accuracy. '
           'Partially observed cells remain partially observed; an all-negative reference does not establish that '
           'the entire physical cell is fire free. Assignment approximates common support, not exact footprint/PSF overlap.','']
    text+=table(r['test']['native_cell'])
    text+=['','## Preprocessing and architecture','',
           '1. Keep the inherited calibrated MODIS radiances, native geolocation inversion, 15-minute maximum '
           'pair interval bound, and QA-based unknown masks. The 20 m inverse residual is numerical consistency, '
           'not absolute geolocation accuracy.',
           '2. Pool valid 250 m and 500 m reflective samples over their aligned 4x4 and 2x2 blocks to match '
           'the native thermal grid. Add seven pooled means, seven local-minus-pooled contrasts, and seven valid fractions. '
           'Retain the original 96 features; invalid values never contribute to means.',
           '3. Add the operational MODIS detection flag. This is an available input from MODIS, not a VIIRS label.',
           '4. Freeze the prior 96→64→32→1 MLP. Train up to 300 shallow logistic boosting rounds with '
           'its raw score as initial score, learning rate 0.03, 15 leaves, depth at most 4, L2=5, '
           'maximum leaf update 1, and validation-AP patience 40.',
           '5. Deploy z = z_MLP + z_residual. The tree export alone is incomplete; the supplied inference '
           'command always adds both. Fit a scalar probability offset and detection cutoff on validation only.','',
           'This is sequential additive fitting: both stages use the training labels, as boosting normally does. '
           'It is not out-of-fold stacking. The old MLP checkpoint was already selected using this validation cohort; '
           'the new early stopping and calibration reuse it. That increases development-set dependence.','',
           '## Validation comparison','']
    text+=table(r['validation']['query'])
    text+=['','## What the research supports','',
           '- [Maeda and Tonooka: contextual MOD14 features plus a learned tree classifier](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823964/) '
           'supports using physical background contrasts as model inputs. Its Himawari study is not evidence of this MODIS hybrid\'s accuracy.',
           '- [Friedman: gradient boosting](https://boosting.org/papers/Fri99.pdf) supplies the general method of learning additive '
           'corrections under classification loss. The particular MLP-plus-MODIS residual architecture here is our adaptation.',
           '- [Schroeder et al.: satellite active-fire validation](https://www.isprs.org/proceedings/2011/isrse-34/211104015Final00178.pdf) '
           'motivates near-coincident observations and comparison at the coarse sensor\'s support.',
           '- [NASA VIIRS active-fire description](https://viirsland.gsfc.nasa.gov/Products/NASA/FireESDR.html) explains '
           'the greater small-fire sensitivity of 375 m VIIRS. Resampling MODIS cannot manufacture that thermal information.',
           '- [LightGBM Dataset API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Dataset.html) '
           'documents initial scores; deployment and validation consistency are also checked numerically in this experiment.','',
           '## Limits and interpretation','',
           'No new satellite data or packages were downloaded. The model is an active-fire reference detector, not '
           'a future-fire forecast. The copied corpus has three training regional cohorts and only one cohort each '
           'for validation and testing. Correlated query centers cannot support an independent-trial confidence claim. '
           'The same test region has influenced earlier architecture discussions, so a fresh held-out event cohort '
           'is necessary before claiming general improvement. This run does not isolate each new feature\'s causal '
           'contribution; the unchanged MLP is the zero-residual comparator. No threshold was retuned after this test.','',
           '![Training and query comparison](comparison.png)','',
           '![Spatial comparison](spatial_comparison.png)','']
    path('results/EVALUATION.md').write_text('\n'.join(text),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    ax[0].plot(np.arange(len(history))+1,history,color='#2563eb')
    ax[0].axvline(r['training']['selected_iteration'],color='#d97706',ls='--',label='Selected round')
    ax[0].set(title='Residual training',xlabel='Boosting round',ylabel='Validation average precision');ax[0].legend()
    keys=['standard_MODIS','previous_MLP_frozen_policy','hybrid_frozen_policy'];labels=['MODIS','Previous MLP','Hybrid']
    vals=[r['test']['query'][k] for k in keys];x=np.arange(3)
    for shift,key,label,color in [(-.18,'precision','Precision','#2563eb'),(.18,'recall','Recall','#d97706')]:
        bars=ax[1].bar(x+shift,[100*v[key] for v in vals],width=.36,label=label,color=color)
        ax[1].bar_label(bars,fmt='%.1f',padding=3)
    ax[1].set_xticks(x,labels);ax[1].set(ylim=(0,110),ylabel='Percent',title='Original query-center benchmark');ax[1].legend(ncol=2)
    bars=ax[2].bar(labels,[v['false_positives'] for v in vals],color=['#64748b','#60a5fa','#2563eb'])
    ax[2].bar_label(bars,fmt='%.0f',padding=3);ax[2].set(title='False positives against VIIRS',ylabel='Query centers')
    ax[2].set_ylim(0,max(v['false_positives'] for v in vals)*1.25)
    fig.suptitle('Frozen validation policies • Reused test cohort • 95 VIIRS-positive query labels')
    fig.savefig(path('results/comparison.png'),dpi=180);plt.close(fig)
    raw=load_split('test');ledger=json.loads((ROOT/'data/features/tiles.json').read_text())
    tile=max((t for t in ledger if t['split']=='test'),key=lambda t:raw['y'][t['start']:t['end']].sum())
    s=slice(tile['start'],tile['end']);flat=raw['flat_index'][s];y=raw['y'][s]>.5
    pred=load('results/test_logits.npy')[s]>=r['operating_policy']['query']['cutoff']
    displays=[y,raw['modis'][s]>.5,pred]
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for axis,v,title in zip(axes,displays,['VIIRS reference','MODIS','Hybrid']):
        grid=np.full(tile['query_shape'],np.nan);grid.ravel()[flat]=v
        axis.imshow(grid,cmap=ListedColormap(['#e2e8f0','#dc2626']),vmin=0,vmax=1,interpolation='nearest')
        axis.set(title=title,xlabel='VIIRS query column',ylabel='VIIRS query row')
    fig.suptitle('Test tile with most known positive labels (display choice only)\nRed: detection/reference positive; gray: clear; white: excluded or unknown')
    fig.savefig(path('results/spatial_comparison.png'),dpi=180);plt.close(fig)
    return {'report':str(path('results/EVALUATION.md'))}
