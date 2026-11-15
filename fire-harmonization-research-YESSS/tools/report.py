"""Report both detection scales and the effect of localization, without choosing a winner."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fireyes.common import read,path

def table(values):
    rows=['| Method | Precision | Recall | TP | FP | FN | F1 | AP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in values.items():
        ap='—' if m['average_precision'] is None else f'{m["average_precision"]:.4f}'
        rows.append(f'| {name} | {100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {m["TP"]:,} | {m["FP"]:,} | {m["FN"]:,} | {m["f1"]:.4f} | {ap} |')
    return rows

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=read('results/evaluation.json');features=read('data/features/receipt.json');p=report['policy']
    text=['# Unsupervised MODIS background learning and two-stage detection','',
          '**Completed fitting without VIIRS or MODIS fire labels. The benchmark was examined in earlier experiments; this is not a new blind test.**','',
          '**This experiment does not improve on standard MODIS.** On the test benchmark, the native anomaly detector recovers '
          f'{report["test"]["query"]["unsupervised_coarse"]["TP"]} of 95 positive query labels, and localization retains '
          f'{report["test"]["query"]["unsupervised_two_stage"]["TP"]}; standard MODIS recovers 76. '
          'A reduction in alert count here is accompanied by a severe loss of recall.','',
          'Two small networks predict withheld MODIS measurements from background context. Positive reconstruction residuals identify thermal anomalies. '
          'The first detects native thermal-cell anomalies. The second narrows supported daytime detections to 500 m samples using band 7 evidence. '
          'These are anomaly scores and relative localization weights, not calibrated fire probabilities.','',
          '## Training','',
          '| Stage | Training samples | Development samples | Parameters | Epochs completed | Selected epoch |',
          '|---|---:|---:|---:|---:|---:|']
    for s,t in report['training'].items():text.append(f'| {s} | {t["training_samples"]:,} | {t["validation_samples"]:,} | {t["parameters"]:,} | {t["epochs_completed"]} | {t["selected_epoch"]} |')
    if path('results/implementation_audit.json').exists():
        text[4:4]=['**Audit correction:** the input requirements exclude 75 of 95 positive test query locations before scoring. '
                   'Seventy-four have detector saturation in band 20 and/or 22; the detector fails to use the available band-21 measurement. '
                   'Its maximum possible recall under these requirements is 21.05%. The reported scores describe a flawed implementation '
                   'and cannot establish that unsupervised learning generally fails. '
                   'See [the implementation audit](IMPLEMENTATION_AUDIT.md).','']
    text+=['','Checkpoint selection minimizes reconstruction loss on unlabeled development MODIS observations. '
           'Feature and target normalization use training measurements only. Neither checkpoint selection nor thresholds use the reference labels. '
           'The training corpus may contain fires: robust SmoothL1 regression limits the influence of large errors but does not guarantee a fire-free learned background.','',
           '## Frozen operating policy','',
           'The native detector flags approximately the top 0.1% of unlabeled development anomaly scores within day/night groups; ties are handled conservatively. '
           'The localization stage requires valid observations in all four child samples and a maximum shortwave residual above the unlabeled 99th percentile. '
           'It then retains the top two child scores, including ties. Without adequate evidence it retains the whole coarse cell and marks localization unresolved. '
           'Alert-volume fractions are engineering choices made before reference evaluation; they are not measured false-positive rates. '
           'A group with fewer than 1,000 development samples uses the pooled development distribution, explicitly recorded in policy.json.','',
           '**Nighttime limitation:** this run has zero eligible nighttime development samples. The nighttime threshold therefore '
           'uses the pooled daytime distribution; independent nighttime calibration has not been established. Training does include '
           'nighttime measurements. This limitation is distinct from the weaker daytime test result.','',
           '## Original query-center benchmark','',
           'Same known VIIRS query centers as earlier comparisons. Model-unusable locations remain in the denominator as undetected; they are not removed to improve scores. '
           'The fine branch produces 500 m localization, sampled at the query centers; it does not create 375 m thermal measurements.','']
    text+=table(report['test']['query'])
    text+=['','VIIRS is the reference: its perfect self-comparison does not establish true sensor accuracy. AP is shown for the coarse continuous anomaly score. '
           'The localization rule produces a gated binary mask, so no separate fine-stage AP or probability-calibration claim is made.','',
           '## Native MODIS-cell benchmark','',
           'A reference cell is positive if any eligible VIIRS center assigned to it is positive. This approximates native-cell support, not exact footprint overlap. '
           'Partially observed cells remain partially observed. Both stages share the same native-cell alert decision; localization only refines where within the cell an alert is placed.','']
    text+=table(report['test']['native_cell'])
    t=report['test']
    text+=['',f'On the query benchmark, localization removed **{t["localization_removed_false_positive_queries"]} false-positive queries** '
           f'and **{t["localization_removed_true_positive_queries"]} true-positive queries** from the coarse-only prediction. '
           f'Across all native test outputs, {t["alerts_with_supported_localization"]} of {t["all_native_alerts"]} alerts had supported localization. '
           f'Model-valid coverage of the known query benchmark was {100*t["known_query_model_coverage"]:.2f}%.','',
           '## Development-set evaluation, performed only after freezing','']
    text+=table(report['validation']['query'])
    text+=['','## Methods and limits','',
           '- Stage 1 predicts central log-radiances for bands 20 and 22 from annular neighborhood summaries, central longwave bands 31/32, and viewing/solar geometry. '
           'All thermal neighborhood summaries exclude the central 3x3 region. The central target-band measurements are absent from the predictor.',
           '- Stage 2 predicts daytime central band-7 log-reflectance from annular context, central visible bands, pooled 250 m bands and geometry. '
           'Its own central 3x3 band-7 region is excluded. Relative child weights sum to one within each native cell; they indicate concentration, not calibrated uncertainty.',
           '- Inputs are strict MODIS-only files. VIIRS references and MODIS fire masks reside in physically separate evaluation files. '
           'Test imagery is not used for feature normalization, fitting, early stopping, residual normalization or thresholds.',
           '- Copying the existing paired-scene dataset does not undo its upstream selection: it is a small curated corpus, with three training regional cohorts '
           'and one each for development and test. Nearby cells are correlated. The unsupervised claim applies to this experiment\'s parameter fitting and policy selection, '
           'not to an uncurated global sampling process.',
           '- Anomalies may be hot ground, reflections, cloud boundaries or calibration artifacts; fire can also be reconstructed as normal. '
           'No performance improvement is assumed, and no reference-guided threshold tuning is performed.','',
           '## Verification and numerical correction','',
           'Twelve focused tests and final end-to-end verification passed. The first verification detected a float32/float64 tie-comparison '
           'mismatch in unlabeled threshold calibration. Residual-score arithmetic was made consistently float64, and the unchanged '
           'unlabeled policy was recomputed. Model weights were preserved. Original code and preliminary results are retained under '
           '`results/pre_precision_fix`, with a repair receipt in `results/precision_repair.json`. The correction did not change '
           'test confusion counts. It used no reference labels for fitting or policy selection.','',
           '## Research basis','',
           '[Masked self-supervised satellite learning (SatMAE)](https://sustainlab-group.github.io/SatMAE/) motivates learning from withheld measurements. '
           'This implementation uses compact blind-spot regressors, not the SatMAE architecture.',
           '[Thermal autoencoder anomaly-detection feasibility study](https://www.mdpi.com/2313-433X/12/6/229) motivates background reconstruction. '
           'That work uses a different sensor and reference-guided development; its reported performance does not transfer here.',
           '[USGS daytime MODIS 500 m fire-location prototype](https://www.usgs.gov/publications/prototype-downscaling-algorithm-modis-satellite-1-km-daytime-active-fire-detections) '
           'supports the physical rationale for using band-7 shortwave evidence within a coarse alert. The present learned residual rule is an experimental adaptation.','',
           '![Training and comparison](training_and_comparison.png)','']
    path('results/EVALUATION.md').write_text('\n'.join(text),encoding='utf-8')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for axis,stage in zip(axes[0],('coarse','fine')):
        h=read(f'results/{stage}_history.json');epochs=[v['epoch'] for v in h]
        axis.plot(epochs,[v['training_reconstruction_loss'] for v in h],label='Training')
        axis.plot(epochs,[v['validation_reconstruction_loss'] for v in h],label='Unlabeled development')
        axis.axvline(report['training'][stage]['selected_epoch'],ls='--',color='gray')
        axis.set(title=stage.capitalize()+' reconstruction',xlabel='Epoch',ylabel='SmoothL1 loss');axis.legend()
    names=['standard_MODIS','unsupervised_coarse','unsupervised_two_stage'];labels=['MODIS','Coarse anomaly','Two-stage anomaly']
    values=[t['query'][k] for k in names];x=np.arange(3)
    for shift,key,label in [(-.18,'precision','Precision'),(.18,'recall','Recall')]:
        bars=axes[1,0].bar(x+shift,[100*v[key] for v in values],width=.36,label=label);axes[1,0].bar_label(bars,fmt='%.1f',padding=3)
    axes[1,0].set_xticks(x,labels);axes[1,0].set(ylim=(0,110),ylabel='Percent',title='Reused query benchmark');axes[1,0].legend()
    bars=axes[1,1].bar(labels,[v['FP'] for v in values],color=['#64748b','#2563eb','#d97706'])
    axes[1,1].bar_label(bars,padding=3);axes[1,1].set(title='Reference false positives',ylabel='Query centers')
    axes[1,1].set_ylim(0,max(1,max(v['FP'] for v in values))*1.2)
    fig.suptitle('Unsupervised two-stage MODIS experiment • No fire-label fitting • Previously examined benchmark')
    fig.savefig(path('results/training_and_comparison.png'),dpi=180);plt.close(fig)
    return str(path('results/EVALUATION.md'))
