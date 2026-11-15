"""Durable comparison with the preserved V1 and MODIS, including class-specific coverage."""
import numpy as np
from common import path,read,config

LABELS={'VIIRS_reference_self_comparison':'VIIRS reference (self-comparison)','standard_MODIS':'Standard MODIS',
        'V1_coarse':'Original V1, coarse','V1_two_stage':'Original V1, two-stage',
        'unsupervised_coarse':'SaturationAwareBackgroundV2, coarse','unsupervised_two_stage':'SaturationAwareBackgroundV2, two-stage'}

def table(values):
    rows=['| Method | Precision | Recall | TP | FP | FN | F1 | AP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in values.items():
        ap='—' if m['average_precision'] is None else f'{m["average_precision"]:.4f}'
        rows.append(f'| {LABELS[name]} | {100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {m["TP"]} | {m["FP"]} | {m["FN"]} | {m["f1"]:.4f} | {ap} |')
    return rows

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read('results/evaluation.json');features=read('data/features/receipt.json');partition=read('data/partition.json')
    order=['VIIRS_reference_self_comparison','standard_MODIS','V1_coarse','V1_two_stage','unsupervised_coarse','unsupervised_two_stage']
    t=r['test'];coverage=t['coverage']
    lines=['# SaturationAwareBackgroundV2: corrected unsupervised experiment','',
           '**The prior test cohort has been examined and its saturation failures informed this repair. This is development evidence, not a new blind test.**','',
           'Both networks are trained from scratch. No VIIRS labels or MODIS fire flags enter fitting, normalization, checkpoint selection or threshold calibration. '
           'The original model, weights, outputs and reports are preserved.','',
           '## Same query-center benchmark','',
           'All methods below use the original 585,464 eligible reference locations and 95 positive labels. Model-invalid queries stay in the denominator as undetected. '
           'VIIRS defines the labels, so its perfect row is a self-comparison, not real-world sensor accuracy.','']
    lines+=table({k:t['query'][k] for k in order})
    lines+=['','## Input coverage','',
            f'The corrected model can score **{coverage["eligible_positives"]}/{coverage["positive_count"]} positive queries '
            f'({100*coverage["positive"]:.2f}%)**, versus 20/95 for V1. Negative-query coverage is '
            f'{100*coverage["negative"]:.2f}%. Positive query channel selection: {coverage["positive_selected_band_counts"]}. '
            'Coverage is not detection: the anomaly score must still exceed its frozen threshold.','',
            '## Native MODIS-cell benchmark','',
            'Reference any-fire labels and maximum model score use the same observed query centers assigned to each native cell. '
            'This approximates common support, not exact physical footprints. Partial observation does not prove a whole cell is fire-free. '
            'The two-stage branch preserves the parent native alert; only within-cell query localization can change.','']
    lines+=table(t['native_cell'])
    lines+=['','## What changed','',
            '- Predict band-21 and band-22 radiances with separate outputs and normalize each from its own observed training values. '
            'Mask unavailable targets from both loss numerator and denominator. Prefer the standardized band-22 anomaly when available, otherwise use band 21. '
            'No simultaneous two-band anomaly is required. No numeric radiance is invented for saturation or fill codes.',
            '- Continue to fit only disjoint halo-16 native interiors. Predict across cells with complete radius-10 background context, '
            'which includes native cells covering the inherited continuous-query halo boundary. This avoids dropping query locations merely due to nearest-cell rounding.',
            '- Deterministically hold out complete training swaths within region/day-night groups, retaining at least one fitting swath in each group. '
            'Add the inherited validation swath to the unlabeled development set. These swaths are excluded from training statistics and optimizer updates. '
            'This repairs missing nighttime calibration but changes the training/development composition, so V1/V2 differences are not a single-factor ablation.',
            '- Keep the 0.1% coarse and 1% localization development alert-volume fractions fixed. No reference-guided threshold search is performed. '
            'Day and night each require at least 1,000 development samples per target channel; no daytime fallback is used for night.',
            '- Retain the experimental daytime 500 m localization rule: require four valid shortwave samples and strong residual evidence, '
            'then keep the highest two child scores including ties; preserve the unresolved full cell otherwise. Relative weights are not calibrated fire probabilities.','',
            '## Fitting and calibration','',
            f'Whole-swath counts by role: {partition["swath_counts"]}. Four existing training swaths are held out when the configured stratum sizes permit it.','',
            '| Stage | Training samples | Development samples | Epochs completed | Selected epoch | Parameters |',
            '|---|---:|---:|---:|---:|---:|']
    for stage,m in r['training'].items():
        lines.append(f'| {stage} | {m["training_samples"]:,} | {m["validation_samples"]:,} | {m["epochs_completed"]} | {m["selected_epoch"]} | {m["parameters"]:,} |')
    night=r['policy']['stages']['coarse']['night']['calibration_samples']
    lines+=['',f'Nighttime calibration now has **{night:,} unlabeled native-cell samples**. Checkpoints minimize observed-target SmoothL1 reconstruction loss. '
            'Anomaly thresholds use the held-out MODIS score distribution. Alert fractions are not false-positive guarantees; anomaly scores are not fire probabilities.','',
            '## Localization effect','',
            f'On the query benchmark, localization removed {t["localization_removed_false_positive_queries"]} false positives and '
            f'{t["localization_removed_true_positive_queries"]} true positives. Across full test tile outputs, '
            f'{t["alerts_with_supported_localization"]} of {t["all_native_alerts"]} native alerts had supported localization.','',
            '## Inherited validation-cohort comparison','',
            'This is only the original validation cohort, not all newly held-out development swaths. Evaluation labels are opened after the models and policies are frozen.','']
    lines+=table({k:r['validation']['query'][k] for k in order})
    lines+=['','## Interpretation limits and research','',
            'The experiment uses the existing selected paired-scene corpus, not a global random MODIS sample. Repeatedly observed regional events '
            'and nearby query labels are correlated. Reconstruction loss is not a direct objective for fire discrimination; anomalies can be clouds, reflections or hot ground. '
            'A successful coverage repair does not by itself establish improvement over MODIS. Independent fire events remain necessary.','',
            '[NASA MODIS calibration ATBD](https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/MODIS_L1B_ATBD_ver8_finalupload.pdf) '
            'describes low-gain band 21 for fire measurements. '
            '[NASA file specification](https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/5/MOD021KM.fs) documents saturation code 65533. '
            '[USGS MODIS 500 m localization prototype](https://www.usgs.gov/publications/prototype-downscaling-algorithm-modis-satellite-1-km-daytime-active-fire-detections) '
            'motivates shortwave localization; the residual rule here is an experimental adaptation.','',
            '![Training and comparison](comparison.png)','']
    path('results/EVALUATION.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for axis,stage in zip(axes[0],('coarse','fine')):
        h=read(f'results/{stage}_history.json');ep=[v['epoch'] for v in h]
        axis.plot(ep,[v['training_reconstruction_loss'] for v in h],label='Training')
        axis.plot(ep,[v['validation_reconstruction_loss'] for v in h],label='Unlabeled development')
        axis.axvline(r['training'][stage]['selected_epoch'],ls='--',color='gray');axis.legend()
        axis.set(title=stage.capitalize()+' reconstruction',xlabel='Epoch',ylabel='Observed-target SmoothL1')
    keys=['standard_MODIS','V1_coarse','unsupervised_coarse','unsupervised_two_stage'];labels=['MODIS','Original V1','V2 coarse','V2 two-stage']
    x=np.arange(len(keys));vals=[t['query'][k] for k in keys]
    for shift,key,label in [(-.18,'precision','Precision'),(.18,'recall','Recall')]:
        bars=axes[1,0].bar(x+shift,[100*v[key] for v in vals],width=.36,label=label);axes[1,0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    axes[1,0].set_xticks(x,labels);axes[1,0].set(ylim=(0,110),title='Reused query benchmark',ylabel='Percent');axes[1,0].legend()
    bars=axes[1,1].bar(labels,[v['FP'] for v in vals],color=['#64748b','#94a3b8','#2563eb','#d97706'])
    axes[1,1].bar_label(bars,padding=3);axes[1,1].set(title='False positives against VIIRS labels',ylabel='Query centers')
    axes[1,1].set_ylim(0,max(1,max(v['FP'] for v in vals))*1.2)
    fig.suptitle('SaturationAwareBackgroundV2 • Frozen unlabeled policies • Previously examined benchmark')
    fig.savefig(path('results/comparison.png'),dpi=180);plt.close(fig)
    return str(path('results/EVALUATION.md'))
