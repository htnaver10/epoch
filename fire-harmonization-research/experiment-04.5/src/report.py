"""Audit frozen holdout outputs and make figures without changing prior experiments."""
import sys
sys.dont_write_bytecode=True
import csv,json,math,os
from datetime import date,timedelta
from pathlib import Path
import numpy as np
from evaluate import ROOT,RESEARCH,output,write_json,sha,AREA_WEIGHT,REGIONS,protected_files,MODEL_FILES

os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
MAIN='boosted_14year';RAW='raw_modis';SCENARIO='paired_active_dates'


def read(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)


def validate():
    rows=read('weekly_predictions.csv');metrics=read('metrics.csv')
    seen=set();refs={}
    for r in rows:
        key=tuple(r[k] for k in ('scenario','model','region','week'));assert key not in seen;seen.add(key)
        rk=tuple(r[k] for k in ('scenario','region','week'))
        value=[r['observed_viirs'],r['observed_modis'],r['valid_days']]
        if rk in refs:
            for a,b in zip(value,refs[rk]):close(a,b)
        refs[rk]=value
    for r in metrics:
        g=[x for x in rows if all(x[k]==r[k] for k in ('scenario','model','region'))]
        y=np.array([float(x['observed_viirs']) for x in g]);p=np.array([float(x['predicted_viirs']) for x in g])
        close(r['weekly_mae'],np.mean(np.abs(y-p)));close(r['weekly_rmse'],np.sqrt(np.mean((y-p)**2)))
        close(r['weekly_wape'],np.sum(np.abs(y-p))/y.sum());close(r['aggregate_bias'],(p.sum()-y.sum())/y.sum())
    binary=read('binary_metrics.csv')
    with np.load(ROOT/'results/prediction_grids_2026.npz') as z:
        static=z['static'];w=AREA_WEIGHT*(~static)
        for scenario in ('paired_active_dates','metadata_window'):
            mask=z[scenario];y=z['viirs'][mask].astype(float)
            for name in ('raw_modis','boosted_14year','boosted_11year','boosted_8year'):
                p=z[name][mask];metric=next(r for r in binary if r['scenario']==scenario and r['model']==name)
                close(metric['brier'],np.sum(w*(p-y)**2)/(w.sum()*mask.sum()))
                yes=p>=.5;tp=np.sum(w*yes*y);fp=np.sum(w*yes*(1-y));fn=np.sum(w*(~yes)*y)
                close(metric['precision'],tp/(tp+fp));close(metric['recall'],tp/(tp+fn));close(metric['f1'],2*tp/(2*tp+fp+fn))
                # Independent tie-group ranking formula, distinct from ROC integration in evaluate.py.
                pred=p[:,~static].ravel();target=y[:,~static].ravel();weights=np.tile(AREA_WEIGHT[~static],int(mask.sum()))
                _,idx=np.unique(pred,return_inverse=True)
                pos=np.bincount(idx,weights=weights*target);neg=np.bincount(idx,weights=weights*(1-target))
                close(metric['roc_auc'],np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum()))
                g=[r for r in rows if r['scenario']==scenario and r['model']==name and r['region']=='canada_rectangle']
                close(sum(float(r['predicted_viirs']) for r in g),np.sum(w*p))
                close(sum(float(r['observed_viirs']) for r in g),np.sum(w*y))
    audit=json.loads((ROOT/'results/frozen_models.json').read_text())
    for name,info in audit.items():
        assert max(info['training_years'])<info['test_year']==2026
        assert sha(Path(info['path']))==info['sha256']
    for path,h in json.loads((ROOT/'results/protocol_hashes.json').read_text()).items():assert sha(ROOT/path)==h
    for path,h in json.loads((ROOT/'results/input_hashes.json').read_text()).items():assert sha(Path(path))==h
    protected=json.loads((ROOT/'results/protected_hashes.json').read_text())
    assert protected_files()==protected['main']
    assert all(sha(RESEARCH/p)==h for p,h in protected['prior_research'].items())
    result={'passed':True,'test_year':2026,'every_test_year_outside_training':True,'frozen_models_unchanged':len(audit),
            'prediction_rows_checked':len(rows),'metric_groups_checked':len(metrics),'binary_metric_rows_checked':len(binary),
            'grid_to_table_totals_reconciled':True,'scenario_inputs_identical':True,'future_modis_invariance_asserted_in_run':True,
            'main_files_unchanged':len(protected['main']),'prior_research_files_unchanged':len(protected['prior_research'])}
    write_json('results/validation.json',result)
    return result


def heading(fig,title,subtitle):
    fig.get_layout_engine().set(rect=(0,.09,1,.85))
    fig.suptitle(title,x=.02,y=1.01,ha='left',fontsize=17,fontweight='bold')
    fig.text(.02,.96,subtitle,va='top',fontsize=10,color='#4B5563')
    fig.text(.02,.02,'NASA FIRMS-derived coarse archive · recorded VIIRS reference · partial 2026 coverage; clear-sky exposure unverified.',fontsize=8.7,color='#4B5563')


def export(fig,name):
    for ext in ('png','svg'):fig.savefig(output(f'visualizations/{name}.{ext}'),dpi=170,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)


def figures():
    metrics=read('metrics.csv');daily=read('daily_predictions.csv')
    names=[RAW,'power_14year_rate','boosted_8year','boosted_11year',MAIN]
    labels=['Raw MODIS','Power law · 14 years','Boosted · 8 years','Boosted · 11 years','Boosted · 14 years']
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
    for ax,scenario,title in zip(axes,('paired_active_dates','metadata_window'),('Both sensors globally active · 51 days','Inferred metadata window · 59 days')):
        vals=[float(next(r for r in metrics if r['scenario']==scenario and r['region']=='canada_rectangle' and r['model']==m)['weekly_mae']) for m in names]
        bars=ax.barh(range(len(names)),vals,color=['#2878B5','#8060A0','#79BCA2','#DB8F61','#D55E00'])
        ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.bar_label(bars,fmt='%.1f',padding=4)
        ax.set_xlim(0,max(vals)*1.2);ax.set_xlabel('Weekly MAE · weighted occupied cell-days');ax.set_title(title,loc='left',fontsize=11)
    heading(fig,'2026 outside-training evaluation: coverage sensitivity','Same frozen models and input features in both panels · nine partial calendar bins · lower error is better')
    export(fig,'01_holdout_error_and_coverage')

    fig,ax=plt.subplots(figsize=(13,5.5),layout='constrained')
    group=[r for r in daily if r['scenario']==SCENARIO and r['region']=='canada_rectangle']
    dates=[date.fromisoformat(r['date']) for r in group]
    # Reindex to the full candidate window so omitted dates appear as gaps, not connecting lines.
    all_dates=[date(2026,6,1)+timedelta(days=i) for i in range(61)]
    by_date={date.fromisoformat(r['date']):r for r in group}
    for key,label,color in [('observed_viirs','Observed VIIRS','#17212B'),(RAW,'Observed MODIS','#2878B5'),(MAIN,'Frozen 14-year model','#D55E00')]:
        ax.plot(all_dates,[float(by_date[d][key]) if d in by_date else np.nan for d in all_dates],color=color,label=label,linewidth=1.7)
    for d in all_dates:
        if d not in by_date:ax.axvspan(d-timedelta(hours=12),d+timedelta(hours=12),color='#E5E7EB',alpha=.6,zorder=-1)
    ax.set_ylabel('Weighted occupied cells per day');ax.set_ylim(bottom=0);ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.2);fig.autofmt_xdate()
    heading(fig,'Unseen-year daily activity: June–July 2026','Primary 51-day diagnostic · gaps/shading denote excluded dates · model fitted only through 2025')
    export(fig,'02_daily_before_after_2026')

    with np.load(ROOT/'results/prediction_grids_2026.npz') as z:
        mask=z[SCENARIO];static=z['static']
        arrays=[z[k][mask].sum(axis=0) for k in ('raw_modis',MAIN,'viirs')]
        norm=PowerNorm(.5,vmin=0,vmax=max(float(a.max()) for a in arrays))
        fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True,layout='constrained')
        cmap=plt.colormaps['magma'].with_extremes(bad='#ADB5BF')
        for ax,a,label in zip(axes,arrays,['Observed MODIS','Frozen 14-year estimate','Observed VIIRS']):
            im=ax.imshow(np.ma.array(a,mask=static),origin='lower',extent=(-140,-60,50,70),aspect=2,norm=norm,cmap=cmap,interpolation='nearest')
            ax.set_title(label,loc='left',fontsize=11);ax.set_xlabel('Longitude');ax.set_xticks([-140,-120,-100,-80,-60]);ax.set_yticks([50,60,70])
        axes[0].set_ylabel('Latitude')
        fig.colorbar(im,ax=list(axes),location='bottom',pad=.12,shrink=.7,aspect=45,label='Occupied days per 0.25° cell · estimate = expected days · shared square-root scale')
        heading(fig,'2026 spatial activity outside the training years','Same 51 dates and frozen source mask in each panel · existing coarse grid; no native-resolution reconstruction')
        export(fig,'03_spatial_before_after_2026')


def report(validation):
    metrics=read('metrics.csv');binary=read('binary_metrics.csv')
    national=lambda scenario:{r['model']:r for r in metrics if r['scenario']==scenario and r['region']=='canada_rectangle'}
    primary=national(SCENARIO);sensitivity=national('metadata_window')
    reduction=1-float(primary[MAIN]['weekly_mae'])/float(primary[RAW]['weekly_mae'])
    reduced_sensitivity=1-float(sensitivity[MAIN]['weekly_mae'])/float(sensitivity[RAW]['weekly_mae'])
    lines=['# Experiment 04.5: frozen model evaluated outside its training years','',
           '**The saved 2012–2025 model was evaluated on 2026 data without retraining, threshold selection or calibration.** '
           'This is an out-of-time partial-archive diagnostic, with substantial coverage limitations.','',
           '## What was unseen','',
           'Every scoring date is in 2026. The frozen 14-year model trained on 2012–2025; the comparator models end in 2022 and 2019. '
           'No 2026 labels fitted their trees or selected their checkpoints. Raw archive coverage had been inspected, so this is not a prospective blinded trial. '
           'All three model hashes stayed unchanged.','',
           '## Coverage policies, frozen before prediction','',
           '- **Primary: 51 dates** between June 3 and July 31 with at least one archived detection from each sensor anywhere in the rectangle. '
           'This excludes globally empty sensor dates but conditions the population on recorded activity. It is not a representative all-days evaluation.',
           '- **Sensitivity: 59 dates** in the inferred June–July download window after removing the two existing gap flags. '
           'It includes eight additional dates with one globally empty sensor archive and treats absent rows as recorded zeros.',
           '- The original per-request manifest is missing, so contiguous coverage of the inferred window cannot be proven. '
           'Both scenarios lack verified per-cell clear-sky coverage. August was not scored.',
           '- The masks affect scoring only. Features were computed once from current/past MODIS and its coverage flags, without VIIRS-derived input selection. '
           'The original frozen 31-cell static mask was retained.','',
           '## Whole-rectangle results','',
           'Units: area-weighted occupied cell-days per calendar bin. Nine partial bins in each scenario. '
           'These MAEs should not be compared directly with Experiment 04\'s longer three-summer benchmark.','',
           '| Model | 51-day MAE | 51-day WAPE | 51-day bias | 59-day MAE | 59-day WAPE | 59-day bias |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in (RAW,'constant_ratio_14year','power_14year_rate','boosted_8year','boosted_11year',MAIN):
        a,b=primary[name],sensitivity[name]
        lines.append(f"| {name} | {float(a['weekly_mae']):.2f} | {float(a['weekly_wape']):.2%} | {float(a['aggregate_bias']):+.2%} | {float(b['weekly_mae']):.2f} | {float(b['weekly_wape']):.2%} | {float(b['aggregate_bias']):+.2%} |")
    lines+=['',f'The frozen final model lowers MAE by **{reduction:.1%}** against raw MODIS on the 51-day mask, '
            f'and by **{reduced_sensitivity:.1%}** on the broader 59-day assumption. '
            'The eleven-year model is slightly better than the fourteen-year model on both masks; this run does not select or replace any model.','',
            'The ratio and power-law comparators are fitted only on 2012–2025 observations. The power-law fit uses weekly mean daily activity '
            'and scales back to each scored bin length, so it is a different comparator from Experiment 04\'s weekly-total power law.','',
            '## Daily recorded-occupancy metrics','',
            '| Mask | Model | Precision | Recall | F1 | Brier | ROC AUC | Average precision |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for scenario in (SCENARIO,'metadata_window'):
        for name in (RAW,MAIN):
            r=next(x for x in binary if x['scenario']==scenario and x['model']==name)
            lines.append(f"| {scenario} | {name} | {float(r['precision']):.3f} | {float(r['recall']):.3f} | {float(r['f1']):.3f} | {float(r['brier']):.6f} | {float(r['roc_auc']):.3f} | {float(r['average_precision']):.3f} |")
    lines+=['','These metrics use the full eligible coarse grid on included dates, area weighting, and threshold 0.5. '
            'They measure recorded VIIRS occupancy; they are not true-fire, forest-only or future-ignition accuracy.','',
            '## Interpretation','',
            'The model was genuinely evaluated outside its training years. The useful agreement on the primary mask is encouraging, '
            'but the large sensitivity to ambiguous dates means this is not a definitive clean holdout validation. '
            'A complete verified 2026 overlap or independently acquired regional/year archive is still required. '
            'Excluding globally empty archives also excludes possible genuine quiet days and cannot be treated as an unbiased fix.','',
            'No native-image data, new geographic domain, precise fire locations, complete summer, or calibrated ground-truth fire probability is demonstrated. '
            'No model or main-app integration was changed.','',
            '## Figures','',
            '- [Coverage sensitivity](../visualizations/01_holdout_error_and_coverage.png).',
            '- [Daily before/after/reference](../visualizations/02_daily_before_after_2026.png).',
            '- [Spatial before/after/reference](../visualizations/03_spatial_before_after_2026.png).','',
            '## Inspectable evidence','',
            '- [Frozen protocol](../PROTOCOL.md), [model hashes and training years](frozen_models.json), [date audit](date_audit.csv).',
            '- [Metrics](metrics.csv), [weekly predictions](weekly_predictions.csv), [daily predictions](daily_predictions.csv), [probability metrics](binary_metrics.csv), [support diagnostics](support_diagnostics.csv).',
            '- [Validation](validation.json), [isolation](isolation.json), [input hashes](input_hashes.json).','',
            f"Validation passed: {validation['prediction_rows_checked']} weekly rows, {validation['metric_groups_checked']} metric groups, "
            f"{validation['frozen_models_unchanged']} unchanged models and {validation['main_files_unchanged']} unchanged main-app files."]
    # Keep generated prose plain; this report is separate from all prior experiment records.
    output('results/REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json('results/readout.json',{'test_year':2026,'primary_days':51,'sensitivity_days':59,
                'primary_mae_reduction_vs_raw':reduction,'sensitivity_mae_reduction_vs_raw':reduced_sensitivity,
                'clean_complete_holdout':False,'boosted_model_retrained':False,'primary_f1':float(next(r for r in binary if r['scenario']==SCENARIO and r['model']==MAIN)['f1'])})
    output('README.md').write_text('# Experiment 04.5 — frozen out-of-time evaluation\n\n'
        'Evaluates the existing Experiment 04 models on partial 2026 data outside all their training years. '
        'No boosted model is retrained. All outputs are isolated here.\n\n'
        '[Results and limitations](results/REPORT.md) · [Frozen protocol](PROTOCOL.md) · [Validation](results/validation.json)\n\n'
        'From the project root:\n\n```sh\n'
        'PYTHONDONTWRITEBYTECODE=1 fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04.5/src/evaluate.py\n'
        'PYTHONDONTWRITEBYTECODE=1 fire-harmonization-research/.venv/bin/python fire-harmonization-research/experiment-04.5/src/report.py\n```\n')


if __name__=='__main__':
    validation=validate();figures();report(validation)
    print(json.dumps(validation,indent=2))
