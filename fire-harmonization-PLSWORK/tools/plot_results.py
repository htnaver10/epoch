"""Standalone scientific figure for the completed local experiment."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    report=json.loads((ROOT/'results/evaluation.json').read_text());history=json.loads((ROOT/'results/mlp_history.json').read_text())
    names=['regular_modis','previous_cnn_raw_0.5','mlp_raw_0.5','mlp_frozen_policy','tree_raw_0.5','tree_frozen_policy']
    labels=['MODIS','Previous\nCNN','MLP\nraw 0.5','MLP\nfrozen policy','Tree\nraw 0.5','Tree\nfrozen policy']
    values=[report['test'][key] for key in names]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    epochs=[r['epoch']+1 for r in history]
    axes[0,0].plot(epochs,[r['training_logloss'] for r in history],label='Training',color='#2563eb')
    axes[0,0].plot(epochs,[r['validation_logloss'] for r in history],label='Validation',color='#d97706')
    axes[0,0].set(title='MLP label loss',xlabel='Completed epoch',ylabel='Binary log loss');axes[0,0].legend(frameon=False)
    axes[0,1].plot(epochs,[r['validation_average_precision'] for r in history],color='#2563eb')
    axes[0,1].set(title='MLP validation average precision',xlabel='Completed epoch',ylabel='AP')
    x=np.arange(len(names))
    for shift,key,label,color in [(-.18,'precision','Precision','#2563eb'),(.18,'recall','Recall','#d97706')]:
        bars=axes[1,0].bar(x+shift,[100*v[key] for v in values],width=.36,label=label,color=color)
        axes[1,0].bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    axes[1,0].set_xticks(x,labels);axes[1,0].set(title='Reused benchmark: precision and recall',ylabel='Percent',ylim=(0,115));axes[1,0].legend(frameon=False,ncol=2)
    bars=axes[1,1].bar(labels,[v['false_positives'] for v in values],color=['#64748b','#94a3b8','#2563eb','#60a5fa','#d97706','#fbbf24'])
    axes[1,1].bar_label(bars,fmt='%.0f',padding=3)
    axes[1,1].set(title='Reused benchmark: false positives',ylabel='Reference query centers')
    axes[1,1].set_ylim(0,max(1,max(v['false_positives'] for v in values))*1.2)
    fig.suptitle('Contextual features: compact MLP and tree baseline\nCopied local data only; previously examined benchmark has 95 positive query labels',fontsize=13)
    dest=ROOT/'results/training_and_benchmark.png';fig.savefig(dest,dpi=180);plt.close(fig);print(dest)

if __name__=='__main__':main()
