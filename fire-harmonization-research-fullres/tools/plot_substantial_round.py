"""Plot completed training and the sealed held-out evaluation without refitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='substantial_round_v3')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    folder=root/'results'/args.run
    report=json.loads((folder/'evaluation.json').read_text())
    records=[json.loads(line) for line in (root/'logs'/args.run/'epochs.jsonl').read_text().splitlines() if line.strip()]
    epochs=np.array([row['epoch']+1 for row in records])
    loss=np.array([row['validation']['logloss'] for row in records])
    best=int(np.argmin(loss))
    keys=['modis','raw_model_at_0_5','validation_adjusted_model_at_frozen_threshold']
    labels=['MODIS','Raw model\nat 0.5','Frozen validation\npolicy']
    colors=['#64748b','#d97706','#2563eb']
    scores=[report['test'][key] for key in keys]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axis=axes[0,0]
    axis.plot(epochs,loss,color='#2563eb',lw=1.8)
    axis.scatter([epochs[best]],[loss[best]],color='#dc2626',zorder=3,label=f'Selected epoch {epochs[best]}')
    axis.set(title='Development-validation log loss',xlabel='Completed epoch',ylabel='Log loss')
    axis.legend(frameon=False)
    for axis,key,title in zip([axes[0,1],axes[1,0],axes[1,1]],
            ['precision','recall','false_positives'],['Test precision','Test recall','Test false positives']):
        values=[score[key]*(100 if key!='false_positives' else 1) for score in scores]
        bars=axis.bar(labels,values,color=colors,width=.6)
        axis.bar_label(bars,labels=[f'{v:.1f}%' if key!='false_positives' else f'{v:,.0f}' for v in values],padding=4)
        axis.set_title(title)
        axis.set_ylabel('Percent' if key!='false_positives' else 'Reference query centers')
        axis.set_ylim(0,max(1,max(values))*1.22)
    positives=scores[0]['observed_positive_weight']
    figure.suptitle(f'{args.run}: {len(records)} epochs; reserved test has {positives:,.0f} positive query labels',fontsize=13)
    output=folder/'training_and_test.png'
    figure.savefig(output,dpi=180)
    plt.close(figure)
    print(str(output))


if __name__=='__main__':main()
