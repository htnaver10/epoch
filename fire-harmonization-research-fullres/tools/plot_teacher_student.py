"""Plot the completed teacher/student fit and the sealed reused benchmark."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='teacher_student_v1')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];folder=root/'results'/args.run
    report=json.loads((folder/'evaluation.json').read_text())
    logs={kind:[json.loads(line) for line in (root/'logs'/(args.run+'_'+kind)/'epochs.jsonl').read_text().splitlines() if line.strip()]
        for kind in ('teacher','student')}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    for kind,color in [('teacher','#0891b2'),('student','#2563eb')]:
        loss=[r['validation']['logloss'] for r in logs[kind]];epochs=[r['epoch']+1 for r in logs[kind]];index=int(np.argmin(loss))
        axes[0,0].plot(epochs,loss,label=kind.capitalize(),color=color,lw=1.5)
        axes[0,0].scatter([epochs[index]],[loss[index]],color=color,s=35)
    axes[0,0].set(title='Development-validation log loss',xlabel='Completed epoch',ylabel='Log loss');axes[0,0].legend(frameon=False)
    records=logs['student'];epochs=[r['epoch']+1 for r in records]
    axes[0,1].plot(epochs,[r['training_hard_logloss'] for r in records],label='Hard-label BCE',color='#2563eb')
    axes[0,1].plot(epochs,[r['training_temperature_scaled_kl'] for r in records],label='Teacher KL, before coefficient 0.5',color='#d97706')
    axes[0,1].set(title='Student training objectives',xlabel='Completed epoch',ylabel='Loss per known query');axes[0,1].legend(frameon=False)
    keys=['modis','supervised_baseline','distilled_student','distilled_student_adjusted']
    labels=['Regular\nMODIS','Previous\nMODIS ML','Distilled\nstudent','Distilled\nfrozen policy']
    values=[report['test'][key] for key in keys];x=np.arange(len(keys))
    for shift,key,label,color in [(-.18,'precision','Precision','#2563eb'),(.18,'recall','Recall','#d97706')]:
        bars=axes[1,0].bar(x+shift,[100*v[key] for v in values],width=.36,label=label,color=color)
        axes[1,0].bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
    axes[1,0].set_xticks(x,labels);axes[1,0].set(title='Reused test benchmark: precision and recall',ylabel='Percent',ylim=(0,110))
    axes[1,0].legend(frameon=False,loc='upper left',ncol=2)
    bars=axes[1,1].bar(labels,[v['false_positives'] for v in values],color=['#64748b','#94a3b8','#2563eb','#d97706'])
    axes[1,1].bar_label(bars,fmt='%.0f',padding=3)
    axes[1,1].set(title='Reused test benchmark: false positives',ylabel='Reference query centers')
    axes[1,1].set_ylim(0,max(1,max(v['false_positives'] for v in values))*1.2)
    figure.suptitle('VIIRS thermal teacher → MODIS-only student\nPreviously examined benchmark; 95 positive reference queries',fontsize=13)
    output=folder/'training_and_benchmark.png';figure.savefig(output,dpi=180);plt.close(figure);print(output)


if __name__=='__main__':main()
