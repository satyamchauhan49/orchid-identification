"""Static scientific figures, with explicit synthetic/development labels."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(folder):
    root=Path(folder)
    data=json.loads((root/'plot_data.json').read_text())
    banner='SYNTHETIC DEMO - NOT RESEARCH RESULTS' if data['data_kind']=='synthetic_demo' else 'DEVELOPMENT VALIDATION - NOT TEST RESULTS'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for name,rows in data['reliability'].items():
        valid=[r for r in rows if r['count']]
        axes[0].plot([r['confidence'] for r in valid],[r['accuracy'] for r in valid],marker='o',label=name)
    axes[0].plot([0,1],[0,1],'--',color='gray',linewidth=1)
    axes[0].set(xlabel='Mean predicted confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title='Reliability (15 equal-width bins)')
    axes[0].legend(fontsize=8)
    for name,rows in data['risk_coverage'].items():
        axes[1].plot([r['coverage'] for r in rows],[r['risk'] for r in rows],label=name)
    axes[1].set(xlabel='Coverage',ylabel='Observed error among accepted cases',xlim=(0,1),ylim=(0,1),title='Risk-coverage on validation')
    axes[1].legend(fontsize=8)
    fig.suptitle(banner,fontsize=13,fontweight='bold')
    fig.savefig(root/'calibration_and_risk.png',dpi=170)
    plt.close(fig)
    with (root/'E3_R3_bn_comparison.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    names=sorted(set(r['name'] for r in rows))
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    x=np.arange(len(names))
    for offset,mask,color in [(-.18,0.,'#2563eb'),(.18,.4,'#0d9488')]:
        values=[float(next(r for r in rows if r['name']==name and float(r['missing_confidence'])==mask)['accuracy']) for name in names]
        ax.bar(x+offset,values,.34,label=f'Missing confidence = {mask}',color=color)
    ax.set_xticks(x,names,rotation=15)
    ax.set(ylabel='Top-1 accuracy',ylim=(0,1),title='E3: BN evidence comparison')
    ax.legend(fontsize=9)
    fig.suptitle(banner,fontsize=12,fontweight='bold')
    fig.savefig(root/'bn_comparison.png',dpi=170)
    plt.close(fig)

    if 'calibration_ablations' in data:
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        pairs=[('DNN','dnn_raw','dnn_temperature'),
               ('BN posterior','bn_calibrated_heads','bn_temperature'),
               ('Fused posterior','fusion_before_temperature','fusion_temperature')]
        for ax,(title,before,after) in zip(axes,pairs):
            for key,label in [(before,'Before temperature'),(after,'After temperature')]:
                points=[r for r in data['calibration_ablations'][key] if r['count']]
                ax.plot([r['confidence'] for r in points],[r['accuracy'] for r in points],marker='o',label=label)
            ax.plot([0,1],[0,1],'--',color='gray',linewidth=1)
            ax.set(title=title,xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1))
            ax.legend(fontsize=8)
        fig.suptitle(banner+'\nE4: independent validation reliability',fontsize=12,fontweight='bold')
        fig.savefig(root/'calibration_ablations.png',dpi=170)
        plt.close(fig)
