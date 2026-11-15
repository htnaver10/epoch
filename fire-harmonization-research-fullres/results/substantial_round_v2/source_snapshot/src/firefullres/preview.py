import os
import numpy as np
from .core import ROOT,path,validate_tiles


def preview(manifest,output,index=0,allow_synthetic=False):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=validate_tiles(manifest,allow_synthetic,False,False)
    if not 0<=index<len(rows):raise ValueError('Tile index outside manifest')
    r=rows[index];p=path(output,True)
    if p.exists():raise ValueError('Preview exists; use a new filename')
    with np.load(path(r['tile'])) as z:
        h,w=z['thermal'].shape[-2:]
        image=np.ma.masked_where(z['thermal'][7]==0,z['thermal'][1])
        grid=z['query_grid'];x=(grid[...,0]+1)*w/2-.5;y=(grid[...,1]+1)*h/2-.5
        observed=z['weight']>0;fire=(z['target']>.5)&observed
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        im=axes[0].imshow(image,cmap='inferno',origin='upper')
        axes[0].scatter(x[fire],y[fire],s=8,facecolors='none',edgecolors='cyan',linewidths=.6,label='VIIRS nominal/high reference')
        axes[0].legend(fontsize=8);axes[0].set_title('Native band 21 radiance and reference alignment')
        fig.colorbar(im,ax=axes[0],label='Calibrated source radiance')
        axes[1].scatter(x[observed],y[observed],c=z['target'][observed],s=3,cmap='coolwarm',vmin=0,vmax=1)
        axes[1].set_title('Known reference queries; blank = unknown/excluded')
        for ax in axes:
            ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_xlabel('MODIS native sample');ax.set_ylabel('MODIS native line')
        fig.suptitle(f"{r['tile_id']} · {r['dataset_kind']} · {r['split']}",fontsize=11)
        fig.savefig(p,dpi=150);plt.close(fig)
    return {'preview':str(p.relative_to(ROOT)),'scope':'alignment inspection, not proof of absolute geolocation or fire-location accuracy'}
