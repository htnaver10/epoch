"""Copy immutable local tiles, then physically separate MODIS inputs and evaluation references."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fireyes.common import config,path,sha,write,read,save_npz
from fireyes.features import INPUT_KEYS

def copy():
    dest='results/copy_receipt.json'
    if path(dest).exists():
        saved=read(dest)
        for f,d in saved['files'].items():
            if sha(path(f))!=d:raise ValueError('Local copy changed: '+f)
        return saved
    cfg=config();source=(ROOT/cfg['source_folder']).resolve()
    if source.parent!=ROOT.parent or source==ROOT:raise ValueError('Expected existing sibling source')
    manifest=source/cfg['source_manifest']
    if sha(manifest)!=cfg['source_manifest_sha256']:raise ValueError('Parent manifest changed')
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    if shutil.disk_usage(ROOT).free<30*2**30:raise ValueError('Insufficient disk reserve')
    names=[];modis_rows=[];reference_rows=[]
    for i,row in enumerate(rows):
        src=(source/row['tile']).resolve()
        if not src.is_relative_to(source):raise ValueError('Source tile outside copied corpus')
        if sha(src)!=row['sha256']:raise ValueError('Original tile checksum failed')
        base=row['tile_id'];filename=f'data/source_tiles/{base}.npz';out=path(filename);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if sha(out)!=row['sha256']:raise ValueError('Existing copied tile differs')
        else:shutil.copy2(src,out)
        if sha(out)!=row['sha256']:raise ValueError('Copy checksum failed')
        with np.load(out,allow_pickle=False) as z:
            inputs={k:z[k] for k in INPUT_KEYS}
            reference={k:z[k] for k in ('target','weight','query_grid','modis_baseline')}
        input_name=f'data/modis/{base}.npz';ref_name=f'data/references/{base}.npz'
        save_npz(input_name,**inputs);save_npz(ref_name,**reference)
        metadata={k:row[k] for k in ('tile_id','split','event_group','source_window')}
        modis_rows.append({**metadata,'input':input_name,'modis_source':row['source_ids'][0]})
        reference_rows.append({**metadata,'reference':ref_name,'input':input_name,'source':filename})
        names.extend([filename,input_name,ref_name])
        if (i+1)%100==0:print(f'Copied and separated {i+1}/{len(rows)} local tiles',flush=True)
    write('data/modis_manifest.json',modis_rows);write('data/reference_manifest.json',reference_rows)
    names+=['data/modis_manifest.json','data/reference_manifest.json']
    # Copy the verified metric implementation as an independent local evaluation dependency.
    metric_src=source/'src/firefullres/calibration.py';metric_dest=path('tools/reference_metrics.py')
    shutil.copy2(metric_src,metric_dest)
    names.append('tools/reference_metrics.py')
    result={'files':{n:sha(path(n)) for n in names},'tile_count':len(rows),'source_manifest_sha256':sha(manifest),
            'bytes':sum(path(n).stat().st_size for n in names),'physical_copies':True,'satellite_downloads':0,
            'source':str(source),'only_prepared_tiles_copied':True,'training_inputs_physically_separated_from_reference_labels':True}
    write(dest,result);return result

if __name__=='__main__':copy()
