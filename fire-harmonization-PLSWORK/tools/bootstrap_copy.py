"""Physically copy frozen local observations and inference dependencies; no network."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def contained(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Copy path escapes its declared source/destination')
    return path


def copy_existing(config=None):
    config=config or json.loads((ROOT/'configs/experiment.json').read_text())
    receipt_path=ROOT/'results/local_copy_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('complete'):
            for row in receipt['files']:
                if digest(contained(ROOT,row['destination']))!=row['sha256']:raise ValueError('Copied local snapshot changed: '+row['destination'])
            print(f'Validated {len(receipt["files"])} existing copied files; no downloads.',flush=True);return receipt
    source=(ROOT/config['source_folder']).resolve()
    if source==ROOT or not source.is_relative_to(ROOT.parent):raise ValueError('Source folder must be the existing sibling research folder')
    manifest=contained(source,config['source_manifest'])
    if digest(manifest)!=config['source_manifest_sha256']:raise ValueError('Frozen source manifest changed')
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or any(row.get('dataset_kind')!='nasa_native' for row in rows):raise ValueError('Require the real native parent dataset')
    dataset_paths=sorted({row['tile'] for row in rows}|{path for row in rows for path in row['source_ids']})
    mapping=[(path,path) for path in dataset_paths]
    mapping += [(config['source_manifest'],'data/manifests/source_v3.jsonl'),
        (config['source_baseline_checkpoint'],'models/reference/supervised_modis.pt')]
    # A local, independent snapshot supports loading the unchanged reference CNN.
    for name in ('__init__','core','network','engine','loading','runtime','metrics','calibration'):
        path=f'src/firefullres/{name}.py';mapping.append((path,path))
    required=sum(contained(source,src).stat().st_size for src,_ in mapping if not contained(ROOT,_).exists())
    if shutil.disk_usage(ROOT).free<required+config['reserve_free_gib']*1024**3:
        raise ValueError('Copy would cross the reserved disk-space limit')
    receipt={'complete':False,'source_root':str(source),'source_manifest_sha256':config['source_manifest_sha256'],
        'network_used':False,'physical_copies':True,'files':[]}
    for index,(src,dst) in enumerate(mapping):
        original=contained(source,src);destination=contained(ROOT,dst);expected=digest(original)
        if src==config['source_baseline_checkpoint'] and expected!=config['source_baseline_sha256']:
            raise ValueError('Supervised baseline checkpoint changed')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if digest(destination)!=expected:raise ValueError('Refusing to overwrite differing copied content: '+dst)
        else:
            temporary=destination.with_name(destination.name+'.partial');shutil.copy2(original,temporary)
            if digest(temporary)!=expected:raise ValueError('Physical copy failed checksum verification')
            temporary.replace(destination)
        receipt['files'].append({'source':src,'destination':dst,'sha256':expected,'bytes':destination.stat().st_size})
        if (index+1)%100==0:print(f'Copied and verified {index+1}/{len(mapping)} local files',flush=True)
    receipt['complete']=True;receipt['bytes']=sum(row['bytes'] for row in receipt['files'])
    write(receipt_path,receipt)
    print(f'Local copy completed: {len(mapping)} files, {receipt["bytes"]/1024**3:.2f} GiB. No network data access.',flush=True)
    return receipt


if __name__=='__main__':copy_existing()
