"""Read-only post-evaluation audit of positive/negative coverage and raw MODIS flags.

This diagnosis does not refit models, thresholds or reported predictions. Raw
HDF inspection uses the already installed native-data environment and reader.
"""
import collections
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fireyes.features import coarse,ring,signed_log,geometric_valid,interior
from fireyes.common import sha

def main():
    fullres=ROOT.parent/'fire-harmonization-research-fullres'
    sys.path.insert(0,str(fullres/'src'))
    from firefullres.native import sds,numbers
    parent={r['tile_id']:r for r in (json.loads(line) for line in
            (fullres/'data/manifests/substantial_round_v3.jsonl').read_text().splitlines() if line.strip())}
    refs=json.loads((ROOT/'data/reference_manifest.json').read_text())
    by_requirement=collections.Counter();combinations=collections.Counter();band_valid=collections.Counter()
    raw_flags={str(b):collections.Counter() for b in (20,21,22,23,31,32)}
    codes=[];positive=negative=valid_positive=valid_negative=0;datasets={};alternative=0
    for row in refs:
        if row['split']!='test':continue
        with np.load(ROOT/row['reference'],allow_pickle=False) as z:
            known=z['weight']>0;y=z['target'][known]>.5;g=z['query_grid'][known]
        with np.load(ROOT/row['input'],allow_pickle=False) as z:
            inputs={k:z[k] for k in ('thermal','reflective_500','reflective_250','geometry')}
        a=inputs['thermal'];geo=inputs['geometry'];h,w=a.shape[-2:]
        iy=np.clip(np.floor((g[:,1]+1)*h/2).astype(int),0,h-1)
        ix=np.clip(np.floor((g[:,0]+1)*w/2).astype(int),0,w-1)
        valid=coarse(inputs)[2][iy,ix]
        positive+=int(y.sum());negative+=int((~y).sum())
        valid_positive+=int((y&valid).sum());valid_negative+=int((~y&valid).sum())
        if not y.any():continue
        iy,ix=iy[y],ix[y]
        checks={'geometry':geometric_valid(geo),'interior':interior((h,w),16)}
        for index,band in ((0,20),(2,22),(4,31),(5,32)):
            checks[f'band_{band}_valid']=a[6+index]>0
        for outer in (7,21):
            for index,band in ((0,20),(2,22)):
                checks[f'band_{band}_ring_{outer}_support']=ring(signed_log(a[index]),a[6+index]>0,outer)[3]>.5
        for name,v in checks.items():by_requirement[name]+=int((~v[iy,ix]).sum())
        for i in range(len(iy)):
            combinations[';'.join(k for k,v in checks.items() if not v[iy[i],ix[i]]) or 'usable']+=1
        for index,band in enumerate((20,21,22,23,31,32)):
            band_valid[str(band)]+=int((a[6+index,iy,ix]>0).sum())
        # Eligibility-only counterfactual; this is NOT a prediction or improvement estimate.
        alternative_mask=interior((h,w),16)&geometric_valid(geo)&(a[7]>0)&(a[10]>0)&(a[11]>0)
        for outer in (7,21):alternative_mask&=ring(signed_log(a[1]),a[7]>0,outer)[3]>.5
        alternative+=int(alternative_mask[iy,ix].sum())
        original=parent[row['tile_id']];src=original['source_ids'][0]
        if src not in datasets:datasets[src]=sds(src,'EV_1KM_Emissive')
        ds=datasets[src];meta=ds.GetMetadata();bands=[b.strip() for b in meta['band_names'].split(',')]
        lo,hi=numbers(meta['valid_range']);r,c,hh,ww=original['source_window']
        for band in (20,21,22,23,31,32):
            raw=ds.GetRasterBand(bands.index(str(band))+1).ReadAsArray(c,r,ww,hh)[iy,ix]
            bad=(raw<lo)|(raw>hi)
            raw_flags[str(band)].update(int(v) for v in raw[bad])
            for v in raw[bad]:codes.append({'tile':row['tile_id'],'band':band,'raw_code':int(v)})
    result={'scope':'Post-hoc diagnosis on previously examined test reference; no fitting or threshold changes',
            'positive_queries':positive,'negative_queries':negative,
            'eligible_positive_queries':valid_positive,'eligible_negative_queries':valid_negative,
            'positive_coverage':valid_positive/positive,'negative_coverage':valid_negative/negative,
            'failures_by_requirement':dict(by_requirement),'failure_combinations':dict(combinations),
            'valid_measurements_at_positive_queries':dict(band_valid),
            'raw_invalid_codes_at_positive_queries':{b:dict(c) for b,c in raw_flags.items()},
            'band21_alternative_eligibility_only':alternative,
            'eligibility_ceiling_on_original_recall':valid_positive/positive,
            'model_export_sha256':{s:sha(ROOT/f'models/{s}/inference.pt') for s in ('coarse','fine')},
            'raw_code_sources':[
                'https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/5/MOD021KM.fs',
                'https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/M1055_PDD_D_072712final.pdf'],
            'raw_samples':codes}
    output=ROOT/'results/implementation_audit.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='raw_samples'},indent=2))

if __name__=='__main__':main()
