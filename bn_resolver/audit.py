"""Read-only development input audit; no test bundle or test metric access."""
import argparse
from collections import Counter
import numpy as np
from .contracts import FEATURES, load_maps, read_manifest, load_bundle, digest, truth, calibration_partition
from .pipeline import write_json


def audit(maps_path, manifest_path, calib_path, val_path):
    maps=load_maps(maps_path)
    rows=read_manifest(manifest_path,maps)
    result=dict(species_count=len(maps['species']),splits={},warnings=[],
                hashes=dict(maps=digest(maps_path),manifest=digest(manifest_path)),
                test_bundle_opened=False)
    for split in ('train','calib','val'):
        subset=[r for r in rows if r['split']==split]
        counts=Counter(r['species'] for r in subset if r['species'])
        support=[counts[s] for s in maps['species']]
        result['splits'][split]=dict(rows=len(subset),species_support=support,
            absent_species=[s for s,n in zip(maps['species'],support) if n==0],
            rare_species=[s for s,n in zip(maps['species'],support) if 0<n<5],
            missing_feature_labels={f:sum(not r[f] for r in subset) for f in FEATURES})
        if not subset: result['warnings'].append(f'{split}: empty split')
        if any(n==0 for n in support): result['warnings'].append(f'{split}: some species have zero support')
    for split,path in [('calib',calib_path),('val',val_path)]:
        ids,arrays=load_bundle(path,rows,maps,split,digest(maps_path))
        result['hashes'][split]=digest(path)
        result['splits'][split]['constant_logit_rows']={
            f:int((np.ptp(a,axis=1)==0).sum()) for f,a in arrays.items()}
        if split=='calib':
            cm=calibration_partition(rows,ids)
            result['cal_model_support']={f:int((truth(rows,ids,maps,f)[cm]>=0).sum()) for f in ('species',*FEATURES)}
            for f,n in result['cal_model_support'].items():
                if n<20: result['warnings'].append(f'{f}: fewer than 20 cal_model labels; temperature stays 1')
    result['warnings'].append('Checkpoint and preprocessing identity are not validated by the current bundle contract; verify separately.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('maps','manifest','calib','val','output'): p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    write_json(a.output,audit(a.maps,a.manifest,a.calib,a.val))
    print(a.output)


if __name__=='__main__': main()
