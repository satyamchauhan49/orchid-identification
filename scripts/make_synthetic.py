"""Create clearly labelled integration fixtures, never scientific results."""
import csv
import json
from pathlib import Path
import numpy as np
from bn_resolver.contracts import FEATURES, digest


def main():
    root=Path('work/synthetic')
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    maps={'species':[f'SYNTHETIC_species_{i:03}' for i in range(156)],
          'features':{f:[f'SYNTHETIC_{f}_{j}' for j in range(k)]
                      for f,k in zip(FEATURES,(3,2,4,3,5,4))}}
    (root/'label_maps.json').write_text(json.dumps(maps,indent=2))
    theta={f:rng.integers(0,len(maps['features'][f]),156) for f in FEATURES}
    rows=[]
    for split,n in [('train',1560),('calib',624),('val',312)]:
        species=np.arange(n)%156
        logits=rng.normal(0,1,(n,156))
        logits[np.arange(n),species]+=4
        wrong = np.arange(n) % 7 == 0
        logits[wrong] = rng.normal(0,1,(int(wrong.sum()),156))
        logits[np.flatnonzero(wrong),(species[wrong]+1)%156] += 5
        uncertain = np.arange(n) % 11 == 0
        logits[uncertain] *= .15
        ids=np.array([f'SYNTHETIC_{split}_{i}' for i in range(n)])
        bundle={'ids':ids,'species_logits':logits,
                'label_maps_sha256':np.array(digest(root/'label_maps.json'))}
        for i in range(n):
            row=dict(image_id=ids[i],path=f'synthetic/{ids[i]}.png',
                     species=maps['species'][species[i]],phash_cluster=ids[i],split=split)
            row.update({f:maps['features'][f][theta[f][species[i]]]
                        if rng.random()>.08 else '' for f in FEATURES})
            rows.append(row)
        for f in FEATURES:
            q=rng.normal(0,1,(n,len(maps['features'][f])))
            q[np.arange(n),theta[f][species]]+=2.5
            q[uncertain] = 0.
            if f in ('CF','CL'):
                q *= .55
            bundle['logits_'+f]=q
        if split!='train':
            np.savez_compressed(root/(split+'.npz'),**bundle)
    with (root/'split_manifest.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/'SYNTHETIC_ONLY.txt').write_text('Generated random data. No real orchids, training, or research findings.\n')
    print(root)


if __name__=='__main__':
    main()
