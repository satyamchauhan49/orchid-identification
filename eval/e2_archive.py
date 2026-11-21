"""Audit supplied E2 development archive and run the prespecified C suite.

Only six allowlisted development members per model are read; test files are never
opened. Original exports lack exporter hashes: see each IMPORT_PROVENANCE.json.
"""
import argparse,zipfile,json
from pathlib import Path
from types import SimpleNamespace
from bn_resolver.import_development import ingest
from .experiments import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','maps','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.output)
    if root.exists():raise ValueError('Choose a new output directory')
    root.mkdir(parents=True)
    with zipfile.ZipFile(a.archive) as z:
        folders=sorted({n.rsplit('/',1)[0] for n in z.namelist() if n.endswith('/best.pt') and not n.startswith('__MACOSX/')})
        for folder in folders:
            name=folder.rsplit('/',1)[-1]
            if not name.startswith('resnet18_e2_') or '/' in name or '..' in name:raise ValueError('Unexpected model folder')
            src=root/name/'original';src.mkdir(parents=True)
            for member in ('best.pt','class_maps.json','normalization.json','calib.npz','val.npz','training_history.csv'):
                (src/member).write_bytes(z.read(folder+'/'+member))
            ingest(src,a.maps,a.manifest,root/name/'audited')
            run(SimpleNamespace(maps=a.maps,manifest=a.manifest,calib=str(root/name/'audited/calib.npz'),
                val=str(root/name/'audited/val.npz'),output=str(root/name/'experiments'),synthetic_only=False,risk=.015,minimum=50))
    print('All E2 development runs completed; no test files opened.')

if __name__=='__main__':main()
