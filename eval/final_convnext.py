"""Evaluate ONLY the agreed ConvNeXt frozen package, after explicit unlock.

No fitting/tuning occurs. Missing/mismatched provenance or code stops before scoring.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bn_resolver.contracts import digest
from bn_resolver.pipeline import evaluate,write_json


def verify(freeze_path,repo):
    freeze=json.loads(Path(freeze_path).read_text());repo=Path(repo)
    for name,item in freeze['files'].items():
        p=repo/item['path']
        if not p.is_file() or digest(p)!=item['sha256']:raise ValueError('Frozen file changed/missing: '+name)
    return freeze


def run(freeze_path,repo,bundle,output,unlock=False):
    if not unlock:raise ValueError('Locked test: explicit --unlock-test required after team freeze')
    repo=Path(repo);freeze=verify(freeze_path,repo)
    output=Path(output)
    if output.exists():raise ValueError('Final results exist; refusing overwrite')
    f=freeze['files']
    with np.load(bundle,allow_pickle=False) as z:
        for key,name in [('checkpoint_sha256','checkpoint'),('label_maps_sha256','maps'),('manifest_sha256','manifest')]:
            if key not in z or str(z[key].item())!=f[name]['sha256']:raise ValueError('Test export provenance mismatch: '+key)
    evaluate(SimpleNamespace(maps=str(repo/f['maps']['path']),manifest=str(repo/f['manifest']['path']),bundle=str(bundle),artifact=str(repo/f['artifact']['path']),
        split='test',unlock_test=True,output=str(output),group_column=freeze['bootstrap_group']))
    write_json(str(output)+'.freeze.json',dict(selection_sha256=digest(freeze_path),test_bundle_sha256=digest(bundle),model='convnext_tiny_lr0.0001',policy_refitted=False,code_verified=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',default='report/CONVNEXT_FINAL_SELECTION.json');p.add_argument('--repo',default='.')
    p.add_argument('--bundle',required=True);p.add_argument('--output',default='eval/final_results.json');p.add_argument('--unlock-test',action='store_true')
    a=p.parse_args();run(a.freeze,a.repo,a.bundle,a.output,a.unlock_test)

if __name__=='__main__':main()
