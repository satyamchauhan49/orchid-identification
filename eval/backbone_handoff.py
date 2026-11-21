"""One-command strict Role B import, C development suite and resolver sidecars."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from bn_resolver.import_backbone import ingest
from bn_resolver.pipeline import evaluate
from .experiments import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','original-maps','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Choose a new output root')
    audited=out/'audited';ingest(a.source,a.original_maps,a.manifest,audited)
    maps=str(audited/'label_maps.json');val=str(audited/'val.npz');exp=out/'experiments'
    run(SimpleNamespace(maps=maps,manifest=a.manifest,calib=str(audited/'calib.npz'),val=val,output=str(exp),synthetic_only=False,risk=.015,minimum=50))
    evaluate(SimpleNamespace(maps=maps,manifest=a.manifest,bundle=val,artifact=str(exp/'frozen_direct_soft.json'),split='val',unlock_test=False,output=str(exp/'validation_evaluation.json')))
    print('Verified development handoff completed. No test inputs used:',out)

if __name__=='__main__':main()
