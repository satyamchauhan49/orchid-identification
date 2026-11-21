
from pathlib import Path, PurePosixPath
from datetime import datetime
from collections import Counter
import hashlib, json, zipfile, shutil, subprocess, sys
import pandas as pd
from google.colab import files
project=Path('/content/orchid-identification')
base=Path('/content/drive/MyDrive/Orchid_Sneha')
assert project.exists() and base.exists(), 'Use existing project runtime and mount Drive.'
config=json.loads((base/'sneha_real_config.json').read_text())
model=base/'real_grid_v1/convnext_tiny_lr0.0001'
assert (model/'best.pt').is_file() and Path(config['data_root']).is_dir(), 'Restore model/images first.'
print('Select convnext_tiny_lr0.0001_handoff.zip from Downloads.')
uploaded=files.upload()
archives=[Path(n).resolve() for n in uploaded if n.lower().endswith('.zip')]
assert len(archives)==1, 'Select exactly one handoff ZIP.'
del uploaded
run=base/('satyam_updated_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
handoff=run/'handoff';handoff.mkdir(parents=True)
with zipfile.ZipFile(archives[0]) as package:
    checksum_name=next(n for n in package.namelist() if n.endswith('/SHA256SUMS.json'))
    prefix=checksum_name.rsplit('/',1)[0]+'/'
    assert sum(i.file_size for i in package.infolist())<50_000_000
    for name,expected in json.loads(package.read(checksum_name)).items():
        assert hashlib.sha256(package.read(prefix+name)).hexdigest()==expected, name
    for item in package.infolist():
        if item.is_dir(): continue
        path=PurePosixPath(item.filename)
        assert not path.is_absolute() and '..' not in path.parts and item.filename.startswith(prefix)
        target=handoff.joinpath(*PurePosixPath(item.filename[len(prefix):]).parts)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(package.read(item))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
exchange=json.loads((handoff/'exchange.json').read_text())
assert exchange['mode']=='real'
assert digest(model/'best.pt')==exchange['provenance']['checkpoint_sha256']
for key,name in [('artifact_sha256','frozen_direct_soft.json'),('bundle_sha256','val.npz'),('label_maps_sha256','label_maps.json')]:
    assert digest(handoff/name)==exchange['provenance'][key],key
assert digest(handoff/'split_manifest.csv')==digest(config['manifest'])
manifest=pd.read_csv(handoff/'split_manifest.csv',dtype=str);lookup=manifest.set_index('image_id')
for case in exchange['cases']:
    assert case['split']=='val' and lookup.loc[case['image_id'],'split']=='val'
    assert lookup.loc[case['image_id'],'path']==case['path']
    assert (Path(config['data_root'])/case['path']).is_file()
references=(manifest[manifest['split']=='train'].sort_values('image_id').drop_duplicates('species').set_index('species')['path'].to_dict())
reference_path=run/'reference_images.json';reference_path.write_text(json.dumps(references,indent=2))
def execute(arguments):
    result=subprocess.run([sys.executable,'-u','-m',*arguments],cwd=project,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);result.check_returncode()
execute(['explain.build_records','--input',str(handoff/'exchange.json'),'--taxonomy',str(handoff/'taxonomy.json'),
         '--references',str(reference_path),'--assets-root',config['data_root'],'--checkpoint',str(model/'best.pt'),
         '--output',str(run/'records')])
records=json.loads((run/'records/records.json').read_text())
assert len(records['records'])==10
stages=dict(Counter(r['decision']['stage'] for r in records['records']))
assert set(stages)=={'trusted','auto_resolved','deferred'},stages
execute(['models_other.release','--comparison',str(base/'real_grid_v1/comparison.json'),
         '--records',str(run/'records/records.json'),'--output',str(run/'site')])
site=run/'site'
for name in ['validation_evaluation.json','IMPORT_PROVENANCE.json','SHA256SUMS.json','calibration_and_risk.png','calibration_ablations.png','bn_comparison.png']:
    shutil.copyfile(handoff/name,site/name)
report=site/'report.md';text=report.read_text()
start=text.index('## Limitations and remaining team work');end=text.index('## Input provenance',start)
text=text[:start]+"## Limitations and remaining team work\nReal validation under Satyam's disjoint-calibration policy: DNN/fusion accuracy 79.42%, accepted accuracy 94.46%, deferral 30.48%. Weight 1.0 means fusion equals DNN; no BN gain is demonstrated. There are 877 trusted, 8 auto_resolved and 388 deferred decisions overall. Accepted error exceeds the 1.5% target. Stage names alone do not prove BN corrections.\n\nTen actual validation cases are illustrated with matching model Grad-CAM. Consistent labellum localization is not established. Taxonomy is empirical TRAIN evidence. These are not final test results. Team freeze, locked evaluation and publication remain pending. Earlier 79.58%/96.86%/39.98% figures belong to a separate policy experiment.\n\n"+text[end:];report.write_text(text)
print('Updated record stages:',stages)
print('Saved separate updated release:',site)
archive=shutil.make_archive('/content/sneha_satyam_updated_website','zip',site)
files.download(archive)
