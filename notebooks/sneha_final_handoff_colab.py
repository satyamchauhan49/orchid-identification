
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
print('Select Sneha_Final_Explanation_Handoff.zip from Downloads.')
uploaded=files.upload()
archives=[Path(n).resolve() for n in uploaded if n.lower().endswith('.zip')]
assert len(archives)==1, 'Select exactly one handoff ZIP.'
del uploaded
run=base/('final_explanations_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
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
for key,name in [('artifact_sha256','frozen_direct_soft.json'),('label_maps_sha256','label_maps.json')]:
    assert digest(handoff/name)==exchange['provenance'][key],key
assert digest(handoff/'split_manifest.csv')==digest(config['manifest'])
manifest=pd.read_csv(handoff/'split_manifest.csv',dtype=str);lookup=manifest.set_index('image_id')
for case in exchange['cases']:
    assert case['split']=='test' and lookup.loc[case['image_id'],'split']=='test'
    assert lookup.loc[case['image_id'],'path']==case['path']
    assert (Path(config['data_root'])/case['path']).is_file()
references=(manifest[manifest['split']=='train'].sort_values('image_id').drop_duplicates('species').set_index('species')['path'].to_dict())
reference_path=run/'reference_images.json';reference_path.write_text(json.dumps(references,indent=2))
def execute(arguments):
    result=subprocess.run([sys.executable,'-u','-m',*arguments],cwd=project,capture_output=True,text=True)
    print(result.stdout);print(result.stderr);result.check_returncode()
execute(['explain.build_records','--input',str(handoff/'exchange.json'),'--taxonomy',str(handoff/'taxonomy.json'),
         '--references',str(reference_path),'--assets-root',config['data_root'],'--checkpoint',str(model/'best.pt'),
         '--output',str(run/'records'),'--locked-test-approved'])
records=json.loads((run/'records/records.json').read_text())
assert len(records['records'])==10
stages=dict(Counter(r['decision']['stage'] for r in records['records']))
assert set(stages)=={'trusted','auto_resolved','deferred'},stages
execute(['models_other.release','--comparison',str(base/'real_grid_v1/comparison.json'),
         '--records',str(run/'records/records.json'),'--output',str(run/'site')])

# Use the supplied locked results. No pipeline fit/evaluate or rescoring occurs.
site=run/'site'
final=json.loads((handoff/'final_results.json').read_text())
freeze=json.loads((handoff/'final_results.json.freeze.json').read_text())
assert final['split']=='test' and final['n']==795
assert final['artifact_sha256']==exchange['provenance']['artifact_sha256']
assert final['bundle_sha256']==freeze['test_bundle_sha256']==exchange['provenance']['bundle_sha256']
assert freeze['policy_refitted'] is False
for name in ['final_results.json','final_results.json.freeze.json','SHA256SUMS.json','READ_ME_FIRST.txt','species_id_to_name.json']:
    shutil.copyfile(handoff/name,site/name)
rows=[]
for key,label in [('dnn','Calibrated ConvNeXt'),('bn','Empirical BN'),('rank_rule','Rank rule'),('fusion','Full-coverage fusion'),('selective','Selective frozen policy')]:
    m=final[key]
    rows.append('<tr><td>'+label+'</td>'+''.join('<td>'+format(100*m[k],'.2f')+'%</td>' for k in ['accuracy','coverage','deferral_rate','accepted_accuracy'])+'</tr>')
section='<section id="final-test"><h2>Final locked test evaluation</h2><p>795 held-out test images; model and policy selected before test. Backbone search below remains validation-only.</p><div class="scroll"><table><thead><tr><th>Method</th><th>Proposed accuracy, all images</th><th>Coverage</th><th>Deferral</th><th>Accepted accuracy</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p>641/795 full-coverage predictions were correct. The selective policy accepted 574 images (541 correct, 33 errors), deferring 221. Accepted error was 5.75%, above the 1.5% target. Fusion weight 1.0 gives identical DNN and fusion predictions; no BN accuracy gain was demonstrated.</p><p>Grouped 95% CI: overall accuracy 77.90–83.48%; accepted accuracy 92.09–96.25%. Macro-F1 0.5502 uses all 156 class slots; test contains 135 species. Auto-resolved is a disagreement tag, not evidence of a DNN correction. No deferred case is credited as human-corrected.</p><p><a href="final_results.json">Final results JSON</a></p></section>'
page=site/'index.html';html=page.read_text()
html=html.replace('<section id="results">',section+'<section id="results">')
html=html.replace('Read Sneha’s generated report →','Read the generated report')
# Override the legacy validation-build messages only for this final artifact.
overrides={
 'notice':'FINAL LOCKED TEST RESULTS — 795 images; validation search shown separately',
 'context':'Frozen ConvNeXt-Tiny LR 0.0001. Final metrics are supplied by the one locked evaluation. The comparison grid below is validation-only; examples below are actual test decisions.',
 'example-note':'Ten actual final-test cases: four trusted, five deferred and one auto-resolved. No invented stage coverage. Maps explain model sensitivity, not verified botanical correctness.',
 'selection':'The team selected ConvNeXt-Tiny LR 0.0001 before test using development comparisons. Final test results were not used to select or retune the model.',
 'limits':'The 1.5% accepted-error target was missed. BN fusion did not improve classification. Labellum localization is not consistently established. Empirical taxonomy, rare classes and fixed-policy uncertainty limit conclusions.',
 'footer-mode':'Local final-results build. Publication and final team review remain separate steps.'}
script='<script>const FINAL_MESSAGES='+json.dumps(overrides).replace('<','\\u003c')+';for(const [id,value] of Object.entries(FINAL_MESSAGES)){document.getElementById(id).textContent=value;}document.querySelector(".pending").textContent="Final locked metrics appear above; the six-run backbone grid remains validation-only.";</script>'
html=html.replace('</body>',script+'</body>');page.write_text(html)
report=site/'report.md';text=report.read_text()
text=text.replace('Validation results. Locked-test analysis remains a separate team stage.','Final locked evaluation supplied by Satyam; backbone search remains validation-only.')
start=text.index('## Limitations and remaining team work');end=text.index('## Input provenance',start)
summary=['## Final locked test results','', '| Method | Proposed accuracy | Coverage | Deferral | Accepted accuracy |','|---|---:|---:|---:|---:|']
for key in ['dnn','bn','rank_rule','fusion','selective']:
    m=final[key];summary.append('| '+key+' | '+' | '.join(format(100*m[k],'.2f')+'%' for k in ['accuracy','coverage','deferral_rate','accepted_accuracy'])+' |')
summary+=['','Model/policy frozen before test. 641/795 full-coverage predictions correct; 541/574 accepted predictions correct; 221 deferred. Test macro-F1 0.5502 over all 156 class slots, zero-division 0. No inference of human correction for deferred cases.','', '## Limitations and remaining team work','The accepted-error target was missed (5.75% observed versus 1.5% intended). Weight 1.0 means fusion equals the DNN; no BN accuracy gain is demonstrated. Auto-resolved stage labels do not establish corrections. Grouped CIs assume independent groups and a fixed selected model/policy. No general superiority over the paper is established.','', 'Ten actual test cases have checkpoint-matched Grad-CAM and training-only references. Visual inspection is still required; consistent anatomical labellum localization cannot be assumed. Taxonomy is empirical TRAIN data, not expert annotation. Team review and publication remain pending.','']
text=text[:start]+chr(10).join(summary)+chr(10)+text[end:]
report.write_text(text)
print('Final example stages:',stages)
print('Saved final website:',site)
archive=shutil.make_archive('/content/sneha_final_website','zip',site)
files.download(archive)
