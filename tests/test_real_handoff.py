import io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from bn_resolver.contracts import FEATURES,load_bundle,read_manifest,digest
from bn_resolver.core import empirical_theta
from bn_resolver.pipeline import write_json
from bn_resolver.export_development import export
from bn_resolver.prepare_metadata import canonicalize
from models_resnet.model import OrchidMultiTaskResNet18
from image_app.server import Predictor

class RealHandoffChecks(unittest.TestCase):
    def test_agrim_export_upload_and_locked_image_not_opened(self):
        import csv
        torch.set_num_threads(2)
        maps=canonicalize(json.loads(Path('outputs/label_maps.json').read_text()))
        labels={'species':maps['species'],**maps['features']}
        sizes={f:len(maps['features'][f]) for f in FEATURES}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp=root/'maps.json';write_json(mp,maps)
            model=OrchidMultiTaskResNet18(156,sizes).eval()
            ck=dict(model_state_dict=model.state_dict(),label_maps={k:{v:i for i,v in enumerate(vs)} for k,vs in labels.items()},num_species=156,feature_classes=sizes,mean=[.5]*3,std=[.2]*3,epoch=0)
            torch.save(ck,root/'untrained.pt')
            rows=[]
            for split in ('train','calib','val','test'):
                row=dict(image_id=split,path=split+'.png',species=maps['species'][0],phash_cluster=split,split=split,**{f:maps['features'][f][0] for f in FEATURES})
                rows.append(row)
                if split!='test':Image.new('RGB',(330,250),'green').save(root/row['path'])
            manifest=root/'manifest.csv'
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            ledger=export(mp,manifest,root/'untrained.pt',root,root/'exports',1)
            self.assertFalse(ledger['test_opened']);self.assertFalse((root/'exports/test.npz').exists())
            ids,arrays=load_bundle(root/'exports/calib.npz',rows,maps,'calib',digest(mp))
            artifact=dict(label_maps_sha256=digest(mp),manifest_sha256=digest(manifest),cal_model_ids=['calib'],cal_policy_ids=[],
                prior=[1/156]*156,theta={f:(np.ones((156,k))/k).tolist() for f,k in sizes.items()},
                feature_priors={f:[1/k]*k for f,k in sizes.items()},matrices={},variant='direct_soft',missing_confidence=.4,
                temperatures={k:{'temperature':1} for k in labels},policy={'weight':.5,'threshold':None})
            write_json(root/'artifact.json',artifact)
            predictor=Predictor(mp,root/'untrained.pt',root/'artifact.json',root/'exports/calib.npz')
            self.assertEqual(predictor.preprocessing,'agrim_v1')
            result=predictor.infer((root/'val.png').read_bytes())
            self.assertIsNone(result['accepted_species'])
            # Direct loader and exported arrays agree for the same development pixels.
            with Image.open(root/'calib.png') as im:x=predictor.transform(im.convert('RGB')).unsqueeze(0)
            with torch.inference_mode():expected,_=predictor.model(x)
            np.testing.assert_allclose(arrays['species'],expected['species'].numpy(),rtol=1e-5,atol=1e-6)
            bad=json.loads(mp.read_text());bad['features']['CF'].reverse();write_json(root/'wrong.json',bad)
            with self.assertRaisesRegex(ValueError,'order'):export(root/'wrong.json',manifest,root/'untrained.pt',root,root/'bad')

if __name__=='__main__':unittest.main()
