import tempfile,unittest,json
from pathlib import Path
import torch
from models_resnet.model_hsv_colab import OrchidHSVResNet18
from explain.checkpoints import load_checkpoint
from bn_resolver.prepare_metadata import canonicalize

class HSVChecks(unittest.TestCase):
    def test_exact_heads_and_embedding_with_48_color_inputs(self):
        torch.set_num_threads(2)
        m=canonicalize(json.loads(Path('outputs/label_maps.json').read_text()));labels={'species':m['species'],**m['features']}
        sizes={k:len(v) for k,v in m['features'].items()}
        model=OrchidHSVResNet18(156,sizes,False,[.5]*3,[.2]*3).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'hsv.pt'
            torch.save(dict(model_state_dict=model.state_dict(),label_maps={k:{v:i for i,v in enumerate(vs)} for k,vs in labels.items()},num_species=156,feature_classes=sizes,mean=[.5]*3,std=[.2]*3),p)
            wrapped,_,_,name,*_=load_checkpoint(p);wrapped.eval()
            x=torch.rand(1,3,224,224)
            with torch.inference_mode():expected=model(x);actual,embedding=wrapped(x)
            self.assertEqual(name,'agrim_hsv_resnet18');self.assertEqual(embedding.shape,(1,512))
            for head in expected:torch.testing.assert_close(expected[head],actual[head])

if __name__=='__main__':unittest.main()
