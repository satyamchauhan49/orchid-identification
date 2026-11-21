"""Integration checks using real metadata and untrained checkpoints; no dataset images."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models_resnet.model import OrchidMultiTaskResNet18
from models_other.preprocessing import geometry
from explain.checkpoints import load_checkpoint
from explain.generate_cam import generate
from models_other.train import MultiTaskModel
from bn_resolver.prepare_metadata import canonicalize
from models_other.label_contract import normalize_maps


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.maps = canonicalize(json.loads(Path("outputs/label_maps.json").read_text()))
        self.labels = normalize_maps(self.maps)

    def test_eval_geometry_matches_agrim(self):
        image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (301, 479, 3), dtype=np.uint8))
        expected = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        actual = transforms.Compose([*geometry(), transforms.ToTensor()])(image)
        torch.testing.assert_close(actual, expected(image), rtol=0, atol=0)

    def test_agrim_logits_and_cam(self):
        sizes = {k: len(v) for k,v in self.maps["features"].items()}
        original = OrchidMultiTaskResNet18(156, sizes).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = dict(model_state_dict=original.state_dict(), label_maps={k:{v:i for i,v in enumerate(values)} for k,values in self.labels.items()}, feature_classes=sizes, num_species=156, mean=[.5]*3, std=[.2]*3, epoch=0)
            torch.save(checkpoint, root/"a.pt")
            model, labels, config, *_ = load_checkpoint(root/"a.pt")
            model.eval(); x=torch.rand(1,3,224,224)
            with torch.no_grad():
                expected = original(x); actual, _ = model(x)
            for head in expected:
                torch.testing.assert_close(expected[head],actual[head])
                self.assertEqual(actual[head].shape[1],len(labels[head]))
            image=Image.fromarray(np.random.default_rng(1).integers(0,256,(280,400,3),dtype=np.uint8)); image.save(root/"image.png")
            result=generate(root/"a.pt",root/"cam",root/"image.png")
            self.assertEqual(result["preprocessing"],"agrim_v1")
            for head in ("species","CF","CL"):
                self.assertTrue((root/"cam"/f"{head}_overlay.png").is_file())
            checkpoint["feature_classes"] = {**sizes,"CF":sizes["CF"]+1}
            torch.save(checkpoint,root/"bad.pt")
            with self.assertRaises(ValueError): load_checkpoint(root/"bad.pt")

    def test_legacy_geometry_preserved(self):
        model=MultiTaskModel("tiny",self.labels)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"b.pt"
            torch.save(dict(model=model.state_dict(),backbone="tiny",labels=self.labels,args={"dummy":True}),path)
            _,_,config,*_=load_checkpoint(path)
            self.assertEqual(config["preprocessing"],"legacy_square")


if __name__ == "__main__": unittest.main()
