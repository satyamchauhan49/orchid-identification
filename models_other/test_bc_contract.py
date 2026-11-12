"""Integration regression tests: use the completed local synthetic B+C run."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from models_other.label_contract import normalize_maps
from explain.from_resolver import convert


class BCContractTests(unittest.TestCase):
    root = Path("outputs/integration_demo_v1")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def inputs(self, evaluation=None, checkpoint=None):
        model = self.root/"models/efficientnet_b0_lr0.0003"
        return [self.root/"data/label_maps.json", self.root/"data/split_manifest.csv", model/"val.npz",
                self.root/"frozen.json", evaluation or self.root/"validation.json", self.work/"adapted",
                checkpoint or model/"best.pt"]

    def test_nested_and_legacy_maps_preserve_order(self):
        raw = json.loads((self.root/"data/label_maps.json").read_text())
        flat = normalize_maps(raw)
        self.assertEqual(normalize_maps(flat), flat)
        self.assertEqual(flat["CF"], raw["features"]["CF"])
        raw["CF"] = list(reversed(flat["CF"]))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            normalize_maps(raw)

    def test_checkpoint_mismatch_is_rejected(self):
        wrong = self.work/"wrong.pt"; wrong.write_bytes(b"not the training checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
            convert(*self.inputs(checkpoint=wrong))

    def copy_evaluation(self):
        target = self.work/"evaluation.json"
        for suffix in ("", ".decisions.json", ".posteriors.npz"):
            shutil.copyfile(str(self.root/"validation.json")+suffix, str(target)+suffix)
        return target

    def test_posteriors_are_joined_by_id_not_row_position(self):
        evaluation = self.copy_evaluation()
        path = str(evaluation)+".posteriors.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        for key in ("ids", "dnn", "bn", "fused"): arrays[key] = arrays[key][::-1]
        np.savez_compressed(path, **arrays)
        exchange = convert(*self.inputs(evaluation=evaluation))
        result = json.loads(exchange.read_text())
        self.assertEqual(len(result["cases"]), 10)
        self.assertEqual(result["provenance"]["resolver"], "c-resolver")

    def test_altered_decisions_are_rejected(self):
        evaluation = self.copy_evaluation()
        path = Path(str(evaluation)+".decisions.json")
        records = json.loads(path.read_text()); records[0]["confidence"] = 0.999999
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "Decision differs"):
            convert(*self.inputs(evaluation=evaluation))


if __name__ == "__main__": unittest.main()
