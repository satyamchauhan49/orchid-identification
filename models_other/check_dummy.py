"""Validate synthetic exports and Grad-CAM. Never use this on real test exports."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models_other.train import FEATURES, MultiTaskModel
from explain.gradcam import gradcam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "run_config.json").read_text())
    if config.get("dummy") is not True:
        raise ValueError("This check may only open synthetic-data outputs")
    for split in ("calib", "val", "test"):
        with np.load(args.run / f"{split}.npz", allow_pickle=False) as data:
            assert len(set(data["ids"])) == 8
            assert all(str(x).startswith(f"dummy-{split}-") for x in data["ids"])
            for key in ("species", *FEATURES):
                values = data["species_logits" if key == "species" else f"logits_{key}"]
                assert values.shape == (8, len(config["labels"][key]))
                assert np.isfinite(values).all()
    model = MultiTaskModel("tiny", config["labels"])
    model.load_state_dict(torch.load(args.run / "best.pt", weights_only=True)["model"])
    for head in ("species", "CF", "CL"):
        heatmap = gradcam(model, torch.rand(1, 3, 224, 224), model.backbone[2], head)
        assert heatmap.shape == (224, 224) and np.isfinite(heatmap).all()
        assert heatmap.min() >= 0 and heatmap.max() <= 1
    print("PASS: synthetic export shapes, IDs, finite logits and Grad-CAM for species/CF/CL")


if __name__ == "__main__":
    main()
