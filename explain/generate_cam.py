"""Save species/CF/CL Grad-CAM overlays from a local training checkpoint.

Run from the project root with python -m explain.generate_cam --help.
Only load checkpoints you trust. No weights are downloaded by this script.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from matplotlib import colormaps
from torchvision import transforms

from explain.gradcam import gradcam
from models_other.train import MultiTaskModel


def select_layer(model, backbone):
    if backbone == "tiny":
        return "backbone.2", model.backbone[2]
    if backbone == "efficientnet_b0":
        return "backbone.conv_head", model.backbone.conv_head
    if backbone == "convnext_tiny":
        return "backbone.stages.3.blocks.2.conv_dw", model.backbone.stages[-1].blocks[-1].conv_dw
    if backbone == "resnet18":
        return "backbone.layer4.1.conv2", model.backbone.layer4[-1].conv2
    raise ValueError(f"Unsupported backbone: {backbone}")


def generate(checkpoint_path, output_dir, image_path=None, demo=False, device="cpu", alpha=0.45):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["args"]
    dummy = config.get("dummy") is True
    demonstration = dummy or config.get("demo_data") is True
    if demo and not dummy:
        raise ValueError("--demo is only permitted with a dummy-data checkpoint")
    if not demo and image_path is None:
        raise ValueError("Supply --image, or use --demo with a dummy checkpoint")
    if not dummy and (not config.get("mean") or not config.get("std")):
        raise ValueError("Checkpoint is missing training mean/std; cannot safely preprocess this image")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output folder is not empty; choose another name to preserve existing explanations")
    model = MultiTaskModel(checkpoint["backbone"], checkpoint["labels"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    layer_name, layer = select_layer(model, checkpoint["backbone"])
    if demo:
        # Synthetic plumbing check, not a flower photo or scientific result.
        rgb = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        source = Image.fromarray(rgb)
    else:
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    resized = source.resize((224, 224), Image.Resampling.BILINEAR)
    preprocessing = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if not dummy:
        preprocessing.append(transforms.Normalize(config["mean"], config["std"]))
    tensor = transforms.Compose(preprocessing)(source).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions, _ = model(tensor)
    output_dir.mkdir(parents=True, exist_ok=True)
    resized.save(output_dir / "input.png")
    metadata = {"checkpoint": str(Path(checkpoint_path).resolve()), "backbone": checkpoint["backbone"],
                "target_layer": layer_name, "image": str(Path(image_path).resolve()) if image_path else "synthetic noise",
                "synthetic_checkpoint": demonstration, "demo_image": demo,
                "note": "Synthetic checkpoint: visual plumbing only, not a meaningful orchid explanation." if demonstration else
                        "Softmax confidence is uncalibrated. Heatmaps highlight model sensitivity, not proof of botanical correctness.",
                "normalization": None if dummy else {"mean": config["mean"], "std": config["std"]}, "heads": {}}
    base = np.asarray(resized, dtype=np.float32) / 255
    for head in ("species", "CF", "CL"):
        probability = predictions[head].softmax(dim=1)[0]
        index = int(probability.argmax().item())
        heatmap = gradcam(model, tensor, layer, head, index)
        if not np.isfinite(heatmap).all():
            raise ValueError(f"Nonfinite Grad-CAM for {head}")
        colored = colormaps["inferno"](heatmap)[..., :3]
        # A zero map remains the unchanged input rather than a misleading colored overlay.
        weight = alpha * heatmap[..., None]
        overlay = np.clip((1 - weight) * base + weight * colored, 0, 1)
        Image.fromarray((heatmap * 255).round().astype(np.uint8)).save(output_dir / f"{head}_heatmap.png")
        Image.fromarray((overlay * 255).round().astype(np.uint8)).save(output_dir / f"{head}_overlay.png")
        metadata["heads"][head] = {"class_index": index, "label": checkpoint["labels"][head][index],
                                    "uncalibrated_confidence": float(probability[index].item()),
                                    "all_zero_heatmap": bool(heatmap.max() == 0),
                                    "heatmap": f"{head}_heatmap.png", "overlay": f"{head}_overlay.png"}
    (output_dir / "explanation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved input, 3 heatmaps, 3 overlays and explanation.json to {output_dir}")
    if demonstration:
        print("DEMO ONLY: this checkpoint was trained on synthetic data.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Use a train/validation image until the locked test evaluation")
    group.add_argument("--demo", action="store_true", help="Generate a synthetic image; requires a dummy checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()
    generate(args.checkpoint, args.output, args.image, args.demo, args.device, args.alpha)


if __name__ == "__main__":
    main()
