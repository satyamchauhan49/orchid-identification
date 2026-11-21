"""Check a trusted checkpoint's class order, normalization and seven output heads.

Artificial inputs test execution only; they do not establish accuracy or CAM quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from explain.checkpoints import load_checkpoint
from models_other.label_contract import read_maps


def check(checkpoint, maps_path, output):
    destination = Path(output)
    if destination.exists():
        raise ValueError("Choose a new report path; existing reports are preserved")
    model, labels, config, backbone, layer, _ = load_checkpoint(checkpoint)
    expected = read_maps(maps_path)
    if labels != expected:
        raise ValueError("Checkpoint class order differs from canonical label maps")
    model.eval()
    with torch.no_grad():
        logits, embedding = model(torch.zeros(2, 3, 224, 224))
    if set(logits) != set(expected):
        raise ValueError("Checkpoint output heads differ from metadata")
    for head, values in logits.items():
        if tuple(values.shape) != (2, len(expected[head])) or not torch.isfinite(values).all():
            raise ValueError(f"Invalid output shape or nonfinite logits for {head}")
    if embedding.ndim != 2 or embedding.shape[0] != 2 or not torch.isfinite(embedding).all():
        raise ValueError("Invalid embedding")
    report = dict(status="passed", scope="software compatibility on artificial inputs only",
        real_image_validation=False, accuracy_evaluated=False, backbone=backbone,
        preprocessing=config["preprocessing"], target_layer=layer,
        normalization={k:config.get(k) for k in ("mean","std")},
        output_shapes={k:list(v.shape) for k,v in logits.items()},
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        label_maps_sha256=hashlib.sha256(Path(maps_path).read_bytes()).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","label-maps","output"): parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(check(args.checkpoint,args.label_maps,args.output),indent=2))


if __name__ == "__main__": main()
