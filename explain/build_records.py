"""Build four-layer explanations from an explicit resolver exchange file."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from explain.generate_cam import generate


def probabilities(values, size, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all() or (array < 0).any() or not np.isclose(array.sum(), 1, atol=1e-5):
        raise ValueError(f"Invalid normalized probability vector: {name}")
    return array


def build(input_path, taxonomy_path, references_path, assets_root, output, checkpoint=None, locked_test=False):
    source = json.loads(Path(input_path).read_text(encoding="utf-8"))
    from models_other.label_contract import normalize_maps
    labels = normalize_maps(source["label_maps"])
    demo = source["mode"] == "demo"
    if source["mode"] not in ("demo", "real"):
        raise ValueError("mode must be demo or real")
    taxonomy = json.loads(Path(taxonomy_path).read_text())
    references = json.loads(Path(references_path).read_text())
    output, assets_root = Path(output), Path(assets_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Explanation output folder must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results, seen = [], set()
    for index, item in enumerate(source["cases"]):
        if item["image_id"] in seen:
            raise ValueError("Duplicate image_id in resolver exchange")
        seen.add(item["image_id"])
        if item["split"] not in ("train", "calib", "val", "test"):
            raise ValueError("Unknown split")
        if not demo and item["split"] == "test" and not locked_test:
            raise ValueError("Real test explanations require the post-Sync-4 --locked-test-approved flag")
        if item["stage"] not in ("trusted", "auto_resolved", "deferred"):
            raise ValueError("Unknown resolver stage")
        dnn = probabilities(item["dnn_probs"], len(labels["species"]), "dnn")
        bn = probabilities(item["bn_probs"], len(labels["species"]), "bn")
        candidates = [labels["species"][int(x.argmax())] for x in (dnn, bn)]
        confidence = float(item["fused_confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("fused_confidence must be between zero and one")
        directory = output / f"case_{index:02d}"
        directory.mkdir()
        image_path = assets_root / item["path"]
        shutil.copyfile(image_path, directory / "input.png")
        signals = []
        for feature in ("T", "NF", "In", "LC", "CF", "CL"):
            evidence = probabilities(item["feature_probs"][feature], len(labels[feature]), feature)
            observed = labels[feature][int(evidence.argmax())]
            distributions = [taxonomy.get(candidate, {}).get(feature) for candidate in candidates]
            # Total variation between expected feature distributions ranks contrasts.
            difference = None
            if all(distributions):
                a, b = [np.array([dist.get(label, 0) for label in labels[feature]], dtype=float) for dist in distributions]
                probabilities(a, len(labels[feature]), "taxonomy")
                probabilities(b, len(labels[feature]), "taxonomy")
                difference = float(np.abs(a - b).sum() / 2)
            signals.append({"feature": feature, "observed_value": observed,
                            "confidence": float(evidence.max()), "unobserved": bool(evidence.max() < 0.4),
                            "expected_dnn": distributions[0], "expected_bn": distributions[1],
                            "candidate_difference": difference})
        contrast = sorted(signals, key=lambda x: x["candidate_difference"] if x["candidate_difference"] is not None else -1, reverse=True)
        reference_files = []
        for role, candidate in zip(("dnn", "bn"), candidates):
            path = references.get(candidate)
            if path and (assets_root / path).is_file():
                destination = f"reference_{role}" + Path(path).suffix
                shutil.copyfile(assets_root / path, directory / destination)
                reference_files.append({"candidate": candidate, "path": destination, "status": "available"})
            else:
                reference_files.append({"candidate": candidate, "path": None, "status": "reference not supplied"})
        visual = {"status": "checkpoint not supplied", "overlays": {}}
        if checkpoint:
            cam = generate(checkpoint, directory / "cam", image_path)
            visual = {"status": "available", "synthetic_checkpoint": cam["synthetic_checkpoint"],
                      "overlays": {head: f"cam/{head}_overlay.png" for head in ("species", "CF", "CL")},
                      "cam_predictions": cam["heads"]}
            # Prevent accidentally illustrating a real resolver record with another model.
            if not demo or source.get("provenance", {}).get("resolver") == "c-resolver":
                for head, expected in (("species", candidates[0]), *[(s["feature"], s["observed_value"]) for s in signals if s["feature"] in ("CF", "CL")]):
                    if cam["heads"][head]["label"] != expected:
                        raise ValueError("Resolver labels differ from checkpoint predictions; confirm model/normalization/class order")
        record = {"image_id": item["image_id"], "mode": source["mode"], "split": item["split"],
                  "image": "input.png", "dnn_candidate": candidates[0], "bn_candidate": candidates[1],
                  "taxonomic": signals, "contrastive": {"signals": contrast, "references": reference_files},
                  "visual": visual, "decision": {"stage": item["stage"], "fused_confidence": confidence,
                    "final_species": item.get("final_species"), "reason": item["reason"],
                    "source": "c-resolver on synthetic model outputs" if source.get("provenance", {}).get("resolver") == "c-resolver" and demo else "simulated resolver; not evaluated" if demo else "supplied by Satyam's resolver"},
                  "directory": directory.name}
        (directory / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        results.append(record)
    (output / "records.json").write_text(json.dumps({"mode": source["mode"], "provenance": source.get("provenance", {}), "records": results}, indent=2), encoding="utf-8")
    return output / "records.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "taxonomy", "references", "assets-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--locked-test-approved", action="store_true")
    a = p.parse_args()
    build(a.input, a.taxonomy, a.references, a.assets_root, a.output, a.checkpoint, a.locked_test_approved)
