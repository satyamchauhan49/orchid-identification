"""Create ten explicitly simulated resolver fixtures to exercise explanation formatting."""
import argparse
import csv
import json
from pathlib import Path


def create(data, output):
    data = Path(data)
    labels = json.loads((data / "label_maps.json").read_text())
    with (data / "split_manifest.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "val"][:10]
    def distribution(length, target, confidence):
        values = [(1-confidence)/(length-1)] * length
        values[target] = confidence
        return values
    cases = []
    for i, row in enumerate(rows):
        dnn = labels["species"].index(row["species"])
        stage = ("trusted", "auto_resolved", "deferred")[i % 3]
        bn = dnn if stage == "trusted" else (dnn+1) % 4
        confidence = {"trusted": 0.96, "auto_resolved": 0.88, "deferred": 0.42}[stage]
        cases.append({"image_id": row["image_id"], "path": row["path"], "split": "val",
                      "dnn_probs": distribution(156, dnn, 0.9), "bn_probs": distribution(156, bn, 0.8),
                      "feature_probs": {k: distribution(len(labels[k]), labels[k].index(row[k]), 0.8)
                                        for k in ("T", "NF", "In", "LC", "CF", "CL")},
                      "stage": stage, "fused_confidence": confidence,
                      "final_species": None if stage == "deferred" else labels["species"][dnn if stage == "trusted" else bn],
                      "reason": f"SIMULATED {stage} case for interface demonstration; no resolver was evaluated."})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"mode": "demo", "label_maps": labels, "cases": cases}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", default="data/sneha_demo"); p.add_argument("--output", default="outputs/demo_resolver.json")
    a = p.parse_args(); create(a.data, a.output)
