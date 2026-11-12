"""Preflight the shared manifest without scoring or reading test image pixels."""
import argparse
import csv
import json
from pathlib import Path
from models_other.train import FEATURES, load_labels


def validate(config):
    if config.get("mode") not in ("demo", "real"):
        raise ValueError("Config mode must be demo or real")
    labels = load_labels(config["label_maps"])
    norm = json.loads(Path(config["normalization"]).read_text())
    if config["mode"] == "real" and norm.get("demo"):
        raise ValueError("Demo normalization cannot be used for real training")
    import math
    if any(len(norm.get(k, [])) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in norm[k]) for k in ("mean", "std")) or any(x <= 0 for x in norm["std"]):
        raise ValueError("Normalization must contain three finite mean/std values; std must be positive")
    with Path(config["manifest"]).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "path", "species", "phash_cluster", "split", *FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest missing columns: " + str(required-set(reader.fieldnames or [])))
        rows = list(reader)
    ids, clusters, splits = set(), {}, {s: 0 for s in ("train", "calib", "val", "test")}
    root = Path(config["data_root"]).resolve()
    for row in rows:
        if not row["image_id"] or row["image_id"] in ids:
            raise ValueError("Empty/duplicate image_id")
        ids.add(row["image_id"])
        split = row["split"]
        if split not in splits: raise ValueError("Unknown split " + split)
        splits[split] += 1
        cluster = row["phash_cluster"]
        if not cluster: raise ValueError("Missing pHash cluster")
        if cluster in clusters and clusters[cluster] != split:
            raise ValueError("A duplicate cluster crosses splits")
        clusters[cluster] = split
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing/out-of-root image " + str(path))
        for key in ("species", *FEATURES):
            if row[key] and row[key] not in labels[key]:
                raise ValueError(f"Unknown {key} label {row[key]}")
    if not all(splits.values()): raise ValueError("Every split must contain images")
    if not any(r["species"] for r in rows if r["split"] == "val"):
        raise ValueError("Validation needs species labels for comparison")
    print("Preflight passed:", splits, "(no test pixels or scores read)")
    return splits


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", required=True)
    validate(json.loads(Path(p.parse_args().config).read_text()))
