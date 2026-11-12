"""Run Sneha's two-backbone LR grid using one shared manifest. No test scoring."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from models_other.train import FEATURES


def run(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    from models_other.validate_data import validate
    validate(config)
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "comparison.json").exists():
        raise ValueError("This grid is already complete; use a new output directory")
    normalization = json.loads(Path(config["normalization"]).read_text())
    rows, details = [], []
    for backbone in config.get("backbones", ["efficientnet_b0", "convnext_tiny"]):
        for lr in config.get("learning_rates", [0.001, 0.0003, 0.0001]):
            name = f"{backbone}_lr{lr:g}"
            output = root / name
            command = [sys.executable, "-u", "-m", "models_other.train", "--backbone", backbone,
                       "--manifest", config["manifest"], "--label-maps", config["label_maps"],
                       "--data-root", config["data_root"], "--mean", *map(str, normalization["mean"]),
                       "--std", *map(str, normalization["std"]), "--epochs", str(config["epochs"]),
                       "--batch-size", str(config.get("batch_size", 4)), "--lr", str(lr),
                       "--loss", config.get("loss", "ce"), "--sampler", config.get("sampler", "plain"),
                       "--threads", str(config.get("threads", 2)), "--output", str(output)]
            for flag in ("pretrained", "amp"):
                if config.get(flag): command.append("--" + flag)
            if config.get("mode") == "demo": command.append("--demo-data")
            print(f"Starting {name}", flush=True)
            # Never silently reuse a partial or stale experiment.
            subprocess.run(command, check=True)
            metrics = json.loads((output / "val_metrics.json").read_text())
            history = json.loads((output / "history.json").read_text())
            row = {"run": name, "backbone": backbone, "lr": lr, "mode": config["mode"],
                   "species_macro_f1": metrics["species"]["macro_f1"],
                   "species_top1": metrics["species"]["top_k"]["1"],
                   "feature_macro_f1": sum(metrics[f]["macro_f1"] for f in FEATURES if metrics[f]["labelled_count"]) /
                       max(1, sum(bool(metrics[f]["labelled_count"]) for f in FEATURES)),
                   "best_val_loss": min(h["val_loss"] for h in history)}
            rows.append(row)
            details.append({"run": name, "metrics": metrics})
    # Predeclared validation criterion: feature macro-F1, then validation loss.
    selected = max(rows, key=lambda r: (r["feature_macro_f1"], -r["best_val_loss"]))
    hashes = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() for key in ("manifest", "label_maps", "normalization")}
    payload = {"mode": config["mode"], "selection_rule": "highest mean feature macro-F1, then lowest validation loss",
               "selected_run": selected["run"], "rows": rows, "details": details, "input_sha256": hashes,
               "note": "Synthetic demonstration only" if config["mode"] == "demo" else "Validation results only; no test metrics"}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Grid complete: {root / 'comparison.csv'}", flush=True)
    return root / "comparison.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
