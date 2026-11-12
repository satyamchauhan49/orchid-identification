"""Local B+C integration: image model -> calibration/BN/resolver -> explanations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
from models_other.demo_data import create
from models_other.experiments import run
from explain.from_resolver import convert
from explain.build_records import build
from models_other.release import build as release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/integration_demo_v1")
    a = p.parse_args()
    root = Path(a.output)
    if root.exists() and any(root.iterdir()): raise ValueError("Choose a new integration output folder")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data = create(root/"data", integration=True)
    config = {"mode": "demo", "manifest": str(data/"split_manifest.csv"), "label_maps": str(data/"label_maps.json"),
              "normalization": str(data/"normalization.json"), "data_root": str(data), "output": str(root/"models"),
              "backbones": ["efficientnet_b0"], "learning_rates": [0.0003], "epochs": 1,
              "batch_size": 4, "threads": 2, "pretrained": False}
    config_path = root/"config.json"; config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    comparison = run(config_path)
    chosen = json.loads(comparison.read_text())["selected_run"]
    model = root/"models"/chosen
    base = [sys.executable, "-u", "-m", "bn_resolver.pipeline"]
    shared = ["--maps", config["label_maps"], "--manifest", config["manifest"]]
    artifact, evaluation = root/"frozen.json", root/"validation.json"
    subprocess.run(base+["fit", *shared, "--bundle", str(model/"calib.npz"), "--output", str(artifact)], check=True)
    subprocess.run(base+["evaluate", *shared, "--bundle", str(model/"val.npz"), "--artifact", str(artifact),
                         "--output", str(evaluation)], check=True)
    exchange = convert(config["label_maps"], config["manifest"], model/"val.npz", artifact, evaluation,
                       root/"adapter", model/"best.pt")
    records = build(exchange, root/"adapter/taxonomy.json", data/"reference_images.json", data,
                    root/"records", model/"best.pt")
    release(comparison, records, root/"site")
    print("B+C integration complete:", root/"site/index.html", flush=True)


if __name__ == "__main__": main()
