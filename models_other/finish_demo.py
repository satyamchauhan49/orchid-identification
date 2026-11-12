"""One command to finish the local demo after the six-run grid."""
import argparse
import hashlib
import json
from pathlib import Path
from models_other.validate_data import validate
from models_other.demo_records import create
from explain.build_records import build as build_records
from models_other.release import build as build_release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="models_other/config_demo.json")
    p.add_argument("--output", default="outputs/demo_delivery")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    if config["mode"] != "demo": raise ValueError("finish_demo accepts only demo configs")
    validate(config)
    comparison_path = Path(config["output"]) / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    for key, digest in comparison["input_sha256"].items():
        if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != digest:
            raise ValueError("Grid inputs changed; run a new grid")
    root = Path(a.output)
    if root.exists() and any(root.iterdir()): raise ValueError("Choose a new output folder")
    root.mkdir(parents=True, exist_ok=True)
    create(config["data_root"], root / "resolver_demo.json")
    checkpoint = Path(config["output"]) / comparison["selected_run"] / "best.pt"
    records = build_records(root / "resolver_demo.json", Path(config["data_root"])/"taxonomy.json",
                            Path(config["data_root"])/"reference_images.json", config["data_root"], root/"records", checkpoint)
    build_release(comparison_path, records, root/"site")
    print("Demo complete. Open", root/"site/index.html")


if __name__ == "__main__": main()
