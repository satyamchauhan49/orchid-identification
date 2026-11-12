"""Verify the generated demo artifacts and important data-contract guards."""
import csv
import json
import tempfile
from pathlib import Path
from PIL import Image
from models_other.validate_data import validate
from explain.build_records import probabilities


def main():
    config = json.loads(Path("models_other/config_demo.json").read_text())
    validate(config)
    comparison = json.loads(Path("outputs/demo_grid/comparison.json").read_text())
    assert comparison["mode"] == "demo"
    assert len(comparison["rows"]) == 6
    assert {r["backbone"] for r in comparison["rows"]} == {"efficientnet_b0", "convnext_tiny"}
    assert {r["lr"] for r in comparison["rows"]} == {0.001, 0.0003, 0.0001}
    root = Path("outputs/demo_delivery/site")
    records = json.loads((root / "records/records.json").read_text())
    assert len(records["records"]) == 10
    assert {r["decision"]["stage"] for r in records["records"]} == {"trusted", "auto_resolved", "deferred"}
    for record in records["records"]:
        directory = root / "records" / record["directory"]
        assert record["mode"] == "demo"
        assert len(record["taxonomic"]) == 6
        assert record["visual"]["synthetic_checkpoint"]
        for path in [directory / "input.png", *[directory / p for p in record["visual"]["overlays"].values()],
                     *[directory / r["path"] for r in record["contrastive"]["references"]]]:
            with Image.open(path) as image: image.verify()
    for path in (root / "figures").glob("*.png"):
        with Image.open(path) as image: image.verify()
    assert len(list((root / "figures").glob("*.png"))) == 8
    assert "SYNTHETIC" in (root / "report.md").read_text()
    for bad in ([0.5, 0.6], [-1, 2], [float("nan"), 0]):
        try: probabilities(bad, 2, "negative test")
        except ValueError: pass
        else: raise AssertionError("Invalid probability vector accepted")
    # Exercise a genuine split-leakage failure using a copy of the manifest.
    with Path(config["manifest"]).open() as stream:
        reader = csv.DictReader(stream); fieldnames = reader.fieldnames; rows = list(reader)
    first = next(r for r in rows if r["split"] == "train")
    other = next(r for r in rows if r["split"] == "val")
    other["phash_cluster"] = first["phash_cluster"]
    with tempfile.TemporaryDirectory(dir="outputs") as temp:
        manifest = Path(temp) / "leaky.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        try: validate({**config, "manifest": str(manifest)})
        except ValueError as error: assert "crosses splits" in str(error)
        else: raise AssertionError("Cross-split duplicate cluster was not rejected")
    print("PASS: six grid runs, ten four-layer records, all decision stages, assets, figures, invalid-probability and split-leakage guards")


if __name__ == "__main__": main()
