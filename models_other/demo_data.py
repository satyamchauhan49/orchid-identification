"""Generate a small explicitly synthetic image dataset with the real input contract."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

FEATURES = ("T", "NF", "In", "LC", "CF", "CL")


def create(root, integration=False):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty demo data folder")
    (root / "images").mkdir(parents=True)
    labels = {"species": [f"DEMO_species_{i:03d}" for i in range(156)],
              **{f: [f"DEMO_{f}_{i}" for i in range(3)] for f in FEATURES}}
    if integration:
        labels = {k: [v.replace("DEMO_", "SYNTHETIC_") for v in values] for k, values in labels.items()}
    rng = np.random.default_rng(42)
    rows, reference = [], {}
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for split, n in (("train", 16), ("calib", 128 if integration else 12), ("val", 12), ("test", 12)):
        for i in range(n):
            species = i % 4
            array = rng.integers(12, 48, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            colors = [(224, 112, 181), (248, 203, 85), (119, 178, 235), (193, 129, 224)]
            dx, dy = int(rng.integers(-12, 12)), int(rng.integers(-12, 12))
            for x, y in ((60, 55), (110, 40), (145, 70), (80, 105)):
                draw.ellipse((x+dx, y+dy, x+dx+55, y+dy+65), fill=colors[species])
            draw.ellipse((94+dx, 107+dy, 135+dx, 174+dy), fill=(238, 238, 219))
            draw.text((8, 205), "SYNTHETIC DEMO", fill="white")
            identifier = f"demo-{split}-{i:03d}"
            relative = f"images/{identifier}.png"
            image.save(root / relative)
            features = {f: labels[f][(species+j) % 3] for j, f in enumerate(FEATURES)}
            rows.append({"image_id": identifier, "path": relative, "species": labels["species"][species],
                         **features, "phash_cluster": identifier, "split": split})
            if split == "train":
                reference.setdefault(labels["species"][species], relative)
                pixels = np.asarray(image, dtype=float).reshape(-1, 3) / 255
                sums += pixels.sum(0)
                squares += (pixels ** 2).sum(0)
                count += len(pixels)
    mean = sums / count
    std = np.sqrt(squares / count - mean ** 2)
    with (root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    taxonomy = {labels["species"][s]: {f: {labels[f][(s+j) % 3]: 1.0} for j, f in enumerate(FEATURES)} for s in range(4)}
    exported_maps = {"species": labels["species"], "features": {f: labels[f] for f in FEATURES}} if integration else labels
    for name, value in (("label_maps.json", exported_maps), ("taxonomy.json", taxonomy),
                        ("reference_images.json", reference), ("normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "source": "train only", "demo": True})):
        (root / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/sneha_demo")
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    create(args.output, args.integration)
