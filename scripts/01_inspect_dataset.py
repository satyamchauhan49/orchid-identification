import argparse, json
from pathlib import Path
import pandas as pd
from PIL import Image
from collections import Counter

FEATURES = ["T", "LC", "In", "NF", "CL", "CF"]

def read_csv(path):
    df = pd.read_csv(path, sep=";")
    if "Texture" in df.columns and "T" not in df.columns:
        df = df.rename(columns={"Texture": "T"})
    if "T" in df.columns:
        df["T"] = df["T"].replace({"NoSpot": "Nospot"})
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()
    root, out = Path(args.dataset_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image_dir = root / "Orchid_Images"
    image_paths = sorted([x for x in image_dir.iterdir() if x.suffix.lower() in {".jpg",".jpeg",".png",".jfif"}])
    csv_dir = root / "Multi-label Classification"
    split_files = {
        "train": csv_dir / "Multilabel_training.csv",
        "val": csv_dir / "Multilabel_validation.csv",
        "test": csv_dir / "Multilabel_testing.csv",
    }

    report = {"image_count": len(image_paths), "splits": {}, "image_extension_counts": Counter(x.suffix.lower() for x in image_paths)}
    frames = []
    for split, path in split_files.items():
        df = read_csv(path)
        df["split"] = split
        report["splits"][split] = {
            "rows": len(df), "unique_images": int(df["Image_Number"].nunique()),
            "species_count": int(df["Species"].nunique()),
            "missing_values": {k:int(v) for k,v in df.isna().sum().items()},
            "columns": list(df.columns)
        }
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    report["total_csv_rows"] = len(all_df)
    report["unique_csv_images"] = int(all_df["Image_Number"].nunique())
    report["species_count_overall"] = int(all_df["Species"].nunique())
    report["feature_classes"] = {f: sorted(all_df[f].dropna().astype(str).unique().tolist()) for f in FEATURES}
    report["feature_value_counts"] = {f: {str(k):int(v) for k,v in all_df[f].value_counts().items()} for f in FEATURES}
    csv_names = set(all_df["Image_Number"].astype(str))
    disk_names = {x.name for x in image_paths}
    report["csv_images_missing_on_disk"] = sorted(csv_names - disk_names)[:100]
    report["images_not_in_multilabel_csv"] = len(disk_names - csv_names)

    # Verify that each image can be decoded.
    bad = []
    for img in image_paths:
        try:
            with Image.open(img) as im: im.verify()
        except Exception as e:
            bad.append({"file": img.name, "error": str(e)})
    report["unreadable_images"] = bad
    (out / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    all_df.to_csv(out / "combined_labels.csv", index=False)
    print(json.dumps({k:v for k,v in report.items() if k not in {"feature_classes","feature_value_counts","unreadable_images"}}, indent=2))
    print(f"\nSaved: {out/'dataset_audit.json'} and {out/'combined_labels.csv'}")

if __name__ == "__main__":
    main()
