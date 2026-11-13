"""Dataset and label encoding helpers for the Orchid multi-task dataset."""
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

FEATURES = ["T", "NF", "In", "LC", "CF", "CL"]


def build_label_maps(frame: pd.DataFrame):
    # Numeric species IDs are retained in the CSV; model targets are zero-based.
    species_values = sorted(frame["species"].dropna().astype(int).unique().tolist())
    maps = {"species": {str(v): i for i, v in enumerate(species_values)}}
    for col in FEATURES:
        values = sorted(frame[col].dropna().astype(str).unique().tolist())
        maps[col] = {v: i for i, v in enumerate(values)}
    return maps


class OrchidDataset(Dataset):
    def __init__(self, manifest_path, dataset_root, split, label_maps, transform=None):
        self.frame = pd.read_csv(manifest_path)
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.maps = label_maps
        self.transform = transform
        if self.frame.empty:
            raise ValueError(f"No rows found for split={split!r}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.dataset_root / str(row["path"])
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if self.transform:
            image = self.transform(image)
        targets = {"species": self.maps["species"][str(int(row["species"]))]}
        for col in FEATURES:
            targets[col] = self.maps[col][str(row[col])]
        return image, targets, str(row["image_id"])


def save_label_maps(label_maps, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(label_maps, indent=2), encoding="utf-8")
