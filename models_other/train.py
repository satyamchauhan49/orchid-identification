"""Multi-task training starter. Synthetic mode needs no dataset or downloads."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

FEATURES = ("T", "NF", "In", "LC", "CF", "CL")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_labels(path):
    """Proposed schema: each key maps to an ordered list of class names."""
    labels = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("species", *FEATURES):
        values = labels.get(key)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must contain a nonempty ordered list of unique labels")
    if len(labels["species"]) != 156:
        raise ValueError("Real-data mode expects 156 species")
    return labels


class OrchidDataset(Dataset):
    def __init__(self, manifest, root, split, labels, mean, std, training=False):
        import pandas as pd
        frame = pd.read_csv(manifest, dtype=str).fillna("")
        required = {"image_id", "path", "species", "split", *FEATURES}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing manifest columns: {required - set(frame.columns)}")
        if frame.image_id.duplicated().any():
            raise ValueError("Manifest image_id values must be unique")
        self.rows = frame[frame.split == split].to_dict("records")
        if not self.rows:
            raise ValueError(f"No rows for split {split}")
        self.root = Path(root)
        self.maps = {key: {name: i for i, name in enumerate(values)} for key, values in labels.items()}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            *([transforms.RandomHorizontalFlip()] if training else []),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        self.species_targets = [self.encode("species", r["species"]) for r in self.rows]

    def encode(self, key, value):
        if not value:
            return -100  # Missing labels never contribute to loss.
        if value not in self.maps[key]:
            raise ValueError(f"Unknown {key} label {value!r}; check label_maps.json")
        return self.maps[key][value]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            image = self.transform(image.convert("RGB"))
        targets = {key: self.encode(key, row[key]) for key in ("species", *FEATURES)}
        return image, targets, row["image_id"]


class DummyDataset(Dataset):
    def __init__(self, count, labels, seed, split):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, 224, 224, generator=generator)
        self.targets = {k: torch.randint(len(v), (count,), generator=generator) for k, v in labels.items()}
        self.species_targets = self.targets["species"].tolist()
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], {k: v[index] for k, v in self.targets.items()}, f"dummy-{self.split}-{index:04d}"


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, labels, pretrained=False):
        super().__init__()
        if backbone == "tiny":
            self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.ReLU(),
                                          nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
            width = 16
        else:
            import timm
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
            width = self.backbone.num_features
        self.heads = nn.ModuleDict({k: nn.Linear(width, len(v)) for k, v in labels.items()})

    def forward(self, images):
        embedding = self.backbone(images)
        return {k: head(embedding) for k, head in self.heads.items()}, embedding


def task_loss(logits, targets, loss_type, gamma):
    valid = targets != -100
    if not valid.any():
        return None
    logits, targets = logits[valid], targets[valid]
    if loss_type == "focal":
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        return (((1 - torch.exp(-ce)) ** gamma) * ce).mean()
    return nn.functional.cross_entropy(logits, targets, label_smoothing=0.1 if loss_type == "smooth" else 0.0)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for images, targets, _ in loader:
        outputs, _ = model(images.to(device))
        losses = [task_loss(outputs[k], targets[k].to(device), "ce", 2) for k in outputs]
        losses = [value for value in losses if value is not None]
        if losses:
            total += sum(losses).item() * len(images)
            count += len(images)
    if not count:
        raise ValueError("Validation split has no labelled samples")
    return total / count


@torch.no_grad()
def export_logits(model, loader, path, device, labels):
    """Only export raw scores; never calculate test metrics here."""
    model.eval()
    ids, blocks, embeddings = [], {k: [] for k in labels}, []
    for images, _, image_ids in loader:
        outputs, embedding = model(images.to(device))
        ids.extend(image_ids)
        embeddings.append(embedding.cpu().numpy())
        for key, values in outputs.items():
            blocks[key].append(values.cpu().numpy())
    arrays = {"ids": np.asarray(ids), "embeddings": np.concatenate(embeddings)}
    arrays.update({"species_logits" if k == "species" else f"logits_{k}": np.concatenate(v) for k, v in blocks.items()})
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--demo-data", action="store_true", help="Mark a physical synthetic dataset as demonstration only")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--backbone", choices=["tiny", "resnet18", "efficientnet_b0", "convnext_tiny"], default="tiny")
    parser.add_argument("--pretrained", action="store_true", help="Downloads pretrained weights when needed")
    parser.add_argument("--manifest")
    parser.add_argument("--label-maps")
    parser.add_argument("--data-root")
    parser.add_argument("--mean", nargs=3, type=float)
    parser.add_argument("--std", nargs=3, type=float)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", choices=["ce", "smooth", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--sampler", choices=["plain", "balanced"], default="plain")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/sneha_dummy")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    seed_everything(args.seed)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.dummy:
        labels = {"species": [f"species_{i}" for i in range(156)], **{k: ["dummy_0", "dummy_1", "dummy_2"] for k in FEATURES}}
        datasets = {split: DummyDataset(8, labels, args.seed + i, split) for i, split in enumerate(("train", "calib", "val", "test"))}
    else:
        if not all((args.manifest, args.label_maps, args.data_root, args.mean, args.std)):
            parser.error("Real data requires --manifest, --label-maps, --data-root, --mean and --std")
        if args.backbone == "tiny":
            parser.error("Select a real backbone for real data")
        if any(x <= 0 for x in args.std):
            parser.error("std must be positive")
        labels = load_labels(args.label_maps)
        datasets = {s: OrchidDataset(args.manifest, args.data_root, s, labels, args.mean, args.std, s == "train") for s in ("train", "calib", "val", "test")}
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty. Choose a new run name to preserve prior results.")
    output.mkdir(parents=True, exist_ok=True)
    sampler = None
    if args.sampler == "balanced":
        y = np.asarray(datasets["train"].species_targets)
        if (y < 0).any():
            parser.error("Balanced species sampling needs species labels for every training row")
        counts = np.bincount(y, minlength=len(labels["species"]))
        sampler = WeightedRandomSampler(torch.as_tensor(1 / counts[y], dtype=torch.double), len(y), replacement=True)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == "train" and sampler is None,
                              sampler=sampler if s == "train" else None, num_workers=0) for s, d in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(args.backbone, labels, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best = [], float("inf")
    print(f"Device: {device}; synthetic data: {args.dummy or args.demo_data}")
    for epoch in range(args.epochs):
        model.train()
        for images, targets, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                predictions, _ = model(images.to(device))
                losses = [task_loss(predictions[k], targets[k].to(device), args.loss, args.gamma) for k in labels]
                losses = [loss for loss in losses if loss is not None]
                if not losses:
                    continue
                loss = sum(losses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_loss = validation_loss(model, loaders["val"], device)
        history.append({"epoch": epoch + 1, "val_loss": val_loss})
        checkpoint = {"model": model.state_dict(), "backbone": args.backbone, "labels": labels,
                      "args": vars(args), "epoch": epoch + 1}
        torch.save(checkpoint, output / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, output / "best.pt")
        print(f"Epoch {epoch + 1}: validation loss {val_loss:.4f}")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])
    for split in ("calib", "val", "test"):
        export_logits(model, loaders[split], output / f"{split}.npz", device, labels)
    # Validation only: the test exports stay locked.
    from models_other.metrics import classification_metrics
    targets_by_head = {key: [] for key in labels}
    for _, targets, _ in loaders["val"]:
        for key in labels:
            targets_by_head[key].append(targets[key].numpy())
    with np.load(output / "val.npz", allow_pickle=False) as data:
        metrics = {key: classification_metrics(data["species_logits" if key == "species" else f"logits_{key}"],
                                               np.concatenate(targets_by_head[key]), labels[key]) for key in labels}
    (output / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps({**vars(args), "labels": labels}, indent=2), encoding="utf-8")
    print(f"Saved checkpoint and raw-logit exports in {output}. No test metrics computed.")


if __name__ == "__main__":
    main()
