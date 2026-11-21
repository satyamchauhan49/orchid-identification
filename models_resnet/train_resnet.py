"""Train a ResNet-18 with one 156-class species head and six feature heads.

Example (Google Colab):
python models_resnet/train_resnet.py \
  --dataset-root data/raw/Orchid\ Flowers\ Dataset-v1.1 \
  --manifest outputs/split_manifest_grouped.csv \
  --run-name resnet18_base --epochs 15 --batch-size 32 --pretrained
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import OrchidDataset, FEATURES, build_label_maps, save_label_maps
from model import OrchidMultiTaskResNet18


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_transforms():
    # Dataset statistics are computed from the training partition only.
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_aug = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    ])
    return train_aug, eval_aug


def compute_mean_std(dataset, batch_size, workers, device):
    # Dataset passed here uses deterministic resize/crop and ToTensor only.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == 'cuda'))
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _, _ in tqdm(loader, desc='Train-only mean/std'):
        x = images.to(dtype=torch.float64)
        channel_sum += x.sum(dim=(0,2,3)).cpu()
        channel_sq += (x*x).sum(dim=(0,2,3)).cpu()
        pixel_count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = channel_sum / pixel_count
    std = (channel_sq / pixel_count - mean**2).clamp_min(1e-12).sqrt()
    return mean.float().tolist(), std.float().tolist()


def make_loader(dataset, batch, workers, shuffle, device):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=(device.type == 'cuda'), persistent_workers=(workers > 0))


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    for images, targets, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(images)
                losses = [criterion(outputs['species'], targets['species'])]
                losses += [criterion(outputs[f], targets[f]) for f in FEATURES]
                loss = torch.stack(losses).mean()
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def export_split(model, loader, split, output_dir, device):
    model.eval()
    ids, logits = [], {"species": []}
    logits.update({f: [] for f in FEATURES})
    targets = {"species": []}; targets.update({f: [] for f in FEATURES})
    for images, y, batch_ids in tqdm(loader, desc=f'Export {split}'):
        outputs = model(images.to(device, non_blocking=True))
        ids.extend(batch_ids)
        for key, value in outputs.items(): logits[key].append(value.float().cpu().numpy())
        for key, value in y.items(): targets[key].append(value.numpy())
    payload = {"ids": np.asarray(ids)}
    payload["species_logits"] = np.concatenate(logits['species'])
    payload["targets_species"] = np.concatenate(targets['species'])
    for f in FEATURES:
        payload[f"logits_{f}"] = np.concatenate(logits[f])
        payload[f"targets_{f}"] = np.concatenate(targets[f])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{split}.npz", **payload)
    print(f"Saved {split}.npz with {len(ids)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--manifest', default='outputs/split_manifest_grouped.csv')
    ap.add_argument('--run-name', default='resnet18_base')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--pretrained', action='store_true')
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    manifest = pd.read_csv(args.manifest)
    if manifest['species'].nunique() != 156:
        raise ValueError(f"Expected 156 species; got {manifest['species'].nunique()}")
    label_maps = build_label_maps(manifest)
    n_species = len(label_maps['species'])
    feature_classes = {f: len(label_maps[f]) for f in FEATURES}
    run_dir = Path('outputs') / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_label_maps(label_maps, run_dir / 'class_maps.json')

    train_transform, eval_transform = make_transforms()
    # Compute train-only channel statistics on a deterministic training view.
    stats_dataset = OrchidDataset(args.manifest, args.dataset_root, 'train', label_maps, eval_transform)
    mean, std = compute_mean_std(stats_dataset, args.batch_size, args.workers, device)
    (run_dir / 'normalization.json').write_text(json.dumps({'mean': mean, 'std': std}, indent=2))
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([train_transform, normalize])
    eval_transform = transforms.Compose([eval_transform, normalize])

    datasets = {s: OrchidDataset(args.manifest, args.dataset_root, s, label_maps,
                                 train_transform if s == 'train' else eval_transform)
                for s in ['train', 'calib', 'val', 'test']}
    loaders = {s: make_loader(ds, args.batch_size, args.workers, s == 'train', device)
               for s, ds in datasets.items()}

    model = OrchidMultiTaskResNet18(n_species, feature_classes, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_val, bad_epochs = float('inf'), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders['train'], criterion, optimizer, scaler, device, True)
        val_loss = run_epoch(model, loaders['val'], criterion, optimizer, scaler, device, False)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch:02d}/{args.epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({'model_state_dict': model.state_dict(), 'label_maps': label_maps,
                        'feature_classes': feature_classes, 'num_species': n_species,
                        'mean': mean, 'std': std, 'epoch': epoch}, run_dir / 'best.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print('Early stopping triggered.')
                break
    pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)

    checkpoint = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Save raw logits for C. Test logits are exported but no test metrics are computed here.
    for split in ['calib', 'val', 'test']:
        export_split(model, loaders[split], split, run_dir, device)
    print('Training complete. Best epoch:', checkpoint['epoch'])
    print('Checkpoint and raw logits:', run_dir.resolve())

if __name__ == '__main__':
    main()
