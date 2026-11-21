# Real checkpoint handoff: ready to run after training

Prepared from integration-b-c d7d6f6b and reviewed against Agrim's a-data e36dd98.
Agrim's new commit adds original-dataset EDA and explicitly says verified baseline
metrics are not available. EDA is descriptive, not a trained-model result.

## Completed locally

- Upload uses Sneha's shared trusted checkpoint loader for both A and B formats.
- Agrim-format checkpoints use Resize(256), CenterCrop(224), stored TRAIN mean/std;
  older B checkpoints retain legacy_square geometry. Label order is checked.
- New C-owned export_development command generates fresh calib/val logits directly
  from the checkpoint and images. It records exact canonical-map, manifest and
  checkpoint hashes; it does not retrofit hashes onto existing arrays.
- Only development image paths are opened. No test export argument exists.
- Test uses an untrained Agrim-format checkpoint and artificial images: confirms
  direct/exported logits agree, upload works, wrong class order fails and missing
  test pixels do not block development export. No real accuracy conclusion follows.

## Commands for the team

First generate canonical maps and TRAIN-only metadata from Agrim's flat map:

```sh
python -m bn_resolver.prepare_metadata --maps outputs/label_maps.json --manifest outputs/split_manifest_grouped.csv --output work/real-metadata-v2 --source-revision e36dd98
```

After a trusted ResNet checkpoint is trained, run fresh exports (new output directory):

```sh
python -m bn_resolver.export_development --maps work/real-metadata-v2/label_maps.json --manifest outputs/split_manifest_grouped.csv --checkpoint outputs/resnet18_base/best.pt --image-root "data/raw/Orchid Flowers Dataset-v1.1" --output outputs/resnet18_base_verified_dev --batch-size 16
```

The exact source checkpoint determines inference; no training or model downloads
occur. Species remain string numeric IDs with a separate species_names.json sidecar.
Do not rename species or reorder feature columns to prettify display labels.

Then validate and run C's reviewed E3-E5 suite:

```sh
python -m bn_resolver.audit --maps work/real-metadata-v2/label_maps.json --manifest outputs/split_manifest_grouped.csv --calib outputs/resnet18_base_verified_dev/calib.npz --val outputs/resnet18_base_verified_dev/val.npz --output work/resnet18-input-audit.json
python -m eval.experiments --maps work/real-metadata-v2/label_maps.json --manifest outputs/split_manifest_grouped.csv --calib outputs/resnet18_base_verified_dev/calib.npz --val outputs/resnet18_base_verified_dev/val.npz --output work/resnet18-real-experiments-01
```

For uploads use image_app.server with those maps, matching best.pt, chosen frozen
artifact and matching calib.npz. This now accepts A/B formats. Upload output displays
the trained IDs/candidates; numerical IDs require the species_names sidecar for a
future human-name display. No dummy artifact should be paired with a real checkpoint.
The default UI remains explicit about synthetic checkpoints; random untrained
compatibility checkpoints are for tests only and must never be presented as trained.

## Remaining real dependencies

Actual dataset pixels and a trained checkpoint are not available in this checkout.
No real exports, BN baseline performance, calibration feedback or E3-E5 findings
were fabricated. Agrim can train now; use this tool to supply the required verified
exports afterward. Final test remains locked until team agreement and model freeze.

## Boundaries

Source: new real-handoff-local branch/worktree from the latest combined branch.
No edits to Agrim's training code, no auto-commit, no push, no main merge. All generated
checkpoints/arrays from tests are temporary. Prior metadata work on c-resolver and
previous image-upload worktree are preserved.
