# orchid-identification

## Confirmed scope
Use **all 156 orchid species** in the supplied multi-label dataset. The 63-species figure in the team work plan is an error. The ResNet-18 species head must therefore have 156 outputs. The six feature targets are `T`, `NF`, `In`, `LC`, `CF`, and `CL`.

## Dataset facts
- 7,156 image records in the multi-label CSVs.
- Published partitions: train 5,119; validation 1,235; test 802.
- Training CSV uses `T`; validation/testing use `Texture`; the manifest script standardises both to `T`.
- Species IDs are numeric in the CSV; `outputs/species_id_to_name.json` maps all 156 IDs to species names.

## Environment setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation — run in this order
```bash
python scripts/01_inspect_dataset.py --dataset-root "data/raw/Orchid Flowers Dataset-v1.1"
python scripts/02_duplicate_audit.py --dataset-root "data/raw/Orchid Flowers Dataset-v1.1" --out outputs --threshold 4
python scripts/03_build_manifest.py --dataset-root "data/raw/Orchid Flowers Dataset-v1.1" --out outputs
python scripts/05_make_group_aware_splits.py --manifest outputs/split_manifest.csv --duplicates outputs/duplicate_pairs.csv --out outputs
python scripts/04_eda.py --manifest outputs/split_manifest_grouped.csv --out outputs/eda_grouped
```

### Why use the group-aware manifest?
The pHash audit found 446 candidate pairs at Hamming distance <= 4. Connected-component analysis found 117 duplicate groups crossing the original train/validation/test partitions, involving 268 images. The generated `outputs/split_manifest_grouped.csv` keeps every connected duplicate group in exactly one partition. The current group-aware partition contains 4,579 train, 509 calibration, 1,273 validation, and 795 test images. All 156 species occur in training; rare species may not occur in every held-out partition.

The pHash algorithm is implemented with OpenCV DCT so the script does not require the `imagehash` package. The threshold is an audit heuristic; manually inspect borderline pairs before making strong claims that every pair is a true duplicate.

## ResNet-18 baseline
Run on Google Colab with a GPU (recommended):
```bash
python models_resnet/train_resnet.py \
  --dataset-root "data/raw/Orchid Flowers Dataset-v1.1" \
  --manifest outputs/split_manifest_grouped.csv \
  --run-name resnet18_base --epochs 15 --batch-size 32 --lr 0.0003 --pretrained
```

The model has one shared ResNet-18 backbone, a 156-class species head, and six feature heads. The script computes channel mean/std from the training partition only, trains using cross-entropy, selects the checkpoint using validation loss, and exports raw logits for `calib`, `val`, and `test`. It does not calculate test metrics. Keep the test outputs unopened until the team's locked evaluation.

Output folder: `outputs/resnet18_base/`
- `best.pt`
- `class_maps.json`
- `normalization.json`
- `training_history.csv`
- `calib.npz`, `val.npz`, `test.npz`

The NPZ files contain `ids`, `species_logits` (N x 156), `logits_T`, `logits_NF`, `logits_In`, `logits_LC`, `logits_CF`, `logits_CL`, and corresponding target arrays. This follows the agreed raw-logit interface while including targets for downstream evaluation.

## Current important dataset/work-plan mismatch
The plan mentions weak categories such as `CL GreenRed`. The supplied multi-label CSVs use the Version 1 label set: CL has 10 classes and does not contain `GreenRed`; CF has 15 classes and `GreenYellow` has 207 examples. Confirm the intended colour-label version with the team before claiming to reproduce the work plan's weak-class baseline figures. For now, the multi-task model uses the actual six labels in the multi-label CSVs.

## Person A ownership
- Dataset audit, pHash duplicate grouping, split manifest, label maps, EDA.
- ResNet-18 baseline and weak-feature experiments (label smoothing, class-balanced sampling, focal loss, HSV branch, labellum crop).
- Confusion matrices, per-class TPR/TNR, macro-F1, R1/R2 report sections, README.

Never use the test partition to select models, tune thresholds, or choose experiments.
