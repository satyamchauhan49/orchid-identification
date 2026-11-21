# Sneha: local completion and real-data handoff

## Completed locally
- Agrim-format ResNet checkpoint adapter for species/CF/CL Grad-CAM.
- agrim_v1 transforms for new B runs, matching A geometry/augmentation.
- Checkpoint-controlled explanation geometry, including legacy B compatibility.
- Actual 156-species metadata tests, checkpoint/logit parity and generated CAM smoke checks.
- Written methods, hyperparameter plan, evaluation and explainability report sections.
- Expanded results-page method flow and repository link.

## Run from the project root
```powershell
python -m unittest models_other.test_checkpoint_compat -v
python -m explain.generate_cam --checkpoint PATH_TO_AGRIM_BEST.pt --image PATH_TO_TRAIN_OR_VAL_IMAGE.jpg --output outputs/agrim_cam_v1
```
The checkpoint test uses random weights and artificial pixels; it measures no model accuracy.
Use trusted checkpoints only. Run CAM on train/validation images until the locked test run.

## Before real experiments
Use bn_resolver.prepare_metadata to create canonical maps from outputs/label_maps.json
and outputs/split_manifest_grouped.csv. Preserve generated map bytes for all exports.
Set config_real.example.json paths to these files, the actual image root and shared
TRAIN-only normalization.json. New runs use agrim_v1. Do not substitute ImageNet
statistics or demo statistics. Defaults now match A: 15 maximum epochs in the real config, batch size 32,
AdamW weight decay 0.0001 and early-stopping patience 4. Keep actual run overrides aligned.
Then run python -m models_other.validate_data --help and follow its configuration interface,
and python -m models_other.experiments --config YOUR_REAL_CONFIG.json.

## Still requires real inputs
Real six-run grid, measured R1 comparison, final model choice, real visual inspection,
final records, team table/plot review and results publication. The record generator uses the same CAM adapter, but its full workflow on A real
outputs remains unverified. A raw exports need fresh provenance-aware export before C can accept them.
Do not patch assumed hashes onto old exports. Keep test evaluation locked.
Nothing in this local work was committed, pushed or published automatically.
