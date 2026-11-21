# Grad-CAM image generation

Run these commands in PyCharm's terminal from the project root, with .venv active.
No training or model download is needed: the script restores the local checkpoint.

## Try it now with your EfficientNet dummy checkpoint

```powershell
python -m explain.generate_cam --checkpoint outputs/effb0_dummy/best.pt --demo --output outputs/cam_my_demo
```

Open the output folder in PyCharm. It contains input.png, species/CF/CL_heatmap.png,
species/CF/CL_overlay.png and explanation.json (predicted labels, raw softmax
confidences, chosen layer and source information). Dummy-checkpoint explanations
are only software checks, even when you supply a real photograph.

For ConvNeXt-T, use outputs/convnext_dummy/best.pt and a different output folder.
Each run requires a new/empty output directory to avoid overwriting prior work.

## Later: a real model and image

```powershell
python -m explain.generate_cam --checkpoint outputs/effb0_base/best.pt --image "C:/path/to/validation_orchid.jpg" --output outputs/cam_orchid_001
```

Use train/validation images until Sync 4; the script cannot infer a standalone image's
split. Real checkpoints must store train-only mean and std. The checkpoint selects its original preprocessing profile and normalization.
New B runs and Agrim models resize the shorter edge to 256 and centre-crop to 224;
legacy B checkpoints retain square resizing. Overlays use the identical model crop.
EXIF orientation is normalized.

Supported local starter checkpoints: tiny, resnet18, efficientnet_b0, convnext_tiny.
Agrim's models_resnet/train_resnet.py checkpoint format is also supported.
Other external formats require an explicit adapter.
Only open trusted PyTorch checkpoints.

The script explains each head's predicted class. Confidences are uncalibrated and
must not be substituted for Satyam's calibrated resolver decisions. An all-zero map
is flagged in the JSON and displays the original image without artificial highlights.
Inspect whether final CL attention lands on the labellum; no botanical claim can be
made from the current synthetic runs.

The four-layer record generator and C-resolver adapter are implemented. Their
real-data verification still requires actual checkpoint-matched exports and images.

## Agrim handoff: run when the checkpoint arrives

First verify software compatibility against the exact canonical maps used by C:

```powershell
python -m explain.check_compatibility --checkpoint outputs/resnet18_base/best.pt --label-maps work/agrim_metadata_20261121/label_maps.json --output outputs/agrim_checks/compatibility_v1.json
python -m explain.generate_cam --checkpoint outputs/resnet18_base/best.pt --image "PATH_TO_TRAIN_OR_VALIDATION_IMAGE.jpg" --output outputs/agrim_checks/cam_v1
```

The first command uses artificial input tensors, compares complete class order,
checks finite output shapes, and records model/map hashes. It does not evaluate
accuracy or open test data. It preserves existing reports. The second command
requires a real image and restores checkpoint normalization automatically.
Do not pass --demo for an Agrim checkpoint.

Request the trusted best.pt, a train/validation image plus its ID/split, and the
matching class_maps.json/normalization.json for cross-checking. Do not request
large model files through a Git commit. The checkpoint already embeds its class
maps and normalization; the sidecars help audit which run was delivered.

Inspect species, CF and CL overlays. Record whether CL attention covers the
labellum, is diffuse, or is all zero. Interpretation requires real images and a
trained model; a successful command alone is not botanical validation.
The optional Grad-CAM-based labellum crop is not implemented as a training transform;
Agrim may continue E2 using the work plan's centre-crop fallback.

For final records, use explain.from_resolver with C's matching artifact, validation
bundle and evaluation, then explain.build_records with taxonomy/reference assets.
Both commands expose --help. A's existing raw exports lack required hashes and
need provenance-aware re-export before this route can accept them. Never attach
assumed hashes. Real test records remain gated until the locked team evaluation.
