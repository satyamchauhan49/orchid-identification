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
split. Real checkpoints must store train-only mean and std. The same 224-pixel resize
and normalization as the training script are applied. The overlay uses the resized
224 x 224 input, so image and heatmap stay aligned. EXIF orientation is normalized.

Supported local starter checkpoints: tiny, resnet18, efficientnet_b0, convnext_tiny.
For Agrim's future model, agree on checkpoint structure first; this loader currently
accepts checkpoints created by models_other.train, not arbitrary external formats.
Only open trusted PyTorch checkpoints.

The script explains each head's predicted class. Confidences are uncalibrated and
must not be substituted for Satyam's calibrated resolver decisions. An all-zero map
is flagged in the JSON and displays the original image without artificial highlights.
Inspect whether final CL attention lands on the labellum; no botanical claim can be
made from the current synthetic runs.

Next integration: taxonomic feature expectations, BN candidate comparisons and
reference images, plus the real resolver's decision record.
