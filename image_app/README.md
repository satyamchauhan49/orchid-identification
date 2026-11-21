# Local image upload

Upload JPEG/PNG from your computer, preview it, and run the frozen neural model and
C's BN/resolver. A deferred result shows candidates without claiming an accepted
species. Images are processed in memory, never persisted, and the server binds only
to 127.0.0.1. Maximum 10 MB / 20 megapixels. No external services or model downloads.

Run from this worktree using the existing integrated-demo environment:

```sh
cd /Users/satyamchauhan/Documents/Codex/orchid-upload-local
/Users/satyamchauhan/Documents/Codex/orchid-identification-c-resolver/work/integration-env/bin/python -m image_app.server \
  --maps /Users/satyamchauhan/Documents/Codex/orchid-identification-c-resolver/outputs/integration_mac_demo_v1/data/label_maps.json \
  --checkpoint /Users/satyamchauhan/Documents/Codex/orchid-identification-c-resolver/outputs/integration_mac_demo_v1/models/efficientnet_b0_lr0.0003/best.pt \
  --artifact /Users/satyamchauhan/Documents/Codex/orchid-identification-c-resolver/outputs/integration_mac_demo_v1/frozen.json \
  --calibration-bundle /Users/satyamchauhan/Documents/Codex/orchid-identification-c-resolver/outputs/integration_mac_demo_v1/models/efficientnet_b0_lr0.0003/calib.npz
```

Open http://127.0.0.1:8765 in a browser. Stop with Ctrl+C. The old static results
HTML is unchanged; uploads require this running Python server, not file:// opening.

The current available checkpoint was trained on synthetic images. The visible demo
banner cannot be disabled from the UI. It returns synthetic labels and cannot reliably
identify real orchid photos. The real-data path needs a compatible B-format checkpoint
with exact label order, stored train mean/std, the matching canonical maps, fitted
resolver artifact and matching calibration export hashes. Replacing data/checkpoint
paths does not establish accuracy; real training/evaluation is still required.

Neither demo nor real mode identifies non-orchids or out-of-set species. Upload only
images you intend to classify; model confidence is not proof of botanical correctness.
The server is a single-user local tool, not a public production hosting service.

Tests:
`python -m unittest discover -s tests -p test_image_upload.py -v`

Verified: JPEG/PNG decoder, corrupt/unsupported file rejection, same-origin request
handling, local HTTP upload, and actual integrated synthetic-checkpoint inference.
No training, commit, push or real test evaluation is performed by this application.

## Updated A/B compatibility

The server now uses explain.checkpoints.load_checkpoint and the shared preprocessing
profile. Agrim-format ResNet-18 checkpoints are supported with their original crop
and stored normalization; B's old/new profile is preserved. Supply verified
fresh calibration exports from bn_resolver.export_development and matching canonical
maps/artifact. See report/REAL_CHECKPOINT_HANDOFF.md. Decode retains raw image pixel
orientation to match training; EXIF auto-rotation is not silently added at inference.

## Received real ConvNeXt development model

After verifying Sneha's handoff and running eval.backbone_handoff, start:

```sh
python -m image_app.start_received --port 8766
```

This uses work/sneha-received/convnext_tiny_lr0.0001 and its matching
work/sneha-analysis artifacts. Open http://127.0.0.1:8766. Class-ID display names
are loaded from outputs/species_id_to_name.json without changing prediction order.
The real-model UI explicitly says development validation only and reports the unmet
risk target. This is not a final-model freeze or a non-orchid detector. For a different
installation, use image_app.server with explicit model/artifact/bundle/name paths.
