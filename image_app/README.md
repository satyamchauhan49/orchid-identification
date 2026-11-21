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

Agrim's currently different ResNet checkpoint format and Resize(256)+CenterCrop(224)
preprocessing are NOT supported by this loader. It intentionally fails on that format
rather than silently applying B's Resize((224,224)) preprocessing. Coordinate a
verified adapter once his trained checkpoint is available. Only trusted project
checkpoints should be loaded; torch weights_only=True is used.

Neither demo nor real mode identifies non-orchids or out-of-set species. Upload only
images you intend to classify; model confidence is not proof of botanical correctness.
The server is a single-user local tool, not a public production hosting service.

Tests:
`python -m unittest discover -s tests -p test_image_upload.py -v`

Verified: JPEG/PNG decoder, corrupt/unsupported file rejection, same-origin request
handling, local HTTP upload, and actual integrated synthetic-checkpoint inference.
No training, commit, push or real test evaluation is performed by this application.
