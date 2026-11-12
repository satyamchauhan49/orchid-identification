# Input from Satyam: proposed exchange format, agree before final integration

Integration update: `python -m explain.from_resolver` now converts C's actual pipeline
evaluation, decisions and posterior sidecar files into this format. See
INTEGRATION_B_C.md for commands and hash/provenance requirements. Nested feature maps
from C are accepted; labels are normalized internally without reordering.

Top-level JSON: mode (real/demo), label_maps (same ordered lists as training), cases.
Each case must include:

- image_id, path relative to assets-root, split (train/calib/val/test).
- dnn_probs and bn_probs: normalized vectors in label_maps.species order.
- feature_probs: keys T,NF,In,LC,CF,CL; normalized vectors in feature label order.
- stage: trusted, auto_resolved or deferred.
- fused_confidence: calibrated fused confidence between 0 and 1.
- final_species: actual selected label, or null when deferred.
- reason: actual decision rule and strongest signals from the resolver.

The formatter does not implement or replace calibration, fusion, or resolver decisions.
Demo fixtures explicitly simulate these inputs. A real final record uses actual inputs.

taxonomy.json: species -> feature -> {feature_value: probability}. Use training-only
taxonomy or a cited domain source. Do not estimate it from the locked test set.
reference_images.json: species -> train/reference image relative path. Never use
test images as references before evaluation. Missing references stay explicitly missing.

Run python -m explain.build_records --help for the batch interface. It uses image_id,
not row positions, and refuses real test cases without --locked-test-approved.
The Grad-CAM checkpoint must be the same DNN model that produced the exchange inputs.
For real cases its predicted labels are checked against the supplied species/CF/CL labels.
