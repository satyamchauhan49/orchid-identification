# Real backbone calibration and resolver results

Received: convnext_tiny_lr0.0001.zip and efficientnet_b0_lr0.0003.zip.
Every supplied SHA-256 matches handoff_sha256.json. Both calib/val bundles match
original map-file, manifest and checkpoint hashes. Ordered checkpoint labels match.
509 calibration and 1273 validation IDs exactly match the grouped split. No test
files were included in these ZIPs or opened by this work.

Canonical map serialization is derived explicitly with original hashes retained;
this is a verified B export import, unlike A's explicitly unverified legacy linkage.
The strict explanation adapters pass on both models without bypassing hash checks.

## Common-metric development comparison

| Model | DNN top-1 | DNN top-3 | DNN top-5 | BN top-1 | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ResNet E1 | 74.94% | 88.06% | 91.83% | 63.00% | 55.38% | 94.89% |
| ResNet smoothing | 76.43% | 88.30% | 91.91% | 65.12% | 48.78% | 95.65% |
| EfficientNet-B0 LR 3e-4 | 75.81% | 89.47% | 92.54% | 64.96% | 0% | NA |
| ConvNeXt-Tiny LR 1e-4 | 79.42% | 93.17% | 96.07% | 70.07% | 69.52% | 94.46% |

BN is the prespecified masked direct-soft empirical TRAIN-only model. Full E3 includes
five evidence constructions with masking on/off; all R3/R4, E4 and per-value tables
are exported. Models use their own calibrated head outputs and calibration policy.
Common macro-F1 averages all 156 classes (zero-division 0): ResNet E1 .5212,
ResNet smoothing .5457, EfficientNet .5242, ConvNeXt .5969. Sneha's original observed-
class macro-F1 differs; do not present those definitions as interchangeable.

All four main policies choose fusion w=1.0, so no gain from BN fusion is established.
ConvNeXt accepts 885 validation cases with 49 errors. Its calibration-threshold
subset accepted 86 with 1 error (1.16%); validation accepted error is 5.54%, so the
nominal 1.5% target is NOT achieved. Threshold=.7338323883066077. Do not describe
selective confidence as a guaranteed 98.5% reliable identifier.

EfficientNet has no feasible threshold meeting the error/support rule on its held-
out calibration-threshold subset; threshold=null and all cases defer. This is a
reported failure, not 100% accepted accuracy or a reason to loosen rules post hoc.

## Calibration and uncertainty

ConvNeXt ECE improves .04158 to .02872, but NLL slightly worsens .70856 to .71146.
EfficientNet ECE worsens .02580 to .03555; NLL .91444 to .91868. Temperatures do not
change DNN argmax accuracy. Calibration is not uniformly beneficial.

ConvNeXt's 1000-resample pHash-group bootstrap (1206 groups) yields conditional
validation accuracy CI 76.98%-81.62%; coverage CI 66.85%-72.06%; accepted accuracy
CI 92.72%-95.99%. These do not include training or model-selection uncertainty.

Paired ConvNeXt-minus-comparator accuracy intervals, percentage points:
- E1 ResNet: +2.43 to +6.65; observed difference +4.48.
- ResNet smoothing: +0.95 to +5.11; observed +2.99.
- EfficientNet: +1.42 to +5.74; observed +3.61.

These are exploratory validation comparisons after epoch/LR/model selection, with
no multiplicity correction, not final-test superiority claims. Paired McNemar is
also exported with its image-independence limitation; grouped intervals assume
independent groups. Do not report these as untouched holdout inference.

## Optional logistic arbiter

ConvNeXt has 23 decisive cal_fusion disagreements and can fit the optional four-
feature logistic arbiter. Validation accuracy is 79.73%, a four-image increase from
79.42%. Tiny fitting support and validation selection limit this observation.
EfficientNet's optional arbiter has 22 training disagreements and drops to 75.18%.
Neither optional arbiter is automatically promoted into the frozen fusion/deferral
policy. It lacks a separately calibrated acceptance rule. Team review is required.

## Explanation handoff

Strict adapter converts ten actual resolver cases per backbone into verified
exchange.json and taxonomy.json. ConvNeXt contains trusted, auto_resolved and deferred
stages; EfficientNet only deferred. Do not fabricate missing stages. Because w=1,
"auto_resolved" is a disagreement tag, not evidence that fusion changed the DNN label.
No raw dataset images arrived here, so new actual Grad-CAM/reference records were
not fabricated. Sneha can use matching source images and her existing model assets
to complete records. Prior website visual examples are not automatically matched
to these new resolver decision IDs.

## Upload feature

Real ConvNeXt checkpoint and verified calibrated artifact now initialize the local
upload server. Botanical names display via the species-ID sidecar without changing
class order. Status is real development, not synthetic. The banner discloses lack
of final-test validation and unmet risk target. JPEG/PNG inference was smoke-tested
on an existing preprocessed website validation image, NOT used to score raw-image
accuracy or claim training-transform reproduction.

Start from this repository:
python -m image_app.start_received --port 8766
Then open http://127.0.0.1:8766. Stop with Ctrl+C. This references the received
ignored work/sneha-received and work/sneha-analysis folders and is local-only.
No non-orchid/OOD rejection is claimed. General uploaded image prediction has not
been clinically/botanically guaranteed or independently validated on new photos.

## Provisional recommendation and remaining work

ConvNeXt is the strongest supplied DNN by validation top-1/top-k, common macro-F1,
calibrated ECE and selective coverage; it is a sensible candidate for TEAM REVIEW.
This is not an autonomous final freeze. Report lower ResNet accepted coverage and
higher accepted accuracy as a trade-off, not a universally inferior system.
Keep DNN-only fallback honest; the planned BN improvement has not been demonstrated.

Remaining: team agrees model/operating point and inference trade-offs; verifies
real per-image explanations and raw image preprocessing; resolves any provenance
limits for A comparisons; freezes all choices; then separately authorizes/export/runs
the locked test once. Neither selected checkpoint is trained here. No test.npz,
final_results.json, commit, push, main merge or public deployment was performed.
