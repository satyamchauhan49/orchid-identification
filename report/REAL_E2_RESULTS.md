# Real E2 resolver comparisons and latest team integration

Development validation only: 509 calibration / 1273 validation images, exact IDs
and all target labels verified against the shared grouped manifest. All six model
class maps/checkpoint maps agree. Original files are retained in ignored work/.
No test files are in the supplied E2 archive and no test predictions were accessed.

## E3-E5 summary

| Model | DNN accuracy | Soft BN accuracy | Fusion accuracy | Weight | Coverage | Accepted accuracy | DNN ECE raw / calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_cf_in | 72.90% | 61.74% | 72.51% | 0.9 | 56.01% | 94.11% | 0.0257 / 0.0315 |
| focal_gamma2 | 74.63% | 62.92% | 74.63% | 1.0 | 60.17% | 93.99% | 0.1006 / 0.0300 |
| hsv_cf_cl | 75.57% | 65.20% | 75.57% | 1.0 | 48.39% | 96.43% | 0.0405 / 0.0378 |
| hsv_smoothing_0p1 | 75.73% | 65.20% | 75.73% | 1.0 | 53.50% | 96.04% | 0.1237 / 0.0645 |
| label_smoothing_0p1 | 76.43% | 65.12% | 76.43% | 1.0 | 48.78% | 95.65% | 0.1297 / 0.0312 |
| labellum_center_crop | 74.47% | 63.08% | 74.47% | 1.0 | 52.40% | 95.95% | 0.0350 / 0.0340 |

Five variants select w=1 (DNN only). The balanced run chooses w=0.9 but fusion
reduces validation accuracy from 72.90% to 72.51%. No BN advantage is established.
None meets the nominal 1.5% accepted validation-error target. Temperatures improve
most DNN calibration metrics; balancing's ECE worsens. Calibration is not uniformly
beneficial, and E4 diagnostic temperatures do not silently replace the E5 policy.

Label smoothing has the strongest ResNet species accuracy (76.43%, +1.49 percentage
points from E1's 74.94%). Its paired group-bootstrap accuracy-difference interval
is about -0.24 to +3.19 percentage points; it includes zero. Treat the gain as a
validation observation, not an established test improvement. All paired comparisons
are exploratory after validation model selection; no multiplicity correction.

HSV/HSV+smoothing reach 65.20% masked direct-soft BN versus E1's 63.00%, but still
trail the DNN substantially. Compare detailed R3 evidence variants and calibration
support before deciding the final model; do not choose solely from a single number.
Selective accepted accuracies near 94-96% require deferring roughly 40-52% of cases.

## Evidence exported

For every run: R3 (five evidence paths and masking ablation), E4 calibration on/off,
R4 resolver baselines, feature-value support/TPR/TNR, reliability/risk/BN charts,
frozen development artifact, 10 decision explanation records, validation evaluation,
posterior/decision sidecars, and grouped bootstrap intervals. Root comparison adds
paired DNN-vs-E1 CIs and McNemar. All outputs are development results, not final test.
No real Grad-CAM was fabricated because dataset images are not in this handoff.

Small CSVs/JSONs are copied into report/real_e2_tables for Git review. Large original
models/logits and generated artifacts stay in work/e2-real-v1, excluded from Git.

## Provenance

NPZ files omit exporter hashes. The existing audited import verifies class order,
IDs, annotations and finite shapes, writes explicit post-hoc map binding, and keeps
logits-to-checkpoint linkage UNVERIFIED. No guessed checkpoint hash is attached.
Fresh image/checkpoint re-export remains necessary for strict explanation-adapter
provenance. Validation was used for epoch/LR/model selection, so it is not an
untouched final holdout. Head/CPT fitting and diagnostic temperature reuse limitations
from E1 still apply. Tests are not a substitute for provenance or real-image validation.

## Sneha's new GitHub work

Combined branch 77c1ff9 includes six real backbone summaries, configs/histories,
feature metrics, qualitative validation Grad-CAM and site/real_validation. Her selected
ConvNeXt-Tiny LR=1e-4 reports 79.42% species accuracy and 0.8971 mean feature macro-F1.
This exceeds current ResNet DNN validation accuracy, but is not yet a C calibrated
comparison: her checkpoint/calibration/validation NPZs remain in Drive, not this repo.
Request that selected run (and optionally EfficientNet LR=3e-4 at 75.81%) with exact
map/manifest/checkpoint provenance. Do not compare total losses directly: head loss
aggregation and macro-F1 class conventions differ across implementations.

Her real site examples are validation visual explanations; they are not our final
frozen-resolver test records. No claim of consistent anatomical localization is made.

## HSV checkpoint integration

Agrim published models_resnet/model_hsv_colab.py at a-data d5e4434. It is copied
into this checkout with ONLY its import changed to package-relative project form.
Shared checkpoint loading detects saved HSV buffers and instantiates the correct
48-bin colour-head architecture. AgrimAdapter now calls the actual model forward,
so HSV colour heads are not bypassed. An exact-forward equality test verifies all
heads and embeddings, and both supplied HSV checkpoints pass finite-head checks.
Crop checkpoints still require review of their special crop preprocessing before
upload/re-export; this work does not infer geometry from weights alone.

## Next team decisions

1. Receive Sneha's real selected-model exports and perform the same audited E3-E5.
2. Review accuracy/calibration/coverage and weak features across the complete team.
3. Verify checkpoint provenance with real images/re-exports; integrate real resolver
   explanations and upload inference for the actually selected model.
4. Agree final policy/model freeze; only then authorize/export/run locked test once.
5. Populate final report/PPT with test findings, including missed targets honestly.

No final_results.json or test metric was produced. No commit or push was performed.
