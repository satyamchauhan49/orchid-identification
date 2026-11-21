# Sneha: backbones and explainability

Status: methods and software implementation; real experimental findings remain pending.
Synthetic checks demonstrate execution and compatibility, not orchid classification quality.

## Models compared
Role B compares EfficientNet-B0 and ConvNeXt-T using timm feature extractors.
Each pooled embedding feeds seven independent linear classifiers: 156 species,
T (2 values), NF (3), In (4), LC (4), CF (15) and CL (10). These dimensions come
from Agrim's committed metadata. Role A supplies the ResNet-18 baseline.
All comparisons must use identical group-isolated train/calib/val/test assignments,
class ordering, image transforms and train-only channel statistics.
Species IDs are model labels; botanical names are display metadata and do not reorder logits.

## Preprocessing and experimental setup
The agrim_v1 profile matches Role A: training uses RandomResizedCrop(224,
scale=(0.75, 1.0)) and horizontal flipping; evaluation resizes the shorter edge to
256 and centre-crops to 224. Both apply ToTensor and train-only mean/std.
Older B checkpoints retain legacy square resizing; they must not be silently
reinterpreted as agrim_v1 runs. Grad-CAM overlays use the same crop as inference.

The real-run configuration proposes 15 maximum epochs, batch size 32, pretrained backbone
weights, seed 42, plain sampling and cross-entropy. The learning-rate grid is
0.001, 0.0003 and 0.0001 for each backbone (six runs). These are planned settings,
not a record of completed experiments. Training uses AdamW (weight decay 0.0001, matching A's default), optional CUDA mixed precision and the sum
of available head losses. Missing targets are ignored. Supported alternatives are
label smoothing (0.1), focal loss (gamma 2) and inverse-frequency species sampling.
These alternatives are options; do not report them as performed ablations.

Checkpoints are selected by minimum validation cross-entropy. The grid ranks runs
by mean feature macro-F1, breaking ties by validation loss. The final team choice
also requires ResNet results and C's calibration analysis; grid selection alone is
not a final-system decision. Actual epochs, hardware, dependency versions, batch
size changes, runtime and chosen configuration must be recorded with real runs.
B now exposes the same default weight decay (0.0001) and validation early-stopping
patience (4 epochs) as A. The team must retain the same settings in actual runs.

## Evaluation and handoff
Each run exports raw species and feature logits, embeddings and image IDs for
calib, val and test. B exports include checkpoint, exact label-map and manifest
hashes. Calibration is C's responsibility; raw softmax confidence is not calibrated.
Validation reports include species/feature macro-F1, top-k accuracy and per-value
TPR/TNR with support. Undefined rates remain null. Current macro-F1 averages
classes present in predictions or truth; state this convention in comparisons.
Test predictions may be exported but are not scored during B's experiments.
The team performs the locked test evaluation once after freezing models and code.

## Four-layer explanation method
1. Taxonomic: compare image feature probabilities with expected feature
   distributions for candidate species. Empirical TRAIN-derived distributions
   must be identified as such, rather than described as expert botanical truth.
2. Contrastive: compare the DNN and BN candidates using feature-distribution
   differences (total variation), with reference images and their provenance.
   Missing references remain explicit; no candidate or reference is fabricated.
3. Visual: compute Grad-CAM from the predicted species, CF and CL logits at
   the final spatial convolution. Average gradients spatially to weight channels,
   apply ReLU and resize the map to the model input. Maps are normalized per
   image and are not comparable as absolute importance across examples.
4. Decision: retain the actual resolver stage, candidate probabilities, confidence
   and decision signals from C's output. Join cases by image ID and verify
   artifact provenance. Feature evidence below the resolver's 0.4 rule is unobserved.

The Grad-CAM command accepts B checkpoints and A's OrchidMultiTaskResNet18
checkpoint format. The resolver-to-record workflow calls the same CAM loader but still requires
compatible hashed exports. The adapter alone does not convert A exports or
establish their provenance; the complete workflow on A's real outputs is pending.
Heatmaps indicate model sensitivity and cannot establish botanical correctness.
CL attention must be inspected on real orchid images to determine whether it
falls on the labellum. Zero maps and failed examples should be reported honestly.

## Results and example records
Real EfficientNet/ConvNeXt runs, R1 contributions and backbone conclusions are
pending dataset access and training. No synthetic score is a measured finding.
After C's locked evaluation, select approximately ten actual cases spanning
trusted, auto-resolved and deferred decisions where those stages occur. For each,
retain image ID, ground truth where available, DNN/BN candidates, actual stage,
confidence, three heatmaps, references and the full explanation JSON. Do not
manufacture coverage of a stage absent from the evaluated predictions.

## Results website
The standalone results page presents the method flow, validation comparison,
figures and explanation records. Demo builds display explicit synthetic-data
banners. Final publication requires real artifacts, links to the repository and
final evaluation, review by A and verification that report numbers match C's
final_results.json. The image-upload application is owned separately by Satyam.

## Remaining dependencies and review
- Obtain images and the shared train-only normalization; agree epochs, early
  stopping and weight decay before the six real experiments.
- Supply real raw exports to C and compare with A's ResNet baseline.
- Verify real Grad-CAM and reference images; generate final records after test freeze.
- Review A's R2 and C's R4, reliability diagrams and risk-coverage curve.
- Insert measured results, choose the final model with the team and publish the page.

## Final locked-test outcome (supersedes pending-result statements above)
The team-selected ConvNeXt-Tiny, learning rate 0.0001, achieved 641/795 correct
full-coverage predictions: 80.63% test accuracy (grouped 95% CI 77.90–83.48%).
Top-3 was 92.20%, top-5 94.97%, and macro-F1 0.5502 over all 156 class slots
(zero-division 0; 135 species represented in test). The frozen selective policy
accepted 574 images: 541 correct and 33 incorrect, giving 94.25% accepted accuracy
at 72.20% coverage, with 221 deferred (27.80%). The 1.5% accepted-error target was
missed (observed 5.75%). Weight 1.0 makes fusion identical to the DNN; no BN gain
was demonstrated, and auto-resolved tags must not be described as DNN corrections.
These figures come from Satyam's supplied final_results.json. No test rescoring,
model change, refitting or threshold tuning was performed for this report import.
Final visual record generation is prepared and requires the matching Colab checkpoint.
See final_test/SATYAM_RESULTS.md for uncertainty, limitations and frozen provenance.
