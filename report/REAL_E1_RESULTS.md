# Role C real E1 development results

Run: selected ResNet18 LR=1e-4, epoch 15, supplied in e36dd98.zip.
509 calibration rows and 1,273 validation rows exactly match the current grouped
manifest, with unique IDs and all seven target arrays matching annotations.
Calibration partitions: 255 cal_model / 126 cal_fusion / 128 cal_threshold.
This report is development analysis, not final test evaluation.

## R3: BN evidence (validation top-1)

| Evidence | No mask | Mask 0.4 |
|---|---:|---:|
| Hard direct | 61.90% | 61.90% |
| Full categorical hard confusion | 62.22% | 62.14% |
| Prior-corrected direct soft | 63.24% | 63.00% |
| Confusion with virtual soft evidence | 62.92% | 62.69% |
| Binary TPR/TNR selected detector | 62.14% | 61.98% |

Full top-3/top-5, NLL, ECE and supports are in the CSVs. The prespecified direct-soft
masked path remains the downstream comparison; the best E3 validation cell was not
silently substituted into the policy. TRAIN-only empirical theta with add-one
smoothing and uniform species prior were used. No claim of expert botanical CPTs
or full replication of the base paper is made.

## E4 calibration feedback

DNN species temperature = 0.9444. Validation accuracy is unchanged at 74.94%.
ECE (15 bins) improves from 0.04385 to 0.03171 and NLL from 0.97420 to 0.96588.
Feature temperatures: T=1.9822, NF=1.5827, In=1.3882, LC=1.6181, CF=1.2801,
CL=1.2199. They soften head outputs; temperature scaling cannot change argmax recall.

For the BN, calibrating heads improves NLL (1.7230 to 1.6379) but worsens top-1
(63.94% to 63.00%) and ECE (0.03259 to 0.06642). Additional BN temperature does
not improve validation metrics. Additional fusion temperature worsens ECE. Do not
claim calibration universally improves every metric/head/system.

Weak observed values: CF Brown recall 1/3; CL WhiteYellow recall 1/4. Supports are
very small, so these are diagnostic observations, not stable comparative findings.
CL GreenRed does not exist in this dataset version and must not be aliased.

## R4 resolver

| Method | Proposed full-coverage accuracy | Coverage | Deferral | Accepted accuracy |
|---|---:|---:|---:|---:|
| Calibrated DNN | 74.94% | 100% | 0% | 74.94% |
| BN direct soft | 63.00% | 100% | 0% | 63.00% |
| Rank rule | 74.94% | 92.93% | 7.07% | 77.68% |
| Full fusion | 74.94% | 100% | 0% | 74.94% |
| Selective fusion | 74.94% | 55.38% | 44.62% | 94.89% |
| Selective DNN | 74.94% | 55.38% | 44.62% | 94.89% |

Calibration selected w=1.0: fusion reduces to the DNN and demonstrates no BN gain.
Threshold tau=0.8216482224506052 accepted 74 calibration-threshold cases with one
error (1.35% empirical risk). Validation accepted 705 cases with 36 errors (5.11%).
The nominal 1.5% error target did NOT generalize to validation. Accepted accuracy
is bought by rejecting 44.62%, not by improving all-case classification.

1,000 paired group-bootstrap resamples over 1,206 validation pHash groups give
DNN/fusion accuracy CI 72.34%-77.46%, coverage CI 52.33%-58.26%, and accepted
accuracy CI 93.12%-96.52%. Fusion-minus-DNN accuracy CI is [0,0]; full-coverage
McNemar p=1 because predictions are identical. These are conditional development
intervals, not training/model-selection uncertainty or final test inference.

## Other E1 runs

| Learning rate | DNN top-1 | Soft BN top-1 | Selective coverage | Accepted accuracy |
|---|---:|---:|---:|---:|
| 3e-4 | 70.23% | 59.54% | 62.45% | 89.69% |
| 1e-3 | 58.99% | 50.75% | 33.07% | 90.74% |
| 1e-4 (selected) | 74.94% | 63.00% | 55.38% | 94.89% |

Each run's policy is fitted to its own calibration exports. None meets 98.5%
accepted validation accuracy. These are E1 backbone/LR outputs, not improved E2
feature models. Do not report baseline-vs-improved results before E2 is available.

## Optional arbiter and explanations

The selected run offers 18 decisive cal_fusion disagreements, below minimum 20;
optional arbiter stays unfitted and falls back to DNN. No support rule was lowered
based on validation results. Ten real-logit taxonomic/contrastive/decision records
are generated. There are no actual image/Grad-CAM assets in these records; those
require dataset pixels and Sneha's visual integration.

## Provenance and limits

Original NPZ exports lack exporter hashes. We preserved them and created derived
analysis bundles only after verifying class/checkpoint map order, exact split IDs,
all target labels and finite shapes. IMPORT_PROVENANCE.json explicitly labels this
post-hoc binding and UNVERIFIED logits-to-checkpoint linkage. No checkpoint hash
was falsely attached to derived arrays. Treat results as analysis of supplied
exports, pending fresh checkpoint/image re-export for strong provenance.

Validation was used by Agrim for checkpoint/LR selection, so it is independent of
C's calibration fit but NOT an untouched model-selection holdout. E4 diagnostic
BN/fusion temperature fitting reuse is also disclosed. Do not interpret plotted
'independent validation' as independent of backbone/checkpoint selection.
Sparse calibration feature states, rare species and source correlations limit
claims. Literature 87.6% and 9.4% are not matched 156-class baselines. No test.npz
was extracted/read, and no eval/final_results.json was created.

## Ready to tell the team

Agrim: your selected exports load after audited class-map adaptation. DNN accuracy
reproduces your 74.94%. Species ECE improves to 0.03171. BN currently trails DNN;
feature calibration is mixed. Please supply E2 improved outputs on the SAME splits
and fresh provenance-verified exports. Preserve original arrays.

Sneha: real R3/R4, calibration/risk plots and decision/posterior sidecars are available
for the results viewer. Do not label them test results or BN improvements. Real
Grad-CAM and uploaded-image claims require matched checkpoint/image verification.

Next: evaluate final/improved exports and other trained backbones when available,
review final policy/model with the team, verify provenance using image re-export,
then freeze before the separately authorized locked test. Do not tune to test.
