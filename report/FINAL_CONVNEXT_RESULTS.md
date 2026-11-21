# Final locked ConvNeXt test results

Model: convnext_tiny_lr0.0001, chosen by the team before test. Original export
checksum confirmed by sender via user and independently matched on receipt.
Checkpoint/map/manifest provenance, all frozen file/code hashes, and exactly 795
unique test IDs passed. One scoring invocation completed. No fitting, model change,
threshold adjustment or repeat scoring was performed after observing test results.

## Final table

| Method | Proposed accuracy on all images | Coverage | Deferral | Accepted accuracy |
|---|---:|---:|---:|---:|
| Calibrated ConvNeXt DNN | 80.63% | 100% | 0% | 80.63% |
| Direct-soft empirical BN | 69.81% | 100% | 0% | 69.81% |
| Rank-only rule | 80.63% | 93.21% | 6.79% | 83.13% |
| Full-coverage fusion | 80.63% | 100% | 0% | 80.63% |
| Selective frozen policy | 80.63% | 72.20% | 27.80% | 94.25% |

Full-coverage DNN: 641/795 correct. Top-3 92.20%, top-5 94.97%, macro-F1 .5502
over all 156 class slots (zero-division 0), NLL .73384, ECE .03332 (15 bins).
Test represents 135 species; macro-F1 convention must be stated when comparing
with observed-class scores. Full-coverage proposed accuracy is not the accuracy
of an automatically accepted answer on every image.

Selective policy accepts 574/795: 541 correct and 33 errors; 221 defer. Accepted
error is 5.75%, so the nominal 1.5% target was NOT met on test (also missed on
validation). Its 94.25% accepted accuracy requires rejecting 27.80% of cases.

## Uncertainty and paired comparison

1000 paired whole-pHash-group bootstrap resamples, seed 42, 753 test groups:
- Full-coverage DNN/fusion accuracy CI: 77.90%-83.48%.
- Coverage CI: 68.68%-75.76%.
- Accepted accuracy CI: 92.09%-96.25%.

Final calibrated fusion weight is exactly 1.0. Full-coverage DNN and fusion
predictions are identical: paired difference CI [0,0], McNemar p=1.0. This does
NOT show a BN fusion gain; no rejected cases are credited as human-corrected.
Grouped intervals assume independent groups and fixed models/policies, omitting
training/selection uncertainty. IID intervals are separately saved; their
independence assumptions may be weaker than grouped sampling here.

## Locked configuration and source of truth

Weight=1.0; threshold=.7338323883066077; direct_soft; mask=.4; uniform species prior;
TRAIN-only empirical theta. Temperatures and weight/threshold were fitted on the
previously saved 255/126/128 calibration memberships, without test use.
Source of truth: eval/final_results.json and its .freeze.json, decision/posterior,
reliability and risk-coverage sidecars. Bootstrap and test plots are descriptive;
test curves must never be used to choose a new operating threshold for this result.

Test NPZ SHA-256:
1a53551ec96b9c7c3d210723187aad775c6e12754bd6e9f522c87cf26f5bac06
Checkpoint SHA-256:
8504b86ee117f0a6d6fe60f3bc0bd0abc4ee126ac39624c5e402336666d8243f

## Best-model paragraph for report/PPT

The team selected ConvNeXt-Tiny (LR 1e-4) using development comparisons before the
locked evaluation. It led the supplied backbones in validation top-1/top-k,
common 156-class macro-F1 and selective coverage. The frozen system achieved
80.63% full-coverage test accuracy (grouped 95% CI 77.90%-83.48%). Selective
prediction achieved 94.25% accuracy on 72.20% of test cases, deferring 27.80%.
Calibration selected DNN-only fusion, so the BN did not improve classification;
the 1.5% accepted-error target did not generalize. These trade-offs, rare class
support, simplified empirical BN assumptions and fixed-policy uncertainty limit
the conclusions. No final-test comparison with unselected models was run.

## Explanation and submission work still pending

Strict converter produced ten actual final resolver exchange cases spanning the
available trusted/auto-resolved/deferred tags. With w=1, auto_resolved is a
 disagreement tag, not evidence fusion corrected the DNN. Sneha must use matching
raw images/checkpoint for real Grad-CAM and four-layer record review. No fabricated
visuals or reference images are supplied. Copy results into report/PPT/site, check
all numbers against JSON, perform team review and second-machine reproduction,
and merge reviewed final changes. Nothing was committed or pushed by this run.
