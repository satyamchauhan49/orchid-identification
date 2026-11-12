# Role C methods draft - fill results only after real-data evaluation

Status: implemented and exercised using synthetic data. No orchid performance
claim is supported by the dummy run.

## Bayesian inference

The model uses the common 156-species label order. Each feature contributes a
species-conditional likelihood under a simplified conditionally independent BN.
TRAIN-only empirical feature distributions receive add-one smoothing; a species
with no feature annotations receives a uniform row and an explicit zero support
count. These distributions are not claimed to be expert botanical descriptions.
The species prior is uniform for the planned balanced task.

E3 compares hard-direct evidence, categorical hard-confusion evidence,
prior-corrected soft evidence, and confusion with virtual soft evidence. The
confusion matrix is P(predicted category | true category), normalized by true-state
columns. The two soft paths are engineering approximations. Masking is compared
at thresholds 0 and 0.4, with missing evidence contributing one to every species.
The paper's binary TPR/TNR construction is not yet reproduced and must not be
identified with our categorical confusion baseline.

## Calibration and policy

Per-head temperature scaling minimizes labelled development NLL. Duplicate
clusters stay together. The original calibration split is deterministically divided
into cal_model (head temperatures/CPTs), cal_fusion (fusion weight), and
cal_threshold (operating threshold). Hash rules and row IDs are saved.
Support below 20 labelled rows or fewer than two observed classes retains T=1
and is flagged. The nominal empirical risk target is 1.5%, with at least 50
accepted calibration cases. Infeasible policies defer every case.

E4 separately compares raw and calibrated heads, BN temperature and fusion
temperature. BN temperature reuses cal_model and fusion temperature reuses
cal_fusion, which can be optimistic within those fitting subsets. Metrics are
reported on independent validation. These E4 diagnostics do not silently change
the E5 frozen policy, which uses calibrated heads and prespecified direct-soft BN.

E5 compares DNN, BN, the rank-only baseline, fusion, selective fusion and selective
DNN. The accepted class is the fused top class whose confidence was tested.
Agreement is a decision explanation tag and never overrides deferral. Null accepted
labels are retained on deferred cases.

## Evaluation and explanation

Report top-1/3/5, macro-F1 (zero division is 0 over the full class map), NLL, ECE
(15 equal-width bins), coverage, deferral and accepted accuracy. Per-value feature
TPR/TNR excludes missing labels and uses null for unsupported denominators.
Full-coverage accuracy is separate from accepted accuracy in every selective row.
Risk-coverage curves and reliability plots are validation diagnostics. A frozen
operating point must be selected before locked test evaluation.

Each example record shows observed model feature probabilities, empirical expected
distributions with support, signed feature log-likelihood contrasts and decision
parameters. No visual/Grad-CAM evidence is fabricated. Those paths require Sneha's
real image and checkpoint integration. Confidence is not a correctness guarantee.

## Real results placeholders

- Dataset/subset and annotation audit: PENDING AGRIM.
- Model/checkpoint and preprocessing hashes: PENDING AGRIM/SNEHA.
- R3/E3 measured comparison: PENDING REAL LOGITS.
- E4 reliability results: PENDING REAL LOGITS.
- R4/E5 accuracy, accepted accuracy and coverage: PENDING REAL LOGITS.
- Final locked test bootstrap/McNemar: NOT RUN.
- Best-model justification: write after development comparisons, then report the
  frozen model's test results without selecting again on test.

Literature values 87.6% and 9.4% come from a different experimental setting and
must be labelled references, not treated as matched 156-species baselines.
