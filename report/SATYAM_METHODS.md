# Role C: Bayesian inference, calibration and resolver

Methods text prepared for the 156-species project. No real-data findings are asserted.
Symbols below follow the implemented code; update artifact choices after development
selection and before the locked test.

## Model and evidence

Let c be one of C=156 species and i index the six features T, NF, In, LC, CF, CL.
The label map fixes the K_i possible values of each feature. The simplified model
uses conditional independence of features given species:

\[
P(c\mid e)=\frac{P(c)\prod_i L_i(c)}{\sum_{c'}P(c')\prod_i L_i(c')},\qquad
L_i(c)=\sum_f\theta_i(c,f)\lambda_i(f).
\]

Inference sums log likelihoods and normalizes with a stable softmax. The current
prior is uniform; this is appropriate to the planned balanced experiment and must
be reviewed if the actual evaluation distribution differs.

The current empirical implementation estimates
\(\theta_i(c,f)=(n_{cif}+1)/(\sum_{f'}n_{cif'}+K_i)\)
from TRAIN image labels only. Missing annotations contribute no counts. Unsupported
species-feature rows are uniform and carry zero support in the artifact. This is
an empirical BN, not a claim of expert botanical knowledge. A complete reviewed
taxonomic table would need a separate provenance-checked integration.

## Calibration

For each head h, probabilities are \(q_h=\operatorname{softmax}(z_h/T_h)\).
A positive temperature minimizes labelled calibration NLL; the implementation
searches log T in [-4,4] and retains T=1 if it gives a better objective. Fewer than
20 labelled samples or fewer than two observed classes retain T=1 with a warning.
This guard is a computational feasibility rule, not evidence of adequate calibration.

Duplicate groups are partitioned reproducibly using seed-tagged SHA-256 parity:
cal_model fits head temperatures and confusion rates; the remaining groups are
split into cal_fusion for weight selection and cal_threshold for threshold selection.
Memberships are exported. Labels from validation/test do not enter these fits.
Broader source groups may require revising the grouping protocol before real runs.

ECE uses 15 equal-width confidence bins:
\(\mathrm{ECE}=\sum_b (n_b/N)|\mathrm{accuracy}_b-\mathrm{confidence}_b|\).
Empty bins are excluded from the sum and exported with null means. Reliability
curves describe finite sample behavior, not a confidence guarantee.

## E3: five evidence constructions

1. Hard direct: \(L_i(c)=\theta_i(c,\hat f_i)\), where \(\hat f_i=\arg\max q_i\).
2. Categorical hard confusion: estimate \(M_i[j,f]=P(J_i=j\mid F_i=f)\)
   from cal_model, adding one to each cell and normalizing true-state columns.
   Then \(L_i(c)=\sum_f\theta_i(c,f)M_i[\hat f_i,f]\).
3. Direct soft: \(L_i(c)=\sum_f\theta_i(c,f)q_i(f)/\pi_i(f)\), where
   \(\pi_i\) is the add-one-smoothed TRAIN feature marginal. The posterior-to-
   likelihood conversion is approximate and assumes compatible priors.
4. Virtual soft: \(r_i(j)=\sum_fM_i[j,f]\pi_i(f)\),
   \(v_i(j)=q_i(j)/r_i(j)\), and
   \(L_i(c)=\sum_f\theta_i(c,f)\sum_jM_i[j,f]v_i(j)\).
   This engineering approximation may smooth evidence twice; it is ablated.
5. Binary selected-detector: observe only \(J_{i,j}=yes\) for hard category j.
   Its likelihood is TPR_j if f=j and FPR_j=1-TNR_j otherwise. Therefore
   \(L_i(c)=\theta_i(c,j)\mathrm{TPR}_j+[1-\theta_i(c,j)](1-\mathrm{TNR}_j)\).
   TPR/TNR use calibration counts with Beta(1,1) smoothing. Empirical unsupported
   rates are null and support is saved. Other detectors remain unobserved; they
   are not multiplied as independent measurements from one argmax prediction.

The binary variant implements the base paper's section 3.5 virtual-evidence
mechanism on our simplified graph. It does not reproduce its learned graph,
historical parameters, or reported numerical results.

Masking compares max(q_i)<0.4 with no confidence masking. Masked features have
L_i(c)=1. All masked evidence returns the prior. Uniform probabilities from a
2-state head have maximum 0.5, so the 0.4 rule does not mask every uniform head.
This behavior is documented rather than silently changed.

## E4: calibration ablations

Compare raw versus calibrated species probabilities, BN using raw versus calibrated
heads, BN posterior temperature, and fused posterior temperature. BN temperature
reuses cal_model and diagnostic fusion temperature reuses cal_fusion, with fitting
reuse disclosed. Report independent validation NLL/ECE and reliability diagrams.
E4 diagnostics do not silently replace the E5 frozen head-calibrated policy.

## E5: fusion and selective decision

\[
s(c)=w\log P_{DNN}(c)+(1-w)\log P_{BN}(c),\qquad p_F=\operatorname{softmax}(s).
\]

Search w in {0,0.1,...,1} using cal_fusion NLL, preferring larger w on equal NLL.
Choose the observed cal_threshold confidence threshold with maximum coverage
whose empirical error is at most 1.5%, requiring at least 50 accepted samples.
Confidence ties are accepted/rejected together. If infeasible, threshold is null
and all cases defer. This selected empirical risk does not guarantee future risk.

Accept the fused argmax only when its confidence meets the frozen threshold.
Tag it trusted_agreement only if that class equals DNN top-1 and DNN top-1 is in
BN top-5; otherwise an accepted answer is auto_resolved. Deferrals have null
accepted species. Agreement never bypasses the threshold.

Compare DNN, BN, rank-only acceptance of DNN when in BN top-5, full-coverage
fusion, selective fusion and selective DNN. Report full-coverage proposed accuracy
separately from accepted accuracy and coverage. No human correction is imputed.
The current development suite prespecifies direct soft downstream; change that
choice only using documented development selection before freezing.

## Final evaluation and uncertainty

Species metrics are top-1/3/5, macro-F1 over the full map (zero division = 0), NLL
and ECE. Per-value TPR/TNR excludes missing labels and uses null for undefined
denominators. Selective metrics include accepted count, deferral, coverage and
accepted accuracy; zero acceptance yields null accepted accuracy.

Use 1,000 paired whole-group bootstrap resamples, seed 42, for baseline/final
accuracy, their difference, coverage and accepted accuracy. Sample groups with
replacement and include all images in each selected group, including repeated
groups. Statistics use image-weighted ratios. Fewer than two groups yields no
estimable CI; replicates with zero accepted samples are excluded from accepted-
accuracy intervals with valid counts disclosed. Group independence is assumed;
training and model-selection uncertainty are not included.

An exact two-sided McNemar binomial test compares full-coverage DNN and fusion on
the same images. Its image-independence assumption can fail under source clusters;
interpret alongside grouped intervals. Do not compare selective predictions by
pretending deferred cases were corrected by humans.

No test bundle is opened until the agreed final sync. Freeze all choices first,
run the final evaluator once, and populate report tables from its saved JSON.
Literature figures 87.6% and 9.4% are external references, not matched 156-species
baselines or measured findings of this implementation.
