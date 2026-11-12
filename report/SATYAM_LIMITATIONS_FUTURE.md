# Limitations and future work

This section describes implementation limitations before real evaluation. Revise
sample sizes and findings only when real data arrives.

- The simplified BN assumes conditional feature independence given species. Shared
  visual backbones can correlate head errors and DNN/BN errors; fusion can be
  confidently wrong when both sources fail together.
- TRAIN-derived species-feature tables are empirical and annotation-dependent.
  Uniform unsupported rows prevent fabricated information but cannot supply missing
  botanical knowledge. Feature-label availability on the 156-species subset needs audit.
- Direct-soft prior correction and virtual-soft evidence are approximations. Dataset
  shift, mismatched priors and repeated smoothing can undermine their interpretation.
- The binary baseline uses the paper's detector-CPT mechanism on a simplified graph;
  it is not an exact full-graph replication or a matched literature comparison.
- Head confidence masking at 0.4 behaves differently across class counts. It does
  not detect every uninformative head and does not establish out-of-distribution detection.
- Temperature scaling changes confidence, not head class rankings. Small, rare or
  unsupported calibration states remain uncertain even with smoothing.
- Diagnostic BN/fusion temperatures reuse fitting subsets. The frozen E5 policy
  separates weight fitting and threshold selection, but threshold search remains
  optimistic about its selected empirical operating point. A 1.5% target can be
  infeasible or unstable with limited independent calibration examples.
- Grouped CIs require genuinely independent groups. pHash detects near duplicates,
  not every photographer, plant, location or acquisition correlation. Bootstrap
  intervals condition on the fitted model/policy and omit training/selection uncertainty.
  Image-wise McNemar can also be affected by within-group dependence.
- No dummy experiment measures orchid performance, calibration quality on orchids,
  biological validity, visual localization or human explanation usefulness.
- Model explanations expose feature evidence and decision arithmetic. They do not
  demonstrate causal botanical reasoning. Reference images/Grad-CAM and expert
  review remain pending Sneha's real-data integration.
- One seed and a short training budget cannot establish broad superiority. Published
  paper numbers use different datasets/protocols and require explicit caveats.

Future work named in the plan: evaluate learned deferral/optional logistic arbiter
on independent development data; validate OOD rejection using labelled unknown
species; conduct a botanist/expert survey; run full grouped cross-validation and
multiple seeds; explore BN structure learning with documented assumptions. These
are future/optional studies, not achievements inferred from dummy runs.
