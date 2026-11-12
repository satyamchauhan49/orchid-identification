# Unfilled report table templates

Every value is intentionally blank and every row is PENDING_REAL_DATA. R1/R2
are shared templates for A/B input; C reviews and supplies the BN/evaluation parts.
R3/R4 are C outputs. FINAL is populated only after the locked test. Optional
interventions can be marked NOT_RUN, with reasons, rather than given fake numbers.
Do not copy synthetic metrics into these templates.

Development suite mapping: E3_R3_bn_comparison.csv supplies accuracy/top3/top5;
E5_R4_resolver.csv supplies accuracy (full-coverage proposal), accepted_accuracy,
coverage and deferral_rate. E4_calibration.csv supplies NLL/ECE diagnostics.
Use saved run IDs, supports and hashes. Literature figures belong in separately
labelled reference text because the protocols/datasets differ.
