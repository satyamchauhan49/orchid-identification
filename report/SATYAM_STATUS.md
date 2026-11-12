# Role C progress

Implemented and locally testable: contracts, calibration, four BN paths, rank baseline,
fusion, deferral, validation CLI and guarded final evaluation.

Pending real inputs: Agrim's audited 156-class maps and manifest; TRAIN feature
annotations or reviewed taxonomic distributions; baseline calibration/validation logits
from Agrim and Sneha. No real locked test evaluation has been run.

Next: review class support; compare E3 variants and masking on development data;
complete E4 BN/fused calibration ablations and charts; freeze E5 policy; integrate
Sneha's explanation records; perform the agreed locked evaluation and write R3/R4,
best-model justification and limitations using measured outputs.

The plan's literature numbers (87.6%, 9.4%) are reference values, not our measured
results, and dataset differences must be stated when comparing the 156-species task.

Synthetic fixtures contain 156 fake species solely to check dimensions and flow.
No claims about orchid recognition accuracy follow from them.
