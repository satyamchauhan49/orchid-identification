# Sneha: real validation results

Six completed experiments on the grouped manifest. Its exact Git blob hash matches the experiment manifest hash. The Windows working copy has CRLF line endings; use the original Colab manifest bytes with hashed bundles. No test predictions or test metrics were opened during this import.

| Backbone | LR | Species top-1 | Species macro-F1 | Feature macro-F1 | Validation loss |
|---|---:|---:|---:|---:|---:|
| efficientnet_b0 | 0.001 | 73.13% | 0.5430 | 0.8516 | 3.2032 |
| efficientnet_b0 | 0.0003 | 75.81% | 0.5380 | 0.8609 | 2.7453 |
| efficientnet_b0 | 0.0001 | 75.02% | 0.5609 | 0.8632 | 2.7985 |
| convnext_tiny | 0.001 | 30.95% | 0.0855 | 0.5321 | 8.0611 |
| convnext_tiny | 0.0003 | 72.43% | 0.4875 | 0.8428 | 3.1707 |
| convnext_tiny | 0.0001 | 79.42% | 0.6126 | 0.8971 | 2.0541 |

Selected B validation run: **convnext_tiny_lr0.0001**. Selection: highest mean feature macro-F1, then lowest validation loss.

All six morphology heads and species were trained together. ConvNeXt at 0.001 performed poorly relative to its lower-rate runs; this unsuccessful configuration remains in the comparison.

## Evidence and limits
Raw summaries, all six run configs, histories and validation metrics are in results/sneha_real_v1. R1_feature_values.csv contains per-value support, TPR and TNR. Eight figures show learning curves, feature comparison and six selected-model confusion matrices.

This is validation selection, not the final calibrated model decision. Compare metric conventions and run settings with Agrim before interpreting cross-model differences. Total loss cannot be compared across implementations without checking head aggregation. B macro-F1 includes classes present in truth or predictions.

Checkpoints and logits remain in Drive. Real Grad-CAM, resolver integration, team freeze, locked evaluation and final explanation examples remain pending.

## Input hashes
```json
{
  "manifest": "bf8d843fc84203cd7aea74904aa153d7d5c71a0e1dd9d2861554c6e9d6e5a7a5",
  "label_maps": "c523cf61d55f30f452653056e95e0af92f8dbca9714749e0afae65dc633708cc",
  "normalization": "f80befd564d608073e700b0370264b191f361968685acaea8f42f78509740e0c"
}
```

## Real Grad-CAM inspection
Ten reproducibly selected validation examples, spanning different species where available,
were qualitatively reviewed. Original inputs, heatmaps, overlays and prediction metadata
are under results/sneha_real_v1/gradcam, alongside selected_validation_images.csv
and qualitative_review.csv. The earlier 11.jpg example is retained separately in that
same directory but is not included in the ten-example review.

Visual inspection showed variable localization. Some maps overlapped floral structures,
while others emphasized background regions. In 2514.jpg, CL attention overlapped the
pink pouch. Consistent labellum localization across the sample was not established.
There are no anatomical masks or localization accuracy scores, so these are qualitative
observations. Maps explain predicted logits, not a verified correct prediction.
Per-image normalization prevents comparison of absolute importance across examples.
Weak and background-focused maps are retained as limitations, not removed.

These are validation Grad-CAM examples, not the final four-layer resolver records.
Next: fit/evaluate C's frozen-policy development workflow on matching calib/val
exports, verify real resolver records, review team comparison conventions, then
perform the team-approved locked evaluation before final test examples.
