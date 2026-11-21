# Person A report — dataset and ResNet-18 experiments

## Problem formulation

The classifier predicts one of 156 orchid species and six morphology attributes (T, NF, In, LC, CF, and CL) from an image. A shared ResNet-18 backbone feeds one classification head per target. The downstream Bayesian network uses the morphology predictions as evidence, so improving recall on weak attributes may improve the evidence available to the resolver. Species accuracy alone does not establish which candidate will perform best in that pipeline.

## Dataset and preprocessing

The multi-label dataset contains 7,156 images, 156 species, and complete labels for all seven targets. The perceptual-hash audit found 446 candidate duplicate pairs at Hamming distance at most 4. Of these, 343 shared the same pHash. In the original partitions, 117 connected groups containing 268 images crossed split boundaries. The revised split keeps each connected pHash group together to reduce leakage from visually similar images. The threshold identifies candidate matches and does not establish that every pair is a duplicate.

| Partition | Images | Species represented |
|---|---:|---:|
| Train | 4,579 | 156 |
| Calibration | 509 | 133 |
| Validation | 1,273 | 152 |
| Test | 795 | 135 |

Training used random resized crops at 224 pixels and random horizontal flips. Evaluation resized the shorter edge to 256 pixels and used a 224-pixel centre crop. Channel means and standard deviations were estimated from the training split only. All E1 and E2 comparisons below use the same grouped split. The labellum-crop trial changed the input crop and recomputed training-only normalization for that view.

The supplied labellum-colour labels contain no GreenRed class. The GreenYellow flower-colour and Panicle inflorescence classes had 38 and 29 validation examples respectively, so a few predictions can move their recall substantially. See the [dataset audit](outputs/DATASET_VALIDATION_REPORT.md) and [grouped split summary](outputs/grouped_split_summary.csv).

## E1 baseline results

The baseline used an ImageNet-pretrained ResNet-18, AdamW, batch size 32, seed 42, plain cross-entropy, and a maximum of 15 epochs. The checkpoint for each learning rate was selected by minimum validation loss across the seven heads.

| Learning rate | Best epoch | Validation loss | Species top-1 |
|---|---:|---:|---:|
| 0.001 | 8 | 0.6045 | 58.99% |
| 0.0003 | 9 | 0.4970 | 70.23% |
| 0.0001 | 15 | 0.4278 | 74.94% |

The selected learning rate was **0.0001**. Its validation species top-3 and top-5 accuracies were 88.06% and 91.83%, and its species macro-F1 was 53.49%. See the [R1 validation summary](outputs/orchid-e1/e36dd98/e36dd98/R1_validation_summary.csv) and [per-class R1 results](outputs/orchid-e1/e36dd98/e36dd98/R1_validation_by_class.csv).

## E2 results and trade-offs

E2 compared label smoothing, class-balanced sampling, focal loss, an HSV colour branch, a labellum centre-crop proxy, and HSV combined with label smoothing. All trials used learning rate 0.0001 and seed 42. The smoothed trials used plain cross-entropy to select checkpoints on validation, matching E1.

- **Label smoothing** improved GreenYellow recall from 71.05% to **81.58%** and species top-1 from 74.94% to **76.43%**, the highest values across the runs. Validation loss increased from 0.4278 to 0.4621.
- **HSV** had the highest Panicle recall (**68.97%**) and lowest validation loss (**0.3994**). Its GreenYellow recall was 73.68%.
- **HSV plus label smoothing** had the highest labellum-colour macro-F1 (**82.99%**), compared with 81.49% for the baseline. Its GreenYellow recall was 71.05%, and its Panicle recall was 65.52%.
- **Class-balanced sampling** improved GreenYellow recall to 76.32%, while species top-1 fell to 72.90%. Focal loss did not improve recall for GreenYellow or Panicle. The centre-crop proxy yielded 81.30% labellum-colour macro-F1, slightly below the baseline.

The [full R2 table](outputs/orchid-e2/e36dd98/Table_R2.md) includes per-class true-negative rates. The ResNet-18 runs serve as baseline and ablation comparisons. See the [E2 handoff guide](E2_HANDOFF.md) for the ResNet training configurations and checkpoint-loading instructions.

## ConvNeXt locked-test handoff

The team selected **ConvNeXt** for the final evaluation. Sneha owns the ConvNeXt checkpoint and test-logit export. Satyam owns the frozen resolver configuration and locked evaluation.

1. **Sneha** records the selected checkpoint and its SHA-256, exact evaluation preprocessing, class maps, and grouped manifest version.
2. **Satyam** confirms the ConvNeXt calibration settings, resolver rule, and operating threshold. The team confirms the freeze before anyone opens or scores the test predictions.
3. **Sneha** exports the selected checkpoint's raw test logits and image IDs and shares the export with Satyam through Drive. The grouped test split has 795 images. **Satyam** verifies the image IDs and class order against the frozen manifest and maps, then runs the locked evaluation.
