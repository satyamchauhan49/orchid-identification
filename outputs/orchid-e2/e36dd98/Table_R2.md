# Table R2: ResNet-18 validation comparison

Grouped validation split, 1,273 images. All runs use seed 42 and learning rate 0.0001.

| Run | GreenYellow TPR | GreenYellow TNR | Panicle TPR | Panicle TNR | CL macro-F1 | Species top-1 | Validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 plain cross-entropy | 71.05% | 99.76% | 51.72% | 99.84% | 81.49% | 74.94% | 0.4278 |
| Label smoothing | 81.58% | 99.43% | 62.07% | 99.84% | 82.41% | 76.43% | 0.4621 |
| Class-balanced sampling | 76.32% | 99.43% | 51.72% | 99.76% | 80.04% | 72.90% | 0.4555 |
| Focal loss | 65.79% | 99.60% | 51.72% | 99.84% | 82.18% | 74.63% | 0.4405 |
| HSV colour branch | 73.68% | 100.00% | 68.97% | 99.52% | 82.07% | 75.57% | 0.3994 |
| Labellum centre crop | 68.42% | 99.60% | 65.52% | 99.52% | 81.30% | 74.47% | 0.4221 |
| HSV + label smoothing | 71.05% | 99.84% | 65.52% | 99.68% | 82.99% | 75.73% | 0.4520 |

- Highest GreenYellow recall: Label smoothing. Highest Panicle recall: HSV colour branch.
- Highest CL macro-F1: HSV + label smoothing. Highest species top-1: Label smoothing. Lowest validation loss: HSV colour branch.
- GreenYellow has 38 validation examples and Panicle has 29. CL GreenRed is absent from this dataset. The crop is a centre-crop proxy for the labellum.
- Metrics are from validation only.
