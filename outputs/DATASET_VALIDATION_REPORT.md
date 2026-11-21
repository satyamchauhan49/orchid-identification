# Agrim — Person A dataset validation report

## Dataset scope

- Images in multi-label dataset: **7,156**
- Species classes: **156** (all retained)
- Feature targets: **T, LC, In, NF, CL, CF**
- Missing values in these seven target columns: **0**

## Duplicate audit

- pHash candidate pairs at Hamming distance <= 4: **446**
- Exact pHash matches (distance 0): **343**
- Near matches (distance 2 or 4): **103**
- Duplicate relationships were converted to connected components so transitive matches stay together.

## Why the published split was not retained unchanged

- The duplicate audit found **117 connected duplicate groups** spanning more than one of the original train/validation/test partitions.
- Those groups contain **268 images**. Keeping the published split unchanged could leak near-identical images across evaluation partitions.
- A group-aware split was therefore generated. No pHash connected component crosses the new partitions.

## Group-aware split

| Split | Images | Species represented |
|---|---:|---:|
| train | 4579 | 156 |
| calib | 509 | 133 |
| val | 1273 | 152 |
| test | 795 | 135 |

- All **156 species** are represented in training.
- Some rare species are absent from calibration, validation, or test because some have only 3 images total and duplicate groups must not be split.
- Calibration is approximately 10% of the remaining training pool, as required by the project plan.

## Feature class counts

### T

| Class | Images |
|---|---:|
| Nospot | 5673 |
| Spot | 1483 |

### LC

| Class | Images |
|---|---:|
| Lobed | 2892 |
| Simple | 1970 |
| Pouched | 1666 |
| Fringed | 628 |

### In

| Class | Images |
|---|---:|
| Raceme | 3954 |
| SingleOrPair | 1607 |
| Spike | 1429 |
| Panicle | 166 |

### NF

| Class | Images |
|---|---:|
| Many | 3320 |
| AFew | 2235 |
| SinglePair | 1601 |

### CL

| Class | Images |
|---|---:|
| White | 2486 |
| PinkRed | 1374 |
| Green | 987 |
| BluePurple | 692 |
| Yellow | 616 |
| PinkRedWhite | 504 |
| Orange | 222 |
| YellowBrown | 131 |
| PurpleYellow | 119 |
| WhiteYellow | 25 |

### CF

| Class | Images |
|---|---:|
| GreenBrown | 1670 |
| White | 1445 |
| Green | 1237 |
| Pink | 857 |
| YellowBrown | 454 |
| Purple | 453 |
| Orange | 222 |
| GreenYellow | 207 |
| Red | 196 |
| YellowPurple | 167 |
| Yellow | 119 |
| GreenWhite | 48 |
| PurpleWhite | 34 |
| PinkWhite | 32 |
| Brown | 15 |

## Important scope mismatch to raise with the team

The project plan refers to weak classes such as **CL GreenRed** and a particular CF GreenYellow/CL GreenRed setup. The supplied multi-label CSVs use the dataset Version 1 labels: CL has 10 classes (including PinkRed and PinkRedWhite) and does **not** contain CL GreenRed. CF has 15 classes and GreenYellow has 207 examples. The exact colour-label version must be confirmed with the group before claiming to reproduce the plan’s reported weak-class figures. The current multi-task pipeline will use the actual labels in the multi-label CSVs unless the group decides otherwise.

## Files

- `duplicate_pairs.csv`: pHash candidate pairs.
- `split_manifest_grouped.csv`: final group-aware manifest.
- `grouped_split_summary.csv`: partition sizes.
- `label_maps.json`: target class order and species name mapping.
- `eda/`: feature distributions and counts.

**Do not tune hyperparameters or select models using test results. The test partition is reserved for the agreed final evaluation.
