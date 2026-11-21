# Initial dataset audit — Agrim

- Images on disk: 7156
- Rows across supplied multilabel CSVs: 7156
- Unique images represented in CSVs: 7156
- Species classes: 156
- Train/validation/test: 5,119 / 1,235 / 802
- Missing labels: none in the supplied multi-label CSVs
- CSV image names missing from image directory: none
- Images not represented in CSVs: 0
- Image extensions: JPG 7,118; JPEG 34; JFIF 4

## Data inconsistencies handled
- Training calls the texture column `T`; validation and test call it `Texture`.
- Training uses `Nospot`; validation/test use `NoSpot`. The scripts normalise both to `Nospot`.

## Important unresolved decisions
1. Work plan specifies 63 species; dataset README and CSVs contain 156. Confirm expected scope with the team.
2. pHash duplicate audit has not yet been run. Install requirements, run script 02, inspect cross-split pairs, then decide whether the official partitions need a group-aware replacement.
3. The work plan's weak label examples (such as CL GreenRed) should be mapped to the exact dataset label vocabulary before reporting comparisons.
