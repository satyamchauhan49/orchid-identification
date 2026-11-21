# Agrim metadata integration: Role C

Source: a-data commit 7ab17b0. Local c-resolver changes only; no teammate branch edits.

Adapted flat numeric species maps to nested resolver maps using string numeric IDs
in the original numeric model order. Botanical names are a separate display sidecar.
Verified alphabetical feature orders against the training code. No columns or species
were silently reordered, no labels renamed, and no manifest splits changed.

TRAIN: 4,579 rows. Calibration: 509 rows. Built add-one-smoothed TRAIN-only theta,
feature priors and support arrays. Uniform species prior is retained pending review
of evaluation goals. This is empirical annotation-based metadata, not a calibrated
classifier or expert botanical table. No model accuracy is available yet.

Duplicate-group isolation passed on manifest metadata. Held-out rows were used only
for structural/group checks; no test images, predictions or test metrics were opened.

## Calibration support

- cal_model: 255 rows, 96 represented species.
  Missing CL values: WhiteYellow.
- cal_fusion: 126 rows, 61 represented species.
  Missing In values: Panicle.
  Missing CF values: Brown, GreenWhite, PinkWhite, PurpleWhite, Yellow.
- cal_threshold: 128 rows, 66 represented species.
  Missing CF values: Brown, PinkWhite, PurpleWhite.

Class support is sparse. Smoothed estimates for absent values are not validated
accuracy estimates. A 50-case minimum accepted threshold may be feasible, but its
1.5% empirical error target cannot be evaluated until genuine logits arrive.

## Remaining export compatibility requirement

Agrim's export_split omits label_maps_sha256. Before using real logits, Agrim should
verify the exact checkpoint class_maps.json matches canonical species/feature orders,
then export fresh arrays with the canonical label-map file SHA-256. Preserve numeric
species IDs as strings; do not replace them with names or change feature order.
Also save manifest/checkpoint/preprocessing provenance for B's integration adapter.
Do not add guessed hashes to old arrays. No automatic training-code edit was made.

CL GreenRed is absent from this dataset version. Do not alias it to PinkRed or claim
matched replication of the plan's literature weak-value results. Use actual labels.

## Reproduce locally

```sh
python -m bn_resolver.prepare_metadata --maps work/agrim-metadata/label_maps.json --manifest work/agrim-metadata/split_manifest_grouped.csv --output work/real-metadata-v2 --source-revision 7ab17b0
```

Source files were read from origin/a-data with git show and stored in ignored work/.
Generated metadata and audit JSON remain there. No dataset images/checkpoints are
committed. Real E3-E5 and final calibration still require matched trained model logits.
