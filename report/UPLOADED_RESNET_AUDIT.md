# Uploaded ResNet: compatibility audit

Status: SPLIT MISMATCH — real BN/calibration experiments paused.

The ZIP contains best.pt, class maps, normalization, training history and calibration/
validation arrays. test.npz was not extracted or opened. baseline_metrics.csv was not
opened because its split scope is unspecified.

Checkpoint uses epoch 5. Checkpoint label maps agree with the supplied class map;
all seven output heads pass an artificial forward test. Export target labels match
current annotation values. These checks establish structural compatibility only.

| Export | Uploaded rows | Current split rows | IDs matching current split |
|---|---:|---:|---:|
| Calibration | 499 | 509 | 44 |
| Validation | 1272 | 1273 | 224 |

The exports appear to be from the older split mentioned in Agrim's README. This
cannot be repaired by renaming files or adding hashes. Using the current TRAIN
annotations for the BN would mix incompatible partitions; using current locked-test
members for development would also contaminate the current protocol.

The reported newly selected E1 run uses epoch 15 and LR 1e-4; this ZIP's checkpoint
is epoch 5 and training history differs. Do not identify it as that selected run.
All exports omit map/manifest/checkpoint hashes, so their same-checkpoint relationship
cannot be proven from those arrays alone.

## Needed to proceed

Preferred: send the selected resnet18_base folder from the new E1 run under
MyDrive/orchid-e1/e36dd98 (the repository paths repeat e36dd98). Include the exact
split_manifest_grouped.csv used for that run, class_maps.json, normalization.json,
selection.json, best.pt, calib.npz and val.npz. Fresh exports with verified hashes
can be produced using bn_resolver.export_development when checkpoint/images are available.

Alternative: supply the original manifest and training provenance for THIS older
checkpoint. The team must explicitly agree whether to use that older protocol and
which test remains held out. We will audit duplicate isolation and source partitions
before any fitting. Merely reconstructing train as the complement of calib/val is
not a valid substitute for actual training membership.

No model training, BN performance claims, calibration fitting, test evaluation,
commit or push was performed. Supplied files remain in ignored work/uploaded-resnet.
