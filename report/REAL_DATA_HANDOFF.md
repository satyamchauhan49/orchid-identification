# Integrating real data later

1. Agrim audits 156 unique species and all six feature domains in fixed order.
   Use the exact `label_maps.json` bytes during model export, including its SHA-256.
2. Agrim supplies the grouped manifest with TRAIN feature annotations. Missing
   annotations stay blank. Do not derive truth from model predictions.
3. Agrim/Sneha supply raw calibration/validation logits using ROLE_C_README.md.
   IDs must match each split exactly. Species logits have 156 columns. The `In`
   spelling and feature class orders must match, including case.
4. Install `requirements-c-resolver.txt` in your environment; run tests.
5. Run the same suite below using real development paths and a new directory:

```sh
python -m eval.experiments --maps data/label_maps.json --manifest data/split_manifest.csv --calib outputs/resnet18_base/calib.npz --val outputs/resnet18_base/val.npz --output work/real-baseline-01
```

No test bundle is accepted by the suite. Review support, missingness and calibration
partition sizes before interpreting results. Freeze the model/evidence choices on
development data. If selecting among many variants on validation, report that
selection and reserve the locked test for the final chosen system.

The current bundle contract does not validate checkpoint/preprocessing hashes;
record those separately until the team agrees the sidecar extension. Do not silently
mix calibration and validation outputs from different checkpoints.

The final test CLI remains guarded by `--unlock-test` and requires frozen matching
map/manifest hashes. It is an accidental-use guard, not a secure vault. Keep test
labels outside ordinary development access. The final evaluator has IID bootstrap
and exact full-coverage McNemar; correlated source clusters require grouped CIs.

Review updated files on c-resolver before committing. Synthetic arrays, dependency
installs and generated run outputs live in ignored work/. Nothing is auto-pushed.
