# Local B+C integration

Branch: integration-b-c. No commits or pushes are performed by the integration tools.

## Completed

The integration trains EfficientNet-B0 on physical synthetic images; exports raw
calib/val/test logits with exact map-file, manifest and checkpoint hashes; fits C's
calibration/BN/resolver; evaluates validation only; adapts actual posterior and decision
artifacts by image ID; and generates ten four-layer records with matching Grad-CAM.

Satyam's BN/resolver implementation is unchanged. New integration code lives in B's
models_other/ and explain/ folders. Both legacy flat demo maps and shared nested maps
are accepted internally. New shared datasets should use C's nested schema:

```json
{"species": ["156 names in agreed order"], "features": {"T": ["..."], "NF": ["..."], "In": ["..."], "LC": ["..."], "CF": ["..."], "CL": ["..."]}}
```

The example above abbreviates domains; use Agrim's exact file without reformatting.
Hashing covers file bytes, not a reserialized or alphabetized version of its labels.
Old exported files without provenance must be re-exported from their checkpoint;
do not attach hashes to old logits on the assumption their class order matches.

## Current outputs

- outputs/integration_demo_v1/validation.json: C's actual synthetic validation results.
- outputs/integration_demo_v1/adapter/exchange.json: verified explanation exchange.
- outputs/integration_demo_v1/records/: ten explanations and Grad-CAM images.
- outputs/integration_demo_v1/site_verified/index.html: refreshed local integration site.

The tiny one-epoch model did not produce a feasible confidence policy, so all validation
cases are deferred. This is a valid integration result, not a real performance claim.
It differs from the earlier independent fixtures, which intentionally simulated all
three stages. The integration never invents missing accepted stages.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m unittest models_other.test_bc_contract -v
```

The second command uses outputs/integration_demo_v1 from the completed local run.
It tests flat/nested map compatibility, ambiguous orders, model mismatch, posterior
row reordering and altered decisions. To regenerate the whole workflow, use a new
output directory (existing outputs are protected):

```powershell
python -m models_other.integrate_demo --output outputs/integration_demo_v2
```

## After Agrim supplies real data

Run the real experiment config with nested maps. Use C's fit/evaluate or development
suite on the raw exported bundles. The simple integration uses bn_resolver.pipeline
fit (its weight and threshold reuse cal_policy, as documented by C). For the final
experimental comparison use Satyam's reviewed eval.experiments protocol, which has
more development partitions; do not present this small smoke run as that experiment.

Use the frozen artifact and matching validation evaluation to run:

```text
python -m explain.from_resolver --maps MAPS --manifest MANIFEST --bundle VAL_NPZ --artifact FROZEN_JSON --evaluation VALIDATION_JSON --checkpoint MATCHING_BEST_PT --output outputs/real_adapter_v1
python -m explain.build_records --input outputs/real_adapter_v1/exchange.json --taxonomy outputs/real_adapter_v1/taxonomy.json --references REFERENCES_JSON --assets-root IMAGE_ROOT --checkpoint MATCHING_BEST_PT --output outputs/real_records_v1
```

The adapter consumes pipeline evaluate's .decisions.json and .posteriors.npz sidecars.
It verifies maps/manifest/artifact/bundle/checkpoint identity and recomputes frozen-policy
predictions, then matches by IDs. If Satyam's experiment-suite outputs use another shape,
produce pipeline evaluate outputs from the chosen frozen artifact or review its adapter
with Satyam. Do not silently substitute a different fitting protocol for final results.

Taxonomy is the artifact's smoothed TRAIN-derived feature distributions, not expert
botanical truth. Reference images must be reviewed training/reference photos. A missing
reference stays missing; it is not synthesized for the real report.

Real test conversion and record generation require --locked-test-approved only after
Sync 4; the evaluator separately requires --split test --unlock-test. No test evaluation
was performed in this integration.

Before pushing: inspect git diff, commit the integration fixes if satisfied, coordinate
the shared schema with Agrim/Satyam, and push only this integration branch when wanted.
