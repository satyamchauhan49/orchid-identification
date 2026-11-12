# Role C evaluation procedure

## Before real experiments

1. Review Agrim's manifest/maps and annotation availability with the team. Confirm
   156 verified species and fixed feature orders. Agree group units and audit cross-
   split duplicates/source leakage. Do not invent absent feature annotations.
2. Record each checkpoint, preprocessing configuration, seed and bundle hash. Confirm
   calibration and validation exports come from the same frozen checkpoint. Our
   loader checks class-map hashes/IDs, not checkpoint identity.
3. Run the input audit, then development E3/E4/E5. Fill R1/R2 jointly from A/B results;
   fill C's BN rows and R3/R4 from the suite. Inspect support and missingness first.
4. Select backbone and evidence variant using development data. E3's five variants
   are not five independent test attempts. If changing the default direct-soft path,
   save the choice and re-fit/freeze the intended pipeline before test.
5. Review E4 fitting reuse and E5 threshold feasibility. Document w, tau, temperature
   fits, empirical risk, accepted support and all calibration membership hashes.

## Sync 4: locked test (requires real inputs and team agreement)

1. Freeze code/model/manifest/maps/artifact; merge only through agreed team review.
2. Confirm output eval/final_results.json does not exist. Store the original test
   bundle and its hash. No parameter tuning or model selection after opening it.
3. Run the existing evaluator once, sharing the screen as specified in the plan:

```sh
python -m bn_resolver.pipeline evaluate --maps data/label_maps.json --manifest data/split_manifest.csv --bundle outputs/resnet18_final/test.npz --artifact work/final-choice/frozen_direct_soft.json --split test --unlock-test --group-column phash_cluster --output eval/final_results.json
```

Paths/model/evidence choices are placeholders for the team's actual frozen run.
Use a broader verified group column when appropriate. Do not execute with dummy
input and present the output as final. The guard refuses an existing output file;
it prevents accidents, not malicious repeated testing.

4. Preserve raw results, grouped CIs, McNemar, decision records, hashes and stdout.
   If code fails, record the failure and correction; do not hide reruns or tune to
   improve test numbers. Any necessary technical rerun must be documented openly.
5. Populate final tables from the JSON. Reconcile n, accepted count, coverage and
   accepted accuracy across the report. Mark infeasible/undefined values as NA.
6. Give Sneha the frozen BN posteriors/decisions to complete ~10 real explanations.
   Review trusted/auto-resolved/deferred cases without cherry-picking only successes.
7. Write the best-model paragraph and finalize limitations honestly. Report missed
   targets. Separate literature values and different dataset protocols.

## Final check

A and C each reproduce README commands from a fresh clone on another machine;
Sneha checks the live website. This chat can verify a fresh local clone only,
not claim the second-machine or live-site checks. Compare every final report number
with eval/final_results.json before submission. Do not fabricate that file early.
