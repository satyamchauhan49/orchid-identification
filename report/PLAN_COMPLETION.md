# Satyam's plan tasks: independent work completion

Scope: the uploaded Three-Person One-Day Work Plan, Role C, with 156 species.
No teammate-owned training/data/site files are edited. Nothing in this checklist
asserts that synthetic performance is measured orchid performance.

| Plan item | Independent deliverable | Status / remaining dependency |
|---|---|---|
| Repo, branch, requirements, README skeleton | c-resolver, requirements-c-resolver.txt, ROLE_C_README.md | Ready; shared requirements/README merge needs team review |
| Temperature scaling, ECE, reliability diagrams | core.py; E4 runner and three calibration-ablation panels | Implemented and dummy-verified |
| NumPy soft-evidence BN | five evidence paths including selected-detector binary baseline | Implemented; actual annotations/CPT review needs Agrim |
| Agreement, fusion and selective threshold | grouped calibration partitions, frozen policy, explicit deferral | Implemented and dummy-verified |
| Unit-test calibration/BN/resolver | numerical, contract, group, integration and stress tests | Passing in clean environment |
| Base-paper rank rule | DNN in BN top-5 acceptance baseline | Implemented |
| Prepare R1-R4 templates | report/templates/R1 through R4 CSVs | Prepared with blank values; R1/R2 populated jointly by A/B/C |
| E3 and R3 | five evidence variants, masking on/off, top-1/3/5 CSV | Runner ready; real comparisons require model logits |
| E4 | DNN/BN/fused calibration on/off metrics and plots | Runner ready; actual calibration findings require logits |
| Missing evidence below 0.4 | mask rule and explicit uniform-head caveat | Implemented |
| Re-run E3/E4 on final feature model | same input interface | Waiting for final model exports |
| E5 / R4 and risk-coverage | rank, fusion, selective fusion and selective DNN comparison | Runner ready; real tuning/results pending |
| Optional logistic-regression arbiter | four-feature NumPy diagnostic, separate command | Implemented/tested; not default policy, real validation pending |
| Locked test once, bootstrap/McNemar, final JSON | guarded evaluator, grouped and IID CIs, paired comparison | Procedure ready; intentionally not run on real test |
| About 10 real explanation records | C candidate/evidence/decision data and posterior NPZ handoff | Dummy examples ready; B's real image/Grad-CAM work required |
| BN/calibration/resolver report sections | SATYAM_METHODS.md with implementation equations | Ready for report merge; update final selected settings later |
| R3/R4 report and best-model paragraph | blank CSVs, evaluation protocol, justification template | Ready to populate; real measured claims cannot be completed now |
| Limitations and future work | SATYAM_LIMITATIONS_FUTURE.md | Written; revise after actual findings |
| Verify code runs from scratch | fresh local clone, isolated venv, requirements install, full dummy flow | Verified locally; second-machine check still needs team |
| Final merge/table cross-check/live website | instructions and source-of-truth mapping | Requires final_results.json, A/B review and Sneha's website |

## What remains after this independent work

Receive verified maps/manifest/annotations and matched real calibration/validation
logits; review supports; perform real E3-E5/model selection; freeze the final choices;
integrate Sneha's real explanation visuals; complete the agreed locked test; populate
real report tables/paragraph; perform final team merge, second-machine check and
website check. No further mandatory independent development item is being hidden
behind the dummy milestone. Unexpected real-input integration fixes can still arise.

Future studies such as OOD rejection, expert survey, full cross-validation and BN
structure learning are documented as future/optional research, not prerequisites
for claiming the required independent implementation finished.
