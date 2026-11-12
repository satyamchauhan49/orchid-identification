# Explainable, self-resolving orchid identification

Local Role C implementation for Satyam Chauhan. Planned dataset: **156 species**.
No real training or test findings are included. Species names and feature class orders
must come from Agrim's verified `label_maps.json`; synthetic names are only test fixtures.

## Collaborating without conflicts

- Agrim: `a-data`; owns `data_prep/`, `models_resnet/`, manifests and maps.
- Sneha: `b-models`; owns `models_other/`, `explain/`, `site/`.
- Satyam: `c-resolver`; owns `bn_resolver/`, `eval/`, Role C tests and report sections.
- Coordinate root README, requirements and interface changes before merging.

This is the shared repository clone on `c-resolver`. Keep Role C changes on this
branch and review shared contracts with Agrim and Sneha before merging to `main`.
New development work is left uncommitted and unpushed for Satyam to review.

## Setup and verification

Python 3.10+; CPU only. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-c-resolver.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.make_synthetic
.venv/bin/python -m bn_resolver.pipeline fit --maps work/synthetic/label_maps.json --manifest work/synthetic/split_manifest.csv --bundle work/synthetic/calib.npz --output work/synthetic/frozen.json
.venv/bin/python -m bn_resolver.pipeline evaluate --maps work/synthetic/label_maps.json --manifest work/synthetic/split_manifest.csv --bundle work/synthetic/val.npz --artifact work/synthetic/frozen.json --split val --output work/synthetic/validation.json
```

Generated synthetic numbers are integration checks, never project results.

## Input contract for Agrim and Sneha

`label_maps.json` has ordered lists: `{"species": [156 verified names], "features":
{"T": [...], "NF": [...], "In": [...], "LC": [...], "CF": [...], "CL": [...]}}`.
The actual feature domains must be audited; none are invented here.

Manifest columns: `image_id,path,species,T,NF,In,LC,CF,CL,phash_cluster,split`.
Splits: `train,calib,val,test`; blank feature labels mean missing supervision.
Each duplicate cluster must stay in one split. Label strings must match the maps.

One NPZ per `calib`, `val`, `test`, with `ids` (unique strings), `species_logits`
of shape N x 156, and `logits_T`, `logits_NF`, `logits_In`, `logits_LC`,
`logits_CF`, `logits_CL` of shape N x the respective feature class count.
Include scalar string `label_maps_sha256` (SHA-256 of the exact label-map file
used by the model). IDs must exactly cover the split; order may differ.
Export **raw logits**, not softmax probabilities. No pickle objects.

## What is implemented

- Temperature scaling with missing-label masks and support diagnostics; ECE and
  reliability-bin exports.
- Smoothed empirical TRAIN-only species-feature distributions, uniform unsupported
  rows, and explicit support counts. These are empirical, not expert botanical descriptions.
- V1 hard direct; V2 hard confusion; V3 prior-corrected soft likelihood;
  V4 confusion plus virtual soft evidence. CPTs are P(predicted | true), column normalized.
- Confidence masking at 0.4; all missing evidence returns the species prior.
- Log-linear fusion, calibration-only weight/threshold selection, trusted versus
  auto-resolved decision tags, and explicit null accepted labels on deferral.
- DNN, BN, rank baseline, fusion and selective metrics, bootstrap CIs, exact McNemar,
  risk-coverage and reliability data, and per-image decisions.
- Hash and duplicate-group checks. Calibration groups are deterministically split
  into `cal_model` and `cal_policy`; the saved IDs make this auditable.

## Development experiment suite

```sh
.venv/bin/python -m scripts.make_synthetic
.venv/bin/python -m eval.experiments --maps work/synthetic/label_maps.json --manifest work/synthetic/split_manifest.csv --calib work/synthetic/calib.npz --val work/synthetic/val.npz --output work/demo-run-01 --synthetic-only
```

Use a fresh output directory for every run; existing results are not overwritten.
The demo uses missing annotations, unequal feature domains, wrong DNN guesses,
and uninformative feature logits. All chart titles and tables mark it synthetic.

The same experiment command accepts real development inputs later: replace the four
input paths and remove `--synthetic-only`. No model/inference code changes are needed.
Never supply test logits under calibration/validation names. Exact manifest ID checks
reject accidental split swaps. No test input argument exists in this development suite.

Outputs: E3/R3 four-variant and masking comparisons; E4 head/BN/fusion temperature
diagnostics; E5/R4 rank/fusion/selective-DNN comparisons; per-feature TPR/TNR;
static charts; ten explanation records; a frozen V3 policy and a run ledger with
input hashes and partition IDs. V3 is prespecified downstream rather than chosen
from validation and silently re-evaluated as a winner on those same labels.

Group-preserving calibration partitions separate head/CPT fitting (`cal_model`),
weight fitting (`cal_fusion`), and threshold selection (`cal_threshold`). Smaller
partitions may make the requested 50-accepted operating point infeasible; that
is exported honestly as null threshold and all-deferred decisions. BN temperature
reuses cal_model and diagnostic fusion temperature reuses cal_fusion; E4 records
these limitations. E4 does not replace the frozen E5 head-calibrated policy.

Still pending: the paper's binary TPR/TNR BN reproduction, real model comparisons,
neural E2 runs (Agrim), visual Grad-CAM integration (Sneha), and final test findings.
Categorical hard-confusion V2 is not falsely labelled the original binary baseline.

## Locked evaluation

The evaluator defaults to validation. Test requires explicit `--split test --unlock-test`
at the agreed final sync, after models, maps, manifest and policy are frozen.
It refuses to overwrite an existing results file. This is an accidental-use guard,
not a secure test vault; keep test labels outside development access.
No test NPZ or final_results.json is generated by the synthetic script.

Only run the final evaluation once the team authorizes it. CIs assume independent
image units; use source/duplicate-cluster resampling if images remain correlated.
The policy's empirical calibration risk is not a guarantee of future or test risk.

## Needed to continue with real experiments

Agrim: verified 156-species maps, grouped manifest and TRAIN image-level feature labels
or a complete reviewed species-description table. Sneha/Agrim: matching raw baseline
logits for calibration and validation. Missing annotations remain missing; no fabricated
species descriptions, checkpoints, or performance figures are used.
