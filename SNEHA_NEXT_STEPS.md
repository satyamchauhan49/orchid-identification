# Sneha's complete local demo workflow

Everything is local. No automatic commits, pushes, test evaluation or publication.

## What is ready

- Physical synthetic dataset using the shared manifest shape and 156-class output.
- Shared-normalization preflight, duplicate-cluster split checks and label validation.
- EfficientNet-B0 / ConvNeXt-T learning-rate grid: 0.001, 0.0003, 0.0001.
- CE, smoothing/focal loss, balanced sampler and mixed precision options.
- Validation macro-F1, top-k, per-value TPR/TNR, confusion matrices and checkpoint selection.
- Raw-logit exports for calib/val/test, selected checkpoint and handoff inventory.
- Species/CF/CL Grad-CAM heatmaps/overlays and batch generation.
- Taxonomic, contrastive, visual and decision records; candidate references.
- Ten demo records covering trusted, auto-resolved and deferred decisions.
- Generated comparison CSV, validation figures, report sections and responsive results site.

## Open your finished demo

Open `outputs/demo_delivery/site/index.html` in your browser (double-click).
Use the metric selector, case-stage filters and species/CF/CL selector. Expand a case
to inspect feature contrasts, reference images and its full JSON record.

The report is `outputs/demo_delivery/site/report.md`.
The handoff inventory is `outputs/demo_delivery/site/handoff_inventory.json`.
The grid summary is `outputs/demo_grid/comparison.csv`.
All demo results are intentionally marked synthetic. Never submit them as real results.

## Reproduce the entire demo in a fresh folder

With .venv active, from this project root:

```powershell
python -m models_other.demo_data
python -m models_other.experiments --config models_other/config_demo.json
python -m models_other.finish_demo
```

Those default outputs already exist after setup. For another run, copy the config,
change its output path, then pass --config to experiments and finish_demo, plus a new
--output folder for finish_demo. Nothing overwrites completed experiments.

## When Agrim's dataset arrives

1. Get the dataset, split_manifest.csv, label_maps.json, and train-only RGB mean/std.
2. Save normalization.json as {"mean":[R,G,B],"std":[R,G,B],"source":"train only"}.
   Those arrays must contain actual numbers supplied/computed from training only.
3. Agree the JSON label representation with Agrim. This loader expects top-level ordered
   lists for species, T,NF,In,LC,CF,CL. Do not reorder labels or regenerate his splits.
4. Copy models_other/config_real.example.json to models_other/config_real.json.
5. Set actual paths, epochs and batch size in that config. Use a GPU environment;
   this local CPU environment cannot use CUDA just by enabling --amp.
6. Run:

```powershell
python -m models_other.validate_data --config models_other/config_real.json
python -m models_other.experiments --config models_other/config_real.json
```

The same scripts work on Colab/Kaggle after uploading/cloning the code and installing
requirements-sneha.txt there. Use relative paths on those environments. The demo
baseline is not a real tuning recommendation; choose epochs/budget with the team.

Share selected model calib/val/test .npz exports and checkpoint through Drive, not git.
Post model name, split, row count and Drive path in the group chat. Test metrics stay
locked until Sync 4. The training script only writes test scores; it never evaluates them.

## When final explanation inputs arrive

Get Agrim's final checkpoint, Satyam's normalized/calibrated BN+resolver exchange,
taxonomy feature distributions and training/reference-image map. Agree the proposed
schema in explain/resolver_contract.md. Agrim's checkpoint may require a loader adapter
if it was not produced by this starter. Check model/feature-head identity with the team.

```text
python -m explain.build_records --input PATH_TO_RESOLVER_JSON --taxonomy PATH_TO_TAXONOMY_JSON --references PATH_TO_REFERENCES_JSON --assets-root PATH_TO_IMAGES --checkpoint PATH_TO_MATCHING_CHECKPOINT --output outputs/real_records_v1
python -m models_other.release --comparison outputs/real_grid_v1/comparison.json --records outputs/real_records_v1/records.json --output outputs/real_release_v1
```

Use train/val examples while developing. After the team runs the locked test at Sync 4,
real test examples require --locked-test-approved. Never use that flag before the sync.
Review all reference images and inspect that CL attention lands on the labellum.
Fixture decisions are simulated; replace them with actual resolver stages and confidence.

## Human/team work that cannot be completed with a demo

- Actual training, learning-rate selection and final results interpretation.
- Compare against Agrim's ResNet; review calibration with Satyam before final selection.
- Review Agrim's Table R2 and Satyam's R4, reliability/risk-coverage figures.
- Use eval/final_results.json to reconcile the final shared report and all website claims.
- Final explanation selection (about ten actual cases) and botanical inspection.
- Merge report sections at Sync 5; replace synthetic prose with real findings/limitations.
- Publish GitHub Pages only when authorized, and verify the live site/repository link.

## Optional GitHub Pages later

No action has been taken on GitHub. When you decide to publish, copy the final release
site files into site/docs (not the model outputs), review them, then follow the team's
Pages setup for that directory/branch. GitHub's normal branch publishing supports only
root or /docs, so agree a /docs root folder or use the team's Actions workflow for
site/docs. Do not enable publication of a synthetic build as if it were final results.

## Demo interpretation

The generated images are stylized shapes, not real flowers. Only four synthetic species
are represented, although the species head has 156 outputs. Training uses one epoch,
and the resolver exchange is a separate simulated fixture. It deliberately exercises
all three stages; its stage distribution is not an empirical deferral rate.
Grad-CAM uses the actual selected demo checkpoint, so its predicted labels can differ
from fixture candidates. For real records the script rejects species/CF/CL mismatch.
