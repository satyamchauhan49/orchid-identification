# Sneha's local starter

Integration update: see INTEGRATION_B_C.md. The loader accepts the shared nested
`features` label map as well as legacy flat demos. New model exports include exact
label-map, manifest and checkpoint hashes for C's loader and explanation verification.

Nothing in this setup is committed or pushed. The Desktop copy is your working copy;
the original Downloads copy is separate. Work in this Desktop folder from now on.

## Open in PyCharm
1. Choose Open and select `orchid-identification` on your Desktop.
2. Open Settings > Project > Python Interpreter (wording can vary).
3. Choose an existing local interpreter: `.venv/Scripts/python.exe` in this project.
4. Open PyCharm's Terminal. Run commands below from the project root.

## First run: no dataset needed
The local environment has CPU PyTorch for a quick smoke test. Install the rest when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r models_other/requirements-sneha.txt
.\.venv\Scripts\python.exe -m models_other.train --dummy --output outputs/my_dummy_run
.\.venv\Scripts\python.exe -m models_other.check_dummy outputs/my_dummy_run
```

Choose a new output folder each run. Dummy mode uses invented labels and random images;
its results are plumbing checks, never research findings. It downloads no model weights.
`tiny` is only a smoke-test backbone. For a full EfficientNet smoke test, add
`--backbone efficientnet_b0` after installing timm.

## What is implemented
- Shared backbone plus a 156-way species head and six feature heads.
- EfficientNet-B0, ConvNeXt-T and ResNet-18 via timm.
- CE, label smoothing, focal loss, optional species-balanced sampling and CUDA AMP.
- Seed 42; partial feature labels are masked out of the loss.
- Best checkpoint selected by validation loss, plus last checkpoint each epoch.
- Contract-shaped raw logits and embeddings for calib/val/test.
- Validation macro-F1, top-1/3/5, per-value TPR/TNR and confusion matrices in val_metrics.json.
- Grad-CAM helper and four-layer explanation-record writer.

## Before real training
Agree on Python/dependency versions with the team. CPU setup is not intended for long
training; use a suitable CUDA environment, Colab or Kaggle for the real experiments.
Get the shared manifest, ordered label maps, dataset path and train-only RGB mean/std.
The work plan does not specify JSON nesting: this starter proposes ordered lists at
top-level keys `species`, `T`, `NF`, `In`, `LC`, `CF`, `CL`. Adapt the loader together
if Agrim uses a different representation; never guess or reorder classes.

Example (replace all paths and normalization placeholders with agreed actual values):

```text
python -m models_other.train --backbone efficientnet_b0 --pretrained --manifest PATH --label-maps PATH --data-root PATH --mean R G B --std R G B --epochs 10 --output outputs/effb0_base
```

Manifest columns: image_id,path,species,T,NF,In,LC,CF,CL,phash_cluster,split.
Splits: train,calib,val,test. Labels must match the JSON strings exactly; missing labels
are blank. All split/duplicate checks remain Agrim's responsibility.

Never inspect real `test.npz` or calculate test metrics before Sync 4. This starter
writes test logits but does not report test metrics. `check_dummy` refuses real runs.

## Next work
1. Agree the data contract with Agrim and Satyam.
2. Review validation metric conventions with Satyam (undefined TPR/TNR is null;
   macro-F1 averages classes present in either predictions or ground truth).
3. Run the learning-rate comparison for EfficientNet-B0 and ConvNeXt-T.
4. Integrate Grad-CAM with real images, correct normalization and final checkpoint.
5. Populate explanation records from Satyam's real BN/resolver results. Reference-image
   selection and contrastive feature comparison still need implementation.
6. Draft report and website after the team has real results.

Proposed dependencies live only in your folder so they do not replace the team's
future requirements.txt. No site has been published and no GitHub changes were sent.
