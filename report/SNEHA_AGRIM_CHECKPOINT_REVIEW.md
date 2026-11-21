# Sneha: Agrim E2 checkpoint loading and Grad-CAM review

## Outcome
All three actual shared checkpoints load strictly and produce finite logits for species (156), T (2), NF (3), In (4), LC (4), CF (15) and CL (10).

| Candidate | Actual checkpoint loading | Real-image forward parity | Grad-CAM |
|---|---|---|---|
| Label smoothing | PASS | Exact equality across all seven heads | Species/CF/CL generated |
| HSV | PASS | Exact equality across all seven heads | Species/CF/CL generated |
| HSV + label smoothing | PASS | Exact equality across all seven heads | Species/CF/CL generated |

Scope: four real validation images per model (2514.jpg, 330.jpg, 720.jpg, 4575.jpg); twelve model-image checks and 36 CAM maps. All maps were finite and none were all zero. These are compatibility and qualitative visual checks, not a new accuracy evaluation. No test pixels or test arrays were opened.

## Loading issue fixed locally
The former local loader instantiated only the plain ResNet and manually applied its heads. HSV checkpoints need 48 extra histogram features for CF/CL and saved rgb_mean/rgb_std buffers. The latest combined-branch loader and package-compatible HSV module were imported into the local checkout. HSV detection now instantiates the correct architecture and calls its actual forward method, preserving the histogram branch. State dicts load with strict=True. No pretrained-weight download is required.

The resulting adapted logits exactly matched each checkpoint model's original forward output on all four real inputs, not merely an artificial tensor. Images use Resize(256), CenterCrop(224) and checkpoint-stored normalization. Downloaded checkpoint hashes, epochs, output sizes, per-image predictions and source-image hashes are recorded in results/agrim_checkpoint_review/loading_and_cam_checks.json.

## Grad-CAM observations
- 2514.jpg: all three candidates show overlap around the pink pouch. CF/CL attention is broad and includes leaves/nearby background, especially in HSV variants.
- 330.jpg: maps overlap the central flower and petals; HSV variants also emphasize surrounding background. Precise labellum localization is not established.
- 720.jpg: maps overlap parts of the flower and stalk, with substantial nearby background attention. Localization is mixed.
- 4575.jpg: broad spike and surrounding-ground attention; fine anatomical labellum localization is not established.

Four comparison figures are provided, each with three model rows and input/species/CF/CL columns. This small intentionally selected qualitative set does not establish which model is better at anatomical localization. Attention-map appearance is not a classifier accuracy measurement.

## HSV explanation limitation
The histogram uses discrete bins and has no spatial attribution in this convolutional Grad-CAM view. The gradients explain the CNN branch's contribution to the predicted colour-head logit, while the full forward prediction also uses the histogram branch. Do not describe these maps as complete explanations of the HSV model. No consistent labellum-localization claim is supported by this review.

## Message for Agrim
Confirmed: label smoothing, HSV and HSV + label smoothing checkpoints all load with the updated shared loader. All seven heads have the correct dimensions; adapter outputs exactly match original model forward outputs on four real validation images per candidate. Species/CF/CL Grad-CAM generation succeeds without zero/nonfinite maps. Maps sometimes overlap flowers/pouch regions but also background; consistent labellum localization is not established. HSV CAM explains only the CNN contribution, not its histogram contribution. No remaining loading error was observed for these three checkpoints. No test evaluation was performed.
