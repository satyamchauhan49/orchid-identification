"""Load trusted Role A/B checkpoints without relabelling their head columns."""
import math
import torch
from torch import nn
from models_other.train import MultiTaskModel, FEATURES


class AgrimAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        features = []
        handle = self.model.backbone.register_forward_hook(lambda module, args, value: features.append(value))
        try:
            outputs = self.model(images)
            return outputs, features[0].flatten(1)
        finally:
            handle.remove()


def validate_normalization(config):
    for key in ("mean", "std"):
        values = config.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3 or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError("Checkpoint normalization must contain three finite mean/std values")
    if any(x <= 0 for x in config["std"]):
        raise ValueError("Checkpoint normalization std must be positive")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        from bn_resolver.prepare_metadata import canonicalize
        from models_resnet.model import OrchidMultiTaskResNet18
        maps = canonicalize(checkpoint["label_maps"])
        labels = {"species": maps["species"], **maps["features"]}
        sizes = {k: len(labels[k]) for k in FEATURES}
        if checkpoint["num_species"] != 156 or checkpoint["feature_classes"] != sizes:
            raise ValueError("Checkpoint dimensions disagree with label maps")
        hsv = 'rgb_mean' in checkpoint['model_state_dict']
        if hsv:
            from models_resnet.model_hsv_colab import OrchidHSVResNet18
            model = OrchidHSVResNet18(156, sizes, False, checkpoint['mean'], checkpoint['std'])
        else:
            model = OrchidMultiTaskResNet18(156, sizes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        wrapped = AgrimAdapter(model)
        config = {"mean": checkpoint["mean"], "std": checkpoint["std"], "preprocessing": "agrim_v1"}
        validate_normalization(config)
        return wrapped, labels, config, "agrim_hsv_resnet18" if hsv else "agrim_resnet18", "model.backbone.7.1.conv2", model.backbone[7][-1].conv2
    from explain.generate_cam import select_layer
    model = MultiTaskModel(checkpoint["backbone"], checkpoint["labels"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    config = dict(checkpoint["args"])
    # Existing checkpoints predate the crop profile and must retain their original geometry.
    config.setdefault("preprocessing", "legacy_square")
    if not config.get("dummy"):
        validate_normalization(config)
    name, layer = select_layer(model, checkpoint["backbone"])
    return model, checkpoint["labels"], config, checkpoint["backbone"], name, layer
