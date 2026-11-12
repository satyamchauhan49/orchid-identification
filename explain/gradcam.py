"""Grad-CAM for convolutional feature maps; returns a normalized 2D heatmap."""
import torch
from torch.nn import functional as F


def gradcam(model, image, target_layer, head="species", class_index=None):
    activations = []
    handle = target_layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            image = image.detach().requires_grad_(True)
            outputs, _ = model(image)
            if len(image) != 1:
                raise ValueError("Explain one image at a time")
            if not activations or activations[0].ndim != 4:
                raise ValueError("Choose a convolutional layer with N,C,H,W output")
            index = int(outputs[head].argmax(1).item()) if class_index is None else class_index
            gradient = torch.autograd.grad(outputs[head][0, index], activations[0])[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = (weights * activations[0]).sum(dim=1, keepdim=True).relu()
            heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()).clamp_min(1e-8)
            return heatmap.detach().cpu().numpy()
    finally:
        handle.remove()
        model.train(was_training)
