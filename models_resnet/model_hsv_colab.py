import torch
from torch import nn

from model import OrchidMultiTaskResNet18


class OrchidHSVResNet18(OrchidMultiTaskResNet18):
    def __init__(self, num_species, feature_classes, pretrained, mean, std):
        super().__init__(num_species, feature_classes, pretrained)
        self.register_buffer('rgb_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor(std).view(1, 3, 1, 1))
        for head in ('CF', 'CL'):
            self.feature_heads[head] = nn.Linear(self.embedding_dim + 48, feature_classes[head])

    def hsv_histogram(self, x):
        rgb = (x[:, :, ::4, ::4].float() * self.rgb_std + self.rgb_mean).clamp(0, 1)
        red, green, blue = rgb.unbind(dim=1)
        value, channel = rgb.max(dim=1)
        chroma = value - rgb.min(dim=1).values
        safe_chroma = chroma.clamp_min(1e-8)
        hue = torch.where(channel == 0, (green - blue) / safe_chroma,
                          torch.where(channel == 1, (blue - red) / safe_chroma + 2,
                                      (red - green) / safe_chroma + 4))
        hue = (hue / 6).remainder(1)
        saturation = chroma / value.clamp_min(1e-8)
        bins = (torch.stack((hue, saturation, value), dim=1).flatten(2) * 16).long().clamp(0, 15)
        histograms = rgb.new_zeros((x.shape[0], 3, 16))
        histograms.scatter_add_(2, bins, rgb.new_ones(bins.shape))
        return (histograms / bins.shape[-1]).flatten(1)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        colour_features = torch.cat((features, self.hsv_histogram(x).to(features.dtype)), dim=1)
        predictions = {'species': self.species_head(features)}
        predictions.update({head: layer(colour_features if head in ('CF', 'CL') else features)
                            for head, layer in self.feature_heads.items()})
        return predictions
