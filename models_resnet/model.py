"""ResNet-18 multi-task orchid classifier: 156 species + six morphology heads."""
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

FEATURES = ["T", "NF", "In", "LC", "CF", "CL"]

class OrchidMultiTaskResNet18(nn.Module):
    def __init__(self, num_species: int, feature_classes: dict, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_dim = backbone.fc.in_features
        self.species_head = nn.Linear(self.embedding_dim, num_species)
        self.feature_heads = nn.ModuleDict({
            name: nn.Linear(self.embedding_dim, feature_classes[name]) for name in FEATURES
        })

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        output = {"species": self.species_head(features)}
        output.update({name: head(features) for name, head in self.feature_heads.items()})
        return output
