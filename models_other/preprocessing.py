"""Versioned image transforms shared by training and explanations."""
from torchvision import transforms


def geometry(profile="agrim_v1", training=False):
    if profile == "agrim_v1":
        return ([transforms.RandomResizedCrop(224, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip()]
                if training else [transforms.Resize(256), transforms.CenterCrop(224)])
    if profile == "legacy_square":
        return [transforms.Resize((224, 224)), *([transforms.RandomHorizontalFlip()] if training else [])]
    raise ValueError(f"Unknown preprocessing profile: {profile}")


def image_transform(mean, std, profile="agrim_v1", training=False):
    return transforms.Compose([*geometry(profile, training), transforms.ToTensor(), transforms.Normalize(mean, std)])
