"""Normalize shared nested maps while accepting legacy flat demo/checkpoint maps."""
import json
from pathlib import Path

FEATURES = ("T", "NF", "In", "LC", "CF", "CL")


def normalize_maps(source):
    features = source.get("features", source)
    result = {"species": source.get("species"), **{key: features.get(key) for key in FEATURES}}
    for key, values in result.items():
        if (not isinstance(values, list) or not values or
            any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
            raise ValueError(f"{key}: expected unique nonempty labels in fixed order")
        if "features" in source and key in FEATURES and key in source and source[key] != values:
            raise ValueError(f"Ambiguous flat/nested class order for {key}")
    if len(result["species"]) != 156:
        raise ValueError("Expected 156 species")
    return result


def read_maps(path):
    return normalize_maps(json.loads(Path(path).read_text(encoding="utf-8")))
