"""Record structure only: populate these fields from actual predictions later."""
import json
from pathlib import Path


def save_record(path, image_id, taxonomic, contrastive, visual, decision):
    record = {"image_id": image_id, "taxonomic": taxonomic, "contrastive": contrastive,
              "visual": visual, "decision": decision}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record
