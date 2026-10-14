"""Select the raw archive or its source-preserving missing-field supplement."""
import json
from pathlib import Path
from attention_observations_v299 import VERSION as ORIGINAL, reconstruct as base
from attention_repair_v299 import VERSION as SUPPLEMENT, reconstruct as supplemented


def reconstruct(directory):
    version = json.loads((Path(directory)/"manifest.json").read_text(encoding="utf-8")).get("version")
    if version == ORIGINAL:
        return base(directory)
    if version == SUPPLEMENT:
        return supplemented(directory)
    raise ValueError("unsupported current attention archive")
