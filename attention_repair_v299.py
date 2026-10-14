"""One finite supplement for explicit source gaps; the parent stays immutable."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

from attention_observations_v299 import ENDPOINTS, fetch, normalize, reconstruct as original_reconstruct, sha, write
from next_session_expectation_contract_v271 import digest
from daily_research_contract_v292 import stamp

VERSION = "v299-missing-attention-supplement-1"


def merged(parent_value, parent_entries, replacement_entries, recorded_at):
    allowed = {(row["code"], kind) for row in parent_value["observations"] for kind in row["errors"]}
    keys = [(e["code"], e["kind"]) for e in replacement_entries]
    if not keys or len(set(keys)) != len(keys) or not set(keys).issubset(allowed):
        raise ValueError("supplement must target explicit parent gaps only, without duplicates")
    index = {(e["code"], e["kind"]): e for e in parent_entries}
    for entry in replacement_entries:
        if stamp(entry["requested_at"]) < stamp(parent_value["recorded_at"]):
            raise ValueError("supplement request cannot precede parent completion")
        index[entry["code"], entry["kind"]] = entry
    combined = [index[e["code"], e["kind"]] for e in parent_entries]
    result = normalize(parent_value["scope"], combined, recorded_at)
    result["supplement"] = {"original_missing_fields": [{"code": c, "kind": k} for c, k in sorted(allowed)],
        "attempted_fields": [{"code": c, "kind": k} for c, k in keys], "original_valid_fields_overwritten": False,
        "parent_failures_retained": True, "new_archive_availability_required": True}
    return result


def reconstruct(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unsupported attention supplement")
    parent = Path(manifest["parent"])
    if sha(parent / "manifest.json") != manifest["parent_manifest_sha256"]:
        raise ValueError("attention parent manifest changed")
    for path, wanted in manifest["source_hashes"].items():
        if sha(path) != wanted:
            raise ValueError("supplement program changed")
    parent_value = original_reconstruct(parent)
    original = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        if "raw_path" in entry and sha(entry["raw_path"]) != entry["raw_sha256"]:
            raise ValueError("supplement raw response changed")
    expected = merged(parent_value, original["entries"], manifest["entries"], manifest["recorded_at"])
    stored = json.loads((directory / "observations.json").read_text(encoding="utf-8"))
    if sha(directory / "observations.json") != manifest["observations_sha256"] or digest(stored) != digest(expected):
        raise ValueError("supplement projection does not reproduce source")
    return expected


def collect(parent, output):
    parent = Path(parent).resolve(); output = Path(output)
    value = original_reconstruct(parent)
    tasks = [(row["code"], kind) for row in value["observations"] for kind in row["errors"] if kind in ENDPOINTS]
    if not tasks:
        raise ValueError("parent has no missing fields to supplement")
    output.mkdir(parents=True, exist_ok=False); (output / "raw").mkdir()
    entries = [fetch(task, output) for task in tasks]  # one attempt each; no retry loop
    original = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    recorded = datetime.now(timezone.utc).isoformat()
    result = merged(value, original["entries"], entries, recorded)
    write(output / "observations.json", result)
    write(output / "manifest.json", {"version": VERSION, "parent": str(parent),
        "parent_manifest_sha256": sha(parent / "manifest.json"), "recorded_at": recorded, "entries": entries,
        "observations_sha256": sha(output / "observations.json"),
        "source_hashes": {str(Path(__file__).resolve()): sha(__file__)}})
    rebuilt = reconstruct(output)
    return {"status": "MISSING_FIELD_SUPPLEMENT_RECONSTRUCTED", "queries": len(entries),
        "remaining_missing_fields": sum(len(r["errors"]) for r in rebuilt["observations"]),
        "original_failures_retained": True, "original_forecast_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.parent, args.output), ensure_ascii=True))


if __name__ == "__main__":
    main()
