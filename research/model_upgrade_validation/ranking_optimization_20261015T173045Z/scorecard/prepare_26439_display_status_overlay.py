"""Append display-only research status, preserving every frozen CSV input cell."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("research/model_upgrade_validation/ranking_optimization_20261015T173045Z/scorecard")
NEW_ARM = "SOURCE_PATCHED_FEATURE_CLASSIFIER"
FAILED = "FAILED_FIXED_SCREEN_NOT_ADOPTED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-receipt", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    receipt = Path(args.source_receipt)
    target = ROOT / "source_patched26439_display_status_overlay.csv"
    manifest = ROOT / "source_patched26439_display_status_overlay_manifest.json"
    assert not target.exists() and not manifest.exists()
    prior_path = ROOT / "final_verified22662_research_view/complete_scorecard.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))["rows"]
    previous_status = {(r["date"], r["code"], r["arm"]): r["research_validation_status"] for r in prior}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        assert "research_validation_status" not in fields, "source already declares status; inspect before changing its display metadata"
        rows = list(reader)
    assert len(rows) == 26439
    counts = {}
    for row in rows:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
        if row["arm"] == NEW_ARM:
            row["research_validation_status"] = FAILED
        else:
            row["research_validation_status"] = previous_status[(row["date"], row["code"], row["arm"])]
    assert len(counts) == 7 and set(counts.values()) == {3777}
    assert sum(row["arm"] != NEW_ARM for row in rows) == len(previous_status)
    with target.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["research_validation_status"])
        writer.writeheader(); writer.writerows(rows)
    with target.open(encoding="utf-8-sig", newline="") as handle:
        copied = list(csv.DictReader(handle))
    for original, emitted in zip(rows, copied):
        assert original == emitted
    metadata = {
        "scope": "DISPLAY_STATUS_METADATA_ONLY_NO_PREDICTION_CHANGE", "rows": len(rows), "rows_per_arm": counts,
        "all_source_csv_cells_preserved_as_text": True, "original6arm_statuses_preserved": True,
        "new_arm_research_validation_status": FAILED, "new_fits": 0,
        "frozen_prediction_inputs_changed": False, "display_overlay_csv_sha256": sha(target),
        "source_bindings": {str(p.resolve()): sha(p) for p in [source, receipt, prior_path, Path(__file__)]},
    }
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "rows_per_arm": counts, "status_overlay": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
