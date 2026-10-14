"""Seal full-pool rank development rows and blind labels into separate files."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE


def _canonical_hash(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare(dataset, split, sealed_at):
    if (dataset.get("training_ready") is not True or dataset.get("full_pool_scope_established") is not True
            or dataset.get("target_definition") != TARGET_DEFINITION
            or dataset.get("selection_scope") != SELECTION_SCOPE):
        raise ValueError("ready full-pool ranking dataset required")
    if (split.get("status") != "READY_FOR_FULL_POOL_RANK_DEVELOPMENT"
            or split.get("blind_labels_not_summarized") is not True
            or split.get("same_signal_date_kept_together") is not True):
        raise ValueError("ready full-pool ranking split required")
    partitions = split.get("partitions", {})
    development_ids = set(partitions.get("train", [])) | set(partitions.get("calibration", [])) | set(partitions.get("validation", []))
    blind_ids = set(partitions.get("blind", []))
    if (not development_ids or not blind_ids or development_ids & blind_ids
            or sum(len(partitions.get(name, [])) for name in ("train", "calibration", "validation")) != len(development_ids)
            or len(partitions.get("blind", [])) != len(blind_ids)):
        raise ValueError("invalid development/blind identities")
    rows = {}
    for row in dataset.get("rows", []):
        candidate_id = row.get("candidate_id")
        if not candidate_id or candidate_id in rows:
            raise ValueError("missing or duplicate dataset candidate id")
        rows[candidate_id] = row
    if not development_ids | blind_ids <= set(rows):
        raise ValueError("split references unknown dataset candidates")

    development_rows = [rows[candidate_id] for candidate_id in sorted(development_ids)]
    blind_features, blind_labels = [], []
    forbidden = {"opportunity_proxy_label", "b2_close_mark_net_return_proxy", "label_available_at", "label_available"}
    for candidate_id in sorted(blind_ids):
        row = rows[candidate_id]
        if (row.get("label_available") is not True or not isinstance(row.get("opportunity_proxy_label"), bool)
                or row.get("issues")):
            raise ValueError("blind row is unresolved or invalid: " + candidate_id)
        feature_row = {key: row.get(key) for key in ("candidate_id", "signal_date", "boards", "board_group", "selected_features")}
        if forbidden & set(feature_row):
            raise AssertionError("target field leaked into blind feature view")
        blind_features.append(feature_row)
        blind_labels.append({"candidate_id": candidate_id, "signal_date": row["signal_date"],
                             "opportunity_proxy_label": row["opportunity_proxy_label"],
                             "label_available_at": row["label_available_at"]})
    blind_commitment = _canonical_hash(blind_labels)
    blind_id_commitment = _canonical_hash(sorted(blind_ids))
    development_dataset = {key: value for key, value in dataset.items() if key != "rows"}
    development_dataset.update({"rows": development_rows, "blind_rows_removed": True,
                                "blind_label_fields_present": False,
                                "blind_candidate_count": len(blind_ids),
                                "blind_candidate_ids_sha256": blind_id_commitment,
                                "blind_labels_sha256_commitment": blind_commitment})
    development_split = {key: value for key, value in split.items() if key not in {"partitions", "purged"}}
    development_split.update({"partitions": {name: list(partitions[name]) for name in ("train", "calibration", "validation")},
                              "blind_count": len(blind_ids), "blind_candidate_ids_sha256": blind_id_commitment,
                              "blind_labels_sha256_commitment": blind_commitment,
                              "blind_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                              "source_dataset_with_blind_labels_not_required_for_training": True})
    feature_view = {"kind": "FULL_POOL_RANK_BLIND_FEATURES", "sealed_at": sealed_at,
                    "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
                    "feature_allowlist": dataset.get("feature_allowlist"), "rows": blind_features,
                    "candidate_count": len(blind_features), "candidate_ids_sha256": blind_id_commitment,
                    "target_fields_absent": True}
    label_vault = {"kind": "FULL_POOL_RANK_BLIND_LABEL_VAULT", "sealed_at": sealed_at,
                   "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
                   "rows": blind_labels, "candidate_count": len(blind_labels),
                   "candidate_ids_sha256": blind_id_commitment, "labels_sha256": blind_commitment,
                   "do_not_open_before_predictions_frozen": True}
    return development_dataset, development_split, feature_view, label_vault


def run(dataset_path, split_path, output_dir):
    dataset_path, split_path, output_dir = map(Path, (dataset_path, split_path, output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new immutable blind-vault output directory")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sealed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    development, dev_split, features, labels = prepare(dataset, split, sealed_at)
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = {"development_dataset.json": development, "development_split.json": dev_split,
                "blind_features.json": features, "blind_labels_VAULT.json": labels}
    hashes = {}
    for name, payload in payloads.items():
        path = output_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"status": "BLIND_PARTITION_SEALED", "sealed_at": sealed_at,
                "development_candidate_count": len(development["rows"]),
                "blind_candidate_count": features["candidate_count"],
                "blind_candidate_ids_sha256": features["candidate_ids_sha256"],
                "blind_labels_sha256_commitment": labels["labels_sha256"],
                "artifact_hashes": hashes,
                "source_hashes": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (dataset_path, split_path, Path(__file__))},
                "operational_limitation": "filesystem owner can still open the vault; independent custody is required for organizational separation"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                         allow_nan=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.split, args.output), ensure_ascii=False, indent=2))
