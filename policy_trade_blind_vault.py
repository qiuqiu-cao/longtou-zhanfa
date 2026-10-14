"""Separate complete-trade development rows from blind features and labels."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_blind_vault import _canonical_hash


def prepare(dataset, split, sealed_at):
    if (dataset.get("training_ready") is not True
            or dataset.get("generalization_scope") != "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY"
            or dataset.get("kind") not in {"paper", "user_confirmed"}):
        raise ValueError("ready policy-conditional complete-trade dataset required")
    if (split.get("status") != "READY_FOR_MODEL_DEVELOPMENT"
            or split.get("blind_labels_not_read_by_audit") is not True):
        raise ValueError("ready complete-trade split required")
    parts = split.get("partitions", {})
    development_ids = set(parts.get("train", [])) | set(parts.get("calibration", [])) | set(parts.get("validation", []))
    blind_ids = set(parts.get("blind", []))
    if (not development_ids or not blind_ids or development_ids & blind_ids
            or sum(len(parts.get(name, [])) for name in ("train", "calibration", "validation")) != len(development_ids)
            or len(parts.get("blind", [])) != len(blind_ids)):
        raise ValueError("invalid development/blind identities")
    row_map = {}
    for row in dataset.get("rows", []):
        decision_id = row.get("decision_id")
        if not decision_id or decision_id in row_map:
            raise ValueError("missing or duplicate decision id")
        row_map[decision_id] = row
    if not development_ids | blind_ids <= set(row_map):
        raise ValueError("split references unknown complete trades")
    development_rows = [row_map[decision_id] for decision_id in sorted(development_ids)]
    blind_features, blind_labels = [], []
    for decision_id in sorted(blind_ids):
        row = row_map[decision_id]
        if (row.get("training_eligible") is not True
                or not isinstance(row.get("complete_trade_profit_label"), bool)
                or not isinstance(row.get("large_loss_label"), bool)
                or row.get("complete_trade_net_return") is None or not row.get("closed_at")
                or row.get("issues")):
            raise ValueError("invalid blind complete-trade row: " + decision_id)
        blind_features.append({key: row.get(key) for key in
                               ("decision_id", "signal_date", "board_group", "emotion_active", "selected_features")})
        blind_labels.append({key: row.get(key) for key in
                             ("decision_id", "signal_date", "board_group", "complete_trade_profit_label",
                              "large_loss_label", "complete_trade_net_return", "closed_at")})
    id_commitment = _canonical_hash(sorted(blind_ids))
    label_commitment = _canonical_hash(blind_labels)
    development = {key: value for key, value in dataset.items() if key != "rows"}
    development.update({"rows": development_rows, "blind_rows_removed": True,
                        "blind_label_fields_present": False, "blind_decision_count": len(blind_ids),
                        "blind_decision_ids_sha256": id_commitment,
                        "blind_labels_sha256_commitment": label_commitment})
    development_split = {key: value for key, value in split.items() if key not in {"partitions", "purged"}}
    development_split.update({"partitions": {name: list(parts[name]) for name in ("train", "calibration", "validation")},
                              "blind_count": len(blind_ids), "blind_decision_ids_sha256": id_commitment,
                              "blind_labels_sha256_commitment": label_commitment,
                              "blind_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
                              "source_dataset_with_blind_labels_not_required_for_training": True})
    feature_view = {"kind": "POLICY_TRADE_BLIND_FEATURES", "sealed_at": sealed_at,
                    "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
                    "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                    "policy_id": dataset.get("policy_id"), "account_kind": dataset.get("kind"),
                    "feature_allowlist": dataset.get("feature_allowlist"),
                    "rows": blind_features, "decision_count": len(blind_features),
                    "decision_ids_sha256": id_commitment, "target_fields_absent": True}
    label_vault = {"kind": "POLICY_TRADE_BLIND_LABEL_VAULT", "sealed_at": sealed_at,
                   "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
                   "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
                   "policy_id": dataset.get("policy_id"), "account_kind": dataset.get("kind"),
                   "execution_independently_verified": dataset.get("execution_independently_verified") is True,
                   "rows": blind_labels,
                   "decision_count": len(blind_labels), "decision_ids_sha256": id_commitment,
                   "labels_sha256": label_commitment, "do_not_open_before_predictions_frozen": True}
    return development, development_split, feature_view, label_vault


def run(dataset_path, split_path, output_dir):
    dataset_path, split_path, output_dir = map(Path, (dataset_path, split_path, output_dir))
    if output_dir.exists():
        raise FileExistsError("choose a new immutable policy blind-vault directory")
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
    manifest = {"status": "POLICY_BLIND_PARTITION_SEALED", "sealed_at": sealed_at,
                "development_decision_count": len(development["rows"]),
                "blind_decision_count": features["decision_count"],
                "blind_decision_ids_sha256": features["decision_ids_sha256"],
                "blind_labels_sha256_commitment": labels["labels_sha256"],
                "artifact_hashes": hashes,
                "source_hashes": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (dataset_path, split_path, Path(__file__))},
                "operational_limitation": "filesystem owner can open the label vault; independent custody remains preferable"}
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
