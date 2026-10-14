"""Freeze policy-conditional profit/risk predictions without blind labels."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from full_pool_rank_blind_vault import _canonical_hash
from leader_target_screen_v17 import finite, timestamp


FORBIDDEN_FIELDS = {"complete_trade_profit_label", "large_loss_label",
                    "complete_trade_net_return", "closed_at", "outcome"}


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _calibrated(model, matrix):
    raw = model["base"].predict_proba(matrix)[:, 1]
    return model["calibrator"].predict_proba(_logit(raw))[:, 1]


def score(bundle, model_card, threshold_report, feature_view, scored_at):
    score_time = timestamp(scored_at)
    if (model_card.get("status") != "DEVELOPMENT_MODEL_ONLY"
            or model_card.get("deployment_allowed") is not False
            or model_card.get("blind_label_access_control") != "SEPARATE_FEATURE_AND_LABEL_VAULT"
            or model_card.get("blind_label_fields_accessed") is not False
            or model_card.get("source_payload_including_blind_rows_parsed") is not False
            or model_card.get("out_of_distribution_gate_frozen") is not True):
        raise ValueError("sealed policy development model card required")
    if (threshold_report.get("status") != "VALIDATION_THRESHOLD_FROZEN"
            or threshold_report.get("deployment_allowed") is not False
            or threshold_report.get("blind_rows_accessed") is not False
            or threshold_report.get("chosen_thresholds") is None
            or threshold_report.get("selection_policy_id") != model_card.get("policy_id")):
        raise ValueError("frozen validation threshold required")
    if not timestamp(model_card["generated_at"]) <= timestamp(threshold_report["validated_at"]) <= score_time:
        raise ValueError("model/threshold/score time sequence invalid")
    if (feature_view.get("kind") != "POLICY_TRADE_BLIND_FEATURES"
            or feature_view.get("target_fields_absent") is not True
            or feature_view.get("target_definition") != TARGET_DEFINITION
            or feature_view.get("selection_scope") != SELECTION_SCOPE
            or feature_view.get("generalization_scope") != "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY"
            or feature_view.get("policy_id") != model_card.get("policy_id")):
        raise ValueError("sealed policy blind feature view required")
    features = model_card.get("feature_allowlist")
    support = model_card.get("feature_support_envelope")
    if (features != bundle.get("feature_allowlist") or features != feature_view.get("feature_allowlist")
            or support != bundle.get("feature_support_envelope") or not isinstance(features, list) or not features
            or set(support or {}) != set(features)):
        raise ValueError("feature schema or frozen support mismatch")
    if (bundle.get("blind_labels_sha256_commitment") != model_card.get("blind_labels_sha256_commitment")
            or bundle.get("blind_decision_ids_sha256") != model_card.get("blind_decision_ids_sha256")
            or feature_view.get("decision_ids_sha256") != model_card.get("blind_decision_ids_sha256")):
        raise ValueError("blind commitment mismatch")
    rows = feature_view.get("rows", [])
    ids = [row.get("decision_id") for row in rows]
    if (not rows or any(not value for value in ids) or len(ids) != len(set(ids))
            or _canonical_hash(sorted(ids)) != feature_view.get("decision_ids_sha256")
            or feature_view.get("decision_count") != len(rows)):
        raise ValueError("invalid blind decision identities")
    matrix, normalized, in_support = [], [], []
    for row in rows:
        if FORBIDDEN_FIELDS & set(row):
            raise ValueError("target field leaked into policy blind features")
        values = row.get("selected_features", {})
        vector = [finite(values.get(name)) for name in features]
        if (any(value is None for value in vector) or not isinstance(row.get("emotion_active"), bool)
                or row.get("board_group") not in {"FIRST_BOARD", "CONSECUTIVE_BOARD"}):
            raise ValueError("invalid policy blind feature row")
        supported = all(support[name]["minimum"] <= value <= support[name]["maximum"]
                        for name, value in zip(features, vector))
        matrix.append(vector)
        in_support.append(supported)
        normalized.append({"decision_id": row["decision_id"], "signal_date": row["signal_date"],
                           "board_group": row["board_group"], "emotion_active": row["emotion_active"],
                           "inside_frozen_feature_support": supported})
    matrix = np.asarray(matrix, dtype=float)
    profit = _calibrated(bundle["profit"], matrix)
    risk = _calibrated(bundle["risk"], matrix)
    chosen = threshold_report["chosen_thresholds"]
    profit_cut, risk_cut = finite(chosen.get("profit_threshold")), finite(chosen.get("large_loss_threshold"))
    if profit_cut is None or risk_cut is None or not 0 <= profit_cut <= 1 or not 0 <= risk_cut <= 1:
        raise ValueError("invalid frozen threshold pair")
    for row, p_profit, p_risk, supported in zip(normalized, profit, risk, in_support):
        if not np.isfinite([p_profit, p_risk]).all() or not 0 <= p_profit <= 1 or not 0 <= p_risk <= 1:
            raise ValueError("invalid calibrated blind score")
        row["complete_trade_profit_probability"] = float(p_profit) if supported else None
        row["large_loss_probability"] = float(p_risk) if supported else None
        row["policy_selected"] = bool(supported and row["emotion_active"]
                                      and p_profit >= profit_cut and p_risk <= risk_cut)
    selected_days = [row["signal_date"] for row in normalized if row["policy_selected"]]
    if len(selected_days) != len(set(selected_days)):
        raise ValueError("one-stock policy selected multiple blind trades on one signal date")
    return {"status": "POLICY_BLIND_PREDICTIONS_FROZEN", "scored_at": score_time.isoformat(),
            "policy_id": model_card["policy_id"], "account_kind": feature_view.get("account_kind"),
            "target_definition": TARGET_DEFINITION, "selection_scope": SELECTION_SCOPE,
            "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
            "rows": normalized, "decision_count": len(normalized),
            "selected_count": sum(row["policy_selected"] for row in normalized),
            "out_of_support_count": sum(not value for value in in_support),
            "frozen_thresholds": {"profit_threshold": profit_cut, "large_loss_threshold": risk_cut},
            "decision_ids_sha256": feature_view["decision_ids_sha256"],
            "blind_labels_sha256_commitment": model_card["blind_labels_sha256_commitment"],
            "blind_labels_loaded": False, "blind_metrics": None, "predictions_frozen": True,
            "limitations": ["predictions apply only to complete trades generated by the frozen selection policy",
                            "out-of-support rows abstain instead of receiving extrapolated probabilities"]}


def run(model_artifact, model_card_path, threshold_path, blind_features_path, output_path):
    paths = list(map(Path, (model_artifact, model_card_path, threshold_path, blind_features_path)))
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable policy blind prediction path")
    artifact, card_path, threshold_file, feature_file = paths
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != card.get("model_artifact_sha256"):
        raise ValueError("model artifact hash mismatch")
    threshold = json.loads(threshold_file.read_text(encoding="utf-8"))
    if threshold.get("model_artifact_sha256") != card.get("model_artifact_sha256"):
        raise ValueError("threshold was not frozen for this model artifact")
    result = score(joblib.load(artifact), card, threshold,
                   json.loads(feature_file.read_text(encoding="utf-8")),
                   datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths + [Path(__file__)]}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--blind-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_artifact, args.model_card, args.threshold, args.blind_features, args.output),
                     ensure_ascii=False, indent=2))
