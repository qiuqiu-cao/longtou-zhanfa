"""Score sealed full-pool blind features without loading the label vault."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from full_pool_rank_blind_vault import _canonical_hash
from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from leader_target_screen_v17 import finite, timestamp


FORBIDDEN_FIELDS = {"opportunity_proxy_label", "b2_close_mark_net_return_proxy",
                    "label_available_at", "label_available", "outcome", "target_label"}


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def score(bundle, model_card, feature_view, scored_at):
    score_time = timestamp(scored_at)
    if (model_card.get("status") != "FULL_POOL_RANK_DEVELOPMENT_MODEL_ONLY"
            or model_card.get("deployment_allowed") is not False
            or model_card.get("blind_label_access_control") != "SEPARATE_FEATURE_AND_LABEL_VAULT"
            or model_card.get("blind_label_fields_accessed") is not False
            or model_card.get("source_payload_including_blind_rows_parsed") is not False):
        raise ValueError("sealed development rank model card required")
    if timestamp(model_card["generated_at"]) > score_time:
        raise ValueError("score time precedes model generation")
    if (feature_view.get("kind") != "FULL_POOL_RANK_BLIND_FEATURES"
            or feature_view.get("target_fields_absent") is not True
            or feature_view.get("target_definition") != TARGET_DEFINITION
            or feature_view.get("selection_scope") != SELECTION_SCOPE):
        raise ValueError("sealed blind feature view required")
    features = model_card.get("feature_allowlist")
    if (features != bundle.get("feature_allowlist") or features != feature_view.get("feature_allowlist")
            or not isinstance(features, list) or not features):
        raise ValueError("feature schema mismatch")
    if (bundle.get("blind_labels_sha256_commitment") != model_card.get("blind_labels_sha256_commitment")
            or bundle.get("blind_candidate_ids_sha256") != model_card.get("blind_candidate_ids_sha256")
            or feature_view.get("candidate_ids_sha256") != model_card.get("blind_candidate_ids_sha256")):
        raise ValueError("blind commitment mismatch")
    rows = feature_view.get("rows", [])
    ids = [row.get("candidate_id") for row in rows]
    if (not rows or any(not candidate_id for candidate_id in ids) or len(ids) != len(set(ids))
            or _canonical_hash(sorted(ids)) != feature_view.get("candidate_ids_sha256")
            or feature_view.get("candidate_count") != len(rows)):
        raise ValueError("invalid blind candidate identities")
    matrix = []
    normalized = []
    for row in rows:
        if FORBIDDEN_FIELDS & set(row):
            raise ValueError("target field leaked into blind feature view")
        values = row.get("selected_features", {})
        vector = [finite(values.get(name)) for name in features]
        boards = finite(row.get("boards"))
        if (any(value is None for value in vector) or boards is None or boards < 1 or not boards.is_integer()
                or row.get("board_group") != ("FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD")):
            raise ValueError("invalid blind feature row")
        matrix.append(vector)
        normalized.append({"candidate_id": row["candidate_id"], "signal_date": row["signal_date"],
                           "boards": int(boards), "board_group": row["board_group"]})
    matrix = np.asarray(matrix, dtype=float)
    raw = bundle["model"]["base"].predict_proba(matrix)[:, 1]
    scores = bundle["model"]["calibrator"].predict_proba(_logit(raw))[:, 1]
    if len(scores) != len(rows) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("invalid model scores")
    by_day = {}
    for row, value in zip(normalized, scores):
        row["opportunity_proxy_score"] = float(value)
        by_day.setdefault(row["signal_date"], []).append(row)
    for day_rows in by_day.values():
        winner = min(day_rows, key=lambda row: (-row["opportunity_proxy_score"], row["candidate_id"]))
        for row in day_rows:
            row["is_daily_top1"] = row["candidate_id"] == winner["candidate_id"]
    return {"status": "BLIND_RANK_PREDICTIONS_FROZEN", "scored_at": score_time.isoformat(),
            "selection_scope": SELECTION_SCOPE, "target_definition": TARGET_DEFINITION,
            "score_name": "opportunity_proxy_score", "score_is_complete_trade_probability": False,
            "rows": normalized, "candidate_count": len(normalized), "session_count": len(by_day),
            "candidate_ids_sha256": feature_view["candidate_ids_sha256"],
            "blind_labels_sha256_commitment": model_card["blind_labels_sha256_commitment"],
            "blind_labels_loaded": False, "blind_metrics": None, "predictions_frozen": True,
            "limitations": ["scores target the opportunity proxy, not actual fills or complete-trade profits"]}


def run(model_artifact, model_card_path, blind_features_path, output_path):
    model_artifact, model_card_path, blind_features_path, target = map(
        Path, (model_artifact, model_card_path, blind_features_path, output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable blind prediction path")
    model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
    if hashlib.sha256(model_artifact.read_bytes()).hexdigest() != model_card.get("model_artifact_sha256"):
        raise ValueError("model artifact hash mismatch")
    bundle = joblib.load(model_artifact)
    feature_view = json.loads(blind_features_path.read_text(encoding="utf-8"))
    scored_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    result = score(bundle, model_card, feature_view, scored_at)
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (model_artifact, model_card_path, blind_features_path, Path(__file__))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-artifact", required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--blind-features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_artifact, args.model_card, args.blind_features, args.output),
                     ensure_ascii=False, indent=2))
