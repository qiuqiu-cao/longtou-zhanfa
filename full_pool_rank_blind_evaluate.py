"""Evaluate already-frozen full-pool blind predictions against the sealed vault."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from full_pool_rank_blind_vault import _canonical_hash
from full_pool_rank_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from leader_target_screen_v17 import finite, timestamp


def _wilson(success, n, z=1.96):
    if not n:
        return None
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - radius), min(1, center + radius)]


def evaluate(predictions, label_vault, evaluated_at):
    evaluation_time = timestamp(evaluated_at)
    if (predictions.get("status") != "BLIND_RANK_PREDICTIONS_FROZEN"
            or predictions.get("predictions_frozen") is not True
            or predictions.get("blind_labels_loaded") is not False
            or predictions.get("blind_metrics") is not None
            or predictions.get("selection_scope") != SELECTION_SCOPE
            or predictions.get("target_definition") != TARGET_DEFINITION):
        raise ValueError("frozen blind predictions required")
    if (label_vault.get("kind") != "FULL_POOL_RANK_BLIND_LABEL_VAULT"
            or label_vault.get("do_not_open_before_predictions_frozen") is not True
            or label_vault.get("selection_scope") != SELECTION_SCOPE
            or label_vault.get("target_definition") != TARGET_DEFINITION):
        raise ValueError("valid blind label vault required")
    if not timestamp(label_vault["sealed_at"]) <= timestamp(predictions["scored_at"]) <= evaluation_time:
        raise ValueError("vault/prediction/evaluation time sequence invalid")
    prediction_rows = predictions.get("rows", [])
    label_rows = label_vault.get("rows", [])
    prediction_ids = [row.get("candidate_id") for row in prediction_rows]
    label_ids = [row.get("candidate_id") for row in label_rows]
    if (len(prediction_ids) != len(set(prediction_ids)) or len(label_ids) != len(set(label_ids))
            or set(prediction_ids) != set(label_ids) or not prediction_ids
            or _canonical_hash(sorted(label_ids)) != label_vault.get("candidate_ids_sha256")
            or label_vault.get("candidate_ids_sha256") != predictions.get("candidate_ids_sha256")
            or _canonical_hash(label_rows) != label_vault.get("labels_sha256")
            or label_vault.get("labels_sha256") != predictions.get("blind_labels_sha256_commitment")):
        raise ValueError("blind prediction/label commitment mismatch")
    labels = {row["candidate_id"]: row for row in label_rows}
    normalized, by_day = [], {}
    for row in prediction_rows:
        score = finite(row.get("opportunity_proxy_score"))
        label = labels[row["candidate_id"]].get("opportunity_proxy_label")
        if (score is None or not 0 <= score <= 1 or not isinstance(label, bool)
                or row.get("board_group") not in {"FIRST_BOARD", "CONSECUTIVE_BOARD"}
                or row.get("signal_date") != labels[row["candidate_id"]].get("signal_date")
                or not isinstance(row.get("is_daily_top1"), bool)):
            raise ValueError("invalid blind evaluation row")
        item = dict(row, opportunity_proxy_label=label)
        normalized.append(item)
        by_day.setdefault(row["signal_date"], []).append(item)
    if any(sum(row["is_daily_top1"] for row in rows) != 1 for rows in by_day.values()):
        raise ValueError("each blind session requires exactly one frozen top choice")
    for rows in by_day.values():
        expected = min(rows, key=lambda row: (-row["opportunity_proxy_score"], row["candidate_id"]))
        actual = next(row for row in rows if row["is_daily_top1"])
        if expected["candidate_id"] != actual["candidate_id"]:
            raise ValueError("top-one marker does not match frozen scores")
    y = np.asarray([int(row["opportunity_proxy_label"]) for row in normalized], dtype=int)
    p = np.asarray([row["opportunity_proxy_score"] for row in normalized], dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("blind pool requires both proxy classes")
    top = [next(row for row in rows if row["is_daily_top1"]) for _, rows in sorted(by_day.items())]
    baseline = [min(rows, key=lambda row: (-row["boards"], row["candidate_id"]))
                for _, rows in sorted(by_day.items())]
    top_wins = sum(row["opportunity_proxy_label"] for row in top)
    baseline_wins = sum(row["opportunity_proxy_label"] for row in baseline)
    session_count = len(by_day)
    top_rate = top_wins / session_count
    baseline_rate = baseline_wins / session_count
    group_metrics = {}
    for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD"):
        selected = [row for row in top if row["board_group"] == group]
        hits = sum(row["opportunity_proxy_label"] for row in selected)
        group_metrics[group] = {"top1_selection_count": len(selected),
                                "top1_proxy_hit_rate": hits / len(selected) if selected else None,
                                "top1_proxy_hit_wilson95": _wilson(hits, len(selected))}
    passed = session_count >= 100 and top_rate > baseline_rate
    return {"status": "BLIND_FULL_POOL_RANK_GATE_PASSED" if passed else "BLIND_FULL_POOL_RANK_GATE_FAILED",
            "selection_scope": SELECTION_SCOPE, "target_definition": TARGET_DEFINITION,
            "actual_trade_target": False, "blind_labels_untouched_until_evaluation": True,
            "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
            "blind_labels_read_by_evaluator": True, "evaluated_at": evaluation_time.isoformat(),
            "candidate_count": len(normalized), "blind_session_count": session_count,
            "candidate_auc": float(roc_auc_score(y, p)), "candidate_brier": float(brier_score_loss(y, p)),
            "blind_top1_proxy_hits": top_wins, "blind_top1_proxy_rate": top_rate,
            "blind_top1_proxy_wilson95": _wilson(top_wins, session_count),
            "blind_baseline_proxy_hits": baseline_wins, "blind_baseline_proxy_rate": baseline_rate,
            "blind_baseline_proxy_wilson95": _wilson(baseline_wins, session_count),
            "baseline_name": "highest_board_then_candidate_id", "baseline_frozen_before_blind": True,
            "outperforms_frozen_baseline": top_rate > baseline_rate,
            "first_and_relay_reported": True, "group_metrics": group_metrics,
            "can_claim_complete_trade_win_probability": False,
            "deployment_allowed": False,
            "limitations": ["blind result evaluates a daily-price opportunity proxy, not executable complete-trade return",
                            "a passing rank gate must still be combined with complete-trade calibration and forward paper validation"]}


def run(prediction_path, label_vault_path, output_path, evaluated_at):
    prediction_path, label_vault_path, target = map(Path, (prediction_path, label_vault_path, output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable blind evaluation path")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    labels = json.loads(label_vault_path.read_text(encoding="utf-8"))
    result = evaluate(predictions, labels, evaluated_at)
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (prediction_path, label_vault_path, Path(__file__))}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label-vault", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluated-at", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions, args.label_vault, args.output, args.evaluated_at),
                     ensure_ascii=False, indent=2))
