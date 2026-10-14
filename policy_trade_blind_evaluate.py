"""Evaluate frozen complete-trade blind predictions after opening label vault."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from full_pool_rank_blind_vault import _canonical_hash
from complete_trade_dataset import TARGET_DEFINITION, SELECTION_SCOPE
from leader_target_screen_v17 import finite, timestamp


def _wilson(success, n, z=1.96):
    if not n:
        return None
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - radius), min(1, center + radius)]


def _metrics(rows):
    n = len(rows)
    returns = [row["complete_trade_net_return"] for row in rows]
    wins = sum(row["complete_trade_profit_label"] for row in rows)
    large = sum(row["large_loss_label"] for row in rows)
    positive = [value for value in returns if value > 0]
    negative = [value for value in returns if value < 0]
    equity, peak, drawdown = 1.0, 1.0, 0.0
    losing_streak = maximum_streak = 0
    for row in sorted(rows, key=lambda item: (timestamp(item["closed_at"]), item["decision_id"])):
        equity *= 1 + row["complete_trade_net_return"]
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
        losing_streak = losing_streak + 1 if row["complete_trade_net_return"] < 0 else 0
        maximum_streak = max(maximum_streak, losing_streak)
    return {"completed_trades": n, "win_rate": wins / n if n else None,
            "win_rate_wilson95": _wilson(wins, n), "big_loss_rate": large / n if n else None,
            "big_loss_rate_wilson95": _wilson(large, n),
            "mean_net_return": sum(returns) / n if n else None,
            "average_winning_return": sum(positive) / len(positive) if positive else None,
            "average_losing_return": sum(negative) / len(negative) if negative else None,
            "profit_factor": sum(positive) / abs(sum(negative)) if negative else None,
            "profit_factor_status": "DEFINED" if negative else "NO_LOSS_DENOMINATOR",
            "worst_net_return": min(returns) if returns else None,
            "trade_sequence_max_drawdown": drawdown if rows else None,
            "maximum_consecutive_losses": maximum_streak}


def evaluate(predictions, label_vault, evaluated_at):
    evaluation_time = timestamp(evaluated_at)
    if (predictions.get("status") != "POLICY_BLIND_PREDICTIONS_FROZEN"
            or predictions.get("predictions_frozen") is not True
            or predictions.get("blind_labels_loaded") is not False
            or predictions.get("blind_metrics") is not None
            or predictions.get("target_definition") != TARGET_DEFINITION
            or predictions.get("selection_scope") != SELECTION_SCOPE):
        raise ValueError("frozen policy blind predictions required")
    if (label_vault.get("kind") != "POLICY_TRADE_BLIND_LABEL_VAULT"
            or label_vault.get("do_not_open_before_predictions_frozen") is not True
            or label_vault.get("target_definition") != TARGET_DEFINITION
            or label_vault.get("selection_scope") != SELECTION_SCOPE
            or label_vault.get("policy_id") != predictions.get("policy_id")
            or label_vault.get("account_kind") != predictions.get("account_kind")):
        raise ValueError("matching policy blind label vault required")
    if not timestamp(label_vault["sealed_at"]) <= timestamp(predictions["scored_at"]) <= evaluation_time:
        raise ValueError("vault/prediction/evaluation time sequence invalid")
    pred_rows, label_rows = predictions.get("rows", []), label_vault.get("rows", [])
    pred_ids = [row.get("decision_id") for row in pred_rows]
    label_ids = [row.get("decision_id") for row in label_rows]
    if (not pred_ids or len(pred_ids) != len(set(pred_ids)) or len(label_ids) != len(set(label_ids))
            or set(pred_ids) != set(label_ids)
            or _canonical_hash(sorted(label_ids)) != label_vault.get("decision_ids_sha256")
            or label_vault.get("decision_ids_sha256") != predictions.get("decision_ids_sha256")
            or _canonical_hash(label_rows) != label_vault.get("labels_sha256")
            or label_vault.get("labels_sha256") != predictions.get("blind_labels_sha256_commitment")):
        raise ValueError("policy blind commitment mismatch")
    label_map = {row["decision_id"]: row for row in label_rows}
    joined = []
    for pred in pred_rows:
        label = label_map[pred["decision_id"]]
        ret = finite(label.get("complete_trade_net_return"))
        profit, large = label.get("complete_trade_profit_label"), label.get("large_loss_label")
        if (ret is None or not isinstance(profit, bool) or not isinstance(large, bool)
                or profit != (ret > 0) or large != (ret <= -.07)
                or pred.get("signal_date") != label.get("signal_date")
                or pred.get("board_group") != label.get("board_group")
                or not isinstance(pred.get("inside_frozen_feature_support"), bool)
                or not isinstance(pred.get("policy_selected"), bool)):
            raise ValueError("invalid or inconsistent complete-trade blind label")
        joined.append({**pred, **label, "complete_trade_net_return": ret})
    selected = [row for row in joined if row["policy_selected"]]
    baseline = [row for row in joined if row["inside_frozen_feature_support"] and row["emotion_active"]]
    if len({row["signal_date"] for row in selected}) != len(selected):
        raise ValueError("one-stock policy contains multiple selected blind trades per signal date")
    metrics, baseline_metrics = _metrics(selected), _metrics(baseline)
    group_metrics = {group: _metrics([row for row in selected if row["board_group"] == group])
                     for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD")}
    beats_baseline = bool(selected and baseline and metrics["mean_net_return"] > baseline_metrics["mean_net_return"]
                          and metrics["win_rate"] >= baseline_metrics["win_rate"]
                          and metrics["big_loss_rate"] <= baseline_metrics["big_loss_rate"])
    passed = (metrics["completed_trades"] >= 100 and metrics["win_rate"] >= .70
              and metrics["big_loss_rate"] <= .10 and metrics["mean_net_return"] > 0
              and metrics["profit_factor"] is not None and metrics["profit_factor"] > 1
              and all(group_metrics[group]["completed_trades"] > 0 for group in group_metrics)
              and beats_baseline)
    return {"status": "BLIND_COMPLETE_TRADE_GATE_PASSED" if passed else "BLIND_COMPLETE_TRADE_GATE_FAILED",
            "policy_id": predictions["policy_id"], "account_kind": predictions.get("account_kind"),
            "selection_scope": SELECTION_SCOPE, "target_definition": TARGET_DEFINITION,
            "generalization_scope": "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
            "blind_labels_untouched_until_evaluation": True, "blind_labels_read_by_evaluator": True,
            "blind_label_access_control": "SEPARATE_FEATURE_AND_LABEL_VAULT",
            "execution_basis_verified": label_vault.get("execution_independently_verified") is True,
            "metrics": metrics, "group_metrics": group_metrics,
            "baseline_name": "all_inside_support_emotion_active_policy_trades",
            "baseline_metrics": baseline_metrics, "outperforms_baseline": beats_baseline,
            "evaluated_at": evaluation_time.isoformat(), "deployment_allowed": False,
            "account_max_drawdown": None,
            "account_drawdown_status": "REQUIRES_SEPARATE_CAPITAL_WEIGHTED_ACCOUNT_LEDGER",
            "limitations": ["trade-sequence drawdown is not a capital-weighted account drawdown",
                            "blind pass still requires system acceptance and frozen forward paper verification"]}


def run(prediction_path, label_vault_path, output_path, evaluated_at):
    prediction_path, label_vault_path, target = map(Path, (prediction_path, label_vault_path, output_path))
    if target.exists():
        raise FileExistsError("choose a new immutable policy blind evaluation path")
    result = evaluate(json.loads(prediction_path.read_text(encoding="utf-8")),
                      json.loads(label_vault_path.read_text(encoding="utf-8")), evaluated_at)
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
