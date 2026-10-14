"""Fail-closed compatibility audit for the two-stage leader model stack.

Stage A ranks every verified first/consecutive closing limit-up using an
opportunity proxy. Stage B estimates complete-trade profit and large-loss risk
only inside the frozen selection/execution policy's empirical support. Neither
stage may silently borrow the other's target or population.
"""
import argparse
import hashlib
import json
from pathlib import Path

from full_pool_rank_dataset import TARGET_DEFINITION as RANK_TARGET
from full_pool_rank_dataset import SELECTION_SCOPE
from complete_trade_dataset import TARGET_DEFINITION as TRADE_TARGET
from leader_target_screen_v17 import finite, timestamp


def _gate(name, passed, observed, requirement):
    return {"name": name, "passed": bool(passed), "observed": observed, "requirement": requirement}


def audit(evidence, asof):
    cutoff = timestamp(asof)
    rank = evidence.get("full_pool_ranker", {})
    trade = evidence.get("policy_conditional_trade_models", {})
    threshold = evidence.get("frozen_threshold", {})
    stack = evidence.get("stack_lock", {})

    rank_top1 = finite(rank.get("blind_top1_proxy_rate"))
    rank_base = finite(rank.get("blind_baseline_proxy_rate"))
    rank_days = finite(rank.get("blind_session_count"))
    trade_n = finite(trade.get("validation_complete_trades"))
    trade_blind_n = finite(trade.get("blind_completed_trades"))
    trade_blind_win = finite(trade.get("blind_win_rate"))
    trade_blind_loss = finite(trade.get("blind_large_loss_rate"))
    trade_blind_mean = finite(trade.get("blind_mean_net_return"))
    trade_blind_pf = finite(trade.get("blind_profit_factor"))
    trade_win = finite(threshold.get("validation_win_rate"))
    trade_loss = finite(threshold.get("validation_large_loss_rate"))
    trade_mean = finite(threshold.get("validation_mean_net_return"))
    selected = finite(threshold.get("validation_selected_trades"))
    try:
        rank_frozen = timestamp(rank["frozen_at"])
        trade_frozen = timestamp(trade["frozen_at"])
        threshold_grid_frozen = timestamp(threshold["frozen_at"])
        threshold_validated = timestamp(threshold["validated_at"])
        stack_frozen = timestamp(stack["frozen_at"])
        time_sequence = (threshold_grid_frozen <= trade_frozen <= threshold_validated <= stack_frozen <= cutoff
                         and rank_frozen <= stack_frozen)
    except (KeyError, TypeError, ValueError):
        time_sequence = False

    rank_gates = [
        _gate("rank_dataset_ready", rank.get("dataset_ready") is True,
              rank.get("dataset_ready"), "true with complete point-in-time daily manifests"),
        _gate("rank_scope_full_pool", rank.get("selection_scope") == SELECTION_SCOPE,
              rank.get("selection_scope"), SELECTION_SCOPE),
        _gate("rank_target_is_proxy", rank.get("target_definition") == RANK_TARGET
              and rank.get("actual_trade_target") is False,
              rank.get("target_definition"), RANK_TARGET + "; actual_trade_target=false"),
        _gate("rank_purged_blind", rank.get("purged_split_ready") is True
              and rank.get("blind_labels_untouched_until_evaluation") is True
              and rank.get("blind_label_access_control") == "SEPARATE_FEATURE_AND_LABEL_VAULT"
              and rank.get("blind_evaluation_status") == "BLIND_FULL_POOL_RANK_GATE_PASSED",
              {"split": rank.get("purged_split_ready"),
               "blind_labels": rank.get("blind_labels_untouched_until_evaluation"),
               "access_control": rank.get("blind_label_access_control"),
               "evaluation": rank.get("blind_evaluation_status")},
              "purged split, separate vault, and frozen blind evaluation passed"),
        _gate("rank_blind_sessions", rank_days is not None and rank_days >= 100,
              rank_days, ">=100 sessions"),
        _gate("rank_beats_frozen_baseline", rank_top1 is not None and rank_base is not None
              and rank_top1 > rank_base and rank.get("baseline_frozen_before_blind") is True,
              {"top1": rank_top1, "baseline": rank_base}, "top1 proxy rate > pre-frozen baseline"),
        _gate("rank_first_and_relay_coverage", rank.get("first_and_relay_reported") is True,
              rank.get("first_and_relay_reported"), "true"),
    ]
    trade_gates = [
        _gate("trade_dataset_ready", trade.get("complete_trade_dataset_ready") is True,
              trade.get("complete_trade_dataset_ready"), "true"),
        _gate("trade_purged_split_ready", trade.get("purged_split_ready") is True,
              trade.get("purged_split_ready"), "true with holding-window purge"),
        _gate("trade_selection_scope", trade.get("selection_scope") == SELECTION_SCOPE,
              trade.get("selection_scope"), SELECTION_SCOPE),
        _gate("trade_target_actual_complete", trade.get("target_definition") == TRADE_TARGET
              and trade.get("execution_basis_verified") is True,
              {"target": trade.get("target_definition"), "execution": trade.get("execution_basis_verified")},
              TRADE_TARGET + "; verified execution basis"),
        _gate("trade_scope_policy_conditional", trade.get("generalization_scope")
              == "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY",
              trade.get("generalization_scope"), "FROZEN_SELECTION_POLICY_CONDITIONAL_ONLY"),
        _gate("trade_models_calibrated", trade.get("profit_model_calibrated") is True
              and trade.get("large_loss_model_calibrated") is True,
              {"profit": trade.get("profit_model_calibrated"),
               "large_loss": trade.get("large_loss_model_calibrated")}, "both true on calibration partition"),
        _gate("trade_validation_sample", trade_n is not None and trade_n >= 30,
              trade_n, ">=30 complete validation trades before threshold selection"),
        _gate("trade_support_gate", trade.get("out_of_distribution_gate_frozen") is True,
              trade.get("out_of_distribution_gate_frozen"), "true; unsupported candidates abstain"),
        _gate("trade_sealed_blind_gate", trade.get("blind_label_access_control")
              == "SEPARATE_FEATURE_AND_LABEL_VAULT"
              and trade.get("blind_labels_untouched_until_evaluation") is True
              and trade.get("blind_evaluation_status") == "BLIND_COMPLETE_TRADE_GATE_PASSED",
              {"access_control": trade.get("blind_label_access_control"),
               "untouched": trade.get("blind_labels_untouched_until_evaluation"),
               "evaluation": trade.get("blind_evaluation_status")},
              "separate vault and frozen complete-trade blind evaluation passed"),
        _gate("trade_blind_complete_gate", trade_blind_n is not None and trade_blind_n >= 100
              and trade_blind_win is not None and trade_blind_win >= .70
              and trade_blind_loss is not None and trade_blind_loss <= .10
              and trade_blind_mean is not None and trade_blind_mean > 0
              and trade_blind_pf is not None and trade_blind_pf > 1
              and trade.get("blind_outperforms_baseline") is True,
              {"n": trade_blind_n, "win": trade_blind_win, "large_loss": trade_blind_loss,
               "mean": trade_blind_mean, "profit_factor": trade_blind_pf,
               "beats_baseline": trade.get("blind_outperforms_baseline")},
              ">=100; win>=70%; large loss<=10%; mean>0; PF>1; beats same-scope baseline"),
    ]
    threshold_gates = [
        _gate("threshold_predeclared", threshold.get("grid_frozen_before_model_output") is True
              and threshold.get("status") == "VALIDATION_THRESHOLD_FROZEN",
              {"grid": threshold.get("grid_frozen_before_model_output"), "status": threshold.get("status")},
              "predeclared grid and VALIDATION_THRESHOLD_FROZEN"),
        _gate("threshold_selected_count", selected is not None and selected >= 30, selected, ">=30"),
        _gate("threshold_win_gate", trade_win is not None and trade_win >= .70, trade_win, ">=70%"),
        _gate("threshold_large_loss_gate", trade_loss is not None and trade_loss <= .10, trade_loss, "<=10%"),
        _gate("threshold_positive_mean", trade_mean is not None and trade_mean > 0, trade_mean, ">0 after all costs"),
        _gate("threshold_signal_frequency", threshold.get("active_weekly_signals_within_3_to_5") is True,
              threshold.get("active_weekly_signals_within_3_to_5"), "true; inactive weeks may be zero"),
    ]
    compatibility_gates = [
        _gate("common_selection_policy", bool(stack.get("selection_policy_id"))
              and stack.get("selection_policy_id") == trade.get("selection_policy_id")
              == threshold.get("selection_policy_id"),
              {"stack": stack.get("selection_policy_id"), "trade": trade.get("selection_policy_id"),
               "threshold": threshold.get("selection_policy_id")}, "same nonempty frozen policy id"),
        _gate("stack_time_sequence", time_sequence, time_sequence,
              "models frozen before threshold; stack frozen before use"),
        _gate("stack_components_hashed", all(isinstance(stack.get(name), str) and len(stack[name]) == 64
              for name in ("rank_model_sha256", "profit_model_sha256", "large_loss_model_sha256",
                           "threshold_sha256", "feature_schema_sha256")),
              {name: bool(stack.get(name)) for name in ("rank_model_sha256", "profit_model_sha256",
               "large_loss_model_sha256", "threshold_sha256", "feature_schema_sha256")},
              "five immutable SHA-256 component hashes"),
        _gate("no_probability_conflation", stack.get("rank_score_named_probability") is False
              and stack.get("complete_trade_probability_from_rank_proxy") is False,
              {"rank_named_probability": stack.get("rank_score_named_probability"),
               "trade_probability_from_proxy": stack.get("complete_trade_probability_from_rank_proxy")},
              "rank output is a score; complete-trade odds come only from calibrated trade models"),
    ]
    gates = rank_gates + trade_gates + threshold_gates + compatibility_gates
    ready = all(gate["passed"] for gate in gates)
    return {
        "status": "MODEL_STACK_READY_FOR_LOCKED_BLIND_PAPER" if ready else "DIAGNOSTIC_ONLY",
        "can_feed_daily_decision_contract": ready,
        "live_trading_authorized": False,
        "direct_buy_recommendation_authorized": False,
        "guarantee": False,
        "rank_gates": rank_gates,
        "trade_gates": trade_gates,
        "threshold_gates": threshold_gates,
        "compatibility_gates": compatibility_gates,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "interpretation": ("the ranker orders the complete verified pool; calibrated complete-trade models may only veto or approve candidates inside policy support"),
    }


def run(input_path, output_path, asof):
    source, target = Path(input_path), Path(output_path)
    if target.exists():
        raise FileExistsError("choose a new immutable model-stack audit path")
    evidence = json.loads(source.read_text(encoding="utf-8"))
    result = audit(evidence, asof)
    result["source_hashes"] = {
        str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.asof), ensure_ascii=False, indent=2))
