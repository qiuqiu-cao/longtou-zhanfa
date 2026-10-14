"""Fail-closed deployment audit for the first-board and relay-leader system."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


MIN_BLIND_COMPLETED_TRADES = 100
MIN_FORWARD_COMPLETED_TRADES = 60
TARGET_WIN_RATE = .70
MAX_BIG_LOSS_RATE = .10
REQUIRED_BOARD_LEVELS = (1, 2, 3, 4, 5)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _gate(name, passed, observed, requirement):
    return dict(name=name, passed=bool(passed), observed=observed, requirement=requirement)


def audit(evidence):
    data = evidence.get("data", {})
    blind = evidence.get("blind_complete_trades", {})
    forward = evidence.get("frozen_forward", {})
    execution_replay = evidence.get("historical_execution_replay", {})
    counts = data.get("board_observations", {})
    data_gates = [
        _gate("historical_trading_days", (_number(data.get("trading_days")) or 0) >= 250,
              data.get("trading_days"), ">=250"),
        _gate("complete_point_in_time_universe", data.get("universe_complete") is True,
              data.get("universe_complete"), "true"),
        _gate("point_in_time_security_status", data.get("point_in_time_status_complete") is True,
              data.get("point_in_time_status_complete"), "true"),
        _gate("point_in_time_cross_theme_competition",
              data.get("cross_theme_competition_ready") is True,
              data.get("cross_theme_competition_ready"),
              "complete full-market primary specific themes and cross-theme factors"),
        _gate("point_in_time_theme_catalyst_archive",
              data.get("theme_catalyst_archive_ready") is True,
              data.get("theme_catalyst_archive_ready"),
              "at least 250 complete sessions with catalyst timing and expectation status"),
        _gate("verified_severe_abnormal_path_model",
              data.get("severe_abnormal_path_ready") is True,
              data.get("severe_abnormal_path_ready"),
              "officially reconciled inputs, verified labels and time-isolated calibration readiness"),
        _gate("validated_two_to_three_day_emotion_path",
              data.get("emotion_path_ready") is True,
              data.get("emotion_path_ready"),
              "purged time-isolated path model beats Markov and prior baselines"),
        _gate("validated_same_height_group_comparison",
              data.get("same_height_group_validation_ready") is True,
              data.get("same_height_group_validation_ready"),
              "at least 100 date clusters, complete five-factor groups and stable out-of-sample uplift"),
        _gate("validated_first_board_dual_target",
              data.get("first_board_dual_target_ready") is True,
              data.get("first_board_dual_target_ready"),
              "separate calibrated B+2 positive and B+1 large-loss models pass forward validation"),
        _gate("complete_five_factor_history",
              data.get("five_factor_history_ready") is True,
              data.get("five_factor_history_ready"),
              "turnover absorption, theme purity, sector leadership, competitor elimination and announcement soft risk complete"),
        _gate("all_observed_board_levels_reported", data.get("all_observed_board_levels_reported") is True,
              data.get("all_observed_board_levels_reported"), "true, including boards above five"),
    ]
    for board in REQUIRED_BOARD_LEVELS:
        observed = counts.get(str(board))
        data_gates.append(_gate(f"board_{board}_observations", (_number(observed) or 0) >= 100,
                                observed, ">=100"))
    blind_count = _number(blind.get("completed_trades")) or 0
    blind_win = _number(blind.get("win_rate"))
    blind_loss = _number(blind.get("big_loss_rate"))
    blind_mean = _number(blind.get("mean_net_return"))
    blind_pf = _number(blind.get("profit_factor"))
    blind_dd = _number(blind.get("account_max_drawdown"))
    blind_dd_cap = _number(blind.get("frozen_max_drawdown_limit"))
    group_metrics = blind.get("group_metrics", {})
    blind_gates = [
        _gate("complete_trade_dataset_ready", blind.get("complete_trade_dataset_ready") is True,
              blind.get("complete_trade_dataset_ready"), "true"),
        _gate("purged_chronological_split_ready", blind.get("purged_split_ready") is True,
              blind.get("purged_split_ready"), "true"),
        _gate("development_model_ready", blind.get("development_model_ready") is True,
              blind.get("development_model_ready"), "true"),
        _gate("validation_threshold_frozen", blind.get("validation_threshold_frozen") is True,
              blind.get("validation_threshold_frozen"), "true without target relaxation"),
        _gate("full_pool_ranking_scope", blind.get("can_rank_full_pool") is True,
              blind.get("can_rank_full_pool"), "true for first and consecutive closing-limit pool"),
        _gate("compatible_two_stage_model_stack", blind.get("model_stack_compatible") is True,
              blind.get("model_stack_compatible"),
              "full-pool ranker plus policy-conditional profit/risk models pass compatibility audit"),
        _gate("independent_blind_window", blind.get("independent_blind_window") is True,
              blind.get("independent_blind_window"), "true and never reused for tuning"),
        _gate("verified_execution_basis", blind.get("execution_basis_verified") is True,
              blind.get("execution_basis_verified"), "true"),
        _gate("historical_full_pool_intraday_coverage",
              execution_replay.get("candidate_universe_coverage_complete") is True,
              execution_replay.get("candidate_universe_coverage_rate"),
              "100% of the frozen first-board and consecutive-board pool"),
        _gate("incomplete_intraday_universe_fails_closed",
              (execution_replay.get("candidate_universe_coverage_complete") is True
               and execution_replay.get("selection_coverage_status") == "COMPLETE_CANDIDATE_UNIVERSE")
              or (execution_replay.get("candidate_universe_coverage_complete") is False
                  and execution_replay.get("selection_coverage_status")
                  == "BLOCKED_INCOMPLETE_CANDIDATE_UNIVERSE"
                  and execution_replay.get("selected_days") == 0),
              {"coverage_complete": execution_replay.get("candidate_universe_coverage_complete"),
               "selection_status": execution_replay.get("selection_coverage_status"),
               "selected_days": execution_replay.get("selected_days")},
              "complete pool may rank; incomplete pool must produce zero selections"),
        _gate("blind_completed_trades", blind_count >= MIN_BLIND_COMPLETED_TRADES,
              blind.get("completed_trades"), f">={MIN_BLIND_COMPLETED_TRADES}"),
        _gate("complete_trade_win_rate", blind_win is not None and blind_win >= TARGET_WIN_RATE,
              blind_win, f">={TARGET_WIN_RATE:.0%}"),
        _gate("large_loss_rate", blind_loss is not None and blind_loss <= MAX_BIG_LOSS_RATE,
              blind_loss, f"<={MAX_BIG_LOSS_RATE:.0%}"),
        _gate("positive_mean_net_return", blind_mean is not None and blind_mean > 0,
              blind_mean, ">0 after all costs"),
        _gate("profit_factor", blind_pf is not None and blind_pf > 1,
              blind_pf, ">1"),
        _gate("account_drawdown_budget", blind_dd is not None and blind_dd_cap is not None
              and blind_dd >= -abs(blind_dd_cap), blind_dd,
              "known and within pre-frozen maximum drawdown"),
        _gate("better_than_simple_baseline", blind.get("outperforms_baseline") is True,
              blind.get("outperforms_baseline"), "true under identical scope/execution/costs"),
        _gate("first_and_relay_reported_separately",
              all(isinstance(group_metrics.get(group), dict) and group_metrics[group].get("completed_trades") is not None
                  for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD")),
              sorted(group_metrics), "both groups reported without selective deletion"),
    ]
    forward_count = _number(forward.get("completed_trades")) or 0
    forward_win = _number(forward.get("win_rate"))
    forward_loss = _number(forward.get("big_loss_rate"))
    forward_mean = _number(forward.get("mean_net_return"))
    forward_dd = _number(forward.get("account_max_drawdown"))
    forward_dd_cap = _number(forward.get("frozen_max_drawdown_limit"))
    forward_gates = [
        _gate("model_and_rules_frozen", forward.get("frozen_before_period") is True,
              forward.get("frozen_before_period"), "true"),
        _gate("no_forward_retuning", forward.get("no_retuning") is True,
              forward.get("no_retuning"), "true"),
        _gate("forward_completed_trades", forward_count >= MIN_FORWARD_COMPLETED_TRADES,
              forward.get("completed_trades"), f">={MIN_FORWARD_COMPLETED_TRADES}"),
        _gate("forward_complete_trade_win_rate", forward_win is not None and forward_win >= TARGET_WIN_RATE,
              forward_win, f">={TARGET_WIN_RATE:.0%}"),
        _gate("forward_large_loss_rate", forward_loss is not None and forward_loss <= MAX_BIG_LOSS_RATE,
              forward_loss, f"<={MAX_BIG_LOSS_RATE:.0%}"),
        _gate("forward_positive_mean_net_return", forward_mean is not None and forward_mean > 0,
              forward_mean, ">0 after all costs"),
        _gate("forward_account_drawdown_budget", forward_dd is not None and forward_dd_cap is not None
              and forward_dd >= -abs(forward_dd_cap), forward_dd,
              "known and within pre-frozen maximum drawdown"),
    ]
    data_pass = all(row["passed"] for row in data_gates)
    blind_pass = data_pass and all(row["passed"] for row in blind_gates)
    forward_pass = blind_pass and all(row["passed"] for row in forward_gates)
    status = "FORWARD_VALIDATED" if forward_pass else "PAPER_TRADE_READY" if blind_pass else "DIAGNOSTIC_ONLY"
    failed = [row["name"] for row in data_gates+blind_gates+forward_gates if not row["passed"]]
    return dict(status=status, deployment_allowed=status == "FORWARD_VALIDATED",
                paper_trade_ready=blind_pass, challenge_target_is_guarantee=False,
                data_gates=data_gates, blind_complete_trade_gates=blind_gates,
                frozen_forward_gates=forward_gates, failed_gates=failed,
                scope="all supplied Shanghai/Shenzhen main-board non-ST closing limit-ups: first and consecutive boards",
                note="70% is a complete-trade challenge gate, not a forecast or return guarantee")


def current_evidence(root):
    root = Path(root)
    latest_outcome_candidates = [
        root / "results" / "v152" / "full_pool_outcomes_b123_touchfixed_cutoff_20251226_v2" / "manifest.json",
        root / "data" / "v16" / "full_pool_outcomes_inventory_cutoff_20260831_v3" / "manifest.json",
        root / "results" / "full_pool_outcomes_20260905" / "manifest.json",
    ]
    outcome_path = next((path for path in latest_outcome_candidates if path.exists()), latest_outcome_candidates[-1])
    paths = {
        "inventory": root/"results"/"first_board_inventory_20260905"/"manifest.json",
        "candidates": root/"results"/"first_board_inventory_20260905"/"candidate_observations.csv",
        "outcomes": outcome_path,
        "emotion": root/"results"/"emotion_broken_feedback_20260905"/"report.json",
        "forward": root/"results"/"v10"/"intraday_forward"/"intraday_forward_monitor.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing current evidence: "+", ".join(missing))
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    outcomes = json.loads(paths["outcomes"].read_text(encoding="utf-8"))
    emotion = json.loads(paths["emotion"].read_text(encoding="utf-8"))
    forward = json.loads(paths["forward"].read_text(encoding="utf-8"))
    feature_manifest_candidates = [
        root / "data" / "v16" / "full_pool_features_inventory_cutoff_20260831_v3" / "manifest.json",
        root / "data" / "v16" / "full_pool_features_inventory_cutoff_20260831_v2" / "manifest.json",
    ]
    feature_manifest_path = next((path for path in feature_manifest_candidates if path.exists()), None)
    feature_manifest = (json.loads(feature_manifest_path.read_text(encoding="utf-8"))
                        if feature_manifest_path else {})
    proxy_candidates = [
        root / "results" / "v16" / "full_pool_proxy_backtest_inventory_cutoff_20260831_v15" / "report.json",
        root / "results" / "v16" / "full_pool_proxy_backtest_inventory_cutoff_20260831_v14" / "report.json",
        root / "results" / "v16" / "full_pool_proxy_backtest_inventory_cutoff_20260831_v13" / "report.json",
        root / "results" / "v16" / "full_pool_proxy_backtest_inventory_cutoff_20260831_v12" / "report.json",
    ]
    proxy_path = next((path for path in proxy_candidates if path.exists()), proxy_candidates[-1])
    role_candidates = [
        root / "results" / "v18" / "full_pool_role_backtest_cutoff_20260831_v11" / "report.json",
        root / "results" / "v18" / "full_pool_role_backtest_cutoff_20260831_v10" / "report.json",
        root / "results" / "v18" / "full_pool_role_backtest_cutoff_20260831_v9" / "report.json",
        root / "results" / "v18" / "full_pool_role_backtest_cutoff_20260831_v8" / "report.json",
        root / "results" / "v18" / "full_pool_role_backtest_cutoff_20260831_v7" / "report.json",
    ]
    role_path = next((path for path in role_candidates if path.exists()), role_candidates[-1])
    walkforward_candidates = [
        root / "results" / "v152" / "group_rank_walkforward_touchfixed_20260917_v4" / "report.json",
        root / "results" / "v17" / "full_pool_proxy_walkforward_cutoff_20260831_v6" / "report.json",
        root / "results" / "v17" / "full_pool_proxy_walkforward_cutoff_20260831_v5" / "report.json",
        root / "results" / "v17" / "full_pool_proxy_walkforward_cutoff_20260831_v4" / "report.json",
    ]
    walkforward_path = next((path for path in walkforward_candidates if path.exists()), walkforward_candidates[-1])
    proxy = json.loads(proxy_path.read_text(encoding="utf-8")) if proxy_path.exists() else {}
    role = json.loads(role_path.read_text(encoding="utf-8")) if role_path.exists() else {}
    walkforward = json.loads(walkforward_path.read_text(encoding="utf-8")) if walkforward_path.exists() else {}
    group_governance_path = (
        root / "results" / "v153" / "group_rank_governance_touchfixed_20260917_v1.json"
    )
    group_governance = (json.loads(group_governance_path.read_text(encoding="utf-8"))
                        if group_governance_path.exists() else {})
    cross_theme_path = (
        root / "results" / "v157" / "cross_theme_competition_20260914_v1" / "manifest.json"
    )
    cross_theme = (json.loads(cross_theme_path.read_text(encoding="utf-8"))
                   if cross_theme_path.exists() else {})
    severe_path = (
        root / "results" / "v161" / "severe_abnormal_readiness_20260917_v2.json"
    )
    severe = (json.loads(severe_path.read_text(encoding="utf-8"))
              if severe_path.exists() else {})
    emotion_path = (
        root / "results" / "v162" / "emotion_path_forecast_20260917_v1" / "report.json"
    )
    emotion_path_report = (json.loads(emotion_path.read_text(encoding="utf-8"))
                           if emotion_path.exists() else {})
    theme_archive_path = (
        root / "results" / "v164" / "point_in_time_theme_archive_20260917_v1" / "report.json"
    )
    theme_archive = (json.loads(theme_archive_path.read_text(encoding="utf-8"))
                     if theme_archive_path.exists() else {})
    dual_target_path = (
        root / "results" / "v206" / "first_board_dual_target_20260920_v5" / "report.json"
    )
    dual_target = (json.loads(dual_target_path.read_text(encoding="utf-8"))
                   if dual_target_path.exists() else {})
    five_factor_path = (
        root / "results" / "v209" / "first_board_five_factor_inputs_20260920_v4" / "report.json"
    )
    five_factor = (json.loads(five_factor_path.read_text(encoding="utf-8"))
                   if five_factor_path.exists() else {})
    same_height_path = (
        root / "results" / "v210" / "same_height_pairwise_validation_20260920_v5" / "report.json"
    )
    same_height = (json.loads(same_height_path.read_text(encoding="utf-8"))
                   if same_height_path.exists() else {})
    execution_replay_candidates = sorted(
        (root / "results" / "v48").glob("jan_feb_full_pool_coverage_*/report.json"))
    execution_replay_path = execution_replay_candidates[-1] if execution_replay_candidates else None
    execution_replay = (json.loads(execution_replay_path.read_text(encoding="utf-8"))
                        if execution_replay_path else {})
    minute_request_candidates = sorted(
        (root / "data" / "v49").glob("minute_request_pack_*/manifest.json"))
    minute_request_path = minute_request_candidates[-1] if minute_request_candidates else None
    minute_request = (json.loads(minute_request_path.read_text(encoding="utf-8"))
                      if minute_request_path else {})
    security_audits = sorted((root/"results"/"v16").glob("full_pool_security_audit_*/security_eligibility_audit.json"))
    # Prefer the audit whose candidate row count matches the immutable inventory.
    # A strict historical cutoff audit may intentionally contain fewer rows and
    # must not silently replace the full-pool status-coverage evidence.
    security_path = None
    matching_security_audits = []
    for candidate_path in security_audits:
        try:
            candidate_security = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if candidate_security.get("candidate_rows") == inventory.get("candidate_rows"):
            matching_security_audits.append(candidate_path)
    if matching_security_audits:
        security_path = matching_security_audits[-1]
    elif security_audits:
        security_path = security_audits[-1]
    security = json.loads(security_path.read_text(encoding="utf-8")) if security_path else {}
    candidates = pd.read_csv(paths["candidates"], usecols=["boards"])
    numeric = pd.to_numeric(candidates["boards"], errors="coerce")
    observed_levels = sorted({int(value) for value in numeric.dropna() if value >= 1 and float(value).is_integer()})
    counts = {str(board): int((numeric == board).sum()) for board in observed_levels}
    evidence = {
        "data": dict(trading_days=emotion.get("days"),
                     universe_complete=inventory.get("candidate_universe_complete") is True
                     and inventory.get("historical_reconstruction_only") is not True,
                     point_in_time_status_complete=(
                         security.get("candidate_rows") == inventory.get("candidate_rows")
                         and security.get("verified_coverage") == 1.0
                         and security.get("eligible_coverage") == 1.0
                     ) if security_path else inventory.get("point_in_time_status_complete") is True,
                     cross_theme_competition_ready=(
                         cross_theme.get("status") == "MODEL_FACTOR_ARCHIVE_READY"
                         and cross_theme.get("cross_theme_scope_complete") is True
                         and cross_theme.get("model_eligible_count")
                         == cross_theme.get("candidate_count")
                     ),
                     theme_catalyst_archive_ready=(
                         theme_archive.get("status") == "READY_FOR_TIME_ISOLATED_THEME_FACTOR_MODEL"
                         and theme_archive.get("model_eligible_rows")
                         == theme_archive.get("candidate_rows")
                         and theme_archive.get("distinct_sessions", 0) >= 250
                     ),
                     severe_abnormal_path_ready=(
                         severe.get("status") == "READY_FOR_TIME_ISOLATED_CALIBRATION"
                         and severe.get("calibrated_hit_probability_allowed") is True
                     ),
                     emotion_path_ready=(
                         emotion_path_report.get("blind", {}).get("promotion_gate_passed") is True
                         and emotion_path_report.get("policy", {}).get(
                             "reliable_two_to_three_day_path_claim_allowed") is True
                     ),
                     same_height_group_validation_ready=(
                         same_height.get("same_height_five_factor_uplift_validated") is True
                         and same_height.get("distinct_signal_dates", 0) >= 100
                     ),
                     first_board_dual_target_ready=(
                         dual_target.get("status") == "FORWARD_VALIDATED_DUAL_TARGET"
                         and dual_target.get("governance", {}).get(
                             "probabilities_deployment_calibrated") is True
                     ),
                     five_factor_history_ready=(
                         five_factor.get("sessions", 0) >= 250
                         and five_factor.get("five_factor_complete_rows")
                         == five_factor.get("candidate_rows")
                     ),
                     all_observed_board_levels_reported=True,
                     board_observations=counts,
                     candidate_rows=inventory.get("candidate_rows")),
        "blind_complete_trades": dict(complete_trade_dataset_ready=False,purged_split_ready=False,
                     development_model_ready=False,validation_threshold_frozen=False,can_rank_full_pool=False,
                     model_stack_compatible=False,
                     independent_blind_window=False, execution_basis_verified=False,
                     completed_trades=outcomes.get("complete_trade_count", 0), win_rate=outcomes.get("complete_trade_win_rate"),
                     big_loss_rate=None, mean_net_return=None, profit_factor=None, account_max_drawdown=None,
                     frozen_max_drawdown_limit=None, outperforms_baseline=None, group_metrics={}),
        "frozen_forward": dict(frozen_before_period=bool(forward.get("model_lock_id"))
                     and forward.get("model_scope") == "first_and_consecutive_limit_up_complete_trade"
                     and forward.get("exit_policy_frozen") is True, no_retuning=None,
                     completed_trades=forward.get("complete_trade_count", 0), win_rate=None,
                     big_loss_rate=forward.get("large_loss_rate"), mean_net_return=forward.get("mean_net_return"),
                     account_max_drawdown=None, frozen_max_drawdown_limit=None),
        "historical_execution_replay": {
            "status": execution_replay.get("status", "UNAVAILABLE"),
            "cohort_rows": execution_replay.get("cohort_rows"),
            "covered_rows": execution_replay.get("usable_signal_rows"),
            "missing_rows": execution_replay.get("missing_rows"),
            "candidate_universe_coverage_complete": execution_replay.get(
                "candidate_universe_coverage_complete", False),
            "candidate_universe_coverage_rate": execution_replay.get(
                "candidate_universe_coverage_rate"),
            "selection_coverage_status": execution_replay.get("selection_coverage_status"),
            "selected_days": execution_replay.get("overall", {}).get("selected_days"),
            "usable_board_group_counts": execution_replay.get("usable_board_group_counts", {}),
            "missing_board_group_counts": execution_replay.get("missing_board_group_counts", {}),
            "actual_verified_fills": execution_replay.get("verified_trades", 0),
            "remediation_request_pack": {
                "status": minute_request.get("status", "UNAVAILABLE"),
                "requested": minute_request.get("requested"),
                "already_complete": minute_request.get("already_complete"),
                "remaining": minute_request.get("remaining"),
                "date_count": minute_request.get("date_count"),
                "claim": minute_request.get("claim"),
            },
        },
        "research_proxy": {
            "status": proxy.get("status") if proxy else "UNAVAILABLE",
            "deployment_authorized": proxy.get("deployment_authorized") if proxy else False,
            "target": proxy.get("target"),
            "target_coverage": proxy.get("target_coverage", {}),
            "role_backtest_status": role.get("status") if role else "UNAVAILABLE",
            "role_target_coverage": role.get("target_coverage", {}),
            "feature_rebuild": {
                "status": feature_manifest.get("status") if feature_manifest else "UNAVAILABLE",
                "emotion_state_one_hot_reconstructed_rows": feature_manifest.get(
                    "emotion_state_one_hot_reconstructed_rows"),
                "emotion_state_unresolved_rows": feature_manifest.get("emotion_state_unresolved_rows"),
                "feature_complete_reconstructed_rows": feature_manifest.get("feature_complete_reconstructed_rows"),
            },
            "walkforward_status": walkforward.get("status") if walkforward else "UNAVAILABLE",
            "explicit_no_touch_negative_target": walkforward.get("target_coverage", {}).get(
                "no_touch_rows_retained_as_negative", False),
            "walkforward_folds": len(walkforward.get("windows", [])) if walkforward else 0,
            "walkforward_models": {
                name: {
                    "selected_sessions": details.get("summary", {}).get("selected_sessions"),
                    "proxy_success_rate": details.get("summary", {}).get("proxy_success_rate"),
                    "b2_positive_rate": details.get("summary", {}).get("path_proxy", {}).get("b2", {}).get("positive_rate"),
                    "b2_large_loss_rate": details.get("summary", {}).get("path_proxy", {}).get("b2", {}).get("large_loss_rate"),
                }
                for name, details in (walkforward.get("aggregate_selected", {}) if walkforward else {}).items()
            },
            "complete_trade_count": 0,
            "group_rank_governance": {
                "status": group_governance.get("status", "UNAVAILABLE"),
                "first_board_probability_improvement_claim_allowed": group_governance.get(
                    "first_board_probability_improvement_claim_allowed", False),
                "relay_probability_improvement_claim_allowed": group_governance.get(
                    "relay_probability_improvement_claim_allowed", False),
                "deployment_allowed": group_governance.get("deployment_allowed", False),
            },
            "interpretation": "diagnostic proxy evidence only; never a complete-trade gate",
        },
    }
    source_paths = list(paths.values()) + ([security_path] if security_path else [])
    if feature_manifest_path:
        source_paths.append(feature_manifest_path)
    source_paths += [path for path in (proxy_path, role_path, walkforward_path) if path.exists()]
    if group_governance_path.exists():
        source_paths.append(group_governance_path)
    if cross_theme_path.exists():
        source_paths.append(cross_theme_path)
    if severe_path.exists():
        source_paths.append(severe_path)
    if emotion_path.exists():
        source_paths.append(emotion_path)
    if theme_archive_path.exists():
        source_paths.append(theme_archive_path)
    for path in (dual_target_path, five_factor_path, same_height_path):
        if path.exists():
            source_paths.append(path)
    if minute_request_path:
        source_paths.append(minute_request_path)
    if execution_replay_path:
        source_paths.append(execution_replay_path)
    return evidence, {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="explicit acceptance evidence JSON; omit for current repository evidence")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("choose a new immutable acceptance report path")
    if args.input:
        source = Path(args.input)
        evidence = json.loads(source.read_text(encoding="utf-8"))
        hashes = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        evidence, hashes = current_evidence(args.root)
    result = audit(evidence)
    result["evidence"] = evidence
    result["source_hashes"] = hashes
    result["source_hashes"][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "failed_gates": result["failed_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
