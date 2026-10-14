"""Register which research artifacts may support current model claims.

The registry is deliberately fail-closed.  Historical reports built before the
explicit no-touch label was added remain reproducible, but cannot be cited as
evidence that the model improves first-board or relay-leader selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(root: Path, relative: str, claim_scope: list[str], status: str,
              reason: str) -> dict:
    path = root / relative
    return {
        "path": relative.replace("\\", "/"),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.is_file() else None,
        "status": status,
        "claim_scope": claim_scope,
        "reason": reason,
    }


def build(root: str | Path) -> dict:
    root = Path(root)
    corrected_outcome = (
        root / "results/v152/full_pool_outcomes_b123_touchfixed_cutoff_20251226_v2/manifest.json"
    )
    corrected_walkforward = (
        root / "results/v152/group_rank_walkforward_touchfixed_20260917_v4/report.json"
    )
    governance_path = root / "results/v153/group_rank_governance_touchfixed_20260917_v1.json"
    acceptance_candidates = [
        root / "results/current_acceptance_report_20260920_v217.json",
        root / "results/current_acceptance_report_20260920_v211.json",
        root / "results/current_acceptance_report_20260917_v164.json",
        root / "results/current_acceptance_report_20260917_v163.json",
        root / "results/current_acceptance_report_20260917_v161.json",
        root / "results/current_acceptance_report_20260917_v159.json",
        root / "results/current_acceptance_report_20260917_v154.json",
    ]
    acceptance_path = next((path for path in acceptance_candidates if path.exists()),
                           acceptance_candidates[-1])
    required = [corrected_outcome, corrected_walkforward, governance_path, acceptance_path]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing corrected evidence: " + ", ".join(missing))

    outcome = _read(corrected_outcome)
    walkforward = _read(corrected_walkforward)
    governance = _read(governance_path)
    acceptance = _read(acceptance_path)
    damage_path = root / "results/v155/emotion_damage_vector_20260917_v1/report.json"
    damage = _read(damage_path) if damage_path.exists() else {}
    severe_path = root / "results/v161/severe_abnormal_readiness_20260917_v2.json"
    severe = _read(severe_path) if severe_path.exists() else {}
    emotion_path_path = root / "results/v162/emotion_path_forecast_20260917_v1/report.json"
    emotion_path = _read(emotion_path_path) if emotion_path_path.exists() else {}
    emotion_dynamics_path = root / "results/v163/emotion_path_dynamics_20260917_v1/report.json"
    emotion_dynamics = _read(emotion_dynamics_path) if emotion_dynamics_path.exists() else {}
    theme_archive_path = root / "results/v164/point_in_time_theme_archive_20260917_v1/report.json"
    theme_archive = _read(theme_archive_path) if theme_archive_path.exists() else {}
    catalyst_backlog_path = root / "results/v165/theme_catalyst_backlog_20260917_v2/report.json"
    catalyst_backlog = _read(catalyst_backlog_path) if catalyst_backlog_path.exists() else {}
    catalyst_validation_path = root / "results/v165/theme_catalyst_validation_20260917_v1/report.json"
    catalyst_validation = _read(catalyst_validation_path) if catalyst_validation_path.exists() else {}
    knowledge_path = root / "results/v158/short_term_knowledge_audit_20260917_v37.json"
    knowledge = _read(knowledge_path) if knowledge_path.exists() else {}
    first_board_outcomes_path = root / "results/v125/first_board_trade_outcomes_20260916_v3/report.json"
    first_board_outcomes = (_read(first_board_outcomes_path)
                            if first_board_outcomes_path.exists() else {})
    first_board_profit_path = root / "results/v126/first_board_profit_link_20260916_v3/report.json"
    first_board_profit = (_read(first_board_profit_path)
                          if first_board_profit_path.exists() else {})
    seal_speed_path = root / "results/v160/seal_speed_tradeoff_20260917_v3/report.json"
    seal_speed = _read(seal_speed_path) if seal_speed_path.exists() else {}
    turnover_path = root / "results/v166/cumulative_turnover_tradeoff_20260917_v1/report.json"
    turnover = _read(turnover_path) if turnover_path.exists() else {}
    failed_seal_path = root / "results/v167/failed_seal_path_tradeoff_20260917_v3/report.json"
    failed_seal = _read(failed_seal_path) if failed_seal_path.exists() else {}
    fast_structure_path = root / "results/v194/fast_board_structure_tradeoff_20260917_v2/report.json"
    fast_structure = _read(fast_structure_path) if fast_structure_path.exists() else {}
    calendar_integrity_path = root / "results/v195/trading_calendar_integrity_20260917_v4/report.json"
    calendar_integrity = (_read(calendar_integrity_path)
                          if calendar_integrity_path.exists() else {})
    first_board_profit_ranker_path = root / "results/v196/first_board_profit_ranker_20260917_v2/report.json"
    first_board_profit_ranker = (_read(first_board_profit_ranker_path)
                                 if first_board_profit_ranker_path.exists() else {})
    theme_profit_coverage_path = root / "results/v197/theme_profit_coverage_20260917_v1/report.json"
    theme_profit_coverage = (_read(theme_profit_coverage_path)
                             if theme_profit_coverage_path.exists() else {})
    payoff_tail_path = root / "results/v198/first_board_payoff_tail_20260917_v1/report.json"
    payoff_tail = _read(payoff_tail_path) if payoff_tail_path.exists() else {}
    big_loss_veto_path = root / "results/v199/first_board_big_loss_veto_20260917_v1/report.json"
    big_loss_veto = _read(big_loss_veto_path) if big_loss_veto_path.exists() else {}
    opening_causality_path = root / "results/v200/opening_information_causality_20260917_v3/report.json"
    opening_causality = (_read(opening_causality_path)
                         if opening_causality_path.exists() else {})
    t1_loss_path = root / "results/v201/t1_unexitable_loss_audit_20260917_v2/report.json"
    t1_loss = _read(t1_loss_path) if t1_loss_path.exists() else {}
    exit_causality_path = root / "results/v202/exit_information_causality_audit_20260917_v1.json"
    exit_causality = (_read(exit_causality_path)
                      if exit_causality_path.exists() else {})
    index_emotion_divergence_path = root / "results/v203/index_emotion_divergence_20260917_v3/report.json"
    index_emotion_divergence = (
        _read(index_emotion_divergence_path)
        if index_emotion_divergence_path.exists() else {})
    theme_confidence_path = root / "results/review_evidence_20260916/theme_roles_v168_v1/manifest.json"
    theme_confidence = _read(theme_confidence_path) if theme_confidence_path.exists() else {}
    theme_membership_path = root / "results/review_evidence_20260916/theme_membership_v169_v2/manifest.json"
    theme_membership = _read(theme_membership_path) if theme_membership_path.exists() else {}
    cross_theme_race_path = root / "results/v170/cross_theme_race_20260916_v3/manifest.json"
    cross_theme_race = _read(cross_theme_race_path) if cross_theme_race_path.exists() else {}
    dual_target_path = root / "results/v206/first_board_dual_target_20260920_v5/report.json"
    dual_target = _read(dual_target_path) if dual_target_path.exists() else {}
    theme_driver_path = root / "results/v207/theme_driver_history_20260917_v1/report.json"
    theme_driver = _read(theme_driver_path) if theme_driver_path.exists() else {}
    severe_label_queue_path = root / "results/v208/severe_abnormal_label_queue_20260917_v1/report.json"
    severe_label_queue = (_read(severe_label_queue_path)
                          if severe_label_queue_path.exists() else {})
    five_factor_path = root / "results/v209/first_board_five_factor_inputs_20260920_v4/report.json"
    five_factor = _read(five_factor_path) if five_factor_path.exists() else {}
    same_height_pair_path = root / "results/v210/same_height_pairwise_validation_20260920_v5/report.json"
    same_height_pair = (_read(same_height_pair_path)
                        if same_height_pair_path.exists() else {})
    announcement_soft_path = root / "results/v212/announcement_soft_risk_history_20260917_v1/report.json"
    announcement_soft = (_read(announcement_soft_path)
                         if announcement_soft_path.exists() else {})
    competitor_elimination_path = root / "results/v216/competitor_elimination_reconstruction_20260920_v1/report.json"
    competitor_elimination = (_read(competitor_elimination_path)
                              if competitor_elimination_path.exists() else {})
    abnormal_role_path = root / "results/v171/abnormal_path_role_20260916_v4/report.json"
    abnormal_role = _read(abnormal_role_path) if abnormal_role_path.exists() else {}
    current_abnormal_path = root / "results/v191/current_abnormal_distance_20260916_v3/report.json"
    current_abnormal = _read(current_abnormal_path) if current_abnormal_path.exists() else {}
    reset_chronology_path = root / "results/v192/current_reset_chronology_20260916_v2/report.json"
    reset_chronology = (_read(reset_chronology_path)
                        if reset_chronology_path.exists() else {})
    reset_overlay_path = root / "results/v193/current_reset_overlay_20260916_v1/report.json"
    reset_overlay = _read(reset_overlay_path) if reset_overlay_path.exists() else {}
    ladder_gap_path = root / "results/v172/ladder_gap_tradeoff_20260917_v1/report.json"
    ladder_gap = _read(ladder_gap_path) if ladder_gap_path.exists() else {}
    first_open_path = root / "results/v173/first_open_price_discovery_20260917_v1/report.json"
    first_open = _read(first_open_path) if first_open_path.exists() else {}
    daily_role_path = root / "results/v174/daily_role_contract_20260916_authority.json"
    daily_role = _read(daily_role_path) if daily_role_path.exists() else {}
    destination_path = root / "results/v175/anchor_capital_destination_20260917_v2/report.json"
    destination = _read(destination_path) if destination_path.exists() else {}
    intraday_causality_path = root / "results/v176/intraday_flow_causality_readiness_20260917_v1.json"
    intraday_causality = _read(intraday_causality_path) if intraday_causality_path.exists() else {}
    profit_concentration_path = root / "results/v177/profit_concentration_emotion_20260917_v1/report.json"
    profit_concentration = (_read(profit_concentration_path)
                            if profit_concentration_path.exists() else {})
    breadth_quality_path = root / "results/v178/breadth_quality_emotion_20260917_v2/report.json"
    breadth_quality = _read(breadth_quality_path) if breadth_quality_path.exists() else {}
    consensus_crowding_path = root / "results/v179/consensus_crowding_tradeoff_20260917_v1/report.json"
    consensus_crowding = (_read(consensus_crowding_path)
                          if consensus_crowding_path.exists() else {})
    breadth_path_path = root / "results/v180/breadth_quality_path_20260917_v1/report.json"
    breadth_path = _read(breadth_path_path) if breadth_path_path.exists() else {}
    height_anchor_path = root / "results/v181/height_anchor_buyability_20260917_v1/report.json"
    height_anchor = _read(height_anchor_path) if height_anchor_path.exists() else {}
    opening_gap_path = root / "results/v182/opening_gap_payoff_tradeoff_20260917_v3/report.json"
    opening_gap = _read(opening_gap_path) if opening_gap_path.exists() else {}
    opening_gap_emotion_path = root / "results/v183/opening_gap_emotion_interaction_20260917_v2/report.json"
    opening_gap_emotion = (_read(opening_gap_emotion_path)
                           if opening_gap_emotion_path.exists() else {})
    weak_to_strong_path = root / "results/v184/weak_to_strong_hindsight_audit_20260917_v1/report.json"
    weak_to_strong = (_read(weak_to_strong_path)
                      if weak_to_strong_path.exists() else {})
    first_board_supply_path = root / "results/v185/first_board_supply_base_rate_audit_20260917_v1/report.json"
    first_board_supply = (_read(first_board_supply_path)
                          if first_board_supply_path.exists() else {})
    promotion_payoff_path = root / "results/v186/promotion_payoff_separation_audit_20260917_v1/report.json"
    promotion_payoff = (_read(promotion_payoff_path)
                        if promotion_payoff_path.exists() else {})
    repair_followthrough_path = root / "results/v187/repair_day_followthrough_audit_20260917_v1/report.json"
    repair_followthrough = (_read(repair_followthrough_path)
                            if repair_followthrough_path.exists() else {})
    repair_layer_path = root / "results/v188/repair_layer_transmission_audit_20260917_v1/report.json"
    repair_layer = _read(repair_layer_path) if repair_layer_path.exists() else {}
    high_to_low_path = root / "results/v189/high_to_low_ecology_audit_20260917_v2/report.json"
    high_to_low = _read(high_to_low_path) if high_to_low_path.exists() else {}
    lane_attention_path = root / "results/v190/lane_attention_competition_audit_20260917_v1/report.json"
    lane_attention = _read(lane_attention_path) if lane_attention_path.exists() else {}
    explicit_no_touch = bool(
        outcome.get("target_contract", {}).get("no_touch_rows_retained_as_negative")
        or outcome.get("no_touch_rows_retained_as_negative")
        or walkforward.get("target_coverage", {}).get("no_touch_rows_retained_as_negative")
    )
    if not explicit_no_touch:
        raise ValueError("corrected evidence must retain next-session no-touch rows as negatives")

    current = [
        _artifact(root,
                  "results/v144/emotion_actionable_risk_20260917_v1/report.json",
                  ["NEXT_SESSION_AGGREGATE_RISK_WARNING"], "CURRENT_DIAGNOSTIC",
                  "May adjust confirmation strictness; cannot rerank stocks or promise entry."),
        _artifact(root,
                  "results/v149/theme_factor_readiness_20260917_v1/report.json",
                  ["THEME_DATA_READINESS"], "CURRENT_DIAGNOSTIC",
                  "Audits whether point-in-time specific-theme evidence is trainable."),
        _artifact(root,
                  "results/v155/emotion_damage_vector_20260917_v1/report.json",
                  ["NEXT_SESSION_DAMAGE_COMPONENT_RESEARCH"], "CURRENT_DIAGNOSTIC",
                  "Only promoted components may corroborate V144; the aggregate gate currently fails."),
        _artifact(root,
                  "results/v156/emotion_scenario_20260916_v1.json",
                  ["EMOTION_FIRST_DAILY_SCENARIO"], "CURRENT_DIAGNOSTIC",
                  "Explains main, support and invalidation paths without authorizing an entry."),
        _artifact(root,
                  "results/v157/cross_theme_competition_20260914_v1/manifest.json",
                  ["CROSS_THEME_COMPETITION_READINESS"], "CURRENT_DESCRIPTIVE_ONLY",
                  "Separates first-board new-cycle races from relay height races; current scope is incomplete."),
        _artifact(root,
                  "results/v158/short_term_knowledge_audit_20260917_v37.json",
                  ["SHORT_TERM_HYPOTHESIS_GOVERNANCE"], "CURRENT_RESEARCH_GOVERNANCE",
                  "Seventy-four sourced hypotheses have proxies and counterexamples; none is production ready."),
        _artifact(root,
                  "results/v124/first_board_promotion_20260916_v5/report.json",
                  ["FIRST_BOARD_CALENDAR_CORRECTED_PROMOTION_LABELS"], "CURRENT_DIAGNOSTIC",
                  "Uses a separately verified trading calendar; missing immediate next-session pools remain unavailable rather than shifted."),
        _artifact(root,
                  "results/v125/first_board_trade_outcomes_20260916_v3/report.json",
                  ["FIRST_BOARD_EXECUTION_AND_B123_OUTCOMES"], "CURRENT_DIAGNOSTIC",
                  "Rebuilt on the verified calendar through 2026-09-16; missing bars stay unavailable and are not failed fills."),
        _artifact(root,
                  "results/v126/first_board_profit_link_20260916_v3/report.json",
                  ["FIRST_BOARD_DAILY_TOP1_EXECUTION_PAYOFF"], "CURRENT_DIAGNOSTIC",
                  "Calendar-corrected daily top-one payoff audit remains below deployment requirements."),
        _artifact(root,
                  "results/v160/seal_speed_tradeoff_20260917_v3/report.json",
                  ["FIRST_BOARD_SEAL_SPEED_TRADEOFF"], "CURRENT_DIAGNOSTIC",
                  "Corrected execution denominators show lower observable fill and higher tail loss for fast boards; neither positive ranking nor hard rejection is authorized."),
        _artifact(root,
                  "results/v166/cumulative_turnover_tradeoff_20260917_v1/report.json",
                  ["RELAY_CUMULATIVE_TURNOVER_TRADEOFF"], "CURRENT_EXPLORATORY",
                  "Identity calibration gain is uncertain and triggered B+2 trade prediction does not improve; factor use is forbidden."),
        _artifact(root,
                  "results/v167/failed_seal_path_tradeoff_20260917_v3/report.json",
                  ["FIRST_BOARD_FAILED_SEAL_PATH_TRADEOFF"], "CURRENT_DIAGNOSTIC",
                  "Quick repair is directionally better than prolonged contest, but date-clustered intervals cross zero and sample sizes are inadequate."),
        _artifact(root,
                  "results/v194/fast_board_structure_tradeoff_20260917_v2/report.json",
                  ["FAST_BOARD_STRUCTURE_VS_CONTROL_INTENT"], "CURRENT_DIAGNOSTIC",
                  "Separates thin/one-price, turnover-validated, other-fast and non-fast structures without inferring an unobservable controlling trader."),
        _artifact(root,
                  "results/v195/trading_calendar_integrity_20260917_v4/report.json",
                  ["TRADING_CALENDAR_AND_LABEL_CHRONOLOGY"], "CURRENT_AUTHORITY",
                  "Finds the sparse-review successor error and freezes the verified 33-session calendar used by the corrected first-board labels."),
        _artifact(root,
                  "results/v196/first_board_profit_ranker_20260917_v2/report.json",
                  ["FIRST_BOARD_EXECUTABLE_B2_PROFIT_RANKING"], "CURRENT_DIAGNOSTIC",
                  "Separates executable B+2 profit and B+1 tail loss from promotion; the ten-session evaluation remains research-only."),
        _artifact(root,
                  "results/v197/theme_profit_coverage_20260917_v1/report.json",
                  ["POINT_IN_TIME_THEME_COVERAGE_VS_EXECUTABLE_PAYOFF"], "CURRENT_DIAGNOSTIC",
                  "Separates roster attention and evidence availability from verified active driver, expectation and B+2 payoff; zero rows are factor eligible."),
        _artifact(root,
                  "results/v198/first_board_payoff_tail_20260917_v1/report.json",
                  ["FIRST_BOARD_TOP1_EXPECTANCY_AND_TAIL_RISK"], "CURRENT_DIAGNOSTIC",
                  "Shows that the higher emotion-ranker hit rate still has negative expectancy, sub-one profit factor and severe drawdown proxy; no ranker is promoted."),
        _artifact(root,
                  "results/v199/first_board_big_loss_veto_20260917_v1/report.json",
                  ["FIRST_BOARD_PREQUENTIAL_BIG_LOSS_VETO"], "CURRENT_DIAGNOSTIC",
                  "Separates B+1 large-loss rejection from profit ranking and audits risk-coverage without replacing vetoed top-one selections; neither model nor veto policy is promoted."),
        _artifact(root,
                  "results/v200/opening_information_causality_20260917_v3/report.json",
                  ["OPENING_INFORMATION_AND_ENTRY_CAUSALITY"], "CURRENT_DIAGNOSTIC",
                  "Forbids using the matched official opening result to filter a fill at that same auction price; later confirmation requires a new timestamped price and outcome label."),
        _artifact(root,
                  "results/v201/t1_unexitable_loss_audit_20260917_v2/report.json",
                  ["T1_UNEXITABLE_LOSS_AND_DOWN_LIMIT_QUEUE"], "CURRENT_DIAGNOSTIC",
                  "Separates B-session unexitable drawdown from the earliest B+1 exit proxy and refuses to assume lower-limit queue fills."),
        _artifact(root,
                  "results/v202/exit_information_causality_audit_20260917_v1.json",
                  ["EXIT_INFORMATION_AND_TIMING_CAUSALITY"], "CURRENT_DIAGNOSTIC",
                  "Forbids an official opening result from conditioning a sale at that same opening price and quantifies hindsight open-versus-close timing optimism."),
        _artifact(root,
                  "results/v203/index_emotion_divergence_20260917_v3/report.json",
                  ["BROAD_INDEX_VS_SHORT_TERM_RELAY_ECOLOGY"], "CURRENT_DIAGNOSTIC",
                  "Separates broad-index direction from prior-limit and failed-board ecology; all replication gates fail, so the divergence remains context only."),
        _artifact(root,
                  "results/review_evidence_20260916/theme_roles_v168_v1/manifest.json",
                  ["SECONDARY_THEME_NARRATIVE_CONFIDENCE"], "CURRENT_DESCRIPTIVE_ONLY",
                  "Covers 52 of 80 closing candidates but preserves all rows as secondary narrative, with zero verified positive-factor or high-low mapping rows."),
        _artifact(root,
                  "results/review_evidence_20260916/theme_membership_v169_v2/manifest.json",
                  ["MEMBER_THEME_BRANCH_AND_DRIVER_SEPARATION"], "CURRENT_DESCRIPTIVE_ONLY",
                  "Verifies three direct-PCB business memberships but keeps active driver, expectation, positive-factor and formal high-low permissions at zero."),
        _artifact(root,
                  "results/review_evidence_20260916/v169_authority.json",
                  ["V169_AUTHORITATIVE_ASOF_POINTER"], "CURRENT_AUTHORITY",
                  "Excludes future-asof exploratory artifacts and points to the 04:35 evidence and daily replay."),
        _artifact(root,
                  "results/v170/cross_theme_race_20260916_v3/manifest.json",
                  ["CONFIDENCE_AWARE_FIRST_RELAY_CROSS_THEME_RACES"],
                  "CURRENT_DESCRIPTIVE_ONLY",
                  "Separates first-board new-cycle races from relay height races, but 28 themes and full-market completeness remain unresolved; zero rows are model eligible."),
        _artifact(root,
                  "results/v170/cross_theme_race_20260916_authority.json",
                  ["V170_AUTHORITATIVE_RACE_POINTER"], "CURRENT_AUTHORITY",
                  "Points to the same-confidence V3 race archive and its bound daily review."),
        _artifact(root,
                  "results/v171/abnormal_path_role_20260916_v4/report.json",
                  ["SEVERE_ABNORMAL_CLOSEST_AND_PATH_ROLE_SEPARATION"],
                  "CURRENT_DIAGNOSTIC",
                  "Replays all 80 candidates after the bounded SSE reset overlay; reset blockers are removed only where official coverage is complete, while verified distances and path roles remain zero."),
        _artifact(root,
                  "results/v171/abnormal_path_role_20260916_authority.json",
                  ["V171_AUTHORITATIVE_ABNORMAL_PATH_POINTER"], "SUPERSEDED_AUTHORITY",
                  "Superseded by the V193 authority that binds the V192 reset evidence, V171 V4 replay and V31 daily review."),
        _artifact(root,
                  "results/v172/ladder_gap_tradeoff_20260917_v1/report.json",
                  ["LADDER_GAP_SCARCITY_VS_MIDDLE_DAMAGE_TRADEOFF"],
                  "CURRENT_DIAGNOSTIC",
                  "Gap-2+ anchors show higher promotion than no-gap anchors, but middle-damage separation is unstable and conditional B+2 evidence is biased; factor promotion is forbidden."),
        _artifact(root,
                  "results/v173/first_open_price_discovery_20260917_v1/report.json",
                  ["FIRST_OPEN_AFTER_ONE_PRICE_DELAYED_DISCOVERY"],
                  "CURRENT_DIAGNOSTIC",
                  "First-open relays retain promotion identity premium, but B+2 advantage is unproved and high turnover is not monotonic; automatic healthy-divergence and entry claims are forbidden."),
        _artifact(root,
                  "results/v174/daily_role_contract_20260916_authority.json",
                  ["DAILY_SIX_ROLE_AND_GROUP_PRIMARY_CONTRACT"],
                  "CURRENT_AUTHORITY",
                  "Separates six research/trading roles, first-board and relay primaries, and the unique comprehensive primary or cash; it authorizes no entry."),
        _artifact(root,
                  "results/v175/anchor_capital_destination_20260917_v2/report.json",
                  ["HEIGHT_ANCHOR_FAILURE_CAPITAL_DESTINATION_LANES"],
                  "CURRENT_DIAGNOSTIC",
                  "Uses lane promotion-rate dominance to describe realized capital destinations; any-stock flags are rejected and no prior-close factor or high-low entry is authorized."),
        _artifact(root,
                  "results/v176/intraday_flow_causality_readiness_20260917_v1.json",
                  ["AUCTION_NET_FLOW_AND_THEME_MINUTE_SEQUENCE_READINESS"],
                  "CURRENT_DATA_COLLECTION_CONTRACT",
                  "Defines Level-1 net matched-amount changes and minute event order without claiming gross cancels, replenishment, manipulation, causal feedback, ranking, or entry."),
        _artifact(root,
                  "results/v177/profit_concentration_emotion_20260917_v1/report.json",
                  ["PROFIT_CONCENTRATION_EMOTION_SCENARIO"],
                  "CURRENT_DIAGNOSTIC",
                  "Separates broad positive premium from a positive mean masked by a non-positive median; historical scenario use is allowed, while stock reranking and entry remain forbidden."),
        _artifact(root,
                  "results/v178/breadth_quality_emotion_20260917_v2/report.json",
                  ["LIMIT_UP_BREADTH_EXPANSION_QUALITY"],
                  "CURRENT_DIAGNOSTIC",
                  "Separates clean from fragile breadth expansion using failed-board pressure; time-segment intervals do not both pass, so only scenario display is allowed."),
        _artifact(root,
                  "results/v179/consensus_crowding_tradeoff_20260917_v1/report.json",
                  ["ONE_PRICE_SHARE_CONSENSUS_CROWDING_TRADEOFF"],
                  "CURRENT_DIAGNOSTIC",
                  "One-price-board share is non-monotonic across development and validation, while formal consensus has only eleven sessions; crowding and buyability description only."),
        _artifact(root,
                  "results/v180/breadth_quality_path_20260917_v1/report.json",
                  ["BREADTH_QUALITY_B1_B2_B3_MARKET_PATH"],
                  "CURRENT_DIAGNOSTIC",
                  "Extends clean versus fragile breadth to three aggregate market sessions; split intervals do not both pass and the path cannot imply candidate holding returns."),
        _artifact(root,
                  "results/v181/height_anchor_buyability_20260917_v1/report.json",
                  ["CORRECTED_LABEL_HEIGHT_ANCHOR_BUYABILITY"],
                  "CURRENT_DIAGNOSTIC",
                  "Corrected no-touch-negative validation finds no daily top-one gain over a boards-only baseline; highest-board identity cannot auto-rank or authorize entry."),
        _artifact(root,
                  "results/v182/opening_gap_payoff_tradeoff_20260917_v3/report.json",
                  ["OPENING_GAP_STRENGTH_PAYOFF_TRADEOFF"],
                  "CURRENT_DIAGNOSTIC",
                  "Opening gap is a 09:30 confirmation feature: above 7% has materially more one-price unbuyability, but payoff-band superiority is not stable and no band authorizes entry."),
        _artifact(root,
                  "results/v183/opening_gap_emotion_interaction_20260917_v2/report.json",
                  ["OPENING_GAP_BY_EMOTION_INTERACTION"],
                  "CURRENT_DIAGNOSTIC",
                  "Opening-gap payoff differences flip across emotion groups and time segments; emotion cannot waive or mechanically trigger the high-open risk rule."),
        _artifact(root,
                  "results/v184/weak_to_strong_hindsight_audit_20260917_v1/report.json",
                  ["LOW_OPEN_WEAK_TO_STRONG_HINDSIGHT_DENOMINATOR"],
                  "CURRENT_DIAGNOSTIC",
                  "Later limit touch is future information: the full low-open denominator is retained, and low opens require live reclaim, VWAP, volume and theme confirmation."),
        _artifact(root,
                  "results/v185/first_board_supply_base_rate_audit_20260917_v1/report.json",
                  ["FIRST_BOARD_SUPPLY_BASE_RATE"],
                  "CURRENT_DIAGNOSTIC",
                  "At-least-one first-board promotion is nearly constant when supply is large; breadth cannot rank an individual candidate or upgrade emotion automatically."),
        _artifact(root,
                  "results/v186/promotion_payoff_separation_audit_20260917_v1/report.json",
                  ["PROMOTION_IDENTITY_VS_BUYABLE_B2_B3_PAYOFF"],
                  "CURRENT_DIAGNOSTIC",
                  "Promotion, buyability, multi-session payoff and big-loss risk are separate targets; promotion probability cannot be reported as a trade win rate."),
        _artifact(root,
                  "results/v187/repair_day_followthrough_audit_20260917_v1/report.json",
                  ["REPAIR_DAY_NEXT_SESSION_FOLLOWTHROUGH"],
                  "CURRENT_DIAGNOSTIC",
                  "Broad repair has lower recent relapse but the effect is not stable across time; only historical scenario context is allowed and next-session confirmation remains mandatory."),
        _artifact(root,
                  "results/v188/repair_layer_transmission_audit_20260917_v1/report.json",
                  ["LOW_MIDDLE_HEIGHT_REPAIR_TRANSMISSION"],
                  "CURRENT_DIAGNOSTIC",
                  "Separates first-board, middle-board and height repair layers; middle confirmation is not stable and recent synchronized support has only three sessions."),
        _artifact(root,
                  "results/v189/high_to_low_ecology_audit_20260917_v2/report.json",
                  ["AGGREGATE_HIGH_TO_LOW_ECOLOGY_WINDOW"],
                  "CURRENT_DIAGNOSTIC",
                  "Separates healthy rotation ecology from collapse-masked low-layer strength; the effect is not stable and cannot identify a same-theme supplement stock."),
        _artifact(root,
                  "results/v190/lane_attention_competition_audit_20260917_v1/report.json",
                  ["FIRST_BOARD_RELAY_LANE_ATTENTION_COMPETITION"],
                  "CURRENT_DIAGNOSTIC",
                  "Separates first-board and relay crowding pressure; neither direction is stable across time and market-level supply cannot name a theme or stock winner."),
        _artifact(root,
                  "results/v191/current_abnormal_distance_20260916_v3/report.json",
                  ["CURRENT_FULL_POOL_UNRECONCILED_SEVERE_ABNORMAL_ARITHMETIC"],
                  "CURRENT_DIAGNOSTIC",
                  "Provides 80/80 provider-aligned research distances while keeping official distance, reset-aware path potential, ranking and entry unavailable."),
        _artifact(root,
                  "results/v191/current_abnormal_distance_20260916_authority.json",
                  ["V191_CURRENT_ABNORMAL_DISTANCE_AUTHORITY"],
                  "SUPERSEDED_AUTHORITY",
                  "Superseded by the V193 authority after bounded official SSE reset chronology was attached."),
        _artifact(root,
                  "results/v192/current_reset_chronology_20260916_v2/report.json",
                  ["CURRENT_BOUNDED_OFFICIAL_RESET_CHRONOLOGY"],
                  "CURRENT_PARTIAL_OFFICIAL_EVIDENCE",
                  "Verifies every SSE exchange session since the active rule date for 32/32 SSE candidates; SZSE company-index absence remains non-exhaustive."),
        _artifact(root,
                  "results/v193/current_reset_overlay_20260916_v1/report.json",
                  ["CURRENT_PARTIALLY_RECONCILED_SEVERE_ABNORMAL_ARITHMETIC"],
                  "CURRENT_DIAGNOSTIC",
                  "Removes the reset blocker only for 32 SSE candidates while preserving all remaining distance, ranking and entry blockers."),
        _artifact(root,
                  "results/v193/current_abnormal_distance_20260916_authority.json",
                  ["V193_CURRENT_ABNORMAL_DISTANCE_AUTHORITY"],
                  "CURRENT_AUTHORITY",
                  "Points to V191 arithmetic, V192 official reset evidence, V193 overlay, V171 replay and V31 daily review."),
        _artifact(root,
                  "results/v161/severe_abnormal_readiness_20260917_v2.json",
                  ["SEVERE_ABNORMAL_PATH_READINESS"], "CURRENT_DIAGNOSTIC",
                  "Rule-distance paths may be shown diagnostically; calibrated hit probability and candidate reranking remain forbidden."),
        _artifact(root,
                  "results/v162/emotion_path_forecast_20260917_v1/report.json",
                  ["TWO_TO_THREE_SESSION_EMOTION_PATH"], "CURRENT_DIAGNOSTIC",
                  "Four-class path model loses to Markov on Brier/log loss; only qualitative scenarios are authorized."),
        _artifact(root,
                  "results/v163/emotion_path_dynamics_20260917_v1/report.json",
                  ["EMOTION_PATH_DYNAMIC_FACTOR_ABLATION"], "CURRENT_DIAGNOSTIC",
                  "Backward-looking deltas and state duration worsen calibration and win only one of six folds."),
        _artifact(root,
                  "results/v164/point_in_time_theme_archive_20260917_v1/report.json",
                  ["POINT_IN_TIME_THEME_CATALYST_ARCHIVE"], "CURRENT_DIAGNOSTIC",
                  "Normalizes all candidates and keeps unknown themes explicit; catalyst timing and expectation fields remain incomplete."),
        _artifact(root,
                  "results/v165/theme_catalyst_backlog_20260917_v2/report.json",
                  ["THEME_CATALYST_EXPECTATION_BACKLOG"], "CURRENT_RESEARCH_GOVERNANCE",
                  "Creates one evidence task per known date/theme; no completed factor row is yet authorized."),
        _artifact(root,
                  "results/v165/theme_catalyst_validation_20260917_v1/report.json",
                  ["THEME_CATALYST_PARTIAL_EVIDENCE"], "CURRENT_DIAGNOSTIC",
                  "Three historical themes have secondary-source reconstructions; none verifies first-public timing sufficiently for factor use."),
        _artifact(root,
                  "results/v206/first_board_dual_target_20260920_v5/report.json",
                  ["FIRST_BOARD_B2_PROFIT_AND_B1_BIG_LOSS_DUAL_TARGET"],
                  "CURRENT_DIAGNOSTIC",
                  "Trains B+2 positive payoff and B+1 large loss separately; both base models remain weak and uncalibrated."),
        _artifact(root,
                  "results/v207/theme_driver_history_20260917_v1/report.json",
                  ["POINT_IN_TIME_ACTIVE_THEME_DRIVER_HISTORY"],
                  "CURRENT_DIAGNOSTIC",
                  "Thirteen active-driver rows are reconstructed, but no session is full-market model eligible."),
        _artifact(root,
                  "results/v208/severe_abnormal_label_queue_20260917_v1/report.json",
                  ["OFFICIAL_SEVERE_ABNORMAL_LABEL_QUEUE"],
                  "CURRENT_RESEARCH_GOVERNANCE",
                  "Separates official positives from threshold hypotheses and emits no probability while labels are unready."),
        _artifact(root,
                  "results/v209/first_board_five_factor_inputs_20260920_v4/report.json",
                  ["FIRST_BOARD_FIVE_FACTOR_POINT_IN_TIME_COVERAGE"],
                  "CURRENT_DIAGNOSTIC",
                  "Competitor elimination now covers all rows descriptively, but complete point-in-time five-factor coverage remains zero."),
        _artifact(root,
                  "results/v210/same_height_pairwise_validation_20260920_v5/report.json",
                  ["SAME_HEIGHT_FORCED_GROUP_VALIDATION"],
                  "CURRENT_DIAGNOSTIC",
                  "Retains all same-height peers and clusters inference by date; zero groups have complete five-factor evidence."),
        _artifact(root,
                  "results/v212/announcement_soft_risk_history_20260917_v1/report.json",
                  ["POINT_IN_TIME_ANNOUNCEMENT_SOFT_RISK_HISTORY"],
                  "CURRENT_DIAGNOSTIC",
                  "Verifies 178 candidate announcement windows without treating failed queries as zero risk."),
        _artifact(root,
                  "results/v216/competitor_elimination_reconstruction_20260920_v1/report.json",
                  ["CURRENT_UNIVERSE_COMPETITOR_ELIMINATION_HISTORY"],
                  "CURRENT_DESCRIPTIVE_ONLY",
                  "Reconstructs all 1,116 candidate rows from adjacent-session board attempts, but none is proven point-in-time full-market complete."),
        _artifact(root,
                  "results/v215/historical_board_attempts_20260920_v2/report.json",
                  ["HISTORICAL_CLOSING_AND_FAILED_BOARD_ATTEMPT_HEIGHTS"],
                  "CURRENT_DESCRIPTIVE_ONLY",
                  "Reconstructs 58,772 closing/failed attempts without skipping missing sessions; current-universe and status bias remain unresolved."),
        _artifact(root,
                  "results/v151/theme_competition_archive_20260914_v1/manifest.json",
                  ["THEME_ARCHIVE_COVERAGE"], "CURRENT_DESCRIPTIVE_ONLY",
                  "Keeps unknown themes explicit; current rows are not model eligible."),
        _artifact(root,
                  "results/v152/full_pool_outcomes_b123_touchfixed_cutoff_20251226_v2/manifest.json",
                  ["CORRECTED_LABEL_DEFINITION"], "CURRENT_RESEARCH",
                  "No-touch candidates are retained as negatives; path returns remain conditional proxies."),
        _artifact(root,
                  "results/v152/group_rank_walkforward_touchfixed_20260917_v4/report.json",
                  ["GROUP_RANK_EXPLORATION"], "CURRENT_RESEARCH",
                  "Corrected group walk-forward; does not prove first-board or relay uplift."),
        _artifact(root,
                  "results/v153/group_rank_governance_touchfixed_20260917_v1.json",
                  ["LEADER_PROBABILITY_CLAIM_GOVERNANCE"], "CURRENT_AUTHORITY",
                  "Authoritative switch for group probability-improvement claims."),
        _artifact(root,
                  str(acceptance_path.relative_to(root)).replace("\\", "/"),
                  ["SYSTEM_DEPLOYMENT_STATUS"], "CURRENT_AUTHORITY",
                  "Authoritative deployment and paper-trade readiness decision."),
    ]
    superseded = [
        _artifact(root,
                  "results/v182/opening_gap_payoff_tradeoff_20260917_v2/report.json",
                  [], "SUPERSEDED_FOR_OPENING_TIME_SEMANTICS",
                  "Used an overly late 09:30 availability description; V3 correctly separates the post-call-auction opening result from the 09:30 continuous auction."),
        _artifact(root,
                  "results/v183/opening_gap_emotion_interaction_20260917_v1/report.json",
                  [], "SUPERSEDED_FOR_OPENING_TIME_SEMANTICS",
                  "Inherited the overly late official-open timestamp; V2 binds the corrected call-auction and continuous-auction timing contract."),
        _artifact(root, "results/full_pool_outcomes_20260905/manifest.json",
                  [], "SUPERSEDED_FOR_PROBABILITY_CLAIMS",
                  "Legacy target construction did not retain explicit no-touch negatives."),
        _artifact(root,
                  "data/v16/full_pool_outcomes_inventory_cutoff_20260831_v3/manifest.json",
                  [], "SUPERSEDED_FOR_PROBABILITY_CLAIMS",
                  "Legacy target construction conditioned the analyzable pool on future touch evidence."),
        _artifact(root,
                  "results/v17/full_pool_proxy_walkforward_cutoff_20260831_v6/report.json",
                  [], "SUPERSEDED_FOR_PROBABILITY_CLAIMS",
                  "Walk-forward used the legacy future-conditioned target universe."),
        _artifact(root,
                  "results/v18/full_pool_role_backtest_cutoff_20260831_v11/report.json",
                  [], "SUPERSEDED_FOR_PROBABILITY_CLAIMS",
                  "Role metrics used the legacy future-conditioned opportunity label."),
        _artifact(root,
                  "results/v132/height_anchor_joint_buy_rowfix_v2/report.json",
                  [], "SUPERSEDED_FOR_STOCK_RANKING_CLAIMS",
                  "Selected-row audit descends from legacy opportunity evidence."),
        _artifact(root,
                  "results/v147/emotion_leader_overlay_20260917_v1/report.json",
                  ["CONSERVATIVE_NO_RERANK_CONCLUSION_ONLY"],
                  "SUPERSEDED_FOR_STOCK_RANKING_CLAIMS",
                  "Its non-authorization conclusion remains conservative, but its stock-level rates cannot support uplift."),
        _artifact(root,
                  "results/v160/seal_speed_tradeoff_20260917_v1/report.json",
                  [], "SUPERSEDED_FOR_EXECUTION_DENOMINATOR",
                  "Missing B-session bars were mixed into the fill denominator; V2 separates unavailable evidence."),
        _artifact(root,
                  "results/v167/failed_seal_path_tradeoff_20260917_v1/report.json",
                  [], "SUPERSEDED_FOR_EXECUTION_DENOMINATOR",
                  "V2 rebuilds the path audit on refreshed bars and explicit execution coverage."),
        _artifact(root,
                  "results/v124/first_board_promotion_20260916_v4/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Used sparse review dates as the successor calendar; V5 uses the separately verified exchange calendar."),
        _artifact(root,
                  "results/v125/first_board_trade_outcomes_20260916_v2/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Its B/B+1/B+2/B+3 chronology inherited the sparse review calendar; V3 is rebuilt."),
        _artifact(root,
                  "results/v126/first_board_profit_link_20260916_v2/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Its selected-trade payoff linkage inherited superseded chronology; V3 is rebuilt."),
        _artifact(root,
                  "results/v160/seal_speed_tradeoff_20260917_v2/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Its payoff horizons inherited superseded chronology; V3 is rebuilt."),
        _artifact(root,
                  "results/v167/failed_seal_path_tradeoff_20260917_v2/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Its payoff horizons inherited superseded chronology; V3 is rebuilt."),
        _artifact(root,
                  "results/v194/fast_board_structure_tradeoff_20260917_v1/report.json",
                  [], "SUPERSEDED_FOR_TRADING_CALENDAR",
                  "Its payoff horizons inherited superseded chronology; V2 is rebuilt."),
    ]

    group_claim = bool(
        governance.get("first_board_probability_improvement_claim_allowed")
        and governance.get("relay_probability_improvement_claim_allowed")
    )
    return {
        "version": "v154-model-evidence-registry-54",
        "status": "DIAGNOSTIC_ONLY",
        "target_fix": {
            "explicit_next_session_touch_field": True,
            "no_touch_rows_retained_as_negative": explicit_no_touch,
            "future_conditioned_pool_forbidden": True,
        },
        "claim_permissions": {
            "next_session_aggregate_emotion_risk_warning": True,
            "next_session_damage_vector_aggregate": bool(
                damage.get("aggregate_gate", {}).get("passed")),
            "emotion_can_directly_rerank_stocks": False,
            "first_board_probability_improvement": bool(
                governance.get("first_board_probability_improvement_claim_allowed")),
            "relay_probability_improvement": bool(
                governance.get("relay_probability_improvement_claim_allowed")),
            "combined_leader_probability_improvement": group_claim,
            "severe_abnormal_counterfactual_path_rating": bool(
                severe.get("diagnostic_counterfactual_path_rating_allowed")),
            "severe_abnormal_calibrated_hit_probability": bool(
                severe.get("calibrated_hit_probability_allowed")),
            "severe_abnormal_candidate_reranking": bool(
                severe.get("candidate_reranking_allowed")),
            "severe_abnormal_closest_distance_role_display": bool(
                abnormal_role.get("status") == "DIAGNOSTIC_ONLY"
                and abnormal_role.get("candidate_reranking_allowed") is False
                and abnormal_role.get("buy_signal") is False),
            "severe_abnormal_path_potential_present": bool(
                abnormal_role.get("hundred_percent_abnormal_path_potential")),
            "severe_abnormal_current_provider_coverage_complete": bool(
                current_abnormal.get("candidate_count", 0) > 0
                and current_abnormal.get("research_distance_count")
                == current_abnormal.get("candidate_count")),
            "severe_abnormal_current_official_distance_complete": bool(
                current_abnormal.get("candidate_count", 0) > 0
                and current_abnormal.get("official_regulatory_distance_count")
                == current_abnormal.get("candidate_count")),
            "severe_abnormal_current_research_arithmetic_display": bool(
                current_abnormal.get("governance", {}).get(
                    "research_arithmetic_may_be_displayed_with_unreconciled_label")),
            "severe_abnormal_current_research_arithmetic_reranking": bool(
                current_abnormal.get("candidate_reranking_allowed")),
            "severe_abnormal_current_sse_reset_coverage_complete": bool(
                reset_chronology.get("sse_candidate_count", 0) > 0
                and reset_chronology.get(
                    "candidate_official_exchange_chronology_complete_count")
                == reset_chronology.get("sse_candidate_count")),
            "severe_abnormal_current_all_exchange_reset_coverage_complete": bool(
                reset_overlay.get("candidate_count", 0) > 0
                and reset_overlay.get("official_reset_evidence_count")
                == reset_overlay.get("candidate_count")),
            "severe_abnormal_current_reset_evidence_reranking": bool(
                reset_overlay.get("candidate_reranking_allowed")),
            "ladder_gap_positive_ranking_factor_use": bool(
                ladder_gap.get("ladder_gap_positive_ranking_factor_allowed")),
            "first_open_delayed_price_discovery_factor_use": bool(
                first_open.get("governance", {}).get(
                    "delayed_price_discovery_factor_allowed")),
            "first_open_automatic_healthy_divergence_claim": bool(
                first_open.get("governance", {}).get(
                    "first_open_is_automatic_healthy_divergence")),
            "daily_six_role_contract_complete": bool(
                daily_role.get("status") == "CURRENT_AUTHORITY"
                and all(daily_role.get("contract", {}).get(key) is True for key in (
                    "emotion_anchor_separate", "current_market_core_separate",
                    "highest_leader_potential_separate",
                    "hundred_percent_path_potential_separate",
                    "current_best_buy_separate", "high_low_supplement_separate"))),
            "daily_group_primary_or_cash_contract_complete": bool(
                daily_role.get("status") == "CURRENT_AUTHORITY"
                and all(daily_role.get("contract", {}).get(key) is True for key in (
                    "first_board_primary_or_none", "relay_primary_or_none",
                    "comprehensive_primary_or_cash",
                    "highest_board_cannot_auto_fill_group_primary",
                    "maximum_one_comprehensive_primary"))),
            "daily_role_contract_authorizes_entry": bool(
                daily_role.get("contract", {}).get("entry_authorized")),
            "anchor_destination_historical_scenario_rates": bool(
                destination.get("governance", {}).get(
                    "historical_scenario_base_rates_allowed")),
            "anchor_destination_prior_close_factor_use": bool(
                destination.get("governance", {}).get(
                    "same_session_destination_as_signal_date_feature_allowed")),
            "anchor_destination_automatic_high_low_entry": bool(
                destination.get("governance", {}).get(
                    "automatic_high_to_low_entry_allowed")),
            "intraday_level1_net_flow_direct_model_use": bool(
                intraday_causality.get("policy", {}).get("direct_model_use_allowed")),
            "intraday_minute_sequence_causal_feedback_claim": bool(
                not intraday_causality.get("policy", {}).get(
                    "minute_sequence_is_not_causality", True)),
            "intraday_flow_or_sequence_entry_authorized": bool(
                intraday_causality.get("policy", {}).get("entry_authorized")),
            "profit_concentration_historical_scenario_use": bool(
                profit_concentration.get("governance", {}).get(
                    "historical_emotion_scenario_use_allowed")),
            "profit_concentration_direct_stock_reranking": bool(
                profit_concentration.get("governance", {}).get(
                    "profit_concentration_direct_stock_reranking_allowed")),
            "profit_concentration_entry_authorized": bool(
                profit_concentration.get("governance", {}).get(
                    "profit_concentration_entry_authorized")),
            "breadth_quality_historical_scenario_use": bool(
                breadth_quality.get("governance", {}).get(
                    "historical_emotion_scenario_use_allowed")),
            "breadth_quality_direct_emotion_gate": bool(
                breadth_quality.get("fragile_expansion_adverse_association_stable")),
            "breadth_quality_direct_stock_reranking": bool(
                breadth_quality.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "breadth_quality_entry_authorized": bool(
                breadth_quality.get("governance", {}).get("entry_authorized")),
            "consensus_crowding_direct_emotion_gate": bool(
                consensus_crowding.get("governance", {}).get(
                    "direct_emotion_gate_allowed")),
            "consensus_crowding_direct_stock_reranking": bool(
                consensus_crowding.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "consensus_crowding_entry_authorized": bool(
                consensus_crowding.get("governance", {}).get("entry_authorized")),
            "breadth_path_historical_scenario_use": bool(
                breadth_path.get("governance", {}).get(
                    "historical_scenario_path_use_allowed")),
            "breadth_path_two_to_three_session_probability": bool(
                breadth_path.get("governance", {}).get(
                    "two_to_three_session_probability_claim_allowed")),
            "breadth_path_candidate_holding_inference": bool(
                breadth_path.get("governance", {}).get(
                    "candidate_holding_period_inference_allowed")),
            "breadth_path_entry_authorized": bool(
                breadth_path.get("governance", {}).get("entry_authorized")),
            "height_anchor_buyability_uplift_supported": bool(
                height_anchor.get("height_anchor_buyability_uplift_supported")),
            "height_anchor_candidate_probability_claim": bool(
                height_anchor.get("governance", {}).get(
                    "candidate_probability_claim_allowed")),
            "height_anchor_auto_relay_primary": bool(
                height_anchor.get("governance", {}).get(
                    "height_anchor_can_auto_fill_relay_primary")),
            "height_anchor_entry_authorized": bool(
                height_anchor.get("governance", {}).get("entry_authorized")),
            "opening_gap_prior_close_model_use": bool(
                opening_gap.get("feature_contract", {}).get(
                    "prior_close_model_use_allowed")),
            "opening_gap_direct_stock_reranking": bool(
                opening_gap.get("governance", {}).get(
                    "opening_gap_direct_stock_reranking_allowed")),
            "opening_gap_direct_entry_authorized": bool(
                opening_gap.get("governance", {}).get(
                    "opening_gap_direct_entry_allowed")),
            "opening_gap_above_seven_risk_flag": bool(
                opening_gap.get("governance", {}).get(
                    "above_seven_default_risk_flag_allowed")),
            "opening_gap_emotion_interaction_factor": bool(
                opening_gap_emotion.get("governance", {}).get(
                    "opening_gap_emotion_interaction_factor_allowed")),
            "emotion_can_override_above_seven_risk_flag": bool(
                opening_gap_emotion.get("governance", {}).get(
                    "emotion_can_override_above_seven_risk_flag")),
            "opening_gap_emotion_direct_reranking": bool(
                opening_gap_emotion.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "opening_gap_emotion_entry_authorized": bool(
                opening_gap_emotion.get("governance", {}).get("entry_authorized")),
            "weak_to_strong_later_touch_as_open_feature": bool(
                weak_to_strong.get("governance", {}).get(
                    "later_touch_may_be_used_as_09_30_feature")),
            "low_open_automatic_weak_to_strong_buy": bool(
                weak_to_strong.get("governance", {}).get(
                    "low_open_automatic_weak_to_strong_buy_allowed")),
            "low_open_direct_reranking": bool(
                weak_to_strong.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "low_open_entry_authorized": bool(
                weak_to_strong.get("governance", {}).get("entry_authorized")),
            "low_open_requires_live_confirmation": bool(
                weak_to_strong.get("governance", {}).get(
                    "requires_live_reclaim_vwap_volume_and_theme_confirmation")),
            "first_board_supply_any_success_candidate_rank": bool(
                first_board_supply.get("governance", {}).get(
                    "any_first_board_promoted_may_rank_a_candidate")),
            "first_board_supply_positive_ranking_factor": bool(
                first_board_supply.get("governance", {}).get(
                    "first_board_supply_positive_ranking_factor_allowed")),
            "first_board_supply_automatic_emotion_upgrade": bool(
                first_board_supply.get("governance", {}).get(
                    "high_supply_automatic_emotion_upgrade_allowed")),
            "first_board_supply_direct_reranking": bool(
                first_board_supply.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "first_board_supply_entry_authorized": bool(
                first_board_supply.get("governance", {}).get("entry_authorized")),
            "promotion_probability_as_trade_win_probability": bool(
                promotion_payoff.get("governance", {}).get(
                    "promotion_probability_may_be_presented_as_trade_win_probability")),
            "promotion_probability_replaces_b2_b3_target": bool(
                promotion_payoff.get("governance", {}).get(
                    "promotion_probability_may_replace_b2_b3_payoff_target")),
            "board_level_promotion_direct_entry_ranking": bool(
                promotion_payoff.get("governance", {}).get(
                    "board_level_promotion_rate_may_directly_rank_entries")),
            "promotion_payoff_direct_stock_reranking": bool(
                promotion_payoff.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "promotion_payoff_entry_authorized": bool(
                promotion_payoff.get("governance", {}).get("entry_authorized")),
            "repair_followthrough_historical_scenario_context": bool(
                repair_followthrough.get("governance", {}).get(
                    "historical_repair_scenario_context_allowed")),
            "repair_followthrough_calibrated_probability": bool(
                repair_followthrough.get("governance", {}).get(
                    "calibrated_next_relapse_probability_allowed")),
            "repair_score_direct_emotion_gate": bool(
                repair_followthrough.get("governance", {}).get(
                    "repair_score_direct_emotion_gate_allowed")),
            "repair_score_direct_stock_reranking": bool(
                repair_followthrough.get("governance", {}).get(
                    "repair_score_direct_stock_reranking_allowed")),
            "repair_score_entry_authorized": bool(
                repair_followthrough.get("governance", {}).get(
                    "repair_score_entry_authorized")),
            "repair_layer_historical_scenario_context": bool(
                repair_layer.get("governance", {}).get(
                    "historical_repair_layer_scenario_context_allowed")),
            "repair_layer_calibrated_probability": bool(
                repair_layer.get("governance", {}).get(
                    "calibrated_layer_followthrough_probability_allowed")),
            "repair_layer_direct_emotion_gate": bool(
                repair_layer.get("governance", {}).get(
                    "repair_layer_score_direct_emotion_gate_allowed")),
            "middle_board_confirmation_positive_ranking_factor": bool(
                repair_layer.get("governance", {}).get(
                    "middle_board_confirmation_positive_ranking_factor_allowed")),
            "repair_layer_direct_stock_reranking": bool(
                repair_layer.get("governance", {}).get(
                    "repair_layer_direct_stock_reranking_allowed")),
            "repair_layer_position_size_increase": bool(
                repair_layer.get("governance", {}).get(
                    "repair_layer_position_size_increase_allowed")),
            "repair_layer_entry_authorized": bool(
                repair_layer.get("governance", {}).get(
                    "repair_layer_entry_authorized")),
            "high_to_low_historical_market_scenario": bool(
                high_to_low.get("governance", {}).get(
                    "historical_high_to_low_market_scenario_allowed")),
            "high_to_low_calibrated_window_probability": bool(
                high_to_low.get("governance", {}).get(
                    "calibrated_high_to_low_window_probability_allowed")),
            "high_to_low_direct_emotion_gate": bool(
                high_to_low.get("governance", {}).get(
                    "high_to_low_window_direct_emotion_gate_allowed")),
            "high_to_low_direct_stock_reranking": bool(
                high_to_low.get("governance", {}).get(
                    "high_to_low_window_direct_stock_reranking_allowed")),
            "high_to_low_stock_selection": bool(
                high_to_low.get("governance", {}).get(
                    "high_to_low_stock_selection_allowed")),
            "high_to_low_same_theme_supplement_mapping": bool(
                high_to_low.get("governance", {}).get(
                    "same_theme_supplement_mapping_allowed")),
            "high_to_low_position_size_increase": bool(
                high_to_low.get("governance", {}).get(
                    "position_size_increase_allowed")),
            "high_to_low_entry_authorized": bool(
                high_to_low.get("governance", {}).get("entry_authorized")),
            "lane_attention_common_winner_score": bool(
                lane_attention.get("governance", {}).get(
                    "common_first_relay_winner_score_allowed")),
            "lane_attention_stock_winner_claim": bool(
                lane_attention.get("governance", {}).get(
                    "market_lane_pressure_may_name_a_stock_winner")),
            "lane_attention_replaces_theme_evidence": bool(
                lane_attention.get("governance", {}).get(
                    "market_lane_pressure_may_replace_specific_theme_evidence")),
            "lane_attention_direct_stock_reranking": bool(
                lane_attention.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "lane_attention_entry_authorized": bool(
                lane_attention.get("governance", {}).get("entry_authorized")),
            "two_to_three_day_emotion_path_probability": bool(
                emotion_path.get("policy", {}).get(
                    "reliable_two_to_three_day_path_claim_allowed")),
            "emotion_dynamic_path_factors_promoted": bool(
                emotion_dynamics.get("promotion_gate_passed")),
            "point_in_time_theme_factor_training": bool(
                theme_archive.get("status") == "READY_FOR_TIME_ISOLATED_THEME_FACTOR_MODEL"),
            "theme_catalyst_expectation_backlog_complete": bool(
                catalyst_backlog.get("theme_tasks", 0) > 0
                and catalyst_backlog.get("completed_factor_rows")
                == catalyst_backlog.get("theme_tasks")),
            "theme_catalyst_partial_evidence_factor_use": bool(
                catalyst_validation.get("factor_schema_ready_rows", 0) > 0),
            "short_term_hypothesis_direct_model_use": bool(
                knowledge.get("direct_model_use_allowed")),
            "sparse_review_dates_as_trading_calendar": bool(
                calendar_integrity.get("governance", {}).get(
                    "daily_review_dates_may_define_trading_calendar")),
            "missing_pool_session_may_shift_labels_forward": bool(
                calendar_integrity.get("governance", {}).get(
                    "missing_pool_session_may_be_skipped_for_promotion_label")),
            "first_board_missing_bar_is_unfilled_trade": bool(
                first_board_profit.get("governance", {}).get(
                    "missing_bar_is_unfilled_trade")),
            "first_board_execution_evidence_complete": bool(
                first_board_outcomes.get("summary", {}).get("missing_or_invalid") == 0),
            "fast_seal_positive_ranking_factor_use": bool(
                seal_speed.get("governance", {}).get(
                    "fast_seal_positive_ranking_factor_allowed")),
            "fast_board_control_intent_inference": bool(
                fast_structure.get("governance", {}).get(
                    "control_intent_inference_allowed")),
            "fast_board_structure_positive_ranking_factor_use": bool(
                fast_structure.get("governance", {}).get(
                    "fast_structure_positive_ranking_factor_allowed")),
            "cumulative_turnover_factor_use": bool(
                turnover.get("direct_model_use_allowed")),
            "failed_seal_path_factor_use": bool(
                failed_seal.get("governance", {}).get(
                    "failed_seal_path_positive_ranking_factor_allowed")),
            "first_board_profit_ranker_direct_stock_reranking": bool(
                first_board_profit_ranker.get("governance", {}).get(
                    "emotion_factor_promoted")
                or first_board_profit_ranker.get("governance", {}).get(
                    "risk_adjusted_ranker_promoted")),
            "first_board_profit_ranker_entry_authorized": bool(
                first_board_profit_ranker.get("governance", {}).get(
                    "stock_entry_authorized")),
            "first_board_dual_target_promoted": bool(
                dual_target.get("status") == "FORWARD_VALIDATED_DUAL_TARGET"
                and dual_target.get("governance", {}).get(
                    "probabilities_deployment_calibrated")),
            "first_board_dual_target_entry_authorized": bool(
                dual_target.get("governance", {}).get("entry_authorized")),
            "five_factor_complete_historical_rows_available": bool(
                five_factor.get("five_factor_complete_rows", 0) > 0),
            "announcement_soft_risk_factor_history_complete": bool(
                announcement_soft.get("verified_soft_risk_rows")
                == announcement_soft.get("candidate_rows")
                and announcement_soft.get("candidate_rows", 0) > 0),
            "competitor_elimination_formal_factor_use": bool(
                competitor_elimination.get("formal_factor_allowed")),
            "same_height_five_factor_uplift_validated": bool(
                same_height_pair.get("same_height_five_factor_uplift_validated")),
            "same_height_fast_seal_or_zero_open_champion_allowed": bool(
                same_height_pair.get("governance", {}).get(
                    "fast_seal_or_zero_open_selects_champion")),
            "theme_driver_formal_factor_use": bool(
                theme_driver.get("formal_theme_factor_allowed")),
            "severe_abnormal_verified_label_calibration": bool(
                severe_label_queue.get("calibrated_probability_allowed")
                and severe_label_queue.get("calibrated_probability_emitted")),
            "theme_evidence_availability_as_strength_factor": bool(
                theme_profit_coverage.get("governance", {}).get(
                    "theme_evidence_availability_is_theme_strength")),
            "theme_roster_direct_profit_factor_use": bool(
                theme_profit_coverage.get("governance", {}).get(
                    "theme_roster_direct_profit_factor_allowed")),
            "theme_profit_coverage_direct_stock_reranking": bool(
                theme_profit_coverage.get("governance", {}).get(
                    "direct_stock_reranking_allowed")),
            "theme_profit_coverage_entry_authorized": bool(
                theme_profit_coverage.get("governance", {}).get(
                    "entry_authorized")),
            "win_rate_alone_sufficient_for_strategy": bool(
                payoff_tail.get("diagnosis", {}).get(
                    "win_rate_alone_sufficient_for_strategy")),
            "payoff_tail_ranker_promotion": bool(
                payoff_tail.get("governance", {}).get(
                    "ranker_promotion_allowed")),
            "payoff_tail_entry_authorized": bool(
                payoff_tail.get("governance", {}).get("entry_authorized")),
            "big_loss_veto_model_promoted": bool(
                big_loss_veto.get("governance", {}).get("risk_model_promoted")),
            "big_loss_veto_policy_promoted": bool(
                big_loss_veto.get("governance", {}).get("veto_policy_promoted")),
            "big_loss_veto_entry_authorized": bool(
                big_loss_veto.get("governance", {}).get("entry_authorized")),
            "risk_probability_as_position_size": bool(
                big_loss_veto.get("governance", {}).get(
                    "risk_probability_is_position_size")),
            "simple_profit_risk_probability_multiplication": bool(
                big_loss_veto.get("diagnosis", {}).get(
                    "simple_probability_multiplication_supported")),
            "opening_gap_as_prior_close_feature": bool(
                opening_causality.get("governance", {}).get(
                    "opening_gap_allowed_in_prior_close_model")),
            "opening_gap_filter_with_same_open_fill": bool(
                opening_causality.get("availability_contract", {}).get(
                    "opening_gap_can_filter_same_official_open_fill")),
            "opening_0935_entry_authorized": bool(
                opening_causality.get("governance", {}).get(
                    "09_35_entry_authorized")),
            "opening_causality_entry_authorized": bool(
                opening_causality.get("governance", {}).get(
                    "entry_authorized")),
            "same_day_stop_as_executed_fill": bool(
                t1_loss.get("governance", {}).get(
                    "same_day_stop_may_be_backtested_as_executed")),
            "B_plus_1_down_limit_queue_assumed_filled": bool(
                t1_loss.get("governance", {}).get(
                    "B_plus_1_down_limit_queue_may_be_assumed_filled")),
            "fixed_stop_loss_guarantee": bool(
                t1_loss.get("governance", {}).get(
                    "fixed_stop_loss_guarantee_allowed")),
            "t1_exit_audit_entry_authorized": bool(
                t1_loss.get("governance", {}).get("entry_authorized")),
            "same_open_conditional_exit_fill": bool(
                exit_causality.get("governance", {}).get(
                    "same_open_conditional_exit_fill_allowed")),
            "hindsight_open_close_exit_oracle": bool(
                exit_causality.get("governance", {}).get(
                    "hindsight_open_close_timing_oracle_allowed")),
            "historical_open_proxy_is_broker_exit": bool(
                exit_causality.get("governance", {}).get(
                    "historical_open_proxy_is_broker_confirmed_exit")),
            "post_open_exit_rule_authorized": bool(
                exit_causality.get("governance", {}).get(
                    "post_open_exit_rule_authorized")),
            "exit_timing_policy_promoted": bool(
                exit_causality.get("governance", {}).get(
                    "entry_or_exit_policy_promoted")),
            "index_emotion_divergence_context_display": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "divergence_may_be_displayed_as_context")),
            "broad_index_equals_short_term_emotion": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "broad_index_direction_equals_short_term_emotion")),
            "index_up_overrides_short_loss_effect": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "index_up_may_override_short_loss_effect")),
            "index_emotion_divergence_direct_gate": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "divergence_direct_emotion_gate_allowed")),
            "index_emotion_divergence_stock_reranking": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "divergence_direct_stock_reranking_allowed")),
            "index_emotion_divergence_entry_authorized": bool(
                index_emotion_divergence.get("governance", {}).get(
                    "divergence_entry_authorized")),
            "secondary_theme_narrative_factor_use": bool(
                theme_confidence.get("verified_positive_factor_rows", 0) > 0),
            "verified_membership_without_driver_factor_use": bool(
                theme_membership.get("positive_factor_rows", 0) > 0),
            "verified_membership_without_driver_high_low_mapping": bool(
                theme_membership.get("formal_high_low_mapping_rows", 0) > 0),
            "cross_theme_race_factor_training": bool(
                cross_theme_race.get("status") == "MODEL_ELIGIBLE_COMPLETE_SCOPE"
                and cross_theme_race.get("model_eligible_count", 0)
                == cross_theme_race.get("closing_candidate_count", -1)),
            "cross_theme_race_winner_or_entry_claim": bool(
                cross_theme_race.get("winner_claimed")
                or cross_theme_race.get("entry_authorized")),
            "complete_trade_win_rate_claim": bool(
                governance.get("complete_trade_probability_claim_allowed")),
            "paper_trade_ready": bool(acceptance.get("paper_trade_ready")),
            "deployment_allowed": bool(acceptance.get("deployment_allowed")),
        },
        "current_artifacts": current,
        "superseded_artifacts": superseded,
        "interpretation": (
            "V144 may forecast aggregate next-session risk only. Leader selection probability "
            "cannot be claimed improved until V150 specific-theme factors pass a new isolated "
            "group walk-forward and complete-trade validation. V161 permits severe-abnormal "
            "rule-distance diagnostics only, not a calibrated hit probability or buy ranking. "
            "V162/V163 do not authorize a two-to-three-day emotion probability; daily output "
            "must remain main/alternative/risk scenarios with invalidation conditions. V164 "
            "keeps theme membership separate from catalyst timing and expectation saturation; "
            "current theme evidence cannot train or rerank candidates. V158/V194/V195/V197/V198/V199/V200/V201/V202/V203 now govern 74 "
            "short-term hypotheses; all remain research-only until isolated out-of-sample tests pass. "
            "V194 corrects the fast-board execution denominator and separates observable thin/one-price "
            "structure from unobservable control intent; neither fast structure nor alleged control may "
            "rank a stock or authorize entry. "
            "V195 proves that sparse review dates cannot define the trading calendar: the "
            "2026-09-01 label previously skipped the actual 09-02 and 09-03 sessions. V124-V126, "
            "V160, V167 and V194 are rebuilt on the verified calendar, and a missing immediate "
            "pool now stays unavailable rather than moving forward. V196 targets executable B+2 "
            "profit and B+1 tail loss separately from promotion. Its ten complete evaluation "
            "sessions show only a diagnostic 40% top-one B+2-positive rate for the emotion model, "
            "with negative mean B+2 return and 30% B+1 large-loss rate; no ranking or entry is authorized. "
            "V197 then joins the profit sample to the point-in-time theme archive. Only 56 of 456 "
            "evaluation rows on four of ten sessions have a pre-open roster, no row has a complete "
            "driver-and-expectation schema, and the covered group does not outperform unknown names. "
            "Roster availability is attention/coverage evidence, not a theme-strength or entry factor. "
            "V198 shows why the 40% emotion top-one hit rate is still unusable: average B+2 return "
            "is -3.77%, payoff ratio 0.66, profit factor 0.44 and the full-capital path drawdown "
            "proxy is about 47.81%. Win rate, payoff magnitude and tail risk remain separate gates. "
            "V199 then builds the tail gate as a separate prequential rejector. Its best candidate-level "
            "AUC is only 0.576, the descriptive highest-risk quintile does not concentrate large losses, "
            "and fixed thresholds that temporarily retain zero large losses keep only two or three losing "
            "trades. Multiplying profit and risk probabilities also worsens both mean payoff and tail loss; "
            "no risk model, veto threshold, position sizing rule or entry is authorized. "
            "V200 separates the opening call-auction result from the later continuous-auction decision. "
            "Five-to-seven and above-seven percent opening groups show about 25% descriptive large-loss "
            "rates in the recent first-board sample, but the three frozen top-one large losses span both "
            "negative and positive opening gaps. The matched opening result cannot filter a fill at that "
            "same auction price; 09:30/09:35 use requires a new execution price, minute evidence and labels, "
            "while current complete intraday sequences remain zero. "
            "V201 then enforces T+1 exit realism: B-session drawdown thresholds are not executable "
            "stops, B+1 one-price lower limits are unsellable, and other lower-limit opening queues "
            "remain fill-unverifiable. Tail protection must act before entry through abstention and "
            "position limits; no fixed-stop guarantee or entry is authorized. "
            "V202 applies the same-event rule to exits. The official open may proxy only a "
            "precommitted auction exit; an open-conditioned sale needs a later price, and ex-post "
            "choice between open and close is a prohibited timing oracle. No historical broker "
            "exit, post-open exit model, or timing policy is promoted. "
            "V203 separates broad-index direction from the board-relay ecology. Index-up with "
            "non-positive prior-limit median feedback shows a descriptive next-adverse-rate gap, "
            "but discovery and validation intervals cross zero, the validation divergence group "
            "has only fourteen sessions, and every replication gate fails. The layer may be shown "
            "as context but cannot override short-term losses, gate emotion, rerank stocks, or authorize entry. "
            "V166 does not authorize relay cumulative turnover as a ranking factor. V167 does "
            "not authorize failed-seal path as a positive ranking factor. V168 prevents "
            "secondary theme rosters from being promoted to verified catalyst or high-low evidence. "
            "V169 separately verifies business membership, active driver and expectation; verified "
            "membership alone remains descriptive and cannot authorize ranking or high-low mapping."
            " V170 keeps first-board and relay cross-theme races in separate lanes and uses "
            "same-confidence peers only; current incomplete scope authorizes neither training, "
            "winner selection nor entry."
            " V171 separates arithmetic threshold proximity from a fully evidenced path role; "
            "the current full-pool replay has zero verified candidate distances and therefore "
            "selects neither a closest candidate nor a path-potential candidate."
            " V172 finds a descriptive promotion premium for two-level-or-greater ladder gaps, "
            "but middle-damage separation and conditional B+2 evidence are unstable; ladder gaps "
            "remain identity context rather than a positive ranking factor."
            " V173 finds that the first turnover board after a one-price streak retains next-day "
            "identity premium, but it does not establish a B+2 payoff edge or monotonic benefit "
            "from higher turnover; first opening remains delayed-price-discovery risk requiring "
            "intraday and theme confirmation. V174 makes all six roles and both group-primary "
            "lanes explicit; absent validated evidence, the unique comprehensive result is cash, "
            "and the contract itself never authorizes entry. V175 rejects an any-low-stock "
            "promotion flag as nearly constant and uses realized lane promotion-rate dominance "
            "only for scenario review; it cannot be leaked into the prior close or authorize a "
            "high-to-low trade. V176 corrects Level-1 auction semantics and freezes a minute "
            "propagation-order contract, but no real forward observations exist; neither net "
            "matched-amount change nor event order proves cancellation intent, causal feedback, "
            "stock ranking, or entry. V177 separates broad from narrow profit distribution: a "
            "positive mean with a non-positive median cannot upgrade the environment by itself. "
            "Its aggregate historical scenario is descriptive and cannot rerank stocks or authorize entry."
            " V178 separates clean and fragile breadth expansion; its full-sample and validation "
            "directions agree, but discovery and validation intervals do not both pass. It may be "
            "displayed as a repair-quality warning, not used as an emotion gate, stock ranker, or entry."
            " V179 finds no monotonic next-session deterioration from aggregate one-price-board "
            "share and only eleven formal-consensus sessions. One-price share may describe crowding, "
            "delayed price discovery, and buyability, but cannot be a direct emotion gate or entry signal."
            " V180 extends breadth quality to aggregate B+1/B+2/B+3 market paths; recent validation "
            "is adverse for fragile expansion, but discovery intervals cross zero. Only an uncalibrated "
            "scenario warning is allowed, never a candidate holding-period or entry inference."
            " V181 rebuilds highest-anchor buyability with no-touch rows and next-session one-price "
            "rows retained as negatives. The identity/emotion/price-volume challenger does not beat "
            "the boards-only baseline, so height identity cannot support a candidate probability, "
            "automatic relay primary, stock reranking, or entry."
            " V182 separates opening strength from post-fill payoff. Above-7% opens show materially "
            "more one-price unbuyability, while payoff advantages between bands are not stable. "
            "Opening gap is available only after the opening call-auction match; continuous auction "
            "starts at 09:30. It cannot enter the prior-close model, and no band "
            "can authorize stock reranking or entry without intraday and theme confirmation."
            " V183 conditions the opening-gap tradeoff on supportive, hot, and adverse emotion "
            "groups. Payoff effects do not replicate across time segments, so emotion cannot waive "
            "the above-7% risk flag, create an automatic veto, rerank stocks, or authorize entry."
            " V184 retains every low-open candidate in the opening denominator. Conditioning on "
            "later limit touch inflates the validation B+2-positive rate by about 10.6 times and "
            "uses future information; low opens are neither automatic buys nor automatic rejects "
            "and require live reclaim, VWAP, volume, and theme confirmation."
            " V185 separates daily first-board supply from individual promotion odds. The "
            "at-least-one-promotion event is nearly constant, while high supply has no stable "
            "positive candidate edge; breadth cannot upgrade emotion, rerank stocks, or authorize entry."
            " V186 shows that promotion identity is not buyable B+2/B+3 payoff: most promoted "
            "names fail the full-denominator buyable B+2-positive path. Promotion probability "
            "cannot replace separate buyability, payoff, big-loss, or complete-trade targets."
            " V187 separates a broad repair close from proven next-session follow-through. The "
            "recent validation segment is directionally better, but discovery and validation "
            "intervals cross zero; repair quality can frame a scenario only and cannot provide "
            "a calibrated relapse probability, emotion gate, stock reranking, position, or entry."
            " V188 separates first-board, middle-board, and height-layer repair transmission. "
            "Middle confirmation loses its discovery advantage in validation, and only three "
            "recent sessions confirm all layers; layer structure cannot become a hard emotion "
            "gate, positive stock factor, position-size increase, or entry signal."
            " V189 separates an aggregate healthy high-to-low ecology window from collapse-masked "
            "low-layer strength. Recent direction does not replicate in development, and no "
            "specific-theme stock identity exists; the window cannot select a supplement, create "
            "a same-theme mapping, increase position size, or authorize entry."
            " V190 separates first-board new-cycle attention from relay height attention. High "
            "relay share does not stably suppress first-board payoff, while high first-board "
            "supply is adverse to relays only in the recent segment. Lane supply is context only: "
            "it cannot create a common winner score, replace specific-theme evidence, rerank a "
            "stock, or authorize entry."
            " V191 attaches provider-aligned 10/30-session arithmetic to all 80 current closing "
            "candidates and matches every provider close to the frozen pool. Exchange-publication "
            "resets were initially unreconciled. V192/V193 now verify every official SSE session "
            "since the active rule date and remove that blocker for all 32 SSE candidates; CNINFO "
            "zero-result searches for 48 SZSE candidates are deliberately not treated as exhaustive "
            "exchange evidence. Corporate actions, historical price-limit eligibility and official "
            "benchmark reconciliation remain incomplete, so official distance is still zero of 80; "
            "arithmetic proximity cannot define path potential, probability, ranking, or entry."
        ),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable registry")
    report = build(args.root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
