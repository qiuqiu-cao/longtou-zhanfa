"""Trainable projection of the corrected causal layer, with no outcome inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

from current_model_diagnostics import RELAY_FIELDS

VERSION = "corrected-causal-trainable-345-v1"
STATIC = [x for x in RELAY_FIELDS if x not in {"signal_one_word", "turnover_vs_prev", "turnover_vs_prev_same_height_percentile"}]
CYCLE_STATES = ["ICE_BREAK_CANDIDATE", "DIVERGENCE_CONTEXT", "CLIMAX_CONTEXT", "EBB_CONTEXT", "MIXED_CONTEXT", "LOW_HEIGHT_WITHOUT_BREAK_CONFIRMATION", "BALANCED_CONTEXT", "UNKNOWN_INCOMPLETE_CONTEXT"]
ROLE_STATES = ["OBSERVED_SPACE_HEIGHT_CANDIDATE", "CROSS_STATE_CONTINUATION_CANDIDATE", "BREAK_FROM_COLD_CANDIDATE", "AUTHOR_SUPPORTED_SUPPLEMENT_CANDIDATE", "RELATIVE_POSITION_USURP_CANDIDATE", "HIGH_NONWORD_PRICE_PATH_CANDIDATE", "RELAY_CONTENDER", "UNKNOWN"]
PARENT_STATES = ["UPPER_PRICE_CONTINUATION", "PRICE_RESTART_AFTER_BREAK", "HEAVY_VOLUME_CONTINUATION_PROXY", "FIRST_PRICE_BREAK_OBSERVED", "POST_BREAK_NEGATIVE_FEEDBACK_PROXY", "NONCLOSING_ANCHOR_WITHOUT_NEW_BREAK", "UNKNOWN_CURRENT_ANCHOR_PRICE", "UNKNOWN_PREVIOUS_ANCHOR_STATE"]
COHORT_FIELDS = ["full_cohort_previous_count", "full_cohort_observed_previous_count", "full_cohort_promoted_count", "full_cohort_failed_touched_count", "full_cohort_failed_no_touch_count", "full_cohort_unknown_count", "full_cohort_observed_fraction", "full_cohort_promotion_rate", "full_cohort_promotion_lower_bound", "full_cohort_promotion_upper_bound", "touch_failure_attempt_count", "touch_failure_closing_count", "touch_failure_failed_count", "touch_failure_rate"]
CYCLE_FIELDS = ["height", "closing_count", "full_promotion", "touch_failure", "premium", "down_count", "one_price_share_proxy", "prior20_breadth_rank", "prior20_promotion_rank"]


def project(artifact, relay, sessions):
    """All targets and execution columns are excluded by explicit allowlists."""
    rows = artifact["rows"]
    keys = ["date", "code"]
    if rows.duplicated(keys).any() or relay.duplicated(keys).any():
        raise ValueError("duplicate causal source")
    base = relay[keys + STATIC].merge(rows[keys], on=keys, validate="one_to_one")
    out = base.copy()
    sem = rows.set_index(keys).loc[pd.MultiIndex.from_frame(out[keys])].reset_index()
    for key in COHORT_FIELDS:
        out["causal_" + key] = pd.to_numeric(sem[key], errors="coerce")
    cycles = artifact["cycles"][["date"] + CYCLE_FIELDS].copy().set_index("date").reindex(sessions)
    for lag in [1, 3, 5]:
        cycles["height_delta_" + str(lag)] = cycles.height - cycles.height.shift(lag)
    for key in CYCLE_FIELDS + ["height_delta_1", "height_delta_3", "height_delta_5"]:
        out["causal_market_" + key] = out.date.map(cycles[key])
    for state in CYCLE_STATES:
        out["causal_cycle_" + state] = sem.review_cycle_state.eq(state).astype(float)
    for state in ROLE_STATES:
        out["causal_role_" + state] = sem.review_roles.map(lambda values: float(state in values))
    out["causal_price_shape"] = pd.to_numeric(sem.current_price_shape, errors="coerce")
    out["causal_cycle_complete"] = sem.cycle_context_complete.astype(float)
    out["causal_continuous_path_observed"] = sem.current_price_path_status.eq("OBSERVED_CONTINUOUS_10PCT_PRICE_PATH_PROXY").astype(float)
    out["causal_anchor_scope_complete"] = sem.active_anchor_discovery_scope.eq("COMPLETE_WITHIN_BOUND_ARCHIVE_PROXY").astype(float)
    out["causal_reported_active_parent_count"] = sem.complete_active_parent_count.astype(float)
    for state in ["COMPLETE_PROXY_ELIGIBLE_COHORT", "PARTIAL_MEMBER_OUTCOMES", "UNKNOWN_PREVIOUS_POOL_SCOPE", "NOT_APPLICABLE_EMPTY_PREVIOUS_COHORT"]:
        out["causal_cohort_" + state] = sem.full_cohort_status.eq(state).astype(float)
    parents = artifact["parents"]
    groups = {k: f for k, f in parents.groupby(keys, sort=False)}
    summaries = []
    for row in sem.itertuples():
        f = groups.get((row.date, row.code), parents.iloc[:0])
        result = {"causal_parent_" + state: float(f.anchor_current_state.eq(state).sum()) for state in PARENT_STATES}
        valid = pd.to_numeric(f.observed_launch_lag, errors="coerce")
        complete = row.active_anchor_discovery_scope == "COMPLETE_WITHIN_BOUND_ARCHIVE_PROXY"
        result.update(causal_parent_observed_lag_count=float(valid.notna().sum()),
            causal_parent_lag_2to6_count=float(valid.between(2, 6).sum()),
            causal_parent_lag_min=valid.min() if valid.notna().any() else np.nan,
            causal_parent_lag_max=valid.max() if valid.notna().any() else np.nan,
            causal_parent_weak_name_count=float(f.shared_characters.map(bool).sum()),
            causal_parent_handover_count=float(f.handover_evidence_dates_in_child_streak.map(bool).sum()),
            causal_parent_observed_fraction=float(f.anchor_current_state.ne("UNKNOWN_CURRENT_ANCHOR_PRICE").mean()) if len(f) else (1. if complete else np.nan))
        summaries.append(result)
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(summaries)], axis=1)
    fields = list(out.columns[2:])
    for field in fields:
        out[field] = pd.to_numeric(out[field], errors="raise").astype(float)
    if not np.isfinite(out[fields].to_numpy()[~out[fields].isna().to_numpy()]).all():
        raise ValueError("infinite causal input")
    forbidden = {"target", "target_date", "base_score", "trade_open", "exit_close", "terminal_date", "next_open_gap"}
    assert not forbidden & set(fields)
    return out, fields
