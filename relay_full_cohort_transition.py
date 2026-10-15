"""Separate prior-session full-cohort transitions from same-day touched failures.

This additive research layer consumes exact-price source-scope attempts.  Its
10% cap is an explicitly declared eligibility proxy, not exchange status truth.
No outcomes, scores, current classifications, or later sessions are consumed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math

import pandas as pd

VERSION = "full-official-prior-cohort-transition-1"
SOURCE_SCOPE = "BOUND_SOURCE_UNIVERSE_ASSUMED_MAINBOARD10PCT_NOT_EXCHANGE_STATUS_CERTIFIED"
KNOWN_STATES = {"PROMOTED_EXACT_CAP", "FAILED_TOUCHED_CAP", "FAILED_NO_CAP_TOUCH"}
ATTEMPT_FIELDS = ("date", "signal_date", "code", "name", "pool_assignment", "board_height",
                  "limit_price_10", "high", "close", "pre_close")
BAR_FIELDS = ("date", "code", "open", "high", "low", "close", "pre_close", "volume",
              "is_suspended", "trade_status")


def _cent(value):
    try:
        number = Decimal(str(value)) * 100
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            return None
        return int(number)
    except (InvalidOperation, ValueError, TypeError):
        return None


def _cap(prior_cents):
    return int((Decimal(prior_cents) * Decimal("1.10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _dated(frame, fields, max_day):
    """Whitelist facts, canonicalize dates/codes, and discard future input first."""
    x = frame[[k for k in fields if k in frame]].copy()
    if "date" not in x:
        if "signal_date" not in x:
            raise ValueError("dated observations required")
        x["date"] = x["signal_date"]
    elif "signal_date" in x:
        x["date"] = x["date"].fillna(x["signal_date"])
    x = x[x.date.notna() & x.date.le(max_day)].copy()
    x["code"] = x.code.astype(str).str.zfill(6)
    if x.duplicated(["date", "code"]).any():
        raise ValueError("duplicate dated source observation")
    return x


def _attempt_state(row):
    hi, cl, prior, upper = [_cent(row.get(k)) for k in ("high", "close", "pre_close", "limit_price_10")]
    if None in (hi, cl, prior, upper) or upper != _cap(prior) or hi != upper or cl > hi:
        return None, "INVALID_EXACT_ATTEMPT_PRICE_OR_CAP_PROXY"
    actual = "CLOSING_LIMIT_UP" if cl == upper else "FAILED_BOARD_OBSERVATION"
    if row.get("pool_assignment") != actual:
        return None, "ATTEMPT_CLASSIFICATION_PRICE_CONFLICT"
    height = _finite(row.get("board_height"))
    if height is None or height < 1 or height != int(height):
        return None, "INVALID_ATTEMPT_HEIGHT"
    return dict(row, board_height=int(height)), None


def _member_state(prior, bar, previous_bar):
    """A missing attempt is never evidence of failure; an observed daily bar is."""
    if bar is None:
        return "UNKNOWN_MISSING_DAILY", "CURRENT_DAILY_ROW_MISSING", None
    if bar.get("is_suspended") is True or str(bar.get("trade_status", "")).upper() == "SUSPENDED":
        return "UNKNOWN_SUSPENDED_EXPLICIT", "EXPLICIT_SOURCE_SUSPENSION_FLAG", None
    volume = _finite(bar.get("volume"))
    if volume is None:
        return "UNKNOWN_VOLUME_MISSING", "DAILY_VOLUME_NOT_OBSERVABLE", None
    if volume <= 0:
        return "UNKNOWN_NONTRADING", "NONPOSITIVE_VOLUME_NOT_A_TRADING_DAY_CERTIFICATE", None
    op, hi, lo, cl = [_cent(bar.get(k)) for k in ("open", "high", "low", "close")]
    pc = _cent(prior.get("close"))
    if None in (op, hi, lo, cl, pc) or not lo <= min(op, cl) <= max(op, cl) <= hi:
        return "UNKNOWN_PRICE_CONFLICT", "INVALID_OR_NONCENT_DAILY_OHLC", None
    if previous_bar is not None and _cent(previous_bar.get("close")) != pc:
        return "UNKNOWN_PRICE_CONFLICT", "PREVIOUS_DAILY_CLOSE_DIFFERS_FROM_BOUND_POOL", None
    stated_preclose = bar.get("pre_close")
    if stated_preclose is not None and not pd.isna(stated_preclose) and _cent(stated_preclose) != pc:
        return "UNKNOWN_PRICE_CONFLICT", "CURRENT_PRECLOSE_DIFFERS_FROM_BOUND_PREVIOUS_CLOSE", None
    upper = _cap(pc)
    if hi > upper:
        return "UNKNOWN_PRICE_CONFLICT", "PRICE_ABOVE_ASSUMED_10PCT_REGIME_UNRESOLVED", upper
    if cl == upper:
        return "PROMOTED_EXACT_CAP", "EXACT_CAP_CLOSE_WITH_POSITIVE_VOLUME", upper
    return ("FAILED_TOUCHED_CAP" if hi == upper else "FAILED_NO_CAP_TOUCH"), "OBSERVED_TRADED_CLOSE_BELOW_CAP", upper


def build_transition_context(candidates, attempts, bars, sessions, *, complete_scope_dates=None):
    """Return ``rows``, ``groups``, ``members``, quarantine, and a scope summary.

    candidates requires date/code/boards. attempts requires exact OHLC summary,
    cap and height fields listed in ATTEMPT_FIELDS. Its full source-scope dates
    must be explicitly supplied or bound in ``failed_scope_complete_dates`` attrs.
    A date having a row does not establish a complete pool. Bars require positive
    volume and exact-cent OHLC; pre_close is checked when supplied. Prior pool
    close is the cap basis, so absence of a redundant previous daily bar alone
    does not destroy an otherwise bound observation. Previous bars, if present,
    must agree. Rates remain null whenever a full denominator or member is unknown.
    """
    sessions = list(sessions)
    if sessions != sorted(set(sessions)):
        raise ValueError("official calendar must be sorted and unique")
    c = candidates[[k for k in ("date", "code", "boards") if k in candidates]].copy()
    if not {"date", "code", "boards"} <= set(c):
        raise ValueError("date/code/boards candidate facts required")
    c["code"] = c.code.astype(str).str.zfill(6)
    if c.duplicated(["date", "code"]).any():
        raise ValueError("duplicate candidate")
    if not set(c.date) <= set(sessions):
        raise ValueError("candidate outside official calendar")
    if c.empty:
        return dict(rows=c, groups=pd.DataFrame(), members=pd.DataFrame(), attempt_quarantine=pd.DataFrame(),
                    summary=dict(version=VERSION, candidate_rows=0, source_scope=SOURCE_SCOPE))
    for value in c.boards:
        if _finite(value) is None or int(value) != value or value < 1:
            raise ValueError("positive integer candidate boards required")
    last_day = max(c.date)
    a = _dated(attempts, ATTEMPT_FIELDS, last_day)
    scope = set(complete_scope_dates if complete_scope_dates is not None else
                attempts.attrs.get("failed_scope_complete_dates", []))
    scope &= set(sessions)
    pools, touched, quarantine = defaultdict(dict), defaultdict(lambda: defaultdict(list)), []
    for row in a.to_dict("records"):
        if row["date"] not in sessions:
            raise ValueError("attempt outside official calendar")
        valid, reason = _attempt_state(row)
        if reason:
            scope.discard(row["date"])
            quarantine.append(dict(row, qualification_reason=reason))
            continue
        touched[row["date"]][valid["board_height"]].append(valid)
        if valid["pool_assignment"] == "CLOSING_LIMIT_UP":
            pools[row["date"]][valid["code"]] = valid
    index = {day: i for i, day in enumerate(sessions)}
    signal_days = sorted(c.date.unique())
    relevant_days = set(signal_days) | {sessions[index[d] - 1] for d in signal_days if index[d]}
    b = _dated(bars[bars.date.isin(relevant_days)], BAR_FIELDS, last_day)
    bar_index = {(r["date"], r["code"]): r for r in b.to_dict("records")}
    group_rows, member_rows = [], []
    for day in signal_days:
        previous_day = sessions[index[day] - 1] if index[day] else None
        prior_pool = pools.get(previous_day, {})
        heights = set(int(v) for v in c.loc[c.date.eq(day), "boards"])
        heights |= {r["board_height"] + 1 for r in prior_pool.values()}
        heights |= set(touched.get(day, {}))
        for height in sorted(heights):
            prior_members = [r for r in prior_pool.values() if r["board_height"] == height - 1] if height >= 2 else []
            states = Counter()
            for prior in prior_members:
                state, reason, upper = _member_state(prior, bar_index.get((day, prior["code"])),
                                                     bar_index.get((previous_day, prior["code"])))
                states[state] += 1
                member_rows.append(dict(date=day, previous_date=previous_day, target_height=height,
                    code=prior["code"], name=prior.get("name", ""), previous_height=height - 1,
                    previous_bound_close=prior["close"], assumed_upper_cents=upper, transition_state=state,
                    qualification_reason=reason, previous_pool_scope_complete=previous_day in scope,
                    source_scope=SOURCE_SCOPE, feature_cutoff=day))
            observed = sum(states[x] for x in KNOWN_STATES)
            unknown = len(prior_members) - observed
            known_scope = previous_day in scope
            if height == 1:
                status = "NOT_APPLICABLE_FIRST_BOARD"
            elif not known_scope:
                status = "UNKNOWN_PREVIOUS_POOL_SCOPE"
            elif not prior_members:
                status = "NOT_APPLICABLE_EMPTY_COHORT"
            elif unknown:
                status = "PARTIAL_MEMBERS_UNKNOWN"
            else:
                status = "COMPLETE_PROXY_ELIGIBLE_COHORT"
            denom = len(prior_members) if known_scope and height >= 2 else None
            success = states["PROMOTED_EXACT_CAP"]
            rate = success / denom if status == "COMPLETE_PROXY_ELIGIBLE_COHORT" else None
            attempts_today = touched.get(day, {}).get(height, [])
            touched_success = sum(x["pool_assignment"] == "CLOSING_LIMIT_UP" for x in attempts_today)
            touched_fail = len(attempts_today) - touched_success
            touch_known = day in scope
            group_rows.append(dict(date=day, target_height=height, full_cohort_previous_date=previous_day,
                full_cohort_status=status, full_cohort_previous_count=denom,
                full_cohort_observed_previous_count=len(prior_members), full_cohort_promoted_count=success,
                full_cohort_failed_touched_count=states["FAILED_TOUCHED_CAP"],
                full_cohort_failed_no_touch_count=states["FAILED_NO_CAP_TOUCH"], full_cohort_unknown_count=unknown,
                full_cohort_observed_fraction=observed / denom if denom else None,
                full_cohort_promotion_rate=rate,
                full_cohort_promotion_lower_bound=success / denom if denom else None,
                full_cohort_promotion_upper_bound=(success + unknown) / denom if denom else None,
                touch_failure_scope_status=("COMPLETE_PROXY_ATTEMPT_SCOPE" if attempts_today else "NOT_APPLICABLE_NO_TOUCHED_ATTEMPTS") if touch_known else "UNKNOWN_CURRENT_ATTEMPT_SCOPE",
                touch_failure_attempt_count=len(attempts_today) if touch_known else None,
                touch_failure_closing_count=touched_success if touch_known else None,
                touch_failure_failed_count=touched_fail if touch_known else None,
                touch_failure_rate=touched_fail / len(attempts_today) if touch_known and attempts_today else None,
                source_scope=SOURCE_SCOPE, exchange_status_verified=False, feature_cutoff=day))
    groups = pd.DataFrame(group_rows)
    # Fixed nullable dtypes preserve exact batch/single-day schema even when a
    # later day introduces a missing source or an empty group.
    for field in groups:
        if field.endswith("_count") or field == "target_height":
            groups[field] = pd.array(groups[field], dtype="Int64")
        elif field.endswith(("_rate", "_fraction", "_bound")):
            groups[field] = pd.array(groups[field], dtype="Float64")
    members = pd.DataFrame(member_rows)
    rows = c.merge(groups, left_on=["date", "boards"], right_on=["date", "target_height"], how="left", validate="many_to_one")
    member_index = {(r["date"], r["target_height"], r["code"]): r for r in member_rows}
    def membership(row):
        key = (row["date"], int(row["boards"]), row["code"])
        if row["boards"] == 1:
            return "NOT_APPLICABLE_FIRST_BOARD", "NOT_APPLICABLE_FIRST_BOARD"
        if key in member_index:
            return "OBSERVED_PREVIOUS_K_MINUS_1_MEMBER", member_index[key]["transition_state"]
        if row["full_cohort_status"] == "UNKNOWN_PREVIOUS_POOL_SCOPE":
            return "UNKNOWN_PREVIOUS_POOL_SCOPE", "UNKNOWN_PREVIOUS_POOL_SCOPE"
        return "NOT_IN_PREVIOUS_K_MINUS_1_COHORT", "UNKNOWN_CANDIDATE_HEIGHT_OR_POOL_CONFLICT"
    if not rows.empty:
        values = [membership(row) for row in rows.to_dict("records")]
        rows["full_cohort_candidate_membership"] = [x[0] for x in values]
        rows["full_cohort_candidate_transition_state"] = [x[1] for x in values]
    summary = dict(version=VERSION, candidate_rows=len(rows), groups=len(groups), members=len(members),
        complete_candidate_rows=int(rows.full_cohort_status.eq("COMPLETE_PROXY_ELIGIBLE_COHORT").sum()),
        candidate_status_counts=rows.full_cohort_status.value_counts().to_dict(),
        member_status_counts=dict(Counter(x["transition_state"] for x in member_rows)),
        quarantined_attempts=len(quarantine), source_scope=SOURCE_SCOPE, exchange_status_verified=False,
        independent_identity_truth_rows=0, complete_scope_dates=sorted(scope),
        event_time_basis="OFFICIAL_SIGNAL_AND_PREVIOUS_SESSIONS", actual_source_clock="CALLER_BINDINGS_REQUIRED_NOT_INFERRED",
        missing_member_never_failure=True, no_targets_or_scores_consumed=True)
    return dict(rows=rows, groups=groups, members=members, attempt_quarantine=pd.DataFrame(quarantine), summary=summary)


def build_for_day(day, candidates, attempts, bars, sessions, *, complete_scope_dates=None):
    """The live/day API uses exactly the same construction as historical batches."""
    return build_transition_context(candidates[candidates.date.eq(day)], attempts, bars, sessions,
                                    complete_scope_dates=complete_scope_dates)
