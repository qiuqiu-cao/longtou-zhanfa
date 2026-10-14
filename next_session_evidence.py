"""Bind a frozen closing pool to next-session snapshots without backdating.

Calendar completeness and verification statuses are caller declarations.
No price-path proxy is converted into an executable entry here.
"""
from datetime import time

from leader_target_screen_v17 import finite, timestamp


def validate_session(pool_asof, asof, calendar):
    prior, current = timestamp(pool_asof), timestamp(asof)
    if prior >= current or prior.time() < time(15):
        raise ValueError("pool cutoff must be after close and before current decision")
    if not time(9, 25) <= current.time() <= time(10):
        raise ValueError("opening decision must be between 09:25 and 10:00")
    if (calendar.get("data_status") != "verified" or calendar.get("complete") is not True
            or not str(calendar.get("source") or "").strip()):
        raise ValueError("a sourced, declared complete trading calendar is required")
    if timestamp(calendar.get("available_at")) > current:
        raise ValueError("calendar was unavailable at decision time")
    dates = calendar.get("sessions", [])
    if not dates or dates != sorted(set(dates)):
        raise ValueError("calendar sessions must be sorted and unique")
    for day in dates:
        if timestamp(day).strftime("%Y-%m-%d") != day:
            raise ValueError("calendar requires ISO dates")
    signal, trade = prior.strftime("%Y-%m-%d"), current.strftime("%Y-%m-%d")
    if signal not in dates or trade not in dates or dates.index(trade) != dates.index(signal) + 1:
        raise ValueError("decision date must be the next session in supplied calendar")
    return signal, trade


def attach(candidates, snapshots, pool_asof, asof, calendar):
    signal, trade = validate_session(pool_asof, asof, calendar)
    cutoff = timestamp(asof)
    by_code = {}
    for item in snapshots:
        code = str(item.get("code", "")).zfill(6)
        if len(code) != 6 or not code.isdigit() or code in by_code:
            raise ValueError("opening snapshots need unique six-digit codes")
        by_code[code] = item
    for candidate in candidates:
        # The old record stays unmodified as signal evidence; current state is nested.
        item = by_code.get(candidate["code"])
        issues = []
        if item is None:
            issues.append("missing_current_snapshot")
        else:
            try:
                observed = timestamp(item.get("observed_at"))
                available = timestamp(item.get("available_at"))
                if observed > available or available > cutoff or observed.strftime("%Y-%m-%d") != trade:
                    issues.append("future_or_wrong_session_snapshot")
                if not time(9, 25) <= observed.time() <= time(10):
                    issues.append("outside_confirmation_window")
                if (cutoff - observed).total_seconds() > 300:
                    issues.append("stale_current_snapshot")
            except (ValueError, TypeError):
                issues.append("invalid_current_timestamp")
            if item.get("data_status") != "verified" or not str(item.get("source") or "").strip():
                issues.append("unverified_current_source")
            price = finite(item.get("price"))
            if price is None or price <= 0:
                issues.append("invalid_current_price")
            if finite(item.get("is_st")) not in (0, 1):
                issues.append("unknown_current_st_status")
        try:
            if timestamp(candidate.get("observed_at")).time() < time(15):
                issues.append("signal_is_not_closing_snapshot")
        except (ValueError, TypeError):
            issues.append("invalid_signal_timestamp")
        eligible = candidate["candidate_status"] == "LIMIT_UP_OBSERVATION_CANDIDATE"
        if not eligible:
            issues.append("ineligible_signal_record")
        state = "DATA_LIMITED" if issues else "CURRENT_EVIDENCE_ONLY"
        trade_state = "UNKNOWN"
        if not issues:
            trade_state = {"buyable": "REQUIRES_LIVE_CONFIRMATION",
                           "locked_limit_up": "UNBUYABLE", "suspended": "UNBUYABLE"}.get(
                               item.get("tradability"), "UNKNOWN")
            if finite(item.get("is_st")) == 1:
                state, trade_state = "CURRENT_OUT_OF_SCOPE", "NO_ENTER"
        candidate["opening_review"] = dict(
            status=state, signal_date=signal, trade_date=trade, asof=cutoff.isoformat(),
            issues=issues, snapshot=item, execution_observation=trade_state,
            execution_permission="NO_ENTER", actual_fill=None,
            limitation="snapshot validity is not entry confirmation; theme, absorption and risk checks remain required")
        candidate["execution_observation"] = trade_state
    return {"signal_date": signal, "trade_date": trade,
            "calendar_verification": "SUPPLIER_DECLARED_NOT_INDEPENDENTLY_VERIFIED",
            "unmatched_opening_codes": sorted(set(by_code) - {r["code"] for r in candidates})}
