"""First-board and relay observation scope, independent of abnormality targets.

Input verification flags are supplier declarations, not independent verification.
This does not establish full-market coverage or authorize an entry.
"""
import pandas as pd
from datetime import time

from leader_target_screen_v17 import finite, timestamp


REQUIRED = {"code", "name", "boards", "is_mainboard", "is_st",
            "identity_status", "observed_at", "available_at", "source"}


def screen(frame, asof):
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    cutoff = timestamp(asof)
    codes = frame.code.astype(str).str.zfill(6)
    if not codes.str.fullmatch(r"\d{6}").all() or codes.duplicated().any():
        raise ValueError("codes must be unique six-digit identifiers")
    output = []
    for (_, row), code in zip(frame.iterrows(), codes):
        issues = []
        observed = available = None
        try:
            observed, available = timestamp(row.observed_at), timestamp(row.available_at)
            declared_signal_date = row.get("signal_date")
            expected_date = (pd.Timestamp(str(declared_signal_date)).date()
                             if pd.notna(declared_signal_date) and str(declared_signal_date).strip()
                             else cutoff.date())
            if observed > available or available > cutoff:
                issues.append("future_or_inconsistent_timestamp")
            if observed.date() != expected_date:
                issues.append("stale_snapshot")
            if observed.time() < time(15) or row.get("snapshot_kind") != "closing":
                issues.append("closing_snapshot_required")
        except (ValueError, TypeError):
            issues.append("missing_or_invalid_timestamp")
        if pd.isna(row.source) or not str(row.source).strip():
            issues.append("missing_source")
        boards = finite(row.boards)
        main, st = finite(row.is_mainboard), finite(row.is_st)
        sealed = finite(row.get("is_limit_up"))
        delisting = finite(row.get("is_delisting_period"))
        unlimited = finite(row.get("no_price_limit"))
        if delisting not in (0, 1) or unlimited not in (0, 1) or row.get("security_status") != "verified":
            issues.append("unverified_security_scope")
        if boards is None or boards < 0 or not boards.is_integer():
            issues.append("invalid_board_count")
        if main not in (0, 1) or st not in (0, 1):
            issues.append("invalid_scope_flags")
        if sealed not in (0, 1) or row.get("limit_up_status") != "verified":
            issues.append("unverified_limit_up_state")
        if sealed == 1 and row.identity_status != "verified":
            issues.append("unverified_board_identity")
        if sealed == 0 and row.identity_status not in {"verified", "not_applicable"}:
            issues.append("invalid_failed_board_identity_status")
        if sealed == 1 and boards == 0:
            issues.append("inconsistent_limit_up_board_count")
        # Prefix is a consistency check, not a substitute for dated security metadata.
        prefix_main = code.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))
        if main == 1 and not prefix_main:
            issues.append("code_scope_conflict")
        status = ("DATA_LIMITED" if issues else "OUT_OF_SCOPE"
                  if main != 1 or st != 0 or delisting != 0 or unlimited != 0 or sealed != 1 or boards < 1
                  else "LIMIT_UP_OBSERVATION_CANDIDATE")
        scope_ok = not issues and main == 1 and st == 0 and delisting == 0 and unlimited == 0
        touched = finite(row.get("touched_limit"))
        touch_known = row.get("touch_status") == "verified" and touched in (0, 1)
        if scope_ok and sealed == 0 and touch_known and touched == 1:
            pool_assignment = "FAILED_BOARD_OBSERVATION"
        elif status == "LIMIT_UP_OBSERVATION_CANDIDATE":
            pool_assignment = "CLOSING_LIMIT_UP"
        elif scope_ok and sealed == 0 and not touch_known:
            pool_assignment = "TOUCH_EVIDENCE_UNAVAILABLE"
        else:
            pool_assignment = "UNASSIGNED"
        trade = row.get("tradability", "unknown")
        execution = ("UNBUYABLE" if trade in ("locked_limit_up", "suspended")
                     else "REQUIRES_LIVE_CONFIRMATION" if trade == "buyable" else "UNKNOWN")
        if issues:
            execution = "UNKNOWN"
        output.append(dict(row.to_dict(), code=code,
                           candidate_status=status, data_issues=";".join(issues),
                           pool_assignment=pool_assignment,
                           board_group=("FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD"
                                        if boards is not None and boards >= 2 and boards.is_integer() else "UNKNOWN"),
                           screen_version="limit-up-universe-2-closing", screen_asof=cutoff.isoformat(),
                           signal_date=str(row.get("signal_date")) if pd.notna(row.get("signal_date")) else None,
                           archive_availability=("DELAYED_BEFORE_DECISION" if available.date() > observed.date()
                                                 else "SAME_SESSION") if observed and available else "UNAVAILABLE",
                           model_status="DIAGNOSTIC_ONLY", execution_permission="NO_ENTER",
                           execution_observation=execution, target_probability=None,
                           risk_review_status="NOT_COMPLETED",
                           abnormality_role="INDEPENDENT_RESEARCH_NOT_SCOPE_GATE"))
    return pd.DataFrame(output)
