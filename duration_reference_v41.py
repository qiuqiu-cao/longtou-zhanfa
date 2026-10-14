"""Expose descriptive historical relay curves, never executable holding forecasts."""
import math
from leader_holding_review_v21 import stamp


def reference(boards, report, asof, report_available_at):
    cutoff = stamp(asof)
    available = stamp(report_available_at)
    if available > cutoff:
        raise ValueError("duration report unavailable at decision")
    if report["training_exclusive_cutoff"] >= cutoff.date().isoformat():
        raise ValueError("duration training overlaps decision date")
    base = dict(status="DIAGNOSTIC_ONLY", expected_holding_sessions=None,
                target_100_probability=None, target_200_probability=None,
                historical_relay_curve=None, report_available_at=available.isoformat(),
                warning="historical board-height reference, not leader-specific calibrated forecast or holding advice")
    try:
        height = float(boards)
        if not math.isfinite(height) or height < 2 or not height.is_integer():
            return dict(base, evidence_state="UNKNOWN_BOARD_HEIGHT")
    except (TypeError, ValueError):
        return dict(base, evidence_state="UNKNOWN_BOARD_HEIGHT")
    group = report.get("per_height", {}).get(str(int(height)))
    if not group:
        return dict(base, evidence_state="NO_HEIGHT_REFERENCE")
    return dict(base, evidence_state="HISTORICAL_REFERENCE_ONLY", boards=int(height),
                training_exclusive_cutoff=report["training_exclusive_cutoff"],
                historical_episodes=group["episodes"],
                historical_relay_curve=group["curve"],
                restricted_mean_future_limit_sessions=group.get("restricted_mean_future_limit_sessions"))
