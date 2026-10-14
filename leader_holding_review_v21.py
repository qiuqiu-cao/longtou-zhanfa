"""Snapshot-driven holding review, separate from frozen T+1 evaluation.

No live feed, order submission, automatic fills, or calibrated probabilities.
Decisions are immutable research records, not executed trades.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def stamp(value):
    time = pd.Timestamp(value)
    if pd.isna(time) or time.tzinfo is None:
        raise ValueError("explicit timezone timestamp required")
    return time.tz_convert("Asia/Shanghai")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("positive finite number required")
    return number


def review(position, snapshot, asof):
    cutoff = stamp(asof)
    entered = stamp(position["entry_at"])
    if entered > cutoff:
        raise ValueError("entry is after review")
    code = str(position["code"]).zfill(6)
    if str(snapshot["code"]).zfill(6) != code:
        raise ValueError("snapshot and holding differ")
    entry, shares = positive(position["entry_price"]), positive(position["shares"])
    if not shares.is_integer():
        raise ValueError("shares must be integer")
    observed, available = stamp(snapshot["observed_at"]), stamp(snapshot["available_at"])
    if observed > available or available > cutoff:
        raise ValueError("future or inconsistent snapshot")
    reasons = []
    fresh = observed.date() == cutoff.date() and (cutoff-observed).total_seconds() <= 300
    verified = snapshot.get("data_status") == "verified" and bool(snapshot.get("source"))
    result = dict(code=code, position_id=position["position_id"], asof=cutoff.isoformat(),
                  expected_remaining_holding_sessions=None, target_probability=None,
                  model_status="DIAGNOSTIC_ONLY", order_submitted=False,
                  entry_at=entered.isoformat(), shares=int(shares),
                  emotion_state=snapshot.get("emotion_state"),
                  theme_feedback=snapshot.get("theme_feedback"),
                  notice_summary=snapshot.get("notice_summary"),
                  source=snapshot.get("source"), observed_at=observed.isoformat())
    if not fresh or not verified:
        result.update(decision="DATA_LIMITED", reasons=["fresh verified snapshot required"],
                      execution_state="NOT_EVALUATED", unrealized_gross_return=None)
        return result
    price = positive(snapshot["price"])
    result["unrealized_gross_return"] = price/entry-1
    result["unrealized_gross_pnl"] = (price-entry)*shares
    identity = snapshot.get("leader_identity")
    # Ordinary notices/emotion alone do not mechanically veto the user's setup.
    target_known = snapshot.get("indicator_status") == "verified"
    values = []
    for field in ("deviation_10d_pct", "deviation_30d_pct"):
        try:
            number = float(snapshot.get(field))
            values.append(number if math.isfinite(number) else None)
        except (TypeError, ValueError):
            values.append(None)
    target_known = target_known and all(v is not None for v in values)
    reached = target_known and (values[0] >= 100 or values[1] >= 200)
    stop = position.get("planned_exit_price")
    stop_hit = stop is not None and price <= positive(stop)
    if reached:
        decision = "EXIT_PLAN"
        reasons.append("100_or_200_target_reached")
    elif stop_hit:
        decision = "EXIT_PLAN"
        reasons.append("user_planned_exit_price_reached")
    elif identity == "lost":
        decision = "EXIT_PLAN"
        reasons.append("leader_identity_lost_requires_exit_plan")
    elif identity != "intact" or not target_known:
        decision = "DATA_LIMITED"
        reasons.append("identity_or_threshold_evidence_missing")
    else:
        decision = "HOLD_REVIEW"
        reasons.append("identity_intact_and_target_not_reached")
    # A changed environment alters expectations even without an automatic veto.
    if snapshot.get("theme_feedback") == "weakening":
        reasons.append("theme_weakening_recheck_identity_and_absorption")
    if snapshot.get("emotion_state") in {"退潮", "大分歧"}:
        reasons.append("market_negative_feedback_expectation_downgrade")
    if entered.date() == cutoff.date():
        execution = "T1_LOCKED"
    elif snapshot.get("sellability") in {"suspended", "locked_limit_down"}:
        execution = "SELL_BLOCKED"
    elif snapshot.get("sellability") == "sellable":
        execution = "SELL_PLAN_ONLY_NOT_FILLED"
    else:
        execution = "SELLABILITY_UNKNOWN"
    result.update(decision=decision, reasons=reasons, execution_state=execution,
                  next_confirmation="refresh price, leader competitors, theme support and actual sellability",
                  profit_note="gross mark-to-market only; costs and actual exit fills not included")
    return result


def save_review(position_path, snapshot_path, asof, output):
    position = json.loads(Path(position_path).read_text(encoding="utf-8"))
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    result = review(position, snapshot, asof)
    payload = dict(position=position, snapshot=snapshot, result=result)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(save_review(args.position, args.snapshot, args.asof, args.output), ensure_ascii=False, indent=2))
