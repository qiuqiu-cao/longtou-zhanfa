"""Aggressive research universe for V16 targets, not an execution signal.

Ordinary notices and emotion states are disclosed but do not veto this universe.
Frozen legacy risk/execution policies remain unchanged for comparison.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

VERSION = "v17-target-screen-1"
REQUIRED = {
    "code", "name", "boards", "is_mainboard", "is_st", "valid_identity",
    "deviation_10d_pct", "deviation_30d_pct", "indicator_status",
    "identity_status", "observed_at", "available_at", "source",
    "tradability", "emotion_state", "notice_titles",
}


def timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("missing timestamp")
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def screen(frame: pd.DataFrame, asof: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    cutoff = timestamp(asof)
    codes = frame["code"].astype(str).str.zfill(6)
    if not codes.str.fullmatch(r"\d{6}").all() or codes.duplicated().any():
        raise ValueError("codes must be unique six-digit identifiers")
    results = []
    for (_, row), code in zip(frame.iterrows(), codes):
        result = row.to_dict()
        result.update(code=code, screen_version=VERSION, screen_asof=cutoff.isoformat(),
                      model_status="DIAGNOSTIC_ONLY", execution_permission="NO_ENTER",
                      target_probability=None)
        issues = []
        try:
            observed, available = timestamp(row.observed_at), timestamp(row.available_at)
            if observed > available or available > cutoff:
                issues.append("future_or_inconsistent_timestamp")
            if observed.date() != cutoff.date():
                issues.append("stale_snapshot")
        except (ValueError, TypeError):
            issues.append("missing_or_invalid_timestamp")
        if pd.isna(row.source) or not str(row.source).strip():
            issues.append("missing_source")
        if row.indicator_status != "verified" or row.identity_status != "verified":
            issues.append("unverified_indicator_or_identity")
        dev10, dev30, boards = [finite(row[k]) for k in
                                ("deviation_10d_pct", "deviation_30d_pct", "boards")]
        flags = [finite(row[k]) for k in ("is_mainboard", "is_st", "valid_identity")]
        if dev10 is None or dev30 is None or boards is None or any(f not in (0, 1) for f in flags):
            issues.append("missing_or_invalid_numeric_field")
        elif boards < 0 or not boards.is_integer():
            issues.append("invalid_board_count")
        if issues:
            status = "DATA_LIMITED"
        elif flags[0] != 1 or flags[1] == 1 or flags[2] != 1 or boards < 2:
            status = "OUT_OF_SCOPE"
        elif dev10 >= 100 or dev30 >= 200:
            status = "TARGET_ALREADY_REACHED"
        else:
            status = "TARGET_RESEARCH_CANDIDATE"
        # Tradability is an independent axis, not a hidden risk/height veto.
        trade = str(row.tradability)
        if trade not in {"buyable", "locked_limit_up", "suspended", "unknown"}:
            trade = "unknown"
        result.update(
            candidate_status=status,
            data_issues=";".join(issues),
            distance_100_pp=None if dev10 is None else 100-dev10,
            distance_200_pp=None if dev30 is None else 200-dev30,
            execution_observation=("REQUIRES_LIVE_CONFIRMATION" if trade == "buyable"
                                   else "UNBUYABLE" if trade in {"locked_limit_up", "suspended"}
                                   else "UNKNOWN"),
            risk_policy="notices_and_emotion_disclosed_not_universe_vetoes",
        )
        results.append(result)
    # No fabricated ranking: board height is descriptive, not calibrated odds.
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(pd.read_csv(args.input, dtype={"code": str}), args.asof)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite screen")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
