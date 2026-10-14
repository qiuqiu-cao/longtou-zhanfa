"""Validate auction net-flow proxies and theme propagation order.

Level-1 auction snapshots reveal only changes in aggregate indicative matched
amount.  They cannot identify gross cancellations, replenishment, participant
identity, or manipulation.  Minute bars can establish a coarse event order but
cannot prove that low-position boards caused a high-position anchor to reseal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

import intraday_evidence_v9 as intraday
import open_confirmation_v4 as opening
from leader_system_v3 import is_mainboard, zfill_code


VERSION = "v176-intraday-flow-causality-1"
ROSTER_REQUIRED = {
    "signal_date", "trade_date", "code", "name", "boards", "theme_group_id",
    "theme_evidence_id", "membership_verified", "driver_verified",
    "roster_complete", "evidence_available_at",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_bool(value, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true"}:
        return True
    if text in {"0", "0.0", "false"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean")


def _clock_timestamp(trade_date: str, asof: str) -> pd.Timestamp:
    return pd.Timestamp(trade_date) + pd.Timedelta(seconds=opening.clock_seconds(asof))


def validate_roster(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    missing = sorted(ROSTER_REQUIRED - set(frame))
    if missing:
        raise ValueError(f"theme roster fields missing: {missing}")
    checked = frame.copy()
    checked["signal_date"] = pd.to_datetime(
        checked.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    checked["trade_date"] = pd.to_datetime(
        checked.trade_date, errors="coerce").dt.strftime("%Y-%m-%d")
    if checked[["signal_date", "trade_date"]].isna().any().any():
        raise ValueError("theme roster contains invalid dates")
    checked["code"] = checked.code.map(zfill_code)
    checked["boards"] = pd.to_numeric(checked.boards, errors="coerce")
    if checked.boards.isna().any() or (checked.boards < 1).any():
        raise ValueError("theme roster boards must be positive")
    if not checked.apply(lambda row: is_mainboard(row.code, row["name"]), axis=1).all():
        raise ValueError("theme roster contains non-mainboard/ST/delisting-risk code")
    if checked.duplicated(["signal_date", "trade_date", "theme_group_id", "code"]).any():
        raise ValueError("theme roster contains duplicate point-in-time keys")
    for field in ("membership_verified", "driver_verified", "roster_complete"):
        checked[field] = checked[field].map(lambda value: _strict_bool(value, field))
    available = pd.to_datetime(checked.evidence_available_at, errors="coerce")
    if available.isna().any():
        raise ValueError("theme roster evidence_available_at must be parseable")
    if available.dt.tz is not None:
        available = available.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if (available > cutoff).any():
        raise ValueError("theme roster contains evidence after asof")
    if checked.theme_group_id.astype(str).str.strip().eq("").any():
        raise ValueError("theme roster requires theme_group_id")
    if checked.theme_evidence_id.astype(str).str.strip().eq("").any():
        raise ValueError("theme roster requires theme_evidence_id")
    checked["_evidence_available_at"] = available
    return checked


def _auction_profile(rows: pd.DataFrame, point: str) -> dict:
    result = {
        "matched_amount_0920": None, "matched_amount_092450": None,
        "matched_amount_0925": None, "indicative_price_0920": None,
        "indicative_price_092450": None, "indicative_price_0925": None,
        "matched_amount_net_change_ratio_0920_to_092450": None,
        "apparent_match_decay_ratio": None, "apparent_match_growth_ratio": None,
        "final_match_to_prior_day_amount": None,
        "auction_flow_status": "UNAVAILABLE_INCOMPLETE_LEVEL1_SNAPSHOTS",
        "gross_cancel_amount": None, "gross_replenishment_amount": None,
        "participant_identity": None, "manipulation_intent": None,
        "level2_queue_status": "UNAVAILABLE_NO_LEVEL2",
    }
    if rows.empty:
        return result
    rows = rows[rows["asof"].map(opening.clock_seconds) <= opening.clock_seconds(point)]
    lookup = rows.sort_values("asof", key=lambda values: values.map(
        opening.clock_seconds)).drop_duplicates("asof", keep="last").set_index("asof")
    mapping = {"09:20": "0920", "09:24:50": "092450", "09:25": "0925"}
    for clock, suffix in mapping.items():
        if clock in lookup.index:
            result[f"matched_amount_{suffix}"] = float(lookup.loc[clock, "matched_amount"])
            result[f"indicative_price_{suffix}"] = float(
                lookup.loc[clock, "indicative_price"])
    if {"09:20", "09:24:50"}.issubset(lookup.index):
        early = float(lookup.loc["09:20", "matched_amount"])
        late = float(lookup.loc["09:24:50", "matched_amount"])
        if early > 0:
            change = (late - early) / early
            result["matched_amount_net_change_ratio_0920_to_092450"] = change
            result["apparent_match_decay_ratio"] = max(0.0, -change)
            result["apparent_match_growth_ratio"] = max(0.0, change)
        result["auction_flow_status"] = (
            "VERIFIED_LEVEL1_NET_MATCH_CHANGE_NOT_GROSS_CANCEL_OR_REPLENISHMENT")
    if "09:25" in lookup.index:
        final = lookup.loc["09:25"]
        prior_amount = float(final["prior_day_amount"])
        if prior_amount > 0:
            result["final_match_to_prior_day_amount"] = float(
                final["matched_amount"] / prior_amount)
    return result


def _minute_profile(rows: pd.DataFrame, point: str) -> dict:
    result = {
        "minute_observed": False, "first_limit_touch_at": None,
        "first_limit_close_at": None, "first_break_after_touch_at": None,
        "first_reseal_after_break_at": None, "amount_observed_to_asof": None,
        "failed_touch_episodes": None,
    }
    if rows.empty:
        return result
    cutoff_seconds = opening.clock_seconds(point)
    ordered = rows[
        (rows.bar_end.dt.hour * 3600 + rows.bar_end.dt.minute * 60
         + rows.bar_end.dt.second) <= cutoff_seconds
    ].sort_values("bar_end").reset_index(drop=True).copy()
    if ordered.empty:
        return result
    result["minute_observed"] = True
    result["amount_observed_to_asof"] = float(ordered.amount_yuan.sum())
    tolerance = 1e-6
    touched = ordered.high.ge(ordered.limit_up_price * (1 - tolerance))
    sealed = ordered.close.ge(ordered.limit_up_price * (1 - tolerance))
    if touched.any():
        first_touch_index = touched[touched].index[0]
        result["first_limit_touch_at"] = pd.Timestamp(
            ordered.loc[first_touch_index, "bar_end"]).isoformat()
        after_touch = ordered.loc[ordered.index >= first_touch_index]
        break_rows = after_touch[after_touch.close.lt(
            after_touch.limit_up_price * (1 - tolerance))]
        if not break_rows.empty:
            break_index = break_rows.index[0]
            result["first_break_after_touch_at"] = pd.Timestamp(
                break_rows.iloc[0].bar_end).isoformat()
            reseal_rows = ordered.loc[ordered.index > break_index]
            reseal_rows = reseal_rows[reseal_rows.close.ge(
                reseal_rows.limit_up_price * (1 - tolerance))]
            if not reseal_rows.empty:
                result["first_reseal_after_break_at"] = pd.Timestamp(
                    reseal_rows.iloc[0].bar_end).isoformat()
    if sealed.any():
        result["first_limit_close_at"] = pd.Timestamp(
            ordered.loc[sealed[sealed].index[0], "bar_end"]).isoformat()
    failed = touched & ~sealed
    result["failed_touch_episodes"] = int((
        failed & ~failed.shift(1, fill_value=False)).sum())
    return result


def _stamp(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _theme_sequences(roster: pd.DataFrame, profiles: dict, point: str) -> list[dict]:
    rows = []
    active = roster[
        roster.membership_verified & roster.driver_verified & roster.roster_complete
    ].copy()
    for theme, members in active.groupby("theme_group_id", sort=True):
        maximum = int(members.boards.max())
        anchors = members[members.boards.eq(maximum) & members.boards.ge(2)]
        low = members[members.boards.le(2) & members.boards.lt(maximum)]
        for anchor in anchors.itertuples():
            observed_codes = [code for code in members.code if profiles.get(code, {}).get(
                "minute_observed")]
            coverage = len(observed_codes) / len(members) if len(members) else 0
            anchor_profile = profiles.get(anchor.code, {})
            anchor_touch = _stamp(anchor_profile.get("first_limit_touch_at"))
            anchor_break = _stamp(anchor_profile.get("first_break_after_touch_at"))
            anchor_reseal = _stamp(anchor_profile.get("first_reseal_after_break_at"))
            low_touch = {
                row.code: _stamp(profiles.get(row.code, {}).get("first_limit_touch_at"))
                for row in low.itertuples()
            }
            low_touch = {code: stamp for code, stamp in low_touch.items()
                         if stamp is not None}
            first_low = min(low_touch.values()) if low_touch else None
            status = "VERIFIED_MINUTE_SEQUENCE_NOT_CAUSAL"
            direction = "UNAVAILABLE_NO_COMPLETE_TOUCH_PAIR"
            if coverage < .67 or len(members) < 3 or len(low) < 1:
                status = "UNAVAILABLE_INSUFFICIENT_THEME_MINUTE_COVERAGE"
            elif anchor_touch is not None and first_low is not None:
                delta = (anchor_touch - first_low).total_seconds()
                if abs(delta) < 60:
                    direction = "NEAR_SIMULTANEOUS_WITHIN_ONE_MINUTE"
                elif delta >= 60:
                    direction = "LOW_POSITION_TOUCH_PRECEDES_ANCHOR"
                else:
                    direction = "ANCHOR_TOUCH_PRECEDES_LOW_POSITION"
            low_before_anchor = sum(
                stamp < anchor_touch for stamp in low_touch.values()) if anchor_touch else 0
            low_between_break_reseal = sum(
                anchor_break <= stamp <= anchor_reseal for stamp in low_touch.values()
            ) if anchor_break is not None and anchor_reseal is not None else 0
            feedback_sequence = bool(
                status == "VERIFIED_MINUTE_SEQUENCE_NOT_CAUSAL"
                and anchor_break is not None and anchor_reseal is not None
                and low_between_break_reseal > 0)
            rows.append({
                "theme_group_id": theme, "theme_evidence_id": anchor.theme_evidence_id,
                "anchor_code": anchor.code, "anchor_name": anchor.name,
                "anchor_boards": maximum, "asof": point,
                "roster_member_count": int(len(members)),
                "observed_member_count": int(len(observed_codes)),
                "minute_member_coverage": coverage,
                "low_position_member_count": int(len(low)),
                "low_position_touched_count": int(len(low_touch)),
                "first_low_position_touch_at": first_low.isoformat()
                if first_low is not None else None,
                "anchor_first_touch_at": anchor_touch.isoformat()
                if anchor_touch is not None else None,
                "anchor_first_break_at": anchor_break.isoformat()
                if anchor_break is not None else None,
                "anchor_first_reseal_at": anchor_reseal.isoformat()
                if anchor_reseal is not None else None,
                "low_touches_before_anchor": int(low_before_anchor),
                "low_touches_between_anchor_break_and_reseal": int(
                    low_between_break_reseal),
                "sequence_direction": direction, "sequence_status": status,
                "low_feedback_sequence_supported": feedback_sequence,
                "causal_feedback_claim_allowed": False,
                "common_catalyst_simultaneity_excluded": False,
                "buy_signal": False,
            })
    return rows


def build(auction: pd.DataFrame, minutes: pd.DataFrame, candidates: pd.DataFrame,
          roster: pd.DataFrame, signal_date: str, trade_date: str,
          asof: str = "10:00") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    point = opening.normalize_asof(asof)
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    cutoff = _clock_timestamp(trade, point)
    checked_auction = intraday.validate_auction(auction)
    checked_minutes = intraday.validate_minutes(minutes)
    checked_roster = validate_roster(roster, cutoff)
    required_candidates = {"signal_date", "code", "name", "boards"}
    if required_candidates - set(candidates):
        raise ValueError("candidate fields missing")
    cand = candidates.copy()
    cand["signal_date"] = pd.to_datetime(
        cand.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    cand["code"] = cand.code.map(zfill_code)
    cand["boards"] = pd.to_numeric(cand.boards, errors="coerce")
    cand = cand[cand.signal_date.eq(signal) & cand.boards.ge(1)].copy()
    if cand.empty or cand.code.duplicated().any():
        raise ValueError("candidate scope is empty or duplicated")
    if not cand.apply(lambda row: is_mainboard(row.code, row["name"]), axis=1).all():
        raise ValueError("candidate scope contains non-mainboard/ST/delisting-risk code")
    checked_auction = checked_auction[
        checked_auction.signal_date.eq(signal)
        & checked_auction.trade_date.eq(trade)
            & (checked_auction["asof"].map(opening.clock_seconds)
           <= opening.clock_seconds(point))
    ]
    checked_minutes = checked_minutes[
        checked_minutes.signal_date.eq(signal)
        & checked_minutes.trade_date.eq(trade)
        & (checked_minutes.bar_end <= cutoff)
        & (checked_minutes._source_timestamp <= cutoff)
    ]
    checked_roster = checked_roster[
        checked_roster.signal_date.eq(signal)
        & checked_roster.trade_date.eq(trade)
    ]
    candidate_rows = []
    profiles = {}
    for row in cand.itertuples():
        auction_profile = _auction_profile(
            checked_auction[checked_auction.code.eq(row.code)], point)
        minute_profile = _minute_profile(
            checked_minutes[checked_minutes.code.eq(row.code)], point)
        profiles[row.code] = minute_profile
        candidate_rows.append({
            "signal_date": signal, "trade_date": trade, "asof": point,
            "code": row.code, "name": row.name, "boards": int(row.boards),
            **auction_profile, **minute_profile,
            "auction_cancel_intent_claim_allowed": False,
            "auction_replenishment_intent_claim_allowed": False,
            "control_or_deception_claim_allowed": False,
            "entry_authorized": False,
        })
    # Theme peers may extend beyond the formal candidate table; profile every
    # verified roster member with a captured minute series.
    for code in checked_roster.code.unique():
        if code not in profiles:
            profiles[code] = _minute_profile(
                checked_minutes[checked_minutes.code.eq(code)], point)
    sequence_rows = _theme_sequences(checked_roster, profiles, point)
    candidate_frame = pd.DataFrame(candidate_rows)
    sequence_frame = pd.DataFrame(sequence_rows)
    verified_sequence = (int(sequence_frame.sequence_status.eq(
        "VERIFIED_MINUTE_SEQUENCE_NOT_CAUSAL").sum()) if not sequence_frame.empty else 0)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal, "trade_date": trade, "asof": point,
        "candidate_count": int(len(candidate_frame)),
        "auction_net_flow_verified_count": int(candidate_frame.auction_flow_status.eq(
            "VERIFIED_LEVEL1_NET_MATCH_CHANGE_NOT_GROSS_CANCEL_OR_REPLENISHMENT").sum()),
        "minute_observed_candidate_count": int(candidate_frame.minute_observed.sum()),
        "theme_sequence_record_count": int(len(sequence_frame)),
        "verified_noncausal_sequence_count": verified_sequence,
        "governance": {
            "level1_net_change_may_be_called_gross_cancel": False,
            "level1_net_change_may_be_called_replenishment": False,
            "minute_order_proves_causal_feedback": False,
            "community_control_or_deception_claim_allowed": False,
            "stock_reranking_allowed": False, "entry_authorized": False,
        },
        "required_missing_for_causal_or_execution_claim": [
            "Level-2 order queue and add/cancel event stream",
            "tick-level participant-agnostic executions after each signal",
            "complete point-in-time active-theme roster and catalyst timestamp",
            "verified fill and T+1 exit path",
        ],
    }
    return candidate_frame, sequence_frame, report


def readiness_report() -> dict:
    return {
        "version": VERSION, "status": "NO_FORWARD_OBSERVATIONS",
        "available_automatic_public_fields": [
            "Level-1 price/OHLC/cumulative volume/cumulative amount/provider timestamp",
        ],
        "broker_or_terminal_export_required": [
            "09:20/09:24:50/09:25 indicative matched amount and price",
            "09:31-10:00 one-minute OHLCV/amount and limit-up price",
            "frozen verified theme-member roster with active driver evidence",
        ],
        "unavailable_without_level2": [
            "gross cancellations", "gross replenishment", "queue ownership",
            "participant identity", "intentional control or deception",
        ],
        "policy": {
            "net_matched_amount_change_is_not_cancel_or_replenishment": True,
            "minute_sequence_is_not_causality": True,
            "missing_values_remain_unavailable": True,
            "direct_model_use_allowed": False, "entry_authorized": False,
        },
    }


def run(auction_path: Path, minutes_path: Path, candidates_path: Path,
        roster_path: Path, output_dir: Path, signal_date: str, trade_date: str,
        asof: str) -> dict:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable V176 output")
    paths = [auction_path, minutes_path, candidates_path, roster_path]
    candidates, sequences, report = build(
        pd.read_csv(auction_path, dtype={"code": str}),
        pd.read_csv(minutes_path, dtype={"code": str}),
        pd.read_csv(candidates_path, dtype={"code": str}),
        pd.read_csv(roster_path, dtype={"code": str}),
        signal_date, trade_date, asof,
    )
    output_dir.mkdir(parents=True)
    candidate_output = output_dir / "candidate_flow_records.csv"
    sequence_output = output_dir / "theme_sequence_records.csv"
    candidates.to_csv(candidate_output, index=False, encoding="utf-8-sig")
    sequences.to_csv(sequence_output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["candidate_records_sha256"] = _sha256(candidate_output)
    report["theme_sequence_records_sha256"] = _sha256(sequence_output)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--auction", type=Path)
    parser.add_argument("--minutes", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--signal-date")
    parser.add_argument("--trade-date")
    parser.add_argument("--asof", default="10:00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable V176 output")
    if args.readiness_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            readiness_report(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(readiness_report(), ensure_ascii=False, indent=2))
    else:
        required = (args.auction, args.minutes, args.candidates, args.roster,
                    args.signal_date, args.trade_date)
        if any(value is None for value in required):
            parser.error("full mode requires auction, minutes, candidates, roster and dates")
        print(json.dumps(run(args.auction, args.minutes, args.candidates,
                             args.roster, args.output, args.signal_date,
                             args.trade_date, args.asof),
                         ensure_ascii=False, indent=2))
