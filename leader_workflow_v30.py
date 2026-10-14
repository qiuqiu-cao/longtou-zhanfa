"""Unified research watchlist and explicit-position review; never submits orders.

No candidate is promoted to a probability winner or an actual holding merely
because it has the highest board height. Frozen legacy modules stay untouched.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from catalyst_evidence_v29 import snapshot as catalyst_snapshot
from leader_holding_review_v21 import review, stamp
from limit_up_universe import screen
from leader_competition_v38 import compare as compare_competition
from duration_reference_v41 import reference as duration_reference
from short_term_environment import analyze as analyze_environment
from theme_identity_review import review as review_theme_identity
from next_session_evidence import attach as attach_opening


def watchlist(frame, asof, ledger=None):
    cutoff = stamp(asof)
    candidates = screen(frame, asof).to_dict("records")
    # Descriptive display order, explicitly not an estimated probability rank.
    for row in candidates:
        row["watch_order_basis"] = "known_board_height_only_NOT_probability"
        try:
            observed = stamp(row["observed_at"])
            stale = (cutoff-observed).total_seconds() > 300
        except (ValueError, TypeError):
            stale = True
        if stale:
            row["execution_observation"] = "REFRESH_REQUIRED"
        row["catalyst_evidence"] = (catalyst_snapshot(ledger, row["code"], asof)
                                    if ledger is not None else None)
        row["position_created"] = False
    def order(row):
        try:
            height = float(row["boards"])
            if not pd.notna(height):
                height = -1
        except (TypeError, ValueError):
            height = -1
        return (row["candidate_status"] != "LIMIT_UP_OBSERVATION_CANDIDATE", -height, row["code"])
    candidates.sort(key=order)
    main_pool = [r for r in candidates if r["pool_assignment"] == "CLOSING_LIMIT_UP"]
    failed_pool = [r["code"] for r in candidates if r["pool_assignment"] == "FAILED_BOARD_OBSERVATION"]
    return dict(status="DIAGNOSTIC_ONLY", asof=cutoff.isoformat(), mode="WATCHLIST",
                probability_first_choice=None, selection_state="NO_VALIDATED_SELECTION_MODEL",
                comprehensive_first_choice=None, leader_potential_first_choice=None,
                emotion_anchors=[], entry_decision="ABSTAIN_PENDING_VALIDATION",
                universe_coverage="SUPPLIED_ROWS_ONLY_NOT_VERIFIED_FULL_MARKET",
                pool_summary=dict(input_count=len(candidates), closing_limit_count=len(main_pool),
                                  first_board_count=sum(r["boards"] == 1 for r in main_pool),
                                  board_level_counts=dict(sorted(Counter(str(int(r["boards"])) for r in main_pool).items(),
                                                                key=lambda item: int(item[0]))),
                                  failed_board_observed_count=len(failed_pool),
                                  data_limited_count=sum(r["candidate_status"] == "DATA_LIMITED" for r in candidates),
                                  touch_evidence_missing_count=sum(r["pool_assignment"] == "TOUCH_EVIDENCE_UNAVAILABLE" for r in candidates),
                                  completeness="NOT_VERIFIED", missing_market_codes=None),
                failed_board_observation_codes=failed_pool,
                candidate_groups={group: [r["code"] for r in candidates
                    if r["board_group"] == group and r["candidate_status"] == "LIMIT_UP_OBSERVATION_CANDIDATE"]
                    for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD")},
                candidates=candidates, order_submitted=False, position_created=False,
                limitations=["display order is not calibrated target odds or a buy instruction",
                             "source data supplied by caller; no live feed or background monitoring",
                             "auction queues, cancel sequences and actual fills not verified here"])


def holding(position, quote, asof, ledger=None, previous=None):
    if position.get("position_kind") not in {"paper", "user_confirmed"}:
        raise ValueError("position_kind must explicitly distinguish paper or user_confirmed")
    result = review(position, quote, asof)
    result["mode"] = "HOLDING_REVIEW"
    result["position_kind"] = position["position_kind"]
    result["catalyst_evidence"] = (catalyst_snapshot(ledger, result["code"], asof)
                                   if ledger is not None else None)
    changes = {}
    if previous is not None:
        if previous.get("mode") != "HOLDING_REVIEW":
            raise ValueError("previous record must be a holding review")
        if previous.get("position_id") != result["position_id"] or previous.get("code") != result["code"]:
            raise ValueError("previous review belongs to another holding")
        if previous.get("position_kind") != result["position_kind"]:
            raise ValueError("paper and actual holding histories cannot be mixed")
        if stamp(previous["asof"]) >= stamp(asof):
            raise ValueError("previous review must precede current review")
        for field in ("decision", "execution_state", "emotion_state", "theme_feedback", "notice_summary"):
            if previous.get(field) != result.get(field):
                changes[field] = {"previous": previous.get(field), "current": result.get(field)}
        old_events = {r["evidence_id"] for r in (previous.get("catalyst_evidence") or {}).get("records", [])}
        events = (result.get("catalyst_evidence") or {}).get("records", [])
        result["new_evidence_ids"] = [r["evidence_id"] for r in events if r["evidence_id"] not in old_events]
    else:
        result["new_evidence_ids"] = None
    result["changes_since_previous_review"] = changes
    result["scheduled_monitoring_enabled"] = False
    return result


def next_session_watchlist(frame, pool_asof, asof, snapshots, calendar, ledger=None):
    result = watchlist(frame, pool_asof)
    result["session_binding"] = attach_opening(result["candidates"], snapshots, pool_asof, asof, calendar)
    result["pool_asof"] = stamp(pool_asof).isoformat()
    result["asof"] = stamp(asof).isoformat()
    result["mode"] = "NEXT_SESSION_WATCHLIST"
    for candidate in result["candidates"]:
        candidate["catalyst_evidence"] = (catalyst_snapshot(ledger, candidate["code"], asof)
                                          if ledger is not None else None)
    return result


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "review"):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, help="candidate CSV or quote JSON")
        command.add_argument("--asof", required=True)
        command.add_argument("--ledger")
        command.add_argument("--output", required=True)
        command.add_argument("--competition", help="scoped competitor snapshot JSON")
        command.add_argument("--previous-competition", help="earlier snapshot of the same scope")
        command.add_argument("--environment", help="per-field timestamped short-term environment JSON")
        command.add_argument("--previous-environment", help="earlier same-session fixed-cohort environment JSON")
        command.add_argument("--theme-registry", help="timestamped membership registry CSV, forward evidence only")
        command.add_argument("--theme-signal-date", help="explicit date of the theme evidence being reviewed")
        command.add_argument("--duration-report", help="historical relay duration reference, NOT holding forecast")
        command.add_argument("--duration-available-at", help="explicit timezone timestamp when report became available")
        if name == "score":
            command.add_argument("--pool-asof", help="prior-session closing pool cutoff")
            command.add_argument("--opening", help="current opening snapshots JSON list")
            command.add_argument("--calendar", help="sourced complete trading-session calendar JSON")
        if name == "review":
            command.add_argument("--position", required=True)
            command.add_argument("--previous")
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    ledger = read(args.ledger)
    if bool(args.theme_registry) != bool(args.theme_signal_date):
        parser.error("--theme-registry and --theme-signal-date must be supplied together")
    if args.previous_environment and not args.environment:
        parser.error("--previous-environment requires --environment")
    if args.command == "score":
        frame = pd.read_csv(args.input, dtype={"code": str})
        binding = [args.pool_asof, args.opening, args.calendar]
        if any(binding) and not all(binding):
            parser.error("--pool-asof, --opening and --calendar must be supplied together")
        result = (next_session_watchlist(frame, args.pool_asof, args.asof, read(args.opening),
                                        read(args.calendar), ledger) if all(binding)
                  else watchlist(frame, args.asof, ledger))
    else:
        result = holding(read(args.position), read(args.input), args.asof, ledger, read(args.previous))
    if args.previous_competition and not args.competition:
        parser.error("--previous-competition requires --competition")
    if args.competition:
        competition = read(args.competition)
        prior_competition = read(args.previous_competition)
        if args.command == "score":
            for candidate in result["candidates"]:
                candidate["competition_review"] = compare_competition(candidate["code"], competition, args.asof, prior_competition)
        else:
            result["competition_review"] = compare_competition(result["code"], competition, args.asof, prior_competition)
    if bool(args.duration_report) != bool(args.duration_available_at):
        parser.error("--duration-report and --duration-available-at must be supplied together")
    if args.duration_report:
        duration = read(args.duration_report)
        if args.command == "score":
            for candidate in result["candidates"]:
                candidate["duration_reference"] = duration_reference(candidate.get("boards"), duration, args.asof, args.duration_available_at)
        else:
            result["duration_reference"] = duration_reference(read(args.input).get("boards"), duration, args.asof, args.duration_available_at)
    result["short_term_environment"] = analyze_environment(read(args.environment), args.asof, read(args.previous_environment))
    result["primary_research_objective"] = "executable_net_returns_from_first_and_consecutive_limit_up_pool"
    result["premium_model_status"] = "NOT_VALIDATED"
    if args.theme_registry:
        registry = pd.read_csv(args.theme_registry, dtype={"code": str})
        subjects = result["candidates"] if args.command == "score" else [result]
        for subject in subjects:
            subject["theme_identity_review"] = review_theme_identity(
                registry, subject["code"], args.theme_signal_date, args.asof)
    result = clean(result)
    paths = [args.input, args.ledger, getattr(args, "position", None), getattr(args, "previous", None), args.competition, args.previous_competition, args.duration_report]
    paths.extend([args.environment, args.previous_environment])
    paths.append(args.theme_registry)
    paths.extend([getattr(args, "opening", None), getattr(args, "calendar", None)])
    result["source_hashes"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths if p}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
