"""Immutable, time-aware next-session expectations and contingency decisions.

This contract does not submit orders. Qualitative expectations can be frozen
without tomorrow's auction; live confirmation and predictive validation remain
separate states. A late snapshot is a replay, never a forward prediction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "v271-expectation-contract-2"
STAGES = {"AUCTION": "09:25:00", "OPEN_0935": "09:35:00", "OPEN_1000": "10:00:00", "CLOSE": "15:00:00"}
STAGE_ORDER = list(STAGES)
EXPECTATION_STATES = {"EXCEEDS", "MEETS", "MISSES", "CONFLICTING", "UNAVAILABLE"}
IDENTITIES = {"LEADER_SEED", "THEME_CORE", "TURNOVER_CORE", "SPACE_ANCHOR", "OLD_LEADER", "SUPPLEMENT", "CHALLENGER", "UNKNOWN"}
TRUTH_LABELS = {"RISK_ON", "REPAIR", "RISK_OFF"}


def stamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.astimezone(ZoneInfo("Asia/Shanghai"))


def now():
    return datetime.now(timezone.utc)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def next_session(calendar, signal_date):
    if calendar.get("status") != "verified" or not calendar.get("source"):
        raise ValueError("independently verified exchange calendar required")
    sessions = calendar.get("sessions", [])
    if sessions != sorted(set(sessions)):
        raise ValueError("calendar sessions must be unique and sorted")
    if signal_date not in sessions or sessions.index(signal_date)+1 >= len(sessions):
        raise ValueError("calendar lacks signal or next session")
    return sessions[sessions.index(signal_date)+1]


def validate_plan(plan, recorded_at):
    recorded = stamp(recorded_at)
    signal, target = plan["signal_date"], plan["target_date"]
    if target != next_session(plan["calendar"], signal):
        raise ValueError("target is not the immediately next verified trading session")
    cutoff = stamp(plan["evidence_cutoff"])
    if cutoff.date().isoformat() != signal or cutoff.time().isoformat() < "15:00:00":
        raise ValueError("EOD cutoff must be on signal session after close")
    if cutoff > recorded:
        raise ValueError("plan frozen before its evidence is available")
    knowledge = stamp(plan.get("knowledge_cutoff", plan["evidence_cutoff"]))
    if knowledge < cutoff or knowledge > recorded:
        raise ValueError("knowledge cutoff must follow the market snapshot and precede registration")
    if not plan.get("model_version") or not plan.get("input_hash"):
        raise ValueError("model version and immutable input hash required")
    if len(plan["input_hash"]) != 64 or any(c not in "0123456789abcdef" for c in plan["input_hash"]):
        raise ValueError("input hash must be SHA-256")
    market = plan["market"]
    if market.get("primary") not in TRUTH_LABELS:
        raise ValueError("one market primary expectation required")
    if market.get("direction") is not None and market["direction"] not in {"IMPROVE", "CONTINUE", "WORSEN"}:
        raise ValueError("invalid market direction forecast")
    if not market.get("opposing_evidence") or not market.get("invalidation"):
        raise ValueError("opposing evidence and invalidation required")
    probability = market.get("probability")
    if probability is not None and (not finite(probability) or not 0 <= probability <= 1
                                    or market.get("calibration_status") != "OUT_OF_SAMPLE_CALIBRATED"):
        raise ValueError("numerical forecast probability requires calibration evidence")
    codes = []
    for row in plan.get("candidates", []):
        code = row["code"]
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit():
            raise ValueError("six-digit stock code required")
        if not code.startswith(("60", "00")):
            raise ValueError("candidate outside default mainboard code scope")
        if not isinstance(row.get("boards"), int) or isinstance(row["boards"], bool) or row["boards"] < 2:
            raise ValueError("relay candidate requires a current consecutive-board count >=2")
        codes.append(code)
        if row.get("identity") not in IDENTITIES or row.get("promotion_call") not in {"PROMOTE", "FAIL", "UNCERTAIN"}:
            raise ValueError("explicit current identity and promotion call required")
        if not row.get("identity_evidence") or not row.get("identity_invalidation"):
            raise ValueError("identity evidence and invalidation required")
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate candidate code")
    if plan.get("primary_candidate") is not None and plan["primary_candidate"] not in codes:
        raise ValueError("primary outside frozen candidate denominator")
    for selection in ["identity_primary_candidate", "non_one_word_primary_candidate"]:
        selected = plan.get(selection)
        if selected is not None:
            candidate = next((r for r in plan["candidates"] if r["code"] == selected), None)
            if candidate is None or candidate["promotion_call"] != "PROMOTE":
                raise ValueError("positive prediction selection requires a frozen PROMOTE call in the denominator")
            if selection == "non_one_word_primary_candidate" and candidate.get("actual_one_word_daily") is not False:
                raise ValueError("non-one-word selection requires verified false daily one-word flag")
    if plan.get("primary_candidate") is not None and (not finite(plan.get("highest_accept_price"))
                                                      or plan["highest_accept_price"] <= 0):
        raise ValueError("primary candidate requires a frozen highest acceptable price")
    if plan.get("denominator_status") != "verified_complete":
        raise ValueError("complete candidate denominator required for a frozen forecast")
    for band in plan.get("bands", []):
        if band.get("stage") not in STAGES or band.get("direction") not in {"HIGHER_BETTER", "LOWER_BETTER"}:
            raise ValueError("invalid expectation stage or direction")
        if not band.get("metric") or not band.get("unit") or not band.get("scope"):
            raise ValueError("metric, unit and cohort/scope must be frozen")
        if not finite(band.get("low")) or not finite(band.get("high")) or band["low"] > band["high"]:
            raise ValueError("finite ordered expected interval required")
        if not band.get("basis"):
            raise ValueError("expected interval needs a recorded basis")
    if not plan.get("bands"):
        raise ValueError("at least one measurable expectation required")
    if not plan.get("evidence"):
        raise ValueError("timestamped evidence required")
    for evidence in plan["evidence"]:
        observed, available = stamp(evidence["observed_at"]), stamp(evidence["available_at"])
        kind = evidence.get("kind", "MARKET_SNAPSHOT")
        if kind not in {"MARKET_SNAPSHOT", "RESEARCH_DERIVED", "ANNOUNCEMENT"}:
            raise ValueError("unknown evidence clock kind")
        if observed > available or available > knowledge:
            raise ValueError("future or unavailable evidence in frozen plan")
        if kind == "MARKET_SNAPSHOT" and observed > cutoff:
            raise ValueError("market snapshot includes post-signal trading observations")
        if not evidence.get("source") or evidence.get("status") not in {"verified", "inferred", "delayed"}:
            raise ValueError("evidence source/status required")
    target_start = stamp(target+"T09:15:00+08:00")
    is_forward = recorded < target_start
    return {"forecast_status": "FORWARD_REGISTERED" if is_forward else "HISTORICAL_REPLAY",
            "market_snapshot_cutoff": plan["evidence_cutoff"],
            "knowledge_cutoff": plan.get("knowledge_cutoff", plan["evidence_cutoff"]),
            "execution_state": "PENDING_OPEN_CONFIRMATION", "accuracy_claim_allowed": False}


def band_result(band, observation):
    if observation is None or observation.get("status") != "verified" or not finite(observation.get("value")):
        return "UNAVAILABLE"
    if (observation.get("unit"), observation.get("scope")) != (band["unit"], band["scope"]):
        raise ValueError("observation definition differs from frozen expectation")
    value = observation["value"]
    if band["low"] <= value <= band["high"]:
        return "MEETS"
    higher = value > band["high"]
    favorable = higher if band["direction"] == "HIGHER_BETTER" else not higher
    return "EXCEEDS" if favorable else "MISSES"


def evaluate_stage(plan, stage, observations, recorded_at, *, previous=None):
    if stage not in STAGES:
        raise ValueError("unknown observation stage")
    recorded = stamp(recorded_at)
    deadline = stamp(plan["target_date"]+"T"+STAGES[stage]+"+08:00")
    if recorded < deadline:
        raise ValueError("stage evaluated before it ends")
    if previous is not None:
        if STAGE_ORDER.index(stage) <= STAGE_ORDER.index(previous["stage"]):
            raise ValueError("stages cannot be rewritten or move backwards")
        if previous.get("primary_candidate") != plan.get("primary_candidate"):
            raise ValueError("previous stage belongs to a different primary")
        if stamp(previous["recorded_at"]) > recorded:
            raise ValueError("previous stage recorded in the future")
    bands = [b for b in plan["bands"] if b["stage"] == stage]
    keyed = {}
    for obs in observations:
        key = (obs.get("metric"), obs.get("scope"))
        if key in keyed:
            raise ValueError("duplicate observation metric/scope")
        observed, available = stamp(obs["observed_at"]), stamp(obs["available_at"])
        if observed > available or available > recorded or observed > deadline:
            raise ValueError("future observation or post-stage information")
        if observed.date().isoformat() != plan["target_date"]:
            raise ValueError("wrong observation session")
        if not obs.get("source"):
            raise ValueError("observation source required")
        keyed[key] = obs
    details = [{"metric": b["metric"], "scope": b["scope"], "expected": [b["low"], b["high"]],
                "result": band_result(b, keyed.get((b["metric"], b["scope"]))),
                "required": b.get("required", True)} for b in bands]
    required = [d["result"] for d in details if d["required"]]
    if not required or "UNAVAILABLE" in required:
        state = "UNAVAILABLE"
    elif "EXCEEDS" in required and "MISSES" in required:
        state = "CONFLICTING"
    elif "MISSES" in required:
        state = "MISSES"
    elif "EXCEEDS" in required:
        state = "EXCEEDS"
    else:
        state = "MEETS"
    # Risk signals are explicit frozen metrics. All post-stage observation extras
    # are excluded; no missing field is silently interpreted as negative.
    hard_failure = any(d["result"] == "MISSES" and b.get("hard_veto", False)
                       for b, d in zip(bands, details))
    prior_cancel = previous and previous.get("execution_state") == "CANCELLED"
    if prior_cancel or hard_failure:
        action = "CANCELLED"
    elif state in {"UNAVAILABLE", "CONFLICTING"}:
        action = "WATCH_DATA_MISSING" if state == "UNAVAILABLE" else "WATCH_CONFLICTING"
    elif state == "MISSES":
        action = "WATCH_BELOW_EXPECTATION"
    else:
        action = "OBSERVATION_CONFIRMED"
    return {"stage": stage, "recorded_at": recorded_at, "expectation_state": state,
            "execution_state": action, "details": details,
            "primary_candidate": plan.get("primary_candidate"),
            "highest_accept_price": plan.get("highest_accept_price"),
            "automatic_replacement_allowed": False, "automatic_entry_allowed": False,
            "earliest_new_order_decision_at": recorded_at}


def register(plan, directory, recorded_at):
    result = validate_plan(plan, recorded_at)
    wrapped = {"version": VERSION, "recorded_at": recorded_at, "plan": deepcopy(plan), **result}
    wrapped["sha256"] = digest(wrapped)
    path = Path(directory) / (plan["signal_date"]+"_"+wrapped["sha256"][:16]+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if result["forecast_status"] == "FORWARD_REGISTERED":
        for existing in path.parent.glob(plan["signal_date"]+"_*.json"):
            prior = load_registered(existing)
            if prior.get("forecast_status") == "FORWARD_REGISTERED":
                raise ValueError("a forward primary expectation is already frozen for this signal date")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(wrapped, ensure_ascii=False, indent=2, allow_nan=False))
    return path, wrapped


def load_registered(path):
    wrapped = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = wrapped.pop("sha256")
    if digest(wrapped) != wanted:
        raise ValueError("frozen plan hash mismatch")
    wrapped["sha256"] = wanted
    return wrapped


def verify_fill(stage_result, fill):
    if stage_result["execution_state"] != "OBSERVATION_CONFIRMED":
        raise ValueError("entry confirmation not present")
    if fill.get("status") != "broker_confirmed" or not fill.get("source"):
        raise ValueError("actual broker evidence required")
    if fill.get("code") != stage_result["primary_candidate"]:
        raise ValueError("fill differs from frozen primary candidate")
    if stamp(fill["filled_at"]) <= stamp(stage_result["recorded_at"]):
        raise ValueError("fill cannot precede the observation-based decision")
    if not finite(fill.get("price")) or fill["price"] <= 0:
        raise ValueError("positive actual fill price required")
    if fill["price"] > stage_result["highest_accept_price"]:
        raise ValueError("actual fill exceeds frozen maximum price")
    return True


def settle(registered, outcomes, recorded_at):
    plan = registered["plan"]
    cutoff = stamp(recorded_at)
    close = stamp(plan["target_date"]+"T15:00:00+08:00")
    if cutoff < close or outcomes.get("date") != plan["target_date"]:
        raise ValueError("settlement requires the exact completed next session")
    for block in [outcomes["market"], outcomes["limit_pool"]]:
        if block.get("status") != "verified" or not block.get("source"):
            raise ValueError("verified market and complete closing pool sources required")
        observed, available = stamp(block["observed_at"]), stamp(block["available_at"])
        if observed < close or observed > available or available > cutoff or observed.date() != close.date():
            raise ValueError("outcome timestamp is not a completed, available target-session result")
    pool = outcomes["limit_pool"]
    if pool.get("scope") != "MAINBOARD_NON_ST_CLOSING_LIMIT_UP" or pool.get("complete") is not True:
        raise ValueError("complete mainboard non-ST closing limit-up denominator required")
    rows = pool.get("rows", [])
    by_code = {}
    for row in rows:
        if row["code"] in by_code or not isinstance(row.get("boards"), int) or row["boards"] < 1:
            raise ValueError("duplicate pool code or invalid consecutive-board count")
        by_code[row["code"]] = row
    market_truth = outcomes["market"].get("primary")
    if market_truth not in TRUTH_LABELS:
        raise ValueError("unknown realized market regime")
    candidate_results = []
    for row in plan.get("candidates", []):
        boards = row.get("boards")
        if not isinstance(boards, int) or boards < 2:
            raise ValueError("frozen relay candidate requires its current board count")
        promoted = row["code"] in by_code and by_code[row["code"]]["boards"] == boards+1
        call = row["promotion_call"]
        candidate_results.append({"code": row["code"], "promoted": promoted, "call": call,
            "correct": (call == "PROMOTE") == promoted if call != "UNCERTAIN" else None})
    primary = next((r for r in candidate_results if r["code"] == plan.get("primary_candidate")), None)
    identity_primary = next((r for r in candidate_results if r["code"] == plan.get("identity_primary_candidate")), None)
    non_word_primary = next((r for r in candidate_results if r["code"] == plan.get("non_one_word_primary_candidate")), None)
    direction_truth = outcomes["market"].get("direction")
    if plan["market"].get("direction") and direction_truth not in {"IMPROVE", "CONTINUE", "WORSEN"}:
        raise ValueError("market movement forecast requires the corresponding realized direction")
    return {"version": VERSION, "registration_hash": registered["sha256"], "signal_date": plan["signal_date"],
        "target_date": plan["target_date"], "recorded_at": recorded_at,
        "forecast_status": registered["forecast_status"],
        "market_primary_correct": plan["market"]["primary"] == market_truth,
        "market_direction_correct": plan["market"]["direction"] == direction_truth if plan["market"].get("direction") else None,
        "primary_candidate_promoted": primary["promoted"] if primary else None,
        "identity_primary_promoted": identity_primary["promoted"] if identity_primary else None,
        "non_one_word_primary_promoted": non_word_primary["promoted"] if non_word_primary else None,
        "candidate_results": candidate_results, "outcome_hash": digest(outcomes),
        "trade_win_rate_claim_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--directory", default="data/v271/expectation_registry")
    evaluate = sub.add_parser("stage")
    evaluate.add_argument("--registered", required=True)
    evaluate.add_argument("--stage", choices=STAGES, required=True)
    evaluate.add_argument("--observations", required=True)
    evaluate.add_argument("--previous")
    evaluate.add_argument("--output", required=True)
    settlement = sub.add_parser("settle")
    settlement.add_argument("--registered", required=True)
    settlement.add_argument("--outcomes", required=True)
    settlement.add_argument("--output", required=True)
    args = parser.parse_args()
    recorded_at = now().isoformat()  # CLI has no backdating argument.
    if args.command == "freeze":
        path, result = register(json.loads(Path(args.plan).read_text(encoding="utf-8")), args.directory, recorded_at)
        print(json.dumps({"path": str(path), "forecast_status": result["forecast_status"]}))
    elif args.command == "stage":
        registered = load_registered(args.registered)
        previous = json.loads(Path(args.previous).read_text(encoding="utf-8")) if args.previous else None
        if previous is not None and previous.get("registration_hash") != registered["sha256"]:
            raise ValueError("previous stage refers to another frozen plan")
        result = evaluate_stage(registered["plan"], args.stage,
            json.loads(Path(args.observations).read_text(encoding="utf-8")), recorded_at, previous=previous)
        result["registration_hash"] = registered["sha256"]
        result["forward_evidence_eligible"] = (registered["forecast_status"] == "FORWARD_REGISTERED"
            and stamp(recorded_at).date().isoformat() == registered["plan"]["target_date"]
            and stamp(recorded_at) <= stamp(registered["plan"]["target_date"]+"T"+STAGES[args.stage]+"+08:00")+timedelta(minutes=5))
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({"expectation_state": result["expectation_state"], "execution_state": result["execution_state"]}))
    else:
        result = settle(load_registered(args.registered), json.loads(Path(args.outcomes).read_text(encoding="utf-8")), recorded_at)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({"market_primary_correct": result["market_primary_correct"],
                          "primary_candidate_promoted": result["primary_candidate_promoted"]}))


if __name__ == "__main__":
    main()
