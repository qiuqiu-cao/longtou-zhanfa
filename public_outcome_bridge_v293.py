"""Reconstruct every candidate's closing truth from independently archived data.

Per-stock missing evidence stays UNKNOWN in the complete original denominator.
An inferred quote field is only accepted when it agrees with Eastmoney's
explicit ztp, actual unadjusted daily prices and the exact quote session.
All truth is provider-scope research; official exchange coverage is separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

from daily_research_contract_v292 import score, stamp
from daily_research_runner_v292 import sha, write_new
from next_session_expectation_contract_v271 import digest, load_registered
from public_outcome_sources_v293 import reconstruct, cent, MAINBOARD

VERSION = "v293-source-bound-outcomes-1"


def legacy_signal(directory, signal):
    """Use the original hash-bound snapshot; never an API date fallback."""
    directory = Path(directory)
    path = directory/"manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "v204-timeout-limit-pool-collector-1" or str(manifest.get("date")) != signal.replace("-", ""):
        raise ValueError("original signal snapshot date/schema mismatch")
    captured = stamp(manifest["collected_at"])
    if captured.date().isoformat() != signal or captured < stamp(signal+"T15:00:00+08:00"):
        raise ValueError("legacy signal snapshot was not captured at the original close")
    item = manifest["datasets"]["limit_up"]
    source = directory/item["path"]
    if sha(source) != item["sha256"]:
        raise ValueError("original signal pool hash mismatch")
    frame = pd.read_csv(source, dtype={"代码": str})
    if len(frame) != item["rows"] or frame["代码"].duplicated().any():
        raise ValueError("original signal count or code-set mismatch")
    rows = []
    for r in frame.to_dict("records"):
        code, name = r["代码"], str(r["名称"])
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit():
            raise ValueError("invalid original signal code")
        if not code.startswith(MAINBOARD) or "ST" in name.upper() or "退" in name:
            continue
        boards = r["连板数"]
        if isinstance(boards, bool) or not float(boards).is_integer() or boards < 1:
            raise ValueError("invalid original board count")
        rows.append({"code": code, "name": name, "price_cents": cent(r["最新价"]), "boards": int(boards)})
    return {"rows": rows, "date": signal, "available_at": manifest["collected_at"], "raw_path": str(source.resolve()),
        "scope_note": "ORIGINAL_HASH_BOUND_PROVIDER_SNAPSHOT_QDATE_NOT_RETAINED_IN_LEGACY_CAPTURE"}, [path, source]


def candidate_truth(candidate, signal_pool, target_pool, feedback, daily, signal, target):
    code, height = candidate["code"], candidate["boards"]
    row = {"code": code, "boards": height, "promoted": None, "status": "unavailable", "checks": {}, "reason": None}
    checks = row["checks"]
    def stop(reason):
        row["reason"] = reason
        return row
    if not signal_pool or not target_pool or not feedback:
        return stop("EXACT_SIGNAL_TARGET_POOL_OR_FEEDBACK_UNAVAILABLE")
    original = next((r for r in signal_pool["rows"] if r["code"] == code), None)
    future = next((r for r in target_pool["rows"] if r["code"] == code), None)
    returned = next((r for r in feedback["rows"] if r["code"] == code), None)
    if not original or original["boards"] != height:
        return stop("FROZEN_SIGNAL_BOARD_DENOMINATOR_DISAGREES")
    if not returned or not returned.get("upper_cents_explicit"):
        return stop("EXPLICIT_TARGET_PRICE_LIMIT_OR_FEEDBACK_MISSING")
    raw_feedback = json.loads(Path(feedback["raw_path"]).read_bytes())["data"]["pool"]
    prior = next(r for r in raw_feedback if str(r["c"]) == code)
    if type(prior.get("ylbc")) is not int or prior["ylbc"] != height:
        return stop("REPORTED_PREVIOUS_BOARD_COUNT_CONFLICT")
    item = daily.get(code)
    if not item or signal not in item["days"] or target not in item["days"]:
        return stop("EXACT_SIGNAL_OR_TARGET_UNADJUSTED_DAY_MISSING")
    s, t = item["days"][signal], item["days"][target]
    checks["signal_close_agrees_original"] = s["close"] == original["price_cents"]
    checks["signal_close_equals_daily_high"] = s["close"] == s["high"]
    if not all(checks.values()):
        return stop("ORIGINAL_SIGNAL_CLOSE_OR_CAP_NECESSARY_CONDITION_CONFLICT")
    quote = item["quote"]
    if quote["status"] != "POSITIONAL_FIELDS_INFERRED":
        return stop("QUOTE_SNAPSHOT_MISSING")
    observed = stamp(quote["observed_at"])
    if observed.date().isoformat() != target or observed < stamp(target+"T15:00:00+08:00") or observed > stamp(item["available_at"]):
        return stop("QUOTE_NOT_AN_AVAILABLE_EXACT_TARGET_CLOSE")
    q = quote["values"]
    checks["quote_matches_unadjusted_target_ohlc"] = all(q[k] == t[k] for k in ["open", "high", "low", "close"])
    checks["quote_previous_close_matches_signal"] = q["previous_close"] == s["close"]
    checks["explicit_limit_agrees_inferred_quote_field"] = q["upper_candidate"] == returned["upper_cents_explicit"]
    checks["feedback_price_agrees_target_close"] = returned["price_cents"] == t["close"]
    if not all(checks.values()):
        return stop("TARGET_PRICE_REFERENCE_OR_LIMIT_SOURCE_CONFLICT")
    upper = returned["upper_cents_explicit"]
    if t["close"] > upper or t["high"] > upper:
        return stop("TARGET_EXCEEDS_REPORTED_PRICE_LIMIT")
    at_cap = t["close"] == upper
    if at_cap:
        if not future or future["price_cents"] != t["close"] or future["boards"] != height+1:
            return stop("CLOSING_PRICE_AND_CONSECUTIVE_BOARD_MEMBERSHIP_CONFLICT")
    elif future:
        return stop("BELOW_LIMIT_STOCK_REPORTED_IN_CLOSING_POOL")
    row.update(status="verified", promoted=at_cap,
        label_scope="PROVIDER_REPORTED_MAINBOARD_NAME_NON_ST_WITH_INDEPENDENT_DAY_AND_PRICE_LIMIT_RECONCILIATION",
        exchange_security_status_verified=False, continuous_board_exchange_certified=False,
        observed_at=target+"T15:00:00+08:00",
        available_at=max([signal_pool["available_at"], target_pool["available_at"], feedback["available_at"], item["available_at"]], key=stamp),
        sources=[signal_pool["raw_path"], target_pool["raw_path"], feedback["raw_path"], item["raw_path"]])
    return row


def assemble(registered, archive, prior_snapshot, assembled_at):
    p = registered["plan"]
    manifest, pools, daily, failures = reconstruct(archive)
    if (manifest["signal_date"], manifest["target_date"]) != (p["signal_date"], p["target_date"]):
        raise ValueError("source archive belongs to a different target")
    if stamp(assembled_at) < stamp(manifest["recorded_at"]):
        raise ValueError("assembly predates receipt archive")
    signal, target = p["signal_date"], p["target_date"]
    source_files = [Path(archive)/"manifest.json", Path(archive)/"normalized.json"]
    source_files += [Path(r["raw_path"]) for r in manifest["requests"] if r.get("raw_path")]
    signal_pool = pools.get((signal, "limit_up"))
    prior_note = "EXACT_QDATE_RAW_SIGNAL_POOL"
    if prior_snapshot:
        original, files = legacy_signal(prior_snapshot, signal)
        source_files += files
        if signal_pool and digest([(r["code"],r["boards"],r["price_cents"]) for r in sorted(signal_pool["rows"],key=lambda r:r["code"])]) != digest([(r["code"],r["boards"],r["price_cents"]) for r in sorted(original["rows"],key=lambda r:r["code"]) ]):
            raise ValueError("new raw signal response contradicts the saved original snapshot")
        signal_pool = original
        prior_note = original["scope_note"]
    if signal_pool:
        expected = {r["code"]:r["boards"] for r in signal_pool["rows"] if r["boards"] >= 2}
        if expected != {r["code"]:r["boards"] for r in p["candidates"]}:
            raise ValueError("original full relay candidate set differs from registration")
    truth = [candidate_truth(c, signal_pool, pools.get((target,"limit_up")), pools.get((target,"previous_limit")), daily, signal, target) for c in p["candidates"]]
    if any(r["status"] == "verified" and stamp(r["available_at"]) > stamp(assembled_at) for r in truth):
        raise ValueError("candidate truth unavailable at assembly")
    sources = {str(path.resolve()): sha(path) for path in source_files}
    for name in ["public_outcome_bridge_v293.py", "public_outcome_sources_v293.py"]:
        path = Path(__file__).resolve().parent/name
        sources[str(path)] = sha(path)
    return {"date": target, "registration_hash": registered["sha256"], "assembled_at": assembled_at,
        "version": VERSION, "archive": str(Path(archive).resolve()),
        "prior_snapshot": str(Path(prior_snapshot).resolve()) if prior_snapshot else None,
        "source_hashes": sources, "candidate_closing_truth": truth,
        "market": {"status": "unavailable", "reason": "完整同口径市场质量与分类证据未重建，不手填标签"},
        "limit_pool": {"status": "unavailable", "scope": "MAINBOARD_NON_ST_CLOSING_LIMIT_UP", "complete": False,
            "reason": "逐股独立价格核验不等于全市场官方涨停池已核验"},
        "labels": [], "observations": [], "error_evidence": [],
        "source_limitations": {"signal_capture": prior_note, "request_failures": failures,
            "quote_field_semantics": "位置字段的含义为推断，必须与明确ztp、日线和实际收盘日期三重吻合",
            "official_security_states_verified": False, "intraday_reseal_available": False,
            "historical_receipt_is_forward_knowledge": False}}


def verify(registered, value):
    if value.get("version") != VERSION or value.get("registration_hash") != registered["sha256"]:
        raise ValueError("wrong source-bound outcome schema or original forecast")
    for path, wanted in value["source_hashes"].items():
        if not Path(path).is_file() or sha(path) != wanted:
            raise ValueError("outcome input/code archive hash mismatch")
    reconstructed = assemble(registered, value["archive"], value.get("prior_snapshot"), value["assembled_at"])
    if digest(reconstructed) != digest(value):
        raise ValueError("outcome differs from a fresh full raw-source reconstruction")
    return reconstructed


def source_score(registered, value, recorded_at):
    verify(registered, value)
    result = score(registered, value, recorded_at)
    truths = {r["code"]:r for r in value["candidate_closing_truth"]}
    for row in result["rows"]:
        if row["layer"] not in {"promotion", "break"}:
            continue
        truth = truths[row["key"]]
        if truth["status"] != "verified":
            row["reason"] = truth["reason"]
            continue
        if stamp(truth["available_at"]) > stamp(recorded_at):
            raise ValueError("candidate truth unavailable at scoring")
        observed = truth["promoted"] if row["layer"] == "promotion" else not truth["promoted"]
        row.update(truth=observed, outcome_known=True, correct=row["call"] == observed if row["forecast_made"] else None,
            reason=None, label_scope=truth["label_scope"])
    result["errors"] = [{"layer":r["layer"],"key":r["key"],"original":r["call"],"actual":r["truth"],
        "observable_by_freeze": None,"evidence":[],"explanation":"归因证据不足，不强行解释"} for r in result["rows"] if r["correct"] is False]
    result["scoring_status"] = "PARTIAL" if any(r["forecast_made"] and not r["outcome_known"] for r in result["rows"]) else "AVAILABLE_DECLARED_CALLS_SCORED"
    result["outcome_verification"] = "PER_CANDIDATE_RAW_SOURCE_RECONSTRUCTION"
    result["exchange_certified_accuracy_claim_allowed"] = False
    return result


def build(registered_path, archive, prior_snapshot, output):
    registered = load_registered(registered_path)
    value = assemble(registered, archive, prior_snapshot, datetime.now(timezone.utc).isoformat())
    write_new(output, value)
    source_score(registered, value, datetime.now(timezone.utc).isoformat())
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--prior-snapshot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.registered, args.archive, args.prior_snapshot, args.output)
    print(json.dumps({"verified":sum(r["status"]=="verified" for r in result["candidate_closing_truth"]),
        "unknown":[{"code":r["code"],"reason":r["reason"]} for r in result["candidate_closing_truth"] if r["status"]!="verified"]}))
