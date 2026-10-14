"""Fixed BASE/G1 prospective registration, source-bound settlement and review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from theme_structure_g1 import load_protocol, read, sha, stamp, run as overlay_run
from next_session_expectation_contract_v271 import digest, load_registered


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_chain(path):
    events, previous = [], None
    if not Path(path).exists():
        return events
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        e = json.loads(line); saved = e.pop("sha256")
        if e.get("previous_sha256") != previous or digest(e) != saved:
            raise ValueError("G1 ledger hash conflict")
        for b in e.get("immutable_sources", []):
            if not Path(b["path"]).is_file() or sha(b["path"]) != b["sha256"]:
                raise ValueError("G1 original artifact hash conflict")
        e["sha256"] = saved; previous = saved; events.append(e)
    return events


def append(path, event):
    chain = read_chain(path)
    event["previous_sha256"] = chain[-1]["sha256"] if chain else None
    event["sha256"] = digest(event)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    return event


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def verify_implementation(protocol_path):
    path = Path(protocol_path).parent/"implementation_BEFORE_FORWARD.json"
    value = read(path)
    if value["protocol_sha256"] != sha(protocol_path) or stamp(value["formed_at"]) > stamp(now()):
        raise ValueError("G1 implementation formation/protocol conflict")
    correction_paths = [path.parent/name for name in ("implementation_correction_BEFORE_FIRST_G1.json", "implementation_correction_FINAL_BEFORE_FIRST_G1.json")]
    for correction_path in correction_paths:
      if correction_path.exists():
        correction = read(correction_path)
        if (correction["prior_implementation_sha256"] != sha(path) or correction["protocol_sha256"] != sha(protocol_path)
                or stamp(correction["formed_at"]) < stamp(value["formed_at"]) or stamp(correction["formed_at"]) > stamp(now())
                or correction.get("numerical_refits") != 0 or correction.get("G1_rule_or_gate_changed") is not False
                or set(correction["source_hashes"]) != set(value["source_hashes"])):
            raise ValueError("G1 pre-forward correction chain conflict")
        changed = {name for name in value["source_hashes"] if value["source_hashes"][name] != correction["source_hashes"][name]}
        if changed != set(correction["prior_source_preimages"]):
            raise ValueError("G1 correction must preserve every changed original implementation source")
        for filename, snapshot in correction["prior_source_preimages"].items():
            if filename not in value["source_hashes"] or snapshot["sha256"] != value["source_hashes"][filename] or sha(snapshot["path"]) != snapshot["sha256"]:
                raise ValueError("G1 corrected source preimage conflict")
        started = read_chain(path.parent/"shadow/registrations.jsonl")
        if started and stamp(correction["formed_at"]) > min(stamp(e["recorded_at"]) for e in started):
            raise ValueError("G1 implementation cannot change after prospective registration starts")
        value = correction
        path = correction_path
    for filename, wanted in value["source_hashes"].items():
        if not Path(filename).is_file() or sha(filename) != wanted:
            raise ValueError("G1 frozen implementation changed; no silent rule migration")
    return binding(path)


def register(protocol_path, calendar_path, registered_path, overlay_path):
    from current_model_diagnostics import _calendar
    p = load_protocol(protocol_path); implementation = verify_implementation(protocol_path)
    registered = load_registered(registered_path); plan = registered["plan"]
    view = read(overlay_path)
    day, target = plan["signal_date"], plan["target_date"]
    sessions = _calendar(calendar_path)["sessions"]
    successor = dict(zip(sessions[:-1], sessions[1:]))
    at, cutoff = now(), registered["knowledge_cutoff"]
    if successor.get(day) != target:
        raise ValueError("G1 target must be exact official successor")
    if (registered["forecast_status"] != "FORWARD_REGISTERED" or not view.get("prospective_eligible")
            or stamp(cutoff) <= stamp(p["formed_at"]) or stamp(cutoff) > stamp(at)
            or stamp(at) >= stamp(target+"T00:00:00+08:00")
            or stamp(view["generated_at"]) > stamp(cutoff)
            or stamp(registered["recorded_at"]) > stamp(at)):
        raise ValueError("G1 must form and register after protocol, before target session; no retrospective entry")
    if view["signal_date"] != day or view["target_date"] != target or view["protocol_binding"]["sha256"] != sha(protocol_path):
        raise ValueError("G1 view scope/protocol mismatch")
    if view["model_binding"] != p["model_binding"] or view.get("ranking_status") != "KNOWN":
        raise ValueError("frozen model changed or scores unavailable")
    for filename, wanted in view["source_hashes"].items():
        if not Path(filename).is_file() or sha(filename) != wanted:
            raise ValueError("G1 view source hash conflict")
    base_path = next((name for name in view["source_hashes"] if Path(name).suffix == ".json" and read(name).get("model_bindings")), None)
    # Recompute all signal-only rules from bound facts; a hand-edited overlay cannot register.
    from theme_structure_g1 import analyse
    theme_path = next((name for name in view["source_hashes"] if Path(name).suffix == ".json" and read(name).get("original_base_prediction_binding")), None)
    if not base_path or not theme_path:
        raise ValueError("bound original theme/BASE sources unavailable")
    recomputed = analyse(read(theme_path), read(base_path), read(p["theme_policy"]["path"]))
    for key in ("all_candidates", "original_model_top5", "G1_shadow_top5", "original_selectors", "G1_shadow_selectors", "full_theme_risk_table"):
        if recomputed[key] != view[key]:
            raise ValueError("G1 view differs from fixed signal-only recomputation")
    originals = {r["code"]: r for r in plan["candidates"]}
    if set(originals) != {r["code"] for r in view["all_candidates"]}:
        raise ValueError("G1 omits full original candidate group")
    for row in view["all_candidates"]:
        if any(originals[row["code"]].get(k) != row.get(k) for k in ("boards", "raw_score", "predicted_positive", "rank")):
            raise ValueError("G1 changed an original call/score/rank")
    from cycle_regime_diagnostics import validate_cycle
    validate_cycle(plan.get("cycle_context"), day, cutoff, registered["recorded_at"])
    path = Path(protocol_path).parent/"shadow/registrations.jsonl"
    chain = read_chain(path)
    if any(e["signal_date"] == day for e in chain):
        raise ValueError("G1 same signal already registered; original retained")
    if chain and day <= chain[-1]["signal_date"]:
        raise ValueError("G1 registrations must follow chronological signal order")
    if not chain and (not view["all_candidates"] or any(view[k]["selection_status"] != "KNOWN"
            or view[k]["non_one_word_rank_first"] is None for k in ("original_selectors", "G1_shadow_selectors"))):
        raise ValueError("first eligible G1 signal needs complete known nonword paired selectors")
    start = chain[0]["signal_date"] if chain else day
    index = sessions.index(day)-sessions.index(start)
    if not 0 <= index < 120:
        raise ValueError("G1 fixed 120-session window ended")
    event = {"event": "REGISTRATION", "recorded_at": at, "signal_date": day, "target_date": target,
        "window_start": start, "window_index": index, "knowledge_cutoff": cutoff,
        "registered_path": str(Path(registered_path).resolve()), "overlay_path": str(Path(overlay_path).resolve()),
        "original_registration_hash": registered["sha256"], "model_sha256": p["model_binding"]["model_sha256"],
        "immutable_sources": [binding(protocol_path), implementation, binding(calendar_path), binding(registered_path), binding(overlay_path)],
        "all_candidates": view["all_candidates"], "BASE": {"selectors": view["original_selectors"], "list": [r["code"] for r in view["original_model_top5"]]},
        "G1": {"selectors": view["G1_shadow_selectors"], "list": [r["code"] for r in view["G1_shadow_top5"]]}}
    return append(path, event)


def settle(protocol_path, production_ledger):
    import hypothesis_bound_daily_runner_v298 as production
    load_protocol(protocol_path); verify_implementation(protocol_path)
    root = Path(protocol_path).parent/"shadow"
    registrations = read_chain(root/"registrations.jsonl")
    old = read_chain(root/"settlements.jsonl")
    events = production.read_chain(production_ledger)
    latest = {e["result"]["registration_hash"]: e for e in events if stamp(e["result"]["recorded_at"]) <= stamp(now())}
    added = []
    for r in registrations:
        if stamp(r["target_date"]+"T15:00:00+08:00") > stamp(now()):
            continue
        e = latest.get(r["original_registration_hash"])
        if not e or any(s["registration_sha256"] == r["sha256"] and s["production_event_sha256"] == e["sha256"] for s in old):
            continue
        registered = load_registered(r["registered_path"])
        result = production.score(registered, read(e["source"]), now())
        rows = [v for v in result["rows"] if v["layer"] == "promotion"]
        saved = [v for v in e["result"]["rows"] if v["layer"] == "promotion"]
        if rows != saved or {v["key"] for v in rows} != {v["code"] for v in r["all_candidates"]}:
            raise ValueError("independent source-bound settlement differs from production/all candidates")
        event = {"event": "SETTLEMENT", "recorded_at": now(), "registration_sha256": r["sha256"],
            "signal_date": r["signal_date"], "target_date": r["target_date"], "production_event_sha256": e["sha256"],
            "immutable_sources": [binding(e["source"])], "outcome_hash": result["outcome_hash"],
            "truths": {v["key"]: v["truth"] if v["outcome_known"] else None for v in rows}}
        added.append(append(root/"settlements.jsonl", event))
    return {"status": "G1_SETTLEMENT_APPENDED" if added else "NO_NEW_MATURE_SOURCE", "new_settlements": len(added)}


def evaluate(days):
    """120 official grid slots including unknown/no-candidate/abstention days."""
    if len(days) != 120:
        raise ValueError("fixed review requires exactly 120 official grid slots")
    pairs, coverage, capture, positive = [], {"eligible": 120, "known": 0}, {"BASE": 0, "G1": 0, "denominator": 0}, {"BASE": 0, "G1": 0}
    status_counts = {}
    for i, day in enumerate(days):
        status = day.get("status", "NO_REGISTRATION")
        status_counts[status] = status_counts.get(status, 0)+1
        registration, truths = day.get("registration"), day.get("truths", {})
        if not registration:
            continue
        codes = {r["code"] for r in registration["all_candidates"]}
        if not codes:
            coverage["eligible"] -= 1
        selected = {a: registration[a]["selectors"]["non_one_word_rank_first"] for a in ("BASE", "G1")}
        candidate_by_code = {r["code"]: r for r in registration["all_candidates"]}
        for a in ("BASE", "G1"):
            positive[a] += registration[a]["selectors"]["non_one_word_positive_first"] is not None
        full_known = set(truths) == codes and all(type(truths[c]) is bool for c in codes)
        paired = full_known and bool(codes) and all(c is not None and type(truths.get(c)) is bool
            and type(candidate_by_code[c].get("predicted_positive")) is bool for c in selected.values())
        if paired:
            coverage["known"] += 1
            pairs.append((i, int(truths[selected["BASE"]]), int(truths[selected["G1"]])))
        if full_known:
            positives = {c for c in codes if truths[c]}
            capture["denominator"] += len(positives)
            for a in ("BASE", "G1"):
                capture[a] += len(positives & set(registration[a]["list"]))
    rate = lambda seq: sum(seq)/len(seq) if seq else None
    base = rate([v[1] for v in pairs]); g1 = rate([v[2] for v in pairs])
    delta = g1-base if base is not None else None
    segments = []
    for j in range(3):
        values = [v[2]-v[1] for v in pairs if j*40 <= v[0] < (j+1)*40]
        segments.append({"start_index": j*40, "paired_days": len(values), "delta": rate(values)})
    rng = np.random.default_rng(20261007)
    per_day = {i: (b, g) for i, b, g in pairs}
    samples = []
    # Circular moving blocks retain missing slots rather than packing known pairs.
    for _ in range(2000):
        draws = [(int(start)+j)%120 for start in rng.integers(0, 120, size=24) for j in range(5)]
        valid = [per_day[i][1]-per_day[i][0] for i in draws if i in per_day]
        if valid:
            samples.append(rate(valid))
    interval = np.quantile(samples, [.00625, .99375]).tolist() if len(samples) == 2000 else [None, None]
    effective = coverage["known"]/coverage["eligible"] if coverage["eligible"] else None
    checks = {"delta_ge_3pp": delta is not None and delta >= .03, "paired_ge_60": len(pairs) >= 60,
        "effective_coverage_ge_90pct": effective is not None and effective >= .9,
        "bootstrap_lower_positive": interval[0] is not None and interval[0] > 0,
        "all_three_segments_nonnegative": all(s["delta"] is not None and s["delta"] >= 0 for s in segments),
        "positive_first_coverage_not_down": positive["G1"] >= positive["BASE"],
        "observation_positive_capture_not_down": capture["denominator"] > 0 and capture["G1"] >= capture["BASE"]}
    return {"BASE_primary_rate": base, "G1_primary_rate": g1, "primary_delta": delta,
        "paired_days": len(pairs), "effective_coverage": effective, "coverage_counts": coverage,
        "bootstrap_interval": interval, "segments": segments, "positive_first_days": positive,
        "positive_first_coverage": {a: positive[a]/120 for a in ("BASE", "G1")},
        "observation_complete_positive_capture": {a: capture[a]/capture["denominator"] if capture["denominator"] else None for a in ("BASE", "G1")},
        "observation_positive_counts": capture, "day_status_counts": status_counts,
        "checks": checks, "passed": all(checks.values()), "paired_rows": pairs}


def review(protocol_path, calendar_path, output):
    from current_model_diagnostics import _calendar
    p = load_protocol(protocol_path); implementation = verify_implementation(protocol_path)
    root = Path(protocol_path).parent/"shadow"
    registrations, settlements = read_chain(root/"registrations.jsonl"), read_chain(root/"settlements.jsonl")
    sessions = _calendar(calendar_path)["sessions"]
    latest = {s["registration_sha256"]: s for s in settlements}
    for reg in registrations:
        original = load_registered(reg["registered_path"])
        overlay = read(reg["overlay_path"])
        if (reg["model_sha256"] != p["model_binding"]["model_sha256"] or reg["original_registration_hash"] != original["sha256"]
                or reg["all_candidates"] != overlay["all_candidates"]):
            raise ValueError("G1 review original model/candidate binding conflict")
        original_rows = {r["code"]: r for r in original["plan"]["candidates"]}
        for row in reg["all_candidates"]:
            if any(row.get(k) != original_rows[row["code"]].get(k) for k in ("boards", "rank", "raw_score", "predicted_positive")):
                raise ValueError("G1 review changed original scores/calls")
        for arm, selector, listing in (("BASE", "original_selectors", "original_model_top5"), ("G1", "G1_shadow_selectors", "G1_shadow_top5")):
            if reg[arm]["selectors"] != overlay[selector] or reg[arm]["list"] != [v["code"] for v in overlay[listing]]:
                raise ValueError("G1 review changed selection")
        if reg["sha256"] in latest:
            settlement = latest[reg["sha256"]]
            import hypothesis_bound_daily_runner_v298 as production
            source = settlement["immutable_sources"][0]["path"]
            checked = production.score(original, read(source), now())
            truths = {r["key"]: r["truth"] if r["outcome_known"] else None for r in checked["rows"] if r["layer"] == "promotion"}
            if truths != settlement["truths"] or checked["outcome_hash"] != settlement["outcome_hash"]:
                raise ValueError("G1 review independent source truth recomputation conflict")
    days, complete = [], False
    if registrations:
        start = sessions.index(registrations[0]["signal_date"])
        grid = sessions[start:start+120]
        complete = len(grid) == 120 and start+120 < len(sessions) and stamp(sessions[start+120]+"T15:00:00+08:00") <= stamp(now())
        by_day = {r["signal_date"]: r for r in registrations}
        for signal in grid:
            reg = by_day.get(signal); settlement = latest.get(reg["sha256"]) if reg else None
            state = "NO_REGISTRATION" if not reg else "NO_CANDIDATES" if not reg["all_candidates"] else "UNKNOWN_IDENTITY" if any(reg[a]["selectors"]["selection_status"] == "UNKNOWN_IDENTITY" for a in ("BASE", "G1")) else "NO_NONWORD_OR_ABSTENTION" if any(reg[a]["selectors"]["non_one_word_rank_first"] is None for a in ("BASE", "G1")) else "MISSING_LABEL" if not settlement or any(v is None for v in settlement["truths"].values()) else "PAIRED_CANDIDATE_DAY"
            if reg and state not in {"NO_CANDIDATES", "UNKNOWN_IDENTITY", "NO_NONWORD_OR_ABSTENTION"}:
                by_code = {r["code"]: r for r in reg["all_candidates"]}
                if any(type(by_code[reg[a]["selectors"]["non_one_word_rank_first"]].get("predicted_positive")) is not bool for a in ("BASE", "G1")):
                    state = "ABSTENTION"
            days.append({"signal_date": signal, "status": state, "registration": reg, "truths": settlement["truths"] if settlement else {}})
    metrics = evaluate(days) if complete else None
    result = {"status": "PASSED_G1_ELIGIBLE_FOR_NEW_PLANS" if metrics and metrics["passed"] else "FAILED_KEEP_BASE" if complete else "WAIT_FIXED_120_OFFICIAL_DAYS",
        "generated_at": now(), "protocol": binding(protocol_path), "implementation": implementation,
        "calendar": binding(calendar_path), "registered_days": len(registrations), "mature_settled_days": len(latest),
        "covered_official_window_days": len(days), "metrics": metrics, "days": days,
        "all_original_calls_scores_unchanged": True, "production_list_changed": False,
        "shadow_ledger_bindings": [binding(root/name) for name in ("registrations.jsonl", "settlements.jsonl") if (root/name).is_file()],
        "pending": "等待真实前向；现有官方日历不足120日时必须先另存官方扩展，禁止推算未来交易日" if not complete else None}
    write_new(Path(output)/"report.json", result)
    return result


def apply_selection(protocol_path, calendar_path, overlay_path, output):
    """Source-bound future list only; never rewrite a registered/old plan."""
    p = load_protocol(protocol_path)
    view = read(overlay_path)
    at = now()
    if (not view.get("prospective_eligible") or stamp(view["knowledge_cutoff"]) <= stamp(p["formed_at"])
            or stamp(at) >= stamp(view["target_date"]+"T00:00:00+08:00")
            or stamp(view["generated_at"]) > stamp(at) or view["model_binding"] != p["model_binding"]):
        raise ValueError("G1 selection applies only to eligible future new plans under the original frozen model")
    registry = Path(__file__).resolve().parent/"data/v271/expectation_registry"
    if any(read(f)["plan"]["signal_date"] == view["signal_date"] for f in registry.glob("*.json")):
        raise ValueError("already frozen signal cannot receive changed observation selection")
    rebuilt = review(protocol_path, calendar_path, Path(output)/"acceptance_rebuilt")
    if rebuilt["status"] != "PASSED_G1_ELIGIBLE_FOR_NEW_PLANS":
        raise ValueError("fixed G1 acceptance has not passed; original list remains")
    for filename, wanted in view["source_hashes"].items():
        if not Path(filename).is_file() or sha(filename) != wanted:
            raise ValueError("new observation source hash conflict")
    from theme_structure_g1 import analyse
    base_path = next((name for name in view["source_hashes"] if Path(name).suffix == ".json" and read(name).get("model_bindings")), None)
    theme_path = next((name for name in view["source_hashes"] if Path(name).suffix == ".json" and read(name).get("original_base_prediction_binding")), None)
    if not base_path or not theme_path:
        raise ValueError("new observation source scope unavailable")
    recomputed = analyse(read(theme_path), read(base_path), read(p["theme_policy"]["path"]))
    for key in ("all_candidates", "original_model_top5", "G1_shadow_top5", "original_selectors", "G1_shadow_selectors", "full_theme_risk_table"):
        if recomputed[key] != view[key]:
            raise ValueError("new observation list differs from fixed G1 recomputation")
    result = {"version": "g1-approved-new-observation-list-1", "signal_date": view["signal_date"],
        "target_date": view["target_date"], "formed_at": now(), "model_binding": p["model_binding"],
        "original_all_candidates": view["all_candidates"], "approved_observation_codes": [r["code"] for r in view["G1_shadow_top5"]],
        "selectors": view["G1_shadow_selectors"], "model_scores_calls_unchanged": True,
        "original_model_primary_preserved": view["original_selectors"]["all_rank_first"],
        "trade_primary_automatic_replacement": False, "overlay": binding(overlay_path),
        "protocol": binding(protocol_path), "acceptance": binding(Path(output)/"acceptance_rebuilt/report.json")}
    write_new(Path(output)/"selection.json", result)
    return {"status": "G1_ACCEPTED_NEW_LIST_FOR_AUTHOR_REVIEW", "selection": str(Path(output)/"selection.json")}


def validate_plan_selection(plan):
    b = plan.get("g1_observation_selection")
    if not b:
        return
    if not Path(b["path"]).is_file() or sha(b["path"]) != b["sha256"]:
        raise ValueError("approved observation selection hash conflict")
    s = read(b["path"])
    if (s["version"] != "g1-approved-new-observation-list-1" or s["signal_date"] != plan["signal_date"]
            or s["target_date"] != plan["target_date"] or stamp(s["formed_at"]) > stamp(plan["knowledge_cutoff"])):
        raise ValueError("approved observation selection scope/cutoff conflict")
    load_protocol(s["protocol"]["path"]); verify_implementation(s["protocol"]["path"])
    for key in ("overlay", "protocol", "acceptance"):
        artifact = s[key]
        if not Path(artifact["path"]).is_file() or sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("approved observation evidence changed")
    acceptance = read(s["acceptance"]["path"])
    if acceptance["status"] != "PASSED_G1_ELIGIBLE_FOR_NEW_PLANS" or not acceptance["metrics"]["passed"]:
        raise ValueError("G1 observation list lacks fixed acceptance")
    independent = evaluate(acceptance["days"])
    if not independent["passed"] or digest(independent) != digest(acceptance["metrics"]):
        raise ValueError("G1 accepted metrics differ from independent paired recomputation")
    for artifact in acceptance.get("shadow_ledger_bindings", []):
        if not Path(artifact["path"]).is_file() or sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("G1 accepted original settlement/registration changed")
    view = read(s["overlay"]["path"])
    if (s["approved_observation_codes"] != [r["code"] for r in view["G1_shadow_top5"]]
            or s["selectors"] != view["G1_shadow_selectors"] or s["model_binding"] != view["model_binding"]):
        raise ValueError("approved G1 list differs from source overlay")
    original = {r["code"]: r for r in s["original_all_candidates"]}
    if set(original) != {r["code"] for r in plan["candidates"]}:
        raise ValueError("approved list plan changed full candidate group")
    for row in plan["candidates"]:
        if any(row.get(k) != original[row["code"]].get(k) for k in ("boards", "raw_score", "predicted_positive", "rank")):
            raise ValueError("approved observation list changed numeric prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    overlay = sub.add_parser("overlay")
    overlay.add_argument("--theme-report", required=True); overlay.add_argument("--base-predictions", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--registered", required=True); reg.add_argument("--overlay", required=True); reg.add_argument("--calendar", required=True)
    settlement = sub.add_parser("settle"); settlement.add_argument("--production-ledger", required=True)
    rev = sub.add_parser("review"); rev.add_argument("--calendar", required=True)
    apply = sub.add_parser("apply"); apply.add_argument("--calendar", required=True); apply.add_argument("--overlay", required=True)
    for command in (overlay, reg, settlement, rev, apply):
        command.add_argument("--protocol", required=True)
    for command in (overlay, rev, apply):
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "overlay":
        result = overlay_run(args.theme_report, args.base_predictions, args.protocol, args.output)
    elif args.command == "register":
        result = register(args.protocol, args.calendar, args.registered, args.overlay)
    elif args.command == "settle":
        result = settle(args.protocol, args.production_ledger)
    elif args.command == "apply":
        result = apply_selection(args.protocol, args.calendar, args.overlay, args.output)
    else:
        result = review(args.protocol, args.calendar, args.output)
    print(json.dumps({"status": result.get("status", result.get("event")), "new_forward_samples": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
