"""Append-only close-scoring runner; never fetch future truth or rewrite plans.

Outcomes are separately archived in the inbox. A rerun is idempotent; a new
source appends a revision and explicitly identifies changed truth. Invalid
inputs are reported per forecast and never silently become failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from daily_research_contract_v292 import VERSION, score, stamp, statistics, validate_extension
from next_session_expectation_contract_v271 import digest, load_registered, register

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_chain(path):
    path = Path(path)
    if not path.exists():
        return []
    events, previous = [], None
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        saved = event.pop("sha256")
        if digest(event) != saved or event.get("previous_sha256") != previous or saved in seen:
            raise ValueError("settlement ledger hash chain mismatch")
        source = Path(event["source"])
        if not source.is_file() or sha(source) != event["source_sha256"]:
            raise ValueError("settlement source archive missing or changed")
        if digest(json.loads(source.read_text(encoding="utf-8"))) != event["result"]["outcome_hash"]:
            raise ValueError("settlement result does not bind its original source content")
        event["sha256"] = saved
        previous = saved
        seen.add(saved)
        events.append(event)
    return events


def append(ledger, result, source_path):
    if digest(json.loads(Path(source_path).read_text(encoding="utf-8"))) != result["outcome_hash"]:
        raise ValueError("scored outcomes differ from the archived input")
    events = read_chain(ledger)
    prior = [e for e in events if e["result"]["registration_hash"] == result["registration_hash"]]
    if any(e["result"]["outcome_hash"] == result["outcome_hash"] for e in prior):
        return "UNCHANGED"
    changes = []
    if prior:
        old = {(r["layer"], r["key"]): r for r in prior[-1]["result"]["rows"]}
        for row in result["rows"]:
            original = old.get((row["layer"], row["key"]))
            if original and original["outcome_known"] and original["truth"] != row["truth"]:
                changes.append({"layer": row["layer"], "key": row["key"], "from": original["truth"], "to": row["truth"]})
        if stamp(result["recorded_at"]) < stamp(prior[-1]["result"]["recorded_at"]):
            raise ValueError("settlement revisions cannot move backwards in time")
    event = {"version": VERSION, "previous_sha256": events[-1]["sha256"] if events else None,
        "revision": len(prior)+1, "source": str(Path(source_path).resolve()), "source_sha256": sha(source_path),
        "truth_changes": changes, "result": result}
    event["sha256"] = digest(event)
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    return "REVISION_APPENDED" if prior else "SETTLEMENT_APPENDED"


def current_results(events, recorded_at):
    latest = {}
    for e in events:
        result = e["result"]
        if stamp(result["recorded_at"]) <= stamp(recorded_at):
            latest[result["registration_hash"]] = result
    return list(latest.values())


def run(registry, inbox, ledger, calendar_path, output, recorded_at=None):
    recorded_at = recorded_at or datetime.now(timezone.utc).isoformat()
    cutoff = stamp(recorded_at)
    calendar = json.loads(Path(calendar_path).read_text(encoding="utf-8"))
    if calendar.get("status") != "verified" or not calendar.get("source"):
        raise ValueError("verified exchange calendar required")
    coverage = calendar.get("scope")
    if coverage and not coverage[0] <= cutoff.date().isoformat() <= coverage[1]:
        raise ValueError("calendar has expired or does not cover this run; refresh official holidays first")
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable run output already exists")
    events = read_chain(ledger)  # fail before writing anything on tampered history
    registrations, jobs, errors = [], [], []
    signal_dates = set()
    for path in sorted(Path(registry).glob("*.json")):
        try:
            registered = load_registered(path)
            validate_extension(registered["plan"], registered["recorded_at"])
            if stamp(registered["recorded_at"]) > cutoff:
                continue
            if registered["forecast_status"] == "FORWARD_REGISTERED":
                date = registered["plan"]["signal_date"]
                if date in signal_dates:
                    raise ValueError("duplicate forward primary for the same signal date")
                signal_dates.add(date)
            registrations.append(registered)
            target = registered["plan"]["target_date"]
            close = stamp(target+"T15:00:00+08:00")
            job = {"registration": str(path.resolve()), "registration_hash": registered["sha256"],
                "signal_date": registered["plan"]["signal_date"], "target_date": target,
                "forecast_status": registered["forecast_status"]}
            if cutoff < close:
                job["status"] = "WAIT_TARGET_CLOSE"
            else:
                sources = sorted((Path(inbox)/target).glob("*.json"))
                acceptable = []
                for source in sources:
                    try:
                        value = json.loads(source.read_text(encoding="utf-8"))
                        if value.get("registration_hash") != registered["sha256"]:
                            continue
                        available = stamp(value["assembled_at"])
                        if available > cutoff:
                            continue
                        acceptable.append((available, source, value))
                    except Exception as exc:
                        errors.append({"source": str(source), "error": str(exc)})
                if not acceptable:
                    job["status"] = "WAIT_VERIFIED_TARGET_OUTCOMES"
                else:
                    _, source, value = max(acceptable, key=lambda t: (t[0], str(t[1])))
                    result = score(registered, value, recorded_at)
                    job["status"] = append(ledger, result, source)
                    job["scoring_status"] = result["scoring_status"]
                    job["source"] = str(source.resolve())
            jobs.append(job)
        except Exception as exc:
            errors.append({"registration": str(path), "error": str(exc)})
    # Do not publish apparently complete statistics when an invalid registration
    # would disappear from the denominator. Retain errors, mark stats withheld.
    all_events = read_chain(ledger)
    summary = statistics(current_results(all_events, recorded_at), registrations, recorded_at, calendar) if not errors else {
        "status": "WITHHELD_INVALID_INPUT", "reason": "invalid entries must be repaired with preserved evidence before aggregate scoring"}
    report = {"version": VERSION, "recorded_at": recorded_at, "status": "ATTENTION_REQUIRED" if errors else "RUN_COMPLETE",
        "jobs": jobs, "errors": errors, "statistics": summary,
        "model_upgrade": "NO_AUTOMATIC_UPGRADE", "trade_execution": "DISABLED",
        "sources": {str(Path(calendar_path).resolve()): sha(calendar_path),
            **{str(Path(j["registration"]).resolve()): sha(j["registration"]) for j in jobs},
            **{str(ROOT/name): sha(ROOT/name) for name in ["daily_research_runner_v292.py", "daily_research_contract_v292.py",
                "daily_research_report_v292.py", "next_session_expectation_contract_v271.py"]}},
        "ledger_sha256": sha(ledger) if Path(ledger).exists() else None}
    write_new(output/"report.json", report)
    from daily_research_report_v292 import render
    results = {s["registration_hash"]: s for s in current_results(all_events, recorded_at)}
    for registered in registrations:
        name = registered["plan"]["signal_date"]+"_"+registered["sha256"][:16]+".md"
        (output/name).write_text(render(registered, report, results.get(registered["sha256"])), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--registry", default=str(ROOT/"data/v271/expectation_registry"))
    tick = commands.add_parser("tick")
    tick.add_argument("--registry", default=str(ROOT/"data/v271/expectation_registry"))
    tick.add_argument("--inbox", default=str(ROOT/"data/v292/outcome_inbox"))
    tick.add_argument("--ledger", default=str(ROOT/"data/v292/settlement_ledger.jsonl"))
    tick.add_argument("--calendar", default=str(ROOT/"data/v279/exchange_calendar_audit_20261002_v1/calendar.json"))
    tick.add_argument("--output", required=True)
    args = parser.parse_args()
    recorded = datetime.now(timezone.utc).isoformat()
    if args.command == "freeze":
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        validate_extension(plan, recorded)
        if "daily_research" not in plan:
            raise ValueError("new freeze entry requires all daily research fields; use original contract for legacy records")
        if plan["daily_research"].get("draft_status") != "AUTHOR_REVIEWED":
            raise ValueError("raw UNKNOWN template cannot be frozen without a completed author review")
        path, registered = register(plan, args.registry, recorded)
        print(json.dumps({"path": str(path), "sha256": registered["sha256"]}))
    else:
        report = run(args.registry, args.inbox, args.ledger, args.calendar, args.output, recorded)
        print(json.dumps({"status": report["status"], "jobs": report["jobs"], "errors": report["errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
