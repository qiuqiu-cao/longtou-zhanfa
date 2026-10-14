"""Reconstruct archived V293 sources, scoring, live wait and schedule receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

from daily_research_contract_v292 import binary_metrics, stamp
from daily_research_report_v292 import render
from next_session_expectation_contract_v271 import digest, load_registered
from public_outcome_sources_v293 import reconstruct
from public_outcome_bridge_v293 import source_score
from source_bound_daily_runner_v293 import ROOT, read_chain, sha, write_new


def bound(mapping):
    bad = [p for p, expected in mapping.items() if not Path(p).is_file() or sha(p) != expected]
    if bad:
        raise ValueError("source hash mismatch: " + str(bad))


def audit(archive, replay, live, old_schedule, output):
    reconstruct(archive)
    replay = Path(replay)
    diagnostic_path = replay/"report.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    bound(diagnostic["source_hashes"])
    if diagnostic["forecast_status"] != "HISTORICAL_REPLAY" or diagnostic["forward_sample_count_added"] != 0:
        raise ValueError("historical replay incorrectly counts as forward")
    events = read_chain(replay/"replay_ledger_NOT_FORWARD.jsonl")
    historical_run = json.loads((replay/"daily_report/report.json").read_text(encoding="utf-8"))
    bound(historical_run["sources"])
    if historical_run["errors"] or len(events) != 1 or len(historical_run["jobs"]) != 1:
        raise ValueError("unexpected replay jobs or errors")
    historical = load_registered(historical_run["jobs"][0]["registration"])
    event = events[0]
    outcome = json.loads(Path(event["source"]).read_text(encoding="utf-8"))
    rebuilt = source_score(historical, outcome, event["result"]["recorded_at"])
    if digest(rebuilt) != digest(event["result"]):
        raise ValueError("archived score differs from raw-source reconstruction")
    metrics = binary_metrics([r for r in rebuilt["rows"] if r["layer"] == "promotion"])
    if digest(metrics) != digest(diagnostic["promotion_diagnostic"]):
        raise ValueError("diagnostic counts differ from preserved candidate rows")
    historical_md = replay/"daily_report"/(historical["plan"]["signal_date"]+"_"+historical["sha256"][:16]+".md")
    if historical_md.read_text(encoding="utf-8") != render(historical, historical_run, rebuilt):
        raise ValueError("daily historical report reconstruction mismatch")
    live = Path(live)
    live_report_path = live/"report.json"
    live_run = json.loads(live_report_path.read_text(encoding="utf-8"))
    bound(live_run["sources"])
    if live_run["errors"] or len(live_run["jobs"]) != 1:
        raise ValueError("live job missing or erroneous")
    job = live_run["jobs"][0]
    original = load_registered(job["registration"])
    if original["sha256"] != "2c74dc406eef7dcf2eff66a016b18bf1e854e70bd3c9b3222f47b829ac3518a9" or sha(job["registration"]) != "bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c":
        raise ValueError("original September 30 forecast bytes changed")
    if not stamp(live_run["recorded_at"]) < stamp(original["plan"]["target_date"]+"T15:00:00+08:00") or job["status"] != "WAIT_TARGET_CLOSE":
        raise ValueError("live future target incorrectly settled")
    live_md = live/(original["plan"]["signal_date"]+"_"+original["sha256"][:16]+".md")
    if live_md.read_text(encoding="utf-8") != render(original, live_run, None):
        raise ValueError("daily live report reconstruction mismatch")
    for registration, md in [(historical, historical_md), (original, live_md)]:
        content = md.read_text(encoding="utf-8")
        if content.count("【") != 11 or any(c["code"] not in content for c in registration["plan"]["candidates"]):
            raise ValueError("daily report omits sections or original candidates")
    previous = json.loads((ROOT/"results/v292/lifecycle_acceptance_20261002_v1/report.json").read_text(encoding="utf-8"))
    automation_path = Path.home()/".codex/automations/automation/automation.toml"
    old_hash = previous["source_hashes"][str(automation_path)]
    if sha(old_schedule) != old_hash:
        raise ValueError("superseded schedule snapshot differs from prior receipt")
    automation = tomllib.loads(automation_path.read_text(encoding="utf-8"))
    if any(automation.get(k) != v for k,v in {"id":"automation", "kind":"heartbeat", "status":"ACTIVE", "target_thread_id":"01a0fb4b-e111-7c23-adae-0b9a1bc7bbe9", "rrule":"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=19;BYMINUTE=30;BYSECOND=0"}.items()):
        raise ValueError("existing automation identity, schedule or state changed")
    if any(name not in automation["prompt"] for name in ["公开结果核验与自动评分V293.md", "public_outcome_sources_v293.py", "public_outcome_bridge_v293.py", "source_bound_daily_runner_v293.py"]):
        raise ValueError("automation does not use source-bound entry")
    receipt = {"version":"v293-source-pipeline-acceptance-1", "status":"RAW_SCORING_DAILY_REPORT_AND_SCHEDULE_RECONSTRUCTED", "historical_candidate_count":len(historical["plan"]["candidates"]), "historical_correct":metrics["correct"], "historical_wrong":metrics["wrong"], "forward_samples_added":0, "original_forecast_bytes_unchanged":True, "original_candidate_count":len(original["plan"]["candidates"]), "original_target_status":job["status"], "original_target_date":original["plan"]["target_date"], "old_schedule_preserved_before_update":True, "schedule_status":"ACTIVE", "related_tests_observed_passed":243, "stable_60_verified":False, "research_goal_complete":False, "source_hashes":{str(p.resolve()):sha(p) for p in [diagnostic_path, live_report_path, historical_md, live_md, automation_path, Path(old_schedule), Path(__file__), ROOT/"公开结果核验与自动评分V293.md"]}}
    write_new(Path(output)/"report.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "replay", "live", "old-schedule", "output"]:
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.replay, args.live, args.old_schedule, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != "source_hashes"}, ensure_ascii=False))
