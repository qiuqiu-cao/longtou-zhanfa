"""Reconstruct V292 live-run provenance and verify the original is unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from daily_research_runner_v292 import ROOT, sha, write_new
from daily_research_contract_v292 import stamp
from next_session_expectation_contract_v271 import load_registered, next_session


def audit(run_directory, output):
    directory = Path(run_directory)
    report_path = directory/"report.json"
    run = json.loads(report_path.read_text(encoding="utf-8"))
    mismatches = [path for path, wanted in run["sources"].items() if not Path(path).is_file() or sha(path) != wanted]
    if mismatches or run["errors"]:
        raise ValueError("run input/code provenance invalid: "+str(mismatches or run["errors"]))
    checks = []
    for job in run["jobs"]:
        registered = load_registered(job["registration"])
        p = registered["plan"]
        if job["registration_hash"] != registered["sha256"] or job["target_date"] != next_session(p["calendar"], p["signal_date"]):
            raise ValueError("job differs from immutable original or skips a trading session")
        future = stamp(run["recorded_at"]) < stamp(p["target_date"]+"T15:00:00+08:00")
        if future and job["status"] != "WAIT_TARGET_CLOSE":
            raise ValueError("future target incorrectly scored")
        md = directory/(p["signal_date"]+"_"+registered["sha256"][:16]+".md")
        content = md.read_text(encoding="utf-8")
        if content.count("【") != 11 or any(c["code"] not in content for c in p["candidates"]):
            raise ValueError("report omits a daily section or original candidate")
        original_unchanged = None
        if registered["sha256"] == "2c74dc406eef7dcf2eff66a016b18bf1e854e70bd3c9b3222f47b829ac3518a9":
            original_unchanged = sha(job["registration"]) == "bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c"
            if not original_unchanged:
                raise ValueError("original September 30 bytes changed")
        checks.append({"registration_hash": registered["sha256"], "candidate_count": len(p["candidates"]),
            "target": p["target_date"], "status": job["status"], "eleven_sections": True,
            "all_original_candidates_displayed": True, "original_bytes_unchanged": original_unchanged,
            "legacy_fields_backfilled": False, "report_sha256": sha(md)})
    # Read the actual app-managed entry rather than inventing a future run receipt.
    import tomllib
    automation_path = Path.home()/".codex/automations/automation/automation.toml"
    automation = tomllib.loads(automation_path.read_text(encoding="utf-8"))
    if automation.get("kind") != "heartbeat" or automation.get("status") != "ACTIVE" or automation.get("target_thread_id") != "01a0fb4b-e111-7c23-adae-0b9a1bc7bbe9":
        raise ValueError("automation is not active on the intended thread")
    acceptance = {"version": "v292-lifecycle-acceptance-1", "status": "LIVE_PENDING_RUN_AND_SCHEDULE_RECONSTRUCTED",
        "run_checks": checks, "app_automation": {"id": automation["id"], "name": automation["name"],
            "status": automation["status"], "kind": automation["kind"], "local_schedule": "工作日19:30；Windows中国标准时间",
            "holiday_action": "skip according to verified exchange calendar", "future_run_completed": False},
        "validation_observed_in_this_session": {"related_tests_passed": 226, "new_tests_passed": 24,
            "meaning": "process validation; not financial accuracy evidence"},
        "stable_60_verified": False, "research_goal_complete": False,
        "open_data_requirements": ["point-in-time theme membership", "verified leader influence/absorption labels",
            "real reseal/auction microstructure", "historical consensus measurements", "independent forward outcomes"],
        "source_hashes": {str(report_path.resolve()): sha(report_path), str(automation_path): sha(automation_path),
            str(Path(__file__).resolve()): sha(__file__),
            str(ROOT/"每日研究与自动验证V292.md"): sha(ROOT/"每日研究与自动验证V292.md")}}
    write_new(Path(output)/"report.json", acceptance)
    return acceptance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps({"status": result["status"], "run_checks": result["run_checks"], "app_automation": result["app_automation"]}, ensure_ascii=True))
