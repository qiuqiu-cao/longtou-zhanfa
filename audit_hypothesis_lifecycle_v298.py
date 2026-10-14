"""Audit real original preservation, legacy raw scores and the fifteen-item bank."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import tomllib

from daily_research_report_v292 import render as old_render
from daily_research_report_v298 import render, hypothesis_table
from daily_research_template_v298 import draft_extension
from hypothesis_review_v298 import attach, check_hashes, load_review, validate_hypotheses
from hypothesis_bound_daily_runner_v298 import read_chain, score, sha, write_new
from next_session_expectation_contract_v271 import digest, load_registered

ROOT = Path(__file__).resolve().parent


def load_run(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    check_hashes(report["sources"])
    if report["errors"] or len(report["jobs"]) != 1:
        raise ValueError("unexpected run errors or denominator")
    registered = load_registered(report["jobs"][0]["registration"])
    validate_hypotheses(registered["plan"])
    return report, registered


def audit(live, historical, ledger, receipt, tests, old_schedule, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("keep existing acceptance immutable")
    reviewed = load_review(receipt)
    live_report, original = load_run(live)
    job = live_report["jobs"][0]
    if job["status"] != "WAIT_TARGET_CLOSE" or original["plan"]["target_date"] != "2026-10-08" or sha(job["registration"]) != "bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c":
        raise ValueError("real original altered or prematurely settled")
    # The actual bank and actual old cutoff are used; only the absent template
    # is built in memory. It is never registered or counted as a new forecast.
    probe = deepcopy(original["plan"])
    probe["daily_research"] = draft_extension(probe)
    validate_hypotheses(probe, require_complete=True)
    try:
        attach(probe, receipt)
    except ValueError as exc:
        rejection = str(exc)
        if "arrived after knowledge cutoff" not in rejection:
            raise
    else:
        raise ValueError("real later bank accepted at original cutoff")
    historical_report, replay = load_run(historical)
    events = read_chain(ledger)
    if len(events) != 1 or historical_report["ledger_sha256"] != sha(ledger) or replay["forecast_status"] != "HISTORICAL_REPLAY":
        raise ValueError("historical replay or ledger scope changed")
    event = events[0]
    rebuilt = score(replay, json.loads(Path(event["source"]).read_text(encoding="utf-8")), event["result"]["recorded_at"])
    if digest(rebuilt) != digest(event["result"]):
        raise ValueError("historical raw truth does not reproduce")
    rows = [r for r in rebuilt["rows"] if r["layer"] == "promotion"]
    previous = json.loads((ROOT / "results/v296/replay_ledger_20261002_v1.jsonl").read_text(encoding="utf-8").splitlines()[0])["result"]
    if len(rows) != 9 or digest(rows) != digest([r for r in previous["rows"] if r["layer"] == "promotion"]):
        raise ValueError("historical candidates or original results changed")
    docs = []
    for directory, registered, report, result in [(Path(live), original, live_report, None), (Path(historical), replay, historical_report, rebuilt)]:
        path = directory / (registered["plan"]["signal_date"] + "_" + registered["sha256"][:16] + ".md")
        content = path.read_text(encoding="utf-8")
        if content != render(registered, report, result) or content != old_render(registered, report, result) or content.count("【") != 11:
            raise ValueError("legacy report changed or eleven-section coverage failed")
        if any(c["code"] not in content for c in registered["plan"]["candidates"]):
            raise ValueError("daily report omitted a candidate")
        docs.append(path)
    test_record = json.loads(Path(tests).read_text(encoding="utf-8"))
    check_hashes(test_record["source_hashes"])
    if test_record.get("passed") is not True or test_record["errors"] or test_record["failures"] or "test_hypothesis_review_v298" not in test_record["modules"]:
        raise ValueError("successful actual related tests required")
    app_path = Path.home() / ".codex/automations/automation/automation.toml"
    old = json.loads((ROOT / "results/v296/context_freeze_acceptance_20261002_v1/report.json").read_text(encoding="utf-8"))
    for name, wanted in old["source_hashes"].items():
        actual_path = Path(old_schedule) if Path(name) == app_path else Path(name)
        if sha(actual_path) != wanted:
            raise ValueError("prior acceptance or exact schedule snapshot changed: " + name)
    app = tomllib.loads(app_path.read_text(encoding="utf-8"))
    before = tomllib.loads(Path(old_schedule).read_text(encoding="utf-8"))
    for field in ["id", "kind", "name", "status", "rrule", "target_thread_id", "created_at"]:
        if app[field] != before[field]:
            raise ValueError("existing schedule preferences changed: " + field)
    for needed in ["hypothesis_bound_daily_runner_v298.py freeze", "hypothesis_bound_daily_runner_v298.py tick", "daily_research_template_v298.py", "hypothesis_review_v298.py", "经验假设逐项留档与冻结核验V298.md"]:
        if needed not in app["prompt"]:
            raise ValueError("new daily entry missing from active automation: " + needed)
    recorded = datetime.now(timezone.utc).isoformat()
    appendix = "\n".join(["# 当前经验研究附页（不属于原预测）", "",
        "生成时间：" + recorded + "；研究验收可得时间：" + reviewed["available_at"] + "。",
        "来源验收：" + reviewed["receipt"] + "；SHA：" + reviewed["receipt_sha256"] + "。",
        "这是取得研究资料后的当前视图，晚于原知识截止。原11只预测没有本表，不能将它并入原预测或提高其概率。",
        "没有新登记，没有新增真实前向结果。", hypothesis_table(reviewed["hypotheses"])]) + "\n"
    if sum(line.startswith("| ") for line in appendix.splitlines()) != 16:
        raise ValueError("current appendix lost a named hypothesis")
    output.mkdir(parents=True, exist_ok=False)
    annex_path = output / "current_research_appendix_NOT_FORECAST.md"
    annex_path.write_text(appendix, encoding="utf-8")
    write_new(output / "real_original_cutoff_rejection.json", {"recorded_at": recorded,
        "original_source": job["registration"], "original_bytes_sha256": sha(job["registration"]),
        "original_knowledge_cutoff": original["plan"].get("knowledge_cutoff", original["plan"]["evidence_cutoff"]),
        "research_receipt_available_at": reviewed["available_at"], "rejection": rejection,
        "new_registration_created": False, "original_modified": False})
    sources = [Path(__file__), ROOT / "hypothesis_review_v298.py", ROOT / "daily_research_template_v298.py", ROOT / "daily_research_report_v298.py",
        ROOT / "hypothesis_bound_daily_runner_v298.py", ROOT / "test_hypothesis_review_v298.py", ROOT / "经验假设逐项留档与冻结核验V298.md",
        Path(live) / "report.json", Path(historical) / "report.json", Path(ledger), Path(receipt), Path(tests), Path(old_schedule), app_path,
        annex_path, output / "real_original_cutoff_rejection.json", *docs]
    report = {"version": "v298-hypothesis-daily-lifecycle-acceptance-1", "recorded_at": recorded,
        "status": "FIFTEEN_HYPOTHESIS_SOURCE_GATES_ORIGINAL_REPORTS_AND_RAW_SCORES_VERIFIED",
        "hypotheses": 15, "tested_proxies": 7, "unavailable_hypotheses": 8, "core_rules": 0,
        "real_original_cutoff_rejection": rejection, "original_bytes_unchanged": True,
        "original_candidates": 11, "original_target": "2026-10-08", "original_status": job["status"],
        "historical_candidates": len(rows), "historical_correct": sum(r["correct"] is True for r in rows),
        "historical_wrong": sum(r["correct"] is False for r in rows), "historical_rows_unchanged": True,
        "related_tests_observed_passed": test_record["tests_run"], "legacy_eleven_section_views_unchanged": True,
        "current_appendix_is_a_frozen_forecast": False, "schedule_active": app["status"] == "ACTIVE", "future_schedule_completed": False,
        "forward_samples_added": 0, "core_model_upgrade": False, "stable_60_verified": False, "goal_complete": False,
        "source_hashes": {str(p.resolve()): sha(p) for p in sources}}
    write_new(output / "report.json", report)
    return {k: report[k] for k in ["status", "hypotheses", "real_original_cutoff_rejection", "original_candidates", "historical_correct", "historical_wrong", "related_tests_observed_passed", "core_rules", "stable_60_verified"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["live", "historical", "ledger", "receipt", "tests", "old-schedule", "output"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.live, args.historical, args.ledger, args.receipt, args.tests, args.old_schedule, args.output), ensure_ascii=True))
