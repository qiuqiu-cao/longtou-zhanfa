"""Verify context freeze rejection, live wait, raw historical score and schedule."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import tomllib

from context_freeze_contract_v296 import validate_context_binding
from context_bound_daily_runner_v296 import score, read_chain, sha, write_new
from daily_research_template_v292 import draft_extension
from daily_research_report_v292 import render
from next_session_expectation_contract_v271 import load_registered,digest
from point_context_v295 import now

ROOT=Path(__file__).resolve().parent


def bound(mapping,substitutions=None):
    substitutions=substitutions or {}
    for path,wanted in mapping.items():
        if sha(substitutions.get(path,path))!=wanted:raise ValueError("source changed: "+path)


def run_report(directory):
    directory=Path(directory)
    report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
    bound(report["sources"])
    if report["errors"] or len(report["jobs"])!=1:raise ValueError("unexpected run error or denominator")
    job=report["jobs"][0];registered=load_registered(job["registration"])
    validate_context_binding(registered["plan"])
    return report,registered


def audit(live,historical,ledger,old_schedule,output):
    live_run,original=run_report(live)
    if sha(live_run["jobs"][0]["registration"])!="bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c" or live_run["jobs"][0]["status"]!="WAIT_TARGET_CLOSE" or original["plan"]["target_date"]!="2026-10-08":raise ValueError("live original changed or settled early")
    contextual_source=ROOT/"data/v295/provider_context_reconciled_20261002_v1/context.json"
    probe=deepcopy(original["plan"]);probe["daily_research"]=draft_extension(probe)
    probe["context_binding"]={"source":str(contextual_source),"source_sha256":sha(contextual_source),"category_codes_filled":[],"attention_codes_filled":[],"history_used_in_live_features":False,"author_review_still_required":True}
    try:validate_context_binding(probe)
    except ValueError as exc:
        rejection=str(exc)
        if "source snapshot arrived after knowledge cutoff" not in rejection:raise
    else:raise ValueError("later real source snapshot accepted for original cutoff")
    historical_run,registration=run_report(historical)
    events=read_chain(ledger)
    if len(events)!=1 or historical_run["ledger_sha256"]!=sha(ledger) or registration["forecast_status"]!="HISTORICAL_REPLAY":raise ValueError("historical ledger scope mismatch")
    event=events[0];value=json.loads(Path(event["source"]).read_text(encoding="utf-8"))
    rebuilt=score(registration,value,event["result"]["recorded_at"])
    if digest(rebuilt)!=digest(event["result"]):raise ValueError("historical score does not reproduce raw truth")
    rows=[r for r in rebuilt["rows"] if r["layer"]=="promotion"]
    if len(rows)!=len(registration["plan"]["candidates"]):raise ValueError("historical candidate lost")
    old=json.loads((ROOT/"results/v293/source_pipeline_replay_20261002_v1/replay_ledger_NOT_FORWARD.jsonl").read_text(encoding="utf-8").splitlines()[0])["result"]
    if digest(rows)!=digest([r for r in old["rows"] if r["layer"]=="promotion"]):raise ValueError("new guard changed original historical scoring rows")
    docs=[]
    for directory,reg,report,result in [(Path(live),original,live_run,None),(Path(historical),registration,historical_run,rebuilt)]:
        md=directory/(reg["plan"]["signal_date"]+"_"+reg["sha256"][:16]+".md")
        content=md.read_text(encoding="utf-8")
        if content!=render(reg,report,result) or content.count("【")!=11 or any(c["code"] not in content for c in reg["plan"]["candidates"]):raise ValueError("daily report changed or omitted original candidates")
        docs.append(md)
    # A preliminary development run is retained with its old guard's exact bytes.
    preliminary=json.loads((ROOT/"results/v296/daily_lifecycle_20261002_v1/report.json").read_text(encoding="utf-8"))
    bound(preliminary["sources"],{str(ROOT/"context_freeze_contract_v296.py"):ROOT/"data/v296/development_sources/context_freeze_contract_v296_before_clock_guard.py"})
    app_path=Path.home()/".codex/automations/automation/automation.toml"
    previous=json.loads((ROOT/"results/v295/context_acceptance_20261002_v1/report.json").read_text(encoding="utf-8"))
    bound(previous["source_hashes"],{str(app_path):Path(old_schedule)})
    app=tomllib.loads(app_path.read_text(encoding="utf-8"))
    if any(app.get(k)!=v for k,v in {"id":"automation","kind":"heartbeat","status":"ACTIVE","target_thread_id":"01a0fb4b-e111-7c23-adae-0b9a1bc7bbe9","rrule":"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=19;BYMINUTE=30;BYSECOND=0"}.items()):raise ValueError("existing schedule changed")
    if "context_bound_daily_runner_v296.py freeze" not in app["prompt"] or "context_bound_daily_runner_v296.py tick" not in app["prompt"] or "背景字段冻结前重建核验V296.md" not in app["prompt"]:raise ValueError("freeze/read gate missing from actual automation")
    output=Path(output)
    report={"version":"v296-freeze-context-lifecycle-acceptance-1","recorded_at":now(),"status":"CONTEXT_FREEZE_AND_READ_GUARDS_LIVE_WAIT_AND_HISTORICAL_SCORES_VERIFIED",
        "original_bytes_unchanged":True,"original_candidates":len(original["plan"]["candidates"]),"original_status":"WAIT_TARGET_CLOSE","original_target":"2026-10-08",
        "real_later_context_rejection":rejection,"late_context_published_as_forecast":False,"historical_candidates":len(rows),
        "historical_correct":sum(r["correct"] is True for r in rows),"historical_wrong":sum(r["correct"] is False for r in rows),
        "historical_rows_unchanged":True,"source_truth_and_daily_report_reproduced":True,"related_tests_observed_passed":285,
        "first_development_source_retained":True,"prior_schedule_retained":True,"schedule_active":True,"future_schedule_completed":False,
        "forward_samples_added":0,"core_model_upgrade":False,"stable_60_verified":False,"goal_complete":False,
        "source_hashes":{str(p.resolve()):sha(p) for p in [Path(live)/"report.json",Path(historical)/"report.json",Path(ledger),Path(old_schedule),app_path,contextual_source,Path(__file__),ROOT/"test_context_freeze_v296.py",ROOT/"背景字段冻结前重建核验V296.md",*docs]}}
    write_new(output/"report.json",report)
    return {k:report[k] for k in ["status","original_bytes_unchanged","original_candidates","original_status","real_later_context_rejection","historical_candidates","historical_correct","historical_wrong","historical_rows_unchanged","related_tests_observed_passed","schedule_active","stable_60_verified"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["live","historical","ledger","old-schedule","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.live,args.historical,args.ledger,args.old_schedule,args.output),ensure_ascii=True))
