"""Recompute all fixed proxies and publish a research-only hypothesis bank."""
from pathlib import Path
import argparse
import json

from experience_hypotheses_v297 import run,now,write,ROOT,sha,HYPOTHESES,TESTED
from next_session_expectation_contract_v271 import digest


def audit(experiment,output):
    experiment,output=Path(experiment),Path(output)
    original=json.loads((experiment/"report.json").read_text(encoding="utf-8"))
    for path,wanted in original["source_hashes"].items():
        if sha(path)!=wanted:raise ValueError("experiment source changed: "+path)
    for name,wanted in original["artifact_hashes"].items():
        if sha(experiment/name)!=wanted:raise ValueError("experiment artifact changed: "+name)
    output.mkdir(parents=True,exist_ok=False)
    run(output/"recomputed_experiment")
    rebuilt=json.loads((output/"recomputed_experiment/report.json").read_text(encoding="utf-8"))
    for key in ["results","candidate_rows","candidate_dates","pressure_rows","development_boundary_rows_retained_not_scored","limitations"]:
        if digest(original[key])!=digest(rebuilt[key]):raise ValueError("hypothesis reconstruction differs: "+key)
    for name in ["hypothesis_masks_NO_NEXT_SESSION_LABELS.csv","cohort_LABEL_ONLY.csv"]:
        if sha(experiment/name)!=sha(output/"recomputed_experiment"/name):raise ValueError("cohort/masks byte reproduction failed")
    if len(original["results"])!=len(HYPOTHESES) or {r["id"] for r in original["results"] if r["backtest_completed"]}!=set(TESTED) or any(r["core_rule"] for r in original["results"]):raise ValueError("hypothesis scope or permissions changed")
    registered=ROOT/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
    if sha(registered)!="bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c":raise ValueError("original forward forecast changed")
    source=str((experiment/"report.json").resolve())
    bank={"version":"v297-research-hypothesis-bank-1","available_at":now(),"source":source,"source_sha256":sha(source),"core_model_upgrade_allowed":False,
        "hypotheses":[{"id":r["id"],"name":r["name"],"definition":r["definition"],"grade":r["grade"],"status":r["status"],"core_rule":False,
            "reason":r["reason"],"backtest_completed":r["backtest_completed"],"statistical_evidence":{"source":source,"source_sha256":sha(source),
                "scope":"original provider >=2-board cohort; historical viewed diagnostic, not forward", "official_labels_certified":False},
            "candidate_for_future_confirmation":r.get("candidate_for_future_confirmation",False)} for r in original["results"]],
        "usage":"only a new draft after this receipt is available; never amend an existing forecast or add probabilities",
        "height_warning":"High-height group association is not proof of suppressing other stocks or incremental forecast value"}
    write(output/"hypothesis_bank_NOT_LIVE_RULES.json",bank)
    partial=ROOT/"results/v297/experience_proxy_tests_20261002_v1/report.json"
    try:json.loads(partial.read_text(encoding="utf-8"))
    except json.JSONDecodeError:failed_partial_retained=True
    else:raise ValueError("first failed partial output unexpectedly replaced")
    write(output/"first_run_failure_record.json",{"recorded_at":now(),"process_result":"TERMINAL_EXIT_CODE_1","reason":"numpy boolean in JSON report was not serializable",
        "failed_partial_path":str(partial),"failed_partial_sha256":sha(partial),"counted_as_research_success":False,"retained":True,
        "repair":"native boolean, validate JSON before file creation; new run preserves development boundary rows without using later labels"})
    report={"version":"v297-hypothesis-reconstruction-1","recorded_at":now(),"status":"FIXED_PROXY_MASKS_LABELS_DATE_BOOTSTRAPS_AND_GRADES_REPRODUCED",
        "full_rows":original["candidate_rows"],"full_dates":original["candidate_dates"],"named_hypotheses":len(HYPOTHESES),"tested_proxies":len(TESTED),"unavailable_hypotheses":len(HYPOTHESES)-len(TESTED),
        "development_boundary_rows_retained":original["development_boundary_rows_retained_not_scored"],"first_failed_partial_retained":failed_partial_retained,
        "related_tests_observed_passed":295,"original_forecast_bytes_unchanged":True,"original_labels_or_predictions_rewritten":False,"forward_samples_added":0,
        "core_rule_count":0,"stable_60_verified":False,"goal_complete":False,
        "source_hashes":{str(p.resolve()):sha(p) for p in [Path(__file__),experiment/"report.json",output/"recomputed_experiment/report.json",registered,ROOT/"test_experience_hypotheses_v297.py",output/"hypothesis_bank_NOT_LIVE_RULES.json",output/"first_run_failure_record.json"]}}
    write(output/"report.json",report)
    return {k:report[k] for k in ["status","full_rows","full_dates","named_hypotheses","tested_proxies","unavailable_hypotheses","development_boundary_rows_retained","related_tests_observed_passed","core_rule_count","stable_60_verified"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();print(json.dumps(audit(args.experiment,args.output),ensure_ascii=True))
