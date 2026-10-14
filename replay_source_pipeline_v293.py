"""Replay already published September calls through source-bound verification.

Registration uses the ACTUAL current clock and is a historical replay. This
cannot manufacture a September forward forecast or certify market/leader truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd

from daily_research_runner_v292 import ROOT, sha, write_new
from next_session_expectation_contract_v271 import digest, next_session, register
from public_outcome_bridge_v293 import build, legacy_signal
from source_bound_daily_runner_v293 import run


def replay(archive, prior_snapshot, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable source replay already exists")
    source_dir = ROOT/"results/v278/september_falsification_20261002_v6"
    report = json.loads((source_dir/"report.json").read_text(encoding="utf-8"))
    if any(sha(ROOT/path) != value for path, value in report["source_hashes"].items()):
        raise ValueError("original model/source evidence changed")
    manifest = json.loads((Path(archive)/"manifest.json").read_text(encoding="utf-8"))
    signal, target = manifest["signal_date"], manifest["target_date"]
    original, files = legacy_signal(prior_snapshot, signal)
    # Explicit prediction-column allowlist: outcomes/next-day prices are not read.
    predictions = pd.read_csv(source_dir/"relay_predictions_LABEL_ONLY.csv", dtype={"code": str},
        usecols=["date","code","name","boards","predicted","actual_one_word_daily"])
    predictions = predictions[predictions.date.eq(signal)]
    market = pd.read_csv(source_dir/"market_predictions_LABEL_ONLY.csv",
        usecols=["date","current_class","predicted","emotion_quality_score"])
    market = market[market.date.eq(signal)]
    expected = {r["code"]:r["boards"] for r in original["rows"] if r["boards"]>=2}
    if len(market)!=1 or predictions.code.duplicated().any() or dict(zip(predictions.code,predictions.boards.astype(int))) != expected:
        raise ValueError("published calls disagree with full original candidate denominator")
    output.mkdir(parents=True)
    knowledge = datetime.now(timezone.utc).isoformat()
    candidates=[]
    for r in predictions.sort_values("code").itertuples():
        candidates.append({"code":r.code,"name":r.name,"boards":int(r.boards),
            "identity":"SPACE_ANCHOR" if r.boards==max(expected.values()) else "CHALLENGER",
            "actual_one_word_daily":bool(r.actual_one_word_daily),"promotion_call":"PROMOTE" if r.predicted==1 else "FAIL",
            "identity_evidence":["Published fixed-model call copied; original provider board count; identity is unconfirmed"],
            "identity_invalidation":["Promotion cannot substitute for leader influence, absorption or theme identity"],"probability":None})
    baseline=float(market.iloc[0].emotion_quality_score)
    movement=str(market.iloc[0].predicted)
    if movement=="WORSEN":lo,hi=-baseline,math.nextafter(-.05,-math.inf)
    elif movement=="IMPROVE":lo,hi=math.nextafter(.05,math.inf),1-baseline
    else:lo,hi=-.05,.05
    source_files=files+[source_dir/"relay_predictions_LABEL_ONLY.csv",source_dir/"market_predictions_LABEL_ONLY.csv",source_dir/"report.json"]
    sources={str(path.resolve()):sha(path) for path in source_files}
    plan={"signal_date":signal,"target_date":target,"calendar":manifest["calendar"],
        "evidence_cutoff":original["available_at"],"knowledge_cutoff":knowledge,
        "model_version":"V278_PUBLISHED_CALL_REPLAY_WITH_UNSCORED_CURRENT_CLASS_PERSISTENCE_BASELINE",
        "input_hash":digest(sources),"denominator_status":"verified_complete",
        "denominator_scope":"PROVIDER_REPORTED_MAINBOARD_NAME_NON_ST_CLOSING_RELAY_POOL_NOT_EXCHANGE_CERTIFIED",
        "primary_candidate":None,"candidates":candidates,
        "market":{"primary":str(market.iloc[0].current_class),"direction":movement,"probability":None,
            "primary_basis":"Explicit current-class persistence baseline for compatibility, not an original historical market forecast",
            "baseline_quality":baseline,"opposing_evidence":["Historical record, unverified realized market truth"],
            "invalidation":["Not eligible for forward accuracy; source-bound true market labels remain unavailable"]},
        "bands":[{"stage":"CLOSE","metric":"quality_delta","scope":"ORIGINAL_V278_BASELINE","unit":"ratio",
            "direction":"HIGHER_BETTER","low":lo,"high":hi,"basis":"Published movement call's previously fixed +/-0.05 descriptive definition"}],
        "evidence":[{"kind":"MARKET_SNAPSHOT","observed_at":signal+"T15:00:00+08:00","available_at":original["available_at"],
            "source":str(files[0].resolve()),"status":"delayed"},
            {"kind":"RESEARCH_DERIVED","observed_at":knowledge,"available_at":knowledge,
                "source":"Published historical replay calls copied with explicit field allowlist","status":"inferred"}]}
    write_new(output/"replay_plan_NOT_FORWARD.json",plan)
    path, registered=register(plan,output/"replay_registry",datetime.now(timezone.utc).isoformat())
    if registered["forecast_status"]!="HISTORICAL_REPLAY":
        raise ValueError("historical test must never be registered as forward")
    inbox=output/"outcome_inbox"/target/"source_bound.json"
    value=build(path,archive,prior_snapshot,inbox)
    write_new(output/"calendar.json",manifest["calendar"])
    result=run(output/"replay_registry",output/"outcome_inbox",output/"replay_ledger_NOT_FORWARD.jsonl",
        output/"calendar.json",output/"daily_report")
    from daily_research_contract_v292 import binary_metrics
    from public_outcome_bridge_v293 import source_score
    settled=source_score(registered,value,datetime.now(timezone.utc).isoformat())
    metrics=binary_metrics([r for r in settled["rows"] if r["layer"]=="promotion"])
    acceptance={"status":"HISTORICAL_SOURCE_PIPELINE_REPLAY_COMPLETED","registration_hash":registered["sha256"],
        "forecast_status":registered["forecast_status"],"original_calls_changed":False,"market_and_leader_truth_invented":False,
        "promotion_diagnostic":metrics,"raw_candidate_verification":value["candidate_closing_truth"],
        "runner_status":result["status"],"forward_sample_count_added":0,"stable_60_verified":False,
        "source_hashes":{str(Path(__file__).resolve()):sha(__file__),str(inbox.resolve()):sha(inbox),str(path.resolve()):sha(path)}}
    write_new(output/"report.json",acceptance)
    return acceptance


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True);p.add_argument("--prior-snapshot",required=True);p.add_argument("--output",required=True)
    args=p.parse_args()
    r=replay(args.archive,args.prior_snapshot,args.output)
    print(json.dumps({"status":r["status"],"diagnostic":{k:r["promotion_diagnostic"][k] for k in ["total_frozen_rows","scored","correct","wrong","unknown_outcomes"]},
        "forward_sample_count_added":0}))
