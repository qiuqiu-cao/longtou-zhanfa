"""Calendar denominator supplement for retrospective signal-date diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

from long_path_challenger_v294 import SPEC, ARMS, binary, ranking, sha, write_json
from next_session_expectation_contract_v271 import next_session


def run(experiment,calendar,output):
    experiment,calendar,output=Path(experiment),Path(calendar),Path(output)
    calendar_value=json.loads(calendar.read_text(encoding="utf-8"))
    if calendar_value.get("status")!="verified" or not calendar_value.get("source"):
        raise ValueError("verified official calendar required")
    report=json.loads((experiment/"report.json").read_text(encoding="utf-8"))
    for name,wanted in report["artifact_hashes"].items():
        if sha(experiment/name)!=wanted:raise ValueError("experiment artifact changed")
    sessions=[d for d in calendar_value["sessions"] if SPEC["test_start"]<=d<=SPEC["test_end"]]
    if not sessions:raise ValueError("calendar does not cover experiment")
    results={};inputs=[calendar,experiment/"report.json",Path(__file__)]
    for arm in ARMS:
        path=experiment/(arm+"_scored_LABEL_ONLY.csv")
        frame=pd.read_csv(path,dtype={"code":str})
        inputs.append(path)
        if set(frame.date)-set(sessions):raise ValueError("candidate signal on a closed date")
        expected={d:next_session(calendar_value,d) for d in frame.date.unique()}
        if not frame.loc[frame.label_valid,"target_date"].eq(frame.loc[frame.label_valid,"date"].map(expected)).all():
            raise ValueError("known original labels skip actual exchange sessions")
        if not pd.to_datetime(frame.loc[frame.label_valid,"target_date"]).dt.date.le(datetime.now(timezone.utc).date()).all():
            raise ValueError("future original label claimed known")
        windows={}
        for key,dates in [("all",sessions),("20",sessions[-20:]),("60",sessions[-60:])]:
            part=frame[frame.date.isin(dates)]
            known_dates=part.groupby("date").label_valid.all()
            windows[key]={"window_basis":"OFFICIAL_SIGNAL_SESSIONS; HISTORICAL_DIAGNOSTIC_NOT_FORWARD", "sessions":len(dates),
                "start":dates[0],"end":dates[-1],"candidate_dates":int(part.date.nunique()),
                "missing_entire_dates":sorted(set(dates)-set(part.date)),
                "incomplete_candidate_dates":known_dates[~known_dates].index.tolist(),
                "complete_label_dates":int(known_dates.sum()),"binary":binary(part),"nonword_top1":ranking(part)[0]}
        results[arm]=windows
    value={"version":"v294-official-signal-calendar-denominators-1","recorded_at":datetime.now(timezone.utc).isoformat(),
        "status":"CALENDAR_GAPS_PRESERVED_AND_KNOWN_SUCCESSORS_CHECKED", "arms":results,
        "missing_dates_not_assumed_zero_candidates":True,"forward_samples_added":0,"stable_60_verified":False,
        "source_hashes":{str(p.resolve()):sha(p) for p in inputs}}
    write_json(output,value)
    return {"status":value["status"],"windows":{k:{field:v[field] for field in ["sessions","candidate_dates","complete_label_dates","missing_entire_dates","incomplete_candidate_dates"]} for k,v in results["BASE"].items()}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["experiment","calendar","output"]:parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.experiment,args.calendar,args.output),ensure_ascii=True))
