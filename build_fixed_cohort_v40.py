"""Lock a continuous historical research interval without inspecting outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from build_opening_cohort_v35 import select


def mapping(calendar, start, end):
    dates = list(calendar)
    if dates != sorted(set(dates)):
        raise ValueError("calendar must be sorted and unique")
    if start > end or not dates or dates[-1] <= end or dates[0] > start:
        raise ValueError("calendar must cover interval and next session")
    return {date:dates[i+1] for i,date in enumerate(dates[:-1]) if start <= date <= end}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dates = mapping(pd.read_csv(args.calendar)["date"],args.start,args.end)
    data = select(pd.read_csv(args.candidates,dtype={"code":str}),dates)
    args.output.mkdir(parents=True,exist_ok=False)
    data.to_csv(args.output/"cohort.csv",index=False)
    for day,group in data.groupby("entry_date"):
        (args.output/(day+"_requests.json")).write_text(json.dumps(group[["provider_code","entry_date"]].values.tolist(),indent=2),encoding="utf-8")
    counts = data.groupby("signal_date").size().to_dict()
    report = dict(status="DIAGNOSTIC_ONLY",start=args.start,end=args.end,
                  observations=len(data),calendar_sessions=len(dates),
                  session_counts={date:int(counts.get(date,0)) for date in dates},
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.candidates,args.calendar)},
                  decision_time="09:35 completed-bar observation; not buy authorization",
                  model_changes_allowed=False,not_blind=True,
                  limitations=["historical period previously explored; no independent blind claim",
                               "all archived candidates retained, but historical universe incomplete",
                               "provider calendar is not exchange-certified; no outcomes read in selection"])
    (args.output/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
