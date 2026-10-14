"""Unique prior-session height benchmark, selected before opening/outcome joins."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pandas as pd


def select(cohort):
    allowed = cohort[["code","signal_date","entry_date","boards","start_known"]].copy()
    if allowed.duplicated(["code","signal_date"]).any():
        raise ValueError("duplicate candidate")
    rows = []
    for day, group in allowed.groupby("entry_date",sort=True):
        base = dict(entry_date=day,pool_count=len(group),selected_code=None)
        known = group.start_known.astype(str).str.lower().isin(["true","1"])
        height = pd.to_numeric(group.boards,errors="coerce")
        if not known.all() or height.isna().any() or not height.ge(2).all():
            rows.append(dict(base,state="HEIGHT_EVIDENCE_INCOMPLETE")); continue
        top = group[height.eq(height.max())]
        if len(top) != 1:
            rows.append(dict(base,state="TOP_HEIGHT_TIE",top_height=int(height.max()))); continue
        candidate = top.iloc[0]
        rows.append(dict(base,state="UNIQUE_HEIGHT_SELECTED",selected_code=candidate.code,
                         signal_date=candidate.signal_date,top_height=int(candidate.boards)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort","replay","scenario","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    chosen = select(pd.read_csv(args.cohort,dtype={"code":str}))
    # Save choices before loading any opening or future outcome records.
    (args.output/"choices_BEFORE_OUTCOMES.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    observations = {(r["code"],r["entry_date"]):r for r in replay["rows"]}
    marks = {(r["code"],r["entry_date"]):r for r in scenario["rows"]}
    results = []
    for choice in chosen:
        row = dict(choice,opening_state=None,promotion_LABEL_ONLY=None,t1_scenario=None)
        code = choice["selected_code"]
        if code is not None:
            key = (code,choice["entry_date"])
            obs = observations.get(key,{})
            row["opening_state"] = obs.get("observation",{}).get("state","MISSING")
            if row["opening_state"] == "CLOSE_AT_LIMIT_PROXY":
                row["promotion_LABEL_ONLY"] = obs.get("promotion_LABEL_ONLY")
                row["t1_scenario"] = marks.get(key)
        results.append(row)
    confirmed = [r for r in results if r["opening_state"] == "CLOSE_AT_LIMIT_PROXY"]
    markable = [r["t1_scenario"] for r in confirmed if r["t1_scenario"] and r["t1_scenario"]["net_t1_mark_return"] is not None]
    report = dict(status="DIAGNOSTIC_ONLY",rows=results,
                  selection_states=dict(Counter(r["state"] for r in results)),opening_confirmed=len(confirmed),
                  confirmed_promotions=sum(r["promotion_LABEL_ONLY"] is True for r in confirmed),
                  hypothetical_markable=len(markable),hypothetical_positive=sum(r["net_t1_mark_return"]>0 for r in markable),
                  hypothetical_large_loss=sum(r["large_loss_proxy"] for r in markable),
                  calibrated_leader_probability=None,verified_trades=0,
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.cohort,args.replay,args.scenario,Path(__file__))},
                  limitations=["highest archived board is not verified theme leadership",
                               "historical exploratory benchmark, not an independent blind trading test",
                               "no fallback to lower boards or outcome-selected replacement",
                               "repeated stock/episode observations and unverified fills prevent win-rate claim"])
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","source_hashes")},indent=2))


if __name__ == "__main__":
    main()
