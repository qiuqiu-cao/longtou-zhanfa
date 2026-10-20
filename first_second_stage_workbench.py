"""Prepare score-masked stage facts, then explicitly rescore a new close."""
from pathlib import Path
import argparse

import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage


def prepare(inputs,signal_date,output):
    target=Path(output)
    if target.exists():raise FileExistsError(target)
    rows,_,sessions,audit=stage.build_rows(inputs,feature_cutoff=signal_date)
    if signal_date not in sessions:raise ValueError("date outside official calendar")
    current=rows[rows.signal_date.eq(signal_date)].copy()
    if current.empty:raise ValueError("no verified first/second-board candidates")
    current=current.rename(columns={"label_end":"target_date"})
    current=current[[c for c in current if "LABEL_ONLY" not in c and c not in ("baseline_score",)]]
    current.to_csv(target,index=False,encoding="utf-8-sig")
    base.write_json(target.with_suffix(".receipt.json"),dict(formed_at=base.now(),signal_date=signal_date,stages=current.groupby("stage").size().to_dict(),
                    source_hashes={str(p):base.sha(p) for p in inputs.values()},signal_day_only=True,model_scores_displayed=False,source_audit=audit))
    return current


def recheck(bundle,features,output,prior_selection=None):
    keys=None
    if prior_selection:
        prior=pd.read_csv(prior_selection,dtype={"code":str},float_precision="round_trip")
        if not {"signal_date","stage","code","model_score","target_date"}.issubset(prior):raise ValueError("prior prediction lacks frozen first-board keys/target date")
        first=base.selected(prior[prior.stage.eq(1)],"model_score",1)
        keys=set(zip(first.target_date,first.code))
    stage.recheck(bundle,features,output)
    out=Path(output)
    scored=pd.read_csv(out,dtype={"code":str},float_precision="round_trip")
    scored["stage_rank"]=scored.groupby(["signal_date","stage"]).model_score.rank(method="first",ascending=False).astype(int)
    if keys is not None:
        scored["was_previous_first_pick"]=[s==2 and (d,c) in keys for d,c,s in zip(scored.signal_date,scored.code,scored.stage)]
    scored.to_csv(out,index=False,encoding="utf-8-sig")
    base.write_json(out.with_suffix(".receipt.json"),dict(formed_at=base.now(),bundle_sha256=base.sha(bundle),features_sha256=base.sha(features),
                    prior_selection_sha256=base.sha(prior_selection) if prior_selection else None,output_sha256=base.sha(out),
                    interpretation="Stage1 known only at first-board close; stage2 updates only second-board close; probability estimates historical-qualified only; research only"))
    return scored


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare");a.add_argument("--bars",required=True);a.add_argument("--recovered",required=True);a.add_argument("--calendar",required=True);a.add_argument("--security-intervals",required=True);a.add_argument("--signal-date",required=True);a.add_argument("--output",required=True)
    b=sub.add_parser("recheck");b.add_argument("--bundle",required=True);b.add_argument("--features",required=True);b.add_argument("--output",required=True);b.add_argument("--prior-selection")
    args=p.parse_args()
    if args.command=="prepare":prepare(dict(bars=Path(args.bars),recovered=Path(args.recovered),calendar=Path(args.calendar),security_intervals=Path(args.security_intervals)),args.signal_date,args.output)
    else:recheck(args.bundle,args.features,args.output,args.prior_selection)
