"""Descriptive paired moving-block intervals; NOT post-selection validation."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGET = "promotion_next_LABEL_ONLY"


def daily(d):
    return d.groupby("signal_date")[TARGET].agg(hits=lambda v: int(v.eq(1).sum()), known="count").sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve existing diagnostic")
    base = pd.read_csv(args.results/"S1_COMPLETE_BASELINE_SELECTIONS.csv", dtype={"code":str})
    methods = ["CORRECT_UNKNOWN_POLICY_CAP8_MATCHED_DIAGNOSTIC", "CORRECT_UNKNOWN_POLICY_FULL_POOL_DIAGNOSTIC"]
    rows = []; rng = np.random.default_rng(20261030)
    for method in methods:
        alt = pd.read_csv(args.results/(method+"_SELECTIONS.csv"), dtype={"code":str})
        for split in ["FULL_286_DAYS", "LATER_83_DAYS"]:
            b = base if split=="FULL_286_DAYS" else base[base.signal_date.ge("2026-05-01")]
            a = alt if split=="FULL_286_DAYS" else alt[alt.signal_date.ge("2026-05-01")]
            db, da = daily(b), daily(a)
            if not db.index.equals(da.index): raise ValueError("Unpaired dates")
            values = np.c_[db.to_numpy(), da.to_numpy()]; n = len(values); length = 25
            starts = rng.integers(0,n-length+1,size=(5000,int(np.ceil(n/length))))
            ix = (starts[:,:,None]+np.arange(length)[None,None,:]).reshape(5000,-1)[:,:n]
            totals = values[ix].sum(axis=1)
            precision = (totals[:,2]/totals[:,3]-totals[:,0]/totals[:,1])*100
            confirmed = (totals[:,2]-totals[:,0])/(3*n)*100
            rows.append(dict(method=method,period=split,days=n,block_days=length,resamples=5000,
                known_precision_delta_pp=float((da.hits.sum()/da.known.sum()-db.hits.sum()/db.known.sum())*100),
                known_precision_delta_95pct_descriptive_interval_pp=np.quantile(precision,[.025,.975]).tolist(),
                confirmed_all_slots_delta_95pct_descriptive_interval_pp=np.quantile(confirmed,[.025,.975]).tolist()))
    result = dict(status="DESCRIPTIVE_ONLY_NOT_SELECTION_ADJUSTED_OR_FORWARD_VALIDATION",seed=20261030,
        method="Paired non-circular moving blocks of consecutive signal days; no independent-stock assumption",results=rows)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__": main()
