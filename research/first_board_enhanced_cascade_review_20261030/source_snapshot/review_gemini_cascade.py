"""Replay already-fitted four-decimal cascade weights; never optimize.

Original weights saw the evaluation labels. Repairs do not make them out of sample.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/"research/first_board_top3_40_20261030/results"
WEIGHTS=HERE/"gemini_pre_fix/MATH_CASCADE_OPTIMIZATION_REPORT.json"
KEYS=["signal_date","code"]
P2="promotion_next_LABEL_ONLY"
P4="reach4_LABEL_ONLY"
FIELDS=["s1_score","height_score","seal_ratio","seal_time_inv","breakout","pos60","amount_log","turnover_val"]

def read(p):
    return pd.read_csv(p,dtype={"code":str},float_precision="round_trip",low_memory=False)

def percentiles(features, missing_policy="neutral_rank"):
    if features.duplicated(KEYS).any():
        raise ValueError("Duplicate pool key")
    d=features[KEYS+["s1_score","height_score","reported_close_seal_log_amount_ratio",
        "reported_first_seal_trading_minutes","breakout_20","position_60","amount","turnover"]].copy()
    d["seal_ratio"]=d.reported_close_seal_log_amount_ratio
    d["seal_time_inv"]=-d.reported_first_seal_trading_minutes
    d["breakout"]=d.breakout_20; d["pos60"]=d.position_60
    d["amount_log"]=np.log1p(d.amount); d["turnover_val"]=d.turnover
    if missing_policy not in ["neutral_rank","original_imputation_diagnostic"]:
        raise ValueError("Unknown missing-data policy")
    if not np.isfinite(d[["s1_score","height_score"]]).all().all():
        raise ValueError("Missing baseline score")
    out=d[KEYS].copy()
    for f in FIELDS:
        values=d[f].replace([np.inf,-np.inf],np.nan)
        out[f+"_missing"]=values.isna()
        if missing_policy=="original_imputation_diagnostic":
            fills={"seal_ratio":0.,"seal_time_inv":-240.,"breakout":0.,"pos60":.5}
            if f in fills:
                values=values.fillna(fills[f])
        out[f+"_pct"]=values.groupby(d.signal_date).rank(method="average",pct=True).fillna(.5)
    return out

def cascade_select(features, weights, missing_policy="neutral_rank"):
    f=percentiles(features,missing_policy)
    cols=[c+"_pct" for c in FIELDS]
    for slot in [1,2,3]:
        w=weights[f"w{slot}"]
        if set(w)!=set(cols) or not np.isfinite(list(w.values())).all() or min(w.values())<0 or sum(w.values())<=0:
            raise ValueError("Invalid saved weight vector")
        f[f"slot{slot}_score"]=f[cols].to_numpy()@np.array([w[c] for c in cols])
    selections=[]
    for date,g in f.groupby("signal_date",sort=True):
        if len(g)<3:
            raise ValueError("Daily pool must have at least three candidates")
        remaining=g.copy()
        for slot in [1,2,3]:
            ordered=remaining.sort_values([f"slot{slot}_score","code"],ascending=[False,True])
            chosen=ordered.iloc[0]
            selections.append(dict(signal_date=date,code=chosen.code,slot=slot,
                saved_weight_score=float(chosen[f"slot{slot}_score"]),
                missing_inputs=int(chosen[[c+"_missing" for c in FIELDS]].sum())))
            remaining=remaining[remaining.code.ne(chosen.code)]
    return pd.DataFrame(selections)

def evaluator():
    p=ROOT/"research/first_board_top3_40_20261030/analyze_top3_gap.py"
    spec=importlib.util.spec_from_file_location("existing_top3_contract",p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.evaluate_selection

def evaluate(pool,picks,dates):
    report=evaluator()(pool,picks[KEYS],dates)
    q=picks.merge(pool[KEYS+[P2,P4]],on=KEYS,validate="one_to_one",how="left")
    report["reach4_hits"]=int(q[P4].eq(1).sum())
    report["reach4_known"]=int(q[P4].notna().sum())
    report["by_slot"]={str(slot):dict(selected=len(g),known=int(g[P2].notna().sum()),
        unknown=int(g[P2].isna().sum()),hits=int(g[P2].eq(1).sum())) for slot,g in q.groupby("slot")}
    return report,q

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"gemini_corrected_results")
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=False)
    paths=[DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv",DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv",WEIGHTS,Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    features,outcomes=map(read,paths[:2]); weights=json.loads(WEIGHTS.read_text(encoding="utf-8"))
    assert len(features)==len(outcomes)==7254
    pool=features.merge(outcomes,on=KEYS,validate="one_to_one")
    assert len(pool)==7254 and pool.signal_date.nunique()==286
    dates=sorted(pool.signal_date.unique())
    original_known_pool=pool[pool[P2].notna()].copy()
    protocols={
        "ORIGINAL_DROPPED_UNKNOWN_ROUNDED_WEIGHTS_DIAGNOSTIC":(original_known_pool,"original_imputation_diagnostic"),
        "FULL_POOL_SAME_IMPUTATION_ROUNDED_WEIGHTS_DIAGNOSTIC":(pool,"original_imputation_diagnostic"),
        "FULL_POOL_NEUTRAL_MISSING_ROUNDED_WEIGHTS_DIAGNOSTIC":(pool,"neutral_rank"),
    }
    (out/"PROTOCOL_BEFORE_REPLAY.json").write_text(json.dumps(dict(
        formed_at=datetime.now(timezone.utc).isoformat(),new_optimization_calls=0,
        original_weights_saw_all_evaluation_labels=True,original_full_precision_weights_available=False,
        methods=list(protocols),source_hashes=hashes),ensure_ascii=False,indent=2),encoding="utf-8")
    results={}
    for name,(frame,policy) in protocols.items():
        picks=cascade_select(frame,weights,policy)
        rep,detail=evaluate(pool,picks,dates)
        results[name]=rep
        detail.to_csv(out/f"{name}_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    baseline=pool.sort_values(["signal_date","s1_score","code"],ascending=[True,False,True]).groupby("signal_date").head(3).copy()
    baseline["slot"]=baseline.groupby("signal_date").cumcount()+1
    results["S1_SAVED_REFERENCE"]=evaluate(pool,baseline[KEYS+["slot"]],dates)[0]
    report=dict(formed_at=datetime.now(timezone.utc).isoformat(),results=results,
        original_claimed_hits=weights["hits"],original_claimed_denominators=weights["slots"],
        actual_original_optimization_pool_rows=len(original_known_pool),
        actual_original_optimization_dates=int(original_known_pool.signal_date.nunique()),
        actual_original_selection_slots=3*int(original_known_pool.signal_date.nunique()),
        source_hashes=hashes,new_optimization_calls=0,production_changed=False,
        historical_weight_search_is_a_fit=True,independent_validation=False,
        full_precision_reproducibility=False,
        issues=["Unknown future outcomes were removed BEFORE percentile construction and selection.",
                "All evaluation days' known outcomes were also used for weight search.",
                "The original 284/283/285 denominators came from S1, not the cascade selections.",
                "Only four-decimal weights were saved; exact original optimizer picks cannot be reconstructed.",
                "Unknown first seal/seal ratio were encoded as last/zero in the original scoring.",
                "Three greedy slot-wise optimizations do not prove optimal joint Top3 count."])
    protection=json.loads((ROOT/"research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json").read_text(encoding="utf-8"))["protected_sha256"]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protection.items())
    report["protected_files_unchanged"]=len(protection)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
