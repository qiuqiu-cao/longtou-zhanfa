"""Compare archived 8- and 13-field cascade weights without refitting.

All saved weights were selected using the evaluation period's labels. Corrected
replays remain in-sample diagnostics, not independent promotion evidence.
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
NEW_WEIGHTS=HERE/"source_snapshot/ENHANCED_CASCADE_OPTIMIZATION_REPORT.json"
OLD_WEIGHTS=ROOT/"research/first_board_math_20261030/gemini_pre_fix/MATH_CASCADE_OPTIMIZATION_REPORT.json"
KEYS=["signal_date","code"]
P2="promotion_next_LABEL_ONLY"
P4="reach4_LABEL_ONLY"
BASE_FIELDS=["s1_score","height_score","seal_ratio","seal_floatcap","seal_time_inv",
             "seal_rank_inv","breakout","pos60","amount_log","turnover_val"]
SCORES=[c+"_pct" for c in BASE_FIELDS]+["seal_dominance_pct","height_synergy_pct","one_p"]

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

OLD=load_module(ROOT/"research/first_board_math_20261030/review_gemini_cascade.py","old_cascade_reference")

def read(p):
    return pd.read_csv(p,dtype={"code":str},float_precision="round_trip",low_memory=False)

def enhanced_features(features,missing_policy="neutral_rank"):
    """S-day whitelist; never inspect or drop rows using future labels."""
    if features.duplicated(KEYS).any():
        raise ValueError("Duplicate candidate key")
    source_fields=KEYS+["s1_score","height_score","reported_close_seal_log_amount_ratio",
        "reported_close_seal_log_floatcap_ratio","reported_first_seal_trading_minutes",
        "reported_first_seal_observed_pool_rank","breakout_20","position_60","amount","turnover","one_price"]
    d=features[source_fields].copy()
    d["seal_ratio"]=d.reported_close_seal_log_amount_ratio
    d["seal_floatcap"]=d.reported_close_seal_log_floatcap_ratio
    d["seal_time_inv"]=-d.reported_first_seal_trading_minutes
    d["seal_rank_inv"]=-d.reported_first_seal_observed_pool_rank
    d["breakout"]=d.breakout_20; d["pos60"]=d.position_60
    d["amount_log"]=np.log1p(d.amount); d["turnover_val"]=d.turnover
    if missing_policy not in ["neutral_rank","original_imputation_diagnostic"]:
        raise ValueError("Unknown missing policy")
    if not np.isfinite(d[["s1_score","height_score"]]).all().all():
        raise ValueError("Missing original model score")
    if not d.one_price.dropna().isin([0,1]).all():
        raise ValueError("Invalid signal-day one-price flag")
    out=d[KEYS].copy()
    original_fill={"seal_ratio":0.,"seal_floatcap":0.,"seal_time_inv":-240.,
                   "seal_rank_inv":-50.,"breakout":0.,"pos60":.5}
    for field in BASE_FIELDS:
        values=d[field].replace([np.inf,-np.inf],np.nan)
        out[field+"_missing"]=values.isna()
        if missing_policy=="original_imputation_diagnostic" and field in original_fill:
            values=values.fillna(original_fill[field])
        out[field+"_pct"]=values.groupby(d.signal_date).rank(method="average",pct=True).fillna(.5)
    for name,left,right in [("seal_dominance","seal_ratio","seal_time_inv"),
                             ("height_synergy","height_score","s1_score")]:
        values=out[left+"_pct"]*out[right+"_pct"]
        missing=out[left+"_missing"]|out[right+"_missing"]
        out[name+"_missing"]=missing
        if missing_policy=="neutral_rank":
            # An interaction with an unknown input is itself unknown. Do not
            # percentile-rank an imputed product as an observed market fact.
            values=values.mask(missing)
        out[name+"_pct"]=values.groupby(d.signal_date).rank(method="average",pct=True).fillna(.5)
    out["one_p_missing"]=d.one_price.isna()
    out["one_p"]=d.one_price.fillna(0. if missing_policy=="original_imputation_diagnostic" else .5)
    return out

def select_features(f,weights,eligible_keys=None):
    """Select distinct slots using only already-constructed score features."""
    cols=list(weights["w1"])
    if any(set(weights[f"w{k}"])!=set(cols) for k in [2,3]) or not set(cols).issubset(f):
        raise ValueError("Inconsistent saved feature schema")
    scored=f.copy()
    for slot in [1,2,3]:
        w=np.array([weights[f"w{slot}"][c] for c in cols],float)
        if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
            raise ValueError("Invalid saved weights")
        scored[f"slot{slot}_score"]=scored[cols].to_numpy()@w
    if eligible_keys is not None:
        scored=scored.merge(eligible_keys[KEYS],on=KEYS,how="inner",validate="one_to_one")
    selections=[]
    for day,g in scored.groupby("signal_date",sort=True):
        remaining=g.copy()
        for slot in [1,2,3]:
            if remaining.empty:
                break
            best=remaining.sort_values([f"slot{slot}_score","code"],ascending=[False,True]).iloc[0]
            selections.append(dict(signal_date=day,code=best.code,slot=slot,
                score=float(best[f"slot{slot}_score"])))
            remaining=remaining[remaining.code.ne(best.code)]
    return pd.DataFrame(selections)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=False)
    paths=[DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv",DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv",
           NEW_WEIGHTS,OLD_WEIGHTS,Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    x,y=map(read,paths[:2]); pool=x.merge(y,on=KEYS,validate="one_to_one")
    assert len(pool)==7254 and pool.signal_date.nunique()==286
    dates=sorted(pool.signal_date.unique())
    new=json.loads(NEW_WEIGHTS.read_text(encoding="utf-8"))
    old=json.loads(OLD_WEIGHTS.read_text(encoding="utf-8"))
    assert set(new["features"])==set(SCORES)
    (out/"PROTOCOL_BEFORE_REPLAY.json").write_text(json.dumps(dict(
        formed_at=datetime.now(timezone.utc).isoformat(),new_fits=0,source_hashes=hashes,
        scope="Saved 4-decimal weights only; already-inspected history, no out-of-sample claim",
        missing_policy="Explicit neutral percentile; unknown interaction stays unknown before imputation",
        comparisons=["Original dropped-known replay","Complete pool original imputation",
                     "Complete pool neutral imputation","Old 8-field matched control","S1 control"],
        non_one_price="Select from S-day non-one-price candidates using percentiles formed on complete pool; same 286 dates"),
        ensure_ascii=False,indent=2),encoding="utf-8")
    dropped=pool[pool[P2].notna()]
    neutral=enhanced_features(x)
    old_f=OLD.percentiles(x)
    picks={
        "NEW13_ORIGINAL_DROPPED_UNKNOWN_DIAGNOSTIC":select_features(enhanced_features(dropped,"original_imputation_diagnostic"),new),
        "NEW13_FULL_ORIGINAL_IMPUTATION_DIAGNOSTIC":select_features(enhanced_features(x,"original_imputation_diagnostic"),new),
        "NEW13_FULL_NEUTRAL_DIAGNOSTIC":select_features(neutral,new),
        "OLD8_FULL_NEUTRAL_DIAGNOSTIC":select_features(old_f,old),
        "NEW13_NON_ONE_PRICE_SAME_POOL_FEATURES":select_features(neutral,new,x[x.one_price.eq(0)]),
        "OLD8_NON_ONE_PRICE_SAME_POOL_FEATURES":select_features(old_f,old,x[x.one_price.eq(0)]),
    }
    for name,frame in [("S1_SAVED_REFERENCE",pool),("S1_NON_ONE_PRICE_REFERENCE",pool[pool.one_price.eq(0)])]:
        q=frame.sort_values(["signal_date","s1_score","code"],ascending=[True,False,True]).groupby("signal_date").head(3).copy()
        q["slot"]=q.groupby("signal_date").cumcount()+1
        picks[name]=q[KEYS+["slot"]]
    reports={}; details={}; months=[]
    for name,take in picks.items():
        rep,detail=OLD.evaluate(pool,take,dates)
        detail=detail.merge(x[KEYS+["one_price"]],on=KEYS,validate="one_to_one")
        rep["selected_one_price"]=int(detail.one_price.eq(1).sum())
        rep["one_price_promotion_hits"]=int(detail.loc[detail.one_price.eq(1),P2].eq(1).sum())
        rep["non_one_price_promotion_hits"]=int(detail.loc[detail.one_price.eq(0),P2].eq(1).sum())
        reports[name]=rep; details[name]=detail
        detail.to_csv(out/f"{name}_SELECTIONS.csv",index=False,encoding="utf-8-sig")
        for month,g in detail.groupby(detail.signal_date.str[:7]):
            months.append(dict(method=name,month=month,selected=len(g),hits=int(g[P2].eq(1).sum()),
                               known=int(g[P2].notna().sum()),reach4_hits=int(g[P4].eq(1).sum())))
    transitions=[]
    for control in ["OLD8_FULL_NEUTRAL_DIAGNOSTIC","S1_SAVED_REFERENCE"]:
        a=set(map(tuple,picks[control][KEYS].to_numpy()))
        b=set(map(tuple,picks["NEW13_FULL_NEUTRAL_DIAGNOSTIC"][KEYS].to_numpy()))
        for direction,keys in [("REMOVED",a-b),("ADDED",b-a)]:
            q=pool[[tuple(k) in keys for k in pool[KEYS].to_numpy()]]
            transitions.append(dict(control=control,direction=direction,rows=len(q),
                hits=int(q[P2].eq(1).sum()),failures=int(q[P2].eq(0).sum()),unknown=int(q[P2].isna().sum()),
                reach4_hits=int(q[P4].eq(1).sum()),reach4_unknown=int(q[P4].isna().sum())))
    pd.DataFrame(months).to_csv(out/"MONTHLY_REPLAY_DIAGNOSTIC.csv",index=False,encoding="utf-8-sig")
    protection=json.loads((ROOT/"research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json").read_text(encoding="utf-8"))["protected_sha256"]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protection.items())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result=dict(formed_at=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,results=reports,
        two_way_replacement_ledger=transitions,original_reported_hits=new["hits"],original_reported_slots=new["slots"],
        original_known_pool_rows=len(dropped),original_known_pool_dates=int(dropped.signal_date.nunique()),
        new_weight_count=39,old_weight_count=24,new_independent_simplex_dimensions=36,old_independent_simplex_dimensions=21,
        historical_optimization_is_fit=True,independent_validation=False,new_optimization_calls=0,
        protected_files_unchanged=len(protection),production_changed=False,
        full_precision_parameters_available=False,
        limitations=["Both sets of weights were optimized on evaluation-period outcomes.",
                     "Original loader removes future-unknown rows before within-day percentiles.",
                     "Original 852 denominator is copied from S1, not calculated from new selections.",
                     "Replay uses rounded weights; no guarantee of original full-precision membership.",
                     "One-price is an S-day flag, not proof that next-day entry is impossible or possible.",
                     "Nonlinear product terms are algebraic transformations, not evidence of a causal mechanism."])
    (out/"REPORT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"results":reports,"two_way_replacement_ledger":transitions,"protected_files_unchanged":len(protection)},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
