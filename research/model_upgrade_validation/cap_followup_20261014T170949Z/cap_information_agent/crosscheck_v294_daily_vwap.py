"""Compare old and daily VWAP inputs; never load labels or fit models."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def description(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"n":len(x), "min":float(x.min()) if len(x) else None,
            "median":float(np.median(x)) if len(x) else None,
            "p99":float(np.quantile(x, .99)) if len(x) else None,
            "max":float(x.max()) if len(x) else None,
            "mean":float(x.mean()) if len(x) else None}


def summarize(frame, tolerances):
    old = np.isfinite(frame.old_v294_close_vs_vwap)
    new = np.isfinite(frame.daily_close_vs_vwap)
    both = old & new
    values = frame.loc[both, "absolute_ratio_spread_difference"]
    relative = frame.loc[both, "relative_ratio_difference"]
    return {"full_denominator":len(frame), "old_known":int(old.sum()), "old_unknown":int((~old).sum()),
        "new_known":int(new.sum()), "new_unknown":int((~new).sum()),
        "both_known":int(both.sum()), "old_known_new_unknown":int((old&~new).sum()),
        "old_unknown_new_known":int((~old&new).sum()), "both_unknown":int((~old&~new).sum()),
        "absolute_ratio_spread_difference":description(values), "relative_ratio_difference":description(relative),
        "fixed_tolerances":[{"tolerance":t,"comparison_n":int(both.sum()),
            "absolute_within":int(values.le(t).sum()),"absolute_outside":int(values.gt(t).sum()),
            "relative_within":int(relative.le(t).sum()),"relative_outside":int(relative.gt(t).sum())} for t in tolerances]}


def main():
    plan = json.loads((HERE/"v294_daily_crosscheck_plan_BEFORE_MEASUREMENT.json").read_text())
    for path,digest in plan["source_bindings"].items():assert sha(path)==digest
    p=list(plan["source_bindings"])
    old=pd.read_csv(p[0],usecols=["date","code","boards","price","m30_close_vs_full_day_vwap"],dtype={"code":str},float_precision="round_trip")
    direct=pd.read_csv(p[1],usecols=["date","code","boards","price","direct_close_over_vwap","direct_vwap_sina_yuan","vwap_within_raw_ohlc_1cent"],dtype={"code":str},float_precision="round_trip")
    september=pd.read_csv(p[2],usecols=["date","code","price","vwap_if_volume_is_100_share_lots","lots_unit_within_ohlc_1cent","candidate_close_matches_raw_1cent"],dtype={"code":str},float_precision="round_trip")
    for frame in [old,direct,september]:
        frame.code=frame.code.str.zfill(6)
        assert not frame.duplicated(["date","code"]).any()
    assert len(old)==2888 and len(direct)==9245 and len(september)==182
    daily=direct[["date","code","boards","price"]].copy()
    daily["daily_close_vs_vwap"]=(direct.direct_close_over_vwap-1).where(direct.vwap_within_raw_ohlc_1cent)
    daily["daily_measurement_policy"]=np.where(daily.daily_close_vs_vwap.notna(),"SINA_DIRECT_AMOUNT_VOLUME","UNKNOWN")
    indexed=september.set_index(["date","code"])
    for i,row in daily.iterrows():
        key=(row.date,row.code)
        if key not in indexed.index:continue
        s=indexed.loc[key]
        assert pd.isna(row.daily_close_vs_vwap)
        if s.lots_unit_within_ohlc_1cent and s.candidate_close_matches_raw_1cent:
            assert np.isclose(row.price,s.price,rtol=0,atol=1e-12)
            daily.loc[i,"daily_close_vs_vwap"]=s.price/s.vwap_if_volume_is_100_share_lots-1
            daily.loc[i,"daily_measurement_policy"]="SEPTEMBER_INFERRED_100_SHARE_LOT_HYPOTHESIS"
    old=old.rename(columns={"m30_close_vs_full_day_vwap":"old_v294_close_vs_vwap","boards":"old_boards","price":"old_price"})
    old["old_roster_order"]=range(len(old))
    cross=old.merge(daily,on=["date","code"],how="left",validate="one_to_one",indicator="original9245_match",sort=False)
    assert len(cross)==2888 and cross.old_roster_order.tolist()==list(range(2888))
    matched=cross.original9245_match.eq("both")
    assert (cross.loc[matched,"old_boards"]==cross.loc[matched,"boards"]).all()
    assert np.isclose(cross.loc[matched,"old_price"],cross.loc[matched,"price"],rtol=0,atol=1e-12).all()
    for frame in [cross]:
        frame["absolute_ratio_spread_difference"]=abs(frame.old_v294_close_vs_vwap-frame.daily_close_vs_vwap)
        frame["relative_ratio_difference"]=abs((frame.old_v294_close_vs_vwap+1)/(frame.daily_close_vs_vwap+1)-1)
    reverse=daily.merge(old[["date","code","old_v294_close_vs_vwap","old_roster_order"]],on=["date","code"],how="left",validate="one_to_one",indicator="old2888_match",sort=False)
    reverse["absolute_ratio_spread_difference"]=abs(reverse.old_v294_close_vs_vwap-reverse.daily_close_vs_vwap)
    reverse["relative_ratio_difference"]=abs((reverse.old_v294_close_vs_vwap+1)/(reverse.daily_close_vs_vwap+1)-1)
    assert len(reverse)==9245 and list(zip(reverse.date,reverse.code))==list(zip(direct.date,direct.code))
    forbidden={"target","target_date","label_valid","truth","prediction","promoted_next_day"}
    assert not forbidden.intersection(cross.columns) and not forbidden.intersection(reverse.columns)
    old_file=HERE/"old2888_v294_vs_daily_vwap_NO_OUTCOMES.csv"
    full_file=HERE/"full9245_v294_vs_daily_vwap_NO_OUTCOMES.csv"
    cross.to_csv(old_file,index=False)
    reverse.to_csv(full_file,index=False)
    report={"formed_at":datetime.now(timezone.utc).isoformat(),"source_bindings":plan["source_bindings"],
        "old2888_full_roster":summarize(cross,plan["tolerances"]),
        "original9245_full_roster":summarize(reverse,plan["tolerances"]),
        "old_roster_not_in_original9245":int((~matched).sum()),
        "old_roster_by_policy":{str(k):summarize(g,plan["tolerances"]) for k,g in cross.groupby("daily_measurement_policy",dropna=False)},
        "daily_without_prior_roster":int((reverse.old2888_match.eq("left_only") & reverse.daily_close_vs_vwap.notna()).sum()),
        "all_unknowns_retained":True,"target_columns_loaded":False,"new_fits":0,"gain_verified":False,
        "no_claim_of_asof_or_independent_unit_certification":True,
        "script_sha256":sha(__file__),"output_sha256":{old_file.name:sha(old_file),full_file.name:sha(full_file)}}
    write(HERE/"v294_daily_vwap_crosscheck_results.json",report)
    # Independent lookup-based numerical replay on saved full old roster.
    saved=pd.read_csv(old_file,dtype={"code":str},float_precision="round_trip")
    old_map=old.set_index(["date","code"])["old_v294_close_vs_vwap"]
    new_map=daily.set_index(["date","code"])["daily_close_vs_vwap"]
    old_values=np.array([old_map.loc[(d,c)] for d,c in zip(saved.date,saved.code)],dtype=float)
    new_values=np.array([new_map.get((d,c),np.nan) for d,c in zip(saved.date,saved.code)],dtype=float)
    np.testing.assert_allclose(saved.old_v294_close_vs_vwap,old_values,rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(saved.daily_close_vs_vwap,new_values,rtol=0,atol=0,equal_nan=True)
    delta=abs(old_values-new_values)
    known=np.isfinite(delta)
    for row in report["old2888_full_roster"]["fixed_tolerances"]:
        assert (delta[known]<=row["tolerance"]).sum()==row["absolute_within"]
        assert (delta[known]>row["tolerance"]).sum()==row["absolute_outside"]
    for path,digest in plan["source_bindings"].items():assert sha(path)==digest
    write(HERE/"v294_daily_crosscheck_verification.json",{"formed_at":datetime.now(timezone.utc).isoformat(),
        "lookup_replay_all2888_passed":True,"fixed_counts_passed":True,"inputs_hashes_unchanged":True,
        "all9245_original_order_and_unknowns_retained":True,"labels_loaded":False,"new_fits":0})
    print(json.dumps({"old":report["old2888_full_roster"],"full":report["original9245_full_roster"],
        "policies":report["old_roster_by_policy"],"daily_without_prior_roster":report["daily_without_prior_roster"]},indent=2))


if __name__=="__main__":main()
