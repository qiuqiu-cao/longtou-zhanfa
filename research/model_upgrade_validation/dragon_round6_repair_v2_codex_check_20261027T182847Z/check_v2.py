"""Snapshot and verify the second repair; prediction only, never estimator fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
SNAP=OUT/"source_snapshot"
FOLDS=[("2024-08-31","2024-09-01","2024-12-31"),("2024-12-31","2025-01-01","2025-05-31"),("2025-05-31","2025-06-01","2025-11-30"),("2025-11-30","2025-12-01","2026-08-31")]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    z=pd.read_csv(p,dtype={"code":str},low_memory=False)
    if "code" in z:z["code"]=z.code.str.zfill(6)
    return z

def pick(z,c):return z.sort_values(["date",c,"code"],ascending=[True,False,True]).groupby("date").head(1).copy()

def transform(raw,m,features):
    z=raw.copy()
    z["is_low_price"]=z.price.lt(m["_head_med_price"]).astype(float)
    z["is_small_cap"]=z.float_cap_log.lt(m["_head_med_cap"]).astype(float)
    z["is_tight_amplitude"]=z.amplitude.lt(5).astype(float)
    z["is_high_churn_risk"]=z.turnover.gt(.20).astype(float)
    z["is_crowded"]=(z.signal_amount.gt(1e9)&z.turnover.gt(.15)).astype(float)
    z["firm_seal_strength"]=z.amplitude.lt(3).astype(float)
    z["healthy_turnover"]=(z.turnover.ge(.08)&z.turnover.le(.20)).astype(float)
    z["theme_assisted_dragon"]=(z.theme_first_board_count.ge(2)&z.theme_zt_share.ge(.3)).astype(float)
    z["lone_wolf_penalty"]=z.is_isolated_lone_wolf*(1-z.unique_space)
    z["theme_diffusion_heating"]=(z.theme_first_board_expansion.gt(0)&z.theme_cluster_expansion.gt(0)).astype(float)
    cols={k:v for k,v in m.items() if not k.startswith("_")}
    z[features]=z[features].fillna(cols)
    return z

def main():
    assert not SNAP.exists(),"Use a new audit directory rather than overwrite a snapshot."
    files=[p for p in SOURCE.iterdir() if p.is_file()]
    hashes={p.name:sha(p) for p in files}
    SNAP.mkdir()
    for p in files:shutil.copy2(p,SNAP/p.name)
    assert all(sha(SNAP/k)==v for k,v in hashes.items())
    d=read(SNAP/"repaired_round6_dataset.csv");o=read(SNAP/"oof_repaired_predictions.csv")
    models=joblib.load(SNAP/"repaired_all_fold_models.joblib")
    meds=json.loads((SNAP/"feature_medians.json").read_text(encoding="utf-8"))
    features=json.loads((SNAP/"feature_order.json").read_text(encoding="utf-8"))
    heads={"3to4":("is_3to4","determined_date_3to4","score_3to4"),"space":("is_market_space_leader","determined_date_final","score_space_leader"),"5b":("is_3to5","det_date_3to5","score_height_5"),"gain46":("target_path_gain_46pct","det_date_gain46","score_path_gain46"),"reb18":("target_rebound_18pct","det_date_reb18","score_rebound18")}
    replay=[]
    for fold,(cut,start,end) in enumerate(FOLDS,1):
        raw=d[d.date.ge(start)&d.date.le(end)].sort_values(["date","code"])
        saved=o[o.fold.eq(fold)].sort_values(["date","code"])
        assert list(zip(raw.date,raw.code))==list(zip(saved.date,saved.code))
        for h,(target,dt,score) in heads.items():
            k=f"fold_{fold}_{h}";med=meds[k]
            train=d[d.date.le(cut)&d[target].notna()&d[dt].le(cut)].copy()
            assert np.isclose(train.price.median(),med["_head_med_price"])
            assert np.isclose(train.float_cap_log.median(),med["_head_med_cap"])
            # Recompute all column-imputation medians from the actual training membership.
            engtrain=transform(train,{"_head_med_price":med["_head_med_price"],"_head_med_cap":med["_head_med_cap"]},features)
            computed=engtrain[features].median()
            assert np.allclose(computed.values,np.array([med[f] for f in features]),equal_nan=True)
            z=transform(raw,med,features)
            delta=float(np.max(np.abs(models[k].predict_proba(z[features])[:,1]-saved[score].values)))
            assert delta<1e-12
            replay.append({"fold":fold,"head":h,"train_n":len(train),"score_max_difference":delta,"all_training_medians_verified":True})
    formula={"arm0":o.score_3to4,"arm1":.45*o.score_3to4+.25*o.score_height_5+.15*o.score_path_gain46+.15*o.score_rebound18,"arm2":.6*o.score_space_leader+.4*o.score_height_5}
    for k,col in [("arm0","score_arm0_r4"),("arm1","score_arm1_full"),("arm2","score_arm2_primary")]:assert np.allclose(formula[k],o[col])
    a=pick(o,"score_arm2_primary")
    a.to_csv(OUT/"arm2_top1.csv",index=False,encoding="utf-8-sig")
    base=a[a.base_historical_score.ge(.5)]
    c2=base[base.base_cohort_match_status.eq("COMPLETE_MATCH")]
    c3=c2[c2.next_open_gap.gt(0)]
    c4=c3[c3.trade_one_word_up.ne(1)]
    c5=c4[c4.execution_status.eq("inferred_daily_limit_touch_fill")]
    stages=[a,c2,c3,c4,c5]
    funnel=[{"n":len(z),"3to4_hits":int(z.is_3to4.sum())} for z in stages]
    ind=a[a.next_open_gap.gt(0)&a.trade_one_word_up.ne(1)&a.execution_status.eq("inferred_daily_limit_touch_fill")]
    old=read(ROOT/"research/model_upgrade_validation/dragon_round6_repair_codex_audit_20261027/all1851_label_recheck.csv")
    z=d.merge(old[["date","code","five_date_expected"]],on=["date","code"],validate="one_to_one")
    disagree=z[z.det_date_3to5.ne(z.five_date_expected)]
    disagree[["date","code","is_3to5","det_date_3to5","five_date_expected"]].to_csv(OUT/"remaining_five_date_disagreements.csv",index=False,encoding="utf-8-sig")
    r4=read(ROOT/"research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z/oof_round4_predictions.csv")
    b=pick(r4,"score_round4_composite").set_index("date");aa=a.set_index("date")
    paired={}
    for target in ["is_3to4","is_market_space_leader","is_3to5"]:
        win=int((aa[target].eq(1)&b[target].eq(0)).sum());loss=int((aa[target].eq(0)&b[target].eq(1)).sum())
        paired[target]={"new_only":win,"old_only":loss,"net":win-loss,"descriptive_p":float(binomtest(win,win+loss).pvalue)}
    manifest=json.loads((SNAP/"repaired_manifest.json").read_text(encoding="utf-8"))
    auc_actual=[float(roc_auc_score(g.is_market_space_leader,g.score_space_leader)) for _,g in o.groupby("fold")]
    auc_manifest=[v["auc_space"] for v in manifest["fold_metrics"]]
    risk_calls=[p.name for p in SNAP.glob("*.py") if "audit_three_board_candidate" in p.read_text(encoding="utf-8")]
    orig_audit=json.loads((ROOT/"research/model_upgrade_validation/dragon_round6_codex_audit_20261027T171732Z/audit.json").read_text(encoding="utf-8"))
    old_changed=[k for k,v in orig_audit["source_sha256"].items() if not (ROOT/k).exists() or sha(ROOT/k)!=v]
    unknown=d[d.is_industry_unknown.eq(1)]
    out={"formed_at_utc":datetime.now(timezone.utc).isoformat(),"fit_count":0,"orders":0,"new_forward_predictions":0,"snapshot_sha256":hashes,"original_baseline_files_changed_since_initial_audit":old_changed,"models":len(models),"all20_replay":replay,"feature_count":len(features),"full_counts":{k:int(a[k].sum()) for k in ["is_3to4","is_market_space_leader","is_3to5"]},"cumulative_with_added_complete_cohort_filter":funnel,"noncumulative206":{"n":len(ind),"BASE_lt50":int(ind.base_historical_score.lt(.5).sum()),"BASE_missing":int(ind.base_historical_score.isna().sum()),"BASE_missing_or_incomplete":int((ind.base_historical_score.isna()|ind.base_cohort_match_status.ne("COMPLETE_MATCH")).sum()),"low_and_incomplete_overlap":int((ind.base_historical_score.lt(.5)&ind.base_cohort_match_status.ne("COMPLETE_MATCH")).sum()),"qualified_complete_ge50":len(c5)},"positive_5b_dates_fixed":int(z[z.is_3to5.eq(1)].det_date_3to5.eq(z[z.is_3to5.eq(1)].five_date_expected).sum()),"remaining_5b_date_mismatch":len(disagree),"paired_vs_R4":paired,"manifest_auc_space":auc_manifest,"actual_auc_space":auc_actual,"six_tag_calls_found":risk_calls,"unknown_industry":{"n":len(unknown),"sole_leader":int(unknown.is_theme_sole_leader.eq(1).sum()),"lone_wolf":int(unknown.is_isolated_lone_wolf.eq(1).sum())},"limitations":["Current recipe changes more than preprocessing consistency; no fixed ablation or fresh forward evidence supplied.","Cumulative daily touch proxies are not actual fills and do not implement six-tag qualification.","Paired tests are descriptive, ignoring serial dependence, repeated research and multiple tests."]}
    assert all(sha(SOURCE/k)==v for k,v in hashes.items()),"Source changed during check."
    (OUT/"check.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ["snapshot_sha256","all20_replay"]},ensure_ascii=False,indent=2))
    print("All 20 heads and training medians independently replayed; max error",max(v["score_max_difference"] for v in replay))

if __name__=="__main__":main()
