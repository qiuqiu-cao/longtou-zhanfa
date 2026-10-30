"""Read-only review: fixed saved-score comparisons, model replay and rule-boundary probes.

Never fit estimators, run the original evaluator, or modify original artifacts.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
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
R7=ROOT/"research/model_upgrade_validation/dragon_round7_capacity_dragon_20261028T094500Z"
R6=ROOT/"research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
FOLDS=[("2024-08-31","2024-09-01","2024-12-31"),("2024-12-31","2025-01-01","2025-05-31"),("2025-05-31","2025-06-01","2025-11-30"),("2025-11-30","2025-12-01","2026-08-31")]
HEADS={"3to4":("is_3to4","determined_date_3to4","score_3to4"),"space":("is_market_space_leader","determined_date_final","score_space_leader"),"5b":("is_3to5","det_date_3to5","score_height_5"),"gain46":("target_path_gain_46pct","det_date_gain46","score_path_gain46"),"reb18":("target_rebound_18pct","det_date_reb18","score_rebound18")}


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(p):
    x=pd.read_csv(p,dtype={"code":str},low_memory=False)
    if "code" in x:x["code"]=x.code.str.zfill(6)
    return x


def pick(x,col):return x.sort_values(["date",col,"code"],ascending=[True,False,True]).groupby("date").head(1).copy()


def stats(x):
    out={"n":len(x)}
    for t in ["is_3to4","is_market_space_leader","is_3to5"]:
        out[t]={"hits":int(x[t].sum()),"rate":float(x[t].mean()) if len(x) else None}
    out["next_day_whole_day_one_price_up"]=int(x.trade_one_word_up.eq(1).sum())
    return out


def paired(a,b):
    z=a.set_index("date").join(b.set_index("date"),lsuffix="_a",rsuffix="_b")
    out={"n":len(z),"top1_changed":int(z.code_a.ne(z.code_b).sum())}
    for t in ["is_3to4","is_market_space_leader","is_3to5"]:
        win=int((z[t+"_a"].eq(1)&z[t+"_b"].eq(0)).sum())
        loss=int((z[t+"_a"].eq(0)&z[t+"_b"].eq(1)).sum())
        out[t]={"a_only":win,"b_only":loss,"net":win-loss,"descriptive_p":float(binomtest(win,win+loss).pvalue) if win+loss else 1.0}
    return out


def transform(x,m,features):
    z=x.copy()
    z["is_low_price"]=z.price.lt(m["_head_med_price"]).astype(float)
    z["is_capacity_volume_leader"]=(z.turnover_cohort_rank.eq(1)|z.is_cohort_volume_leader.eq(1)).astype(float)
    z["is_tight_amplitude"]=z.amplitude.lt(5).astype(float)
    z["is_high_churn_risk"]=(z.turnover.gt(.25)&z.previous_limit_tail_loss.gt(0)).astype(float)
    z["is_capacity_dragon"]=(z.signal_amount.ge(8e8)&z.turnover.ge(.12)&z.turnover.le(.28)&z.amplitude.lt(5)).astype(float)
    z["firm_seal_strength"]=z.amplitude.lt(3).astype(float)
    z["healthy_turnover"]=(z.turnover.ge(.10)&z.turnover.le(.28)).astype(float)
    z["theme_assisted_dragon"]=(z.theme_first_board_count.ge(2)&z.theme_zt_share.ge(.3)).astype(float)
    z["lone_wolf_penalty"]=z.is_isolated_lone_wolf*(1-z.unique_space)
    z["theme_diffusion_heating"]=(z.theme_first_board_expansion.gt(0)&z.theme_cluster_expansion.gt(0)).astype(float)
    z[features]=z[features].fillna({k:v for k,v in m.items() if not k.startswith("_")})
    return z


def main():
    snap=OUT/"source_snapshot"
    assert not snap.exists(),"Do not overwrite an existing review snapshot."
    src=[p for p in R7.iterdir() if p.is_file()]
    src += [R6/"oof_repaired_predictions.csv",R6/"repaired_round6_dataset.csv",R6/"repaired_comprehensive_evaluation.json",ROOT/"audit_three_board_traps.py",ROOT/"AGENTS.md",ROOT/"leader_system_v3.py",ROOT/"compute_round3_dataset.py",ROOT/"relay_review_workbench.py"]
    before={str(p.relative_to(ROOT)):sha(p) for p in src}
    snap.mkdir()
    for p in src:
        dest=snap/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        assert sha(dest)==before[str(p.relative_to(ROOT))]
    s7=snap/R7.relative_to(ROOT);s6=snap/R6.relative_to(ROOT)
    d=read(s7/"round7_dataset.csv");o=read(s7/"oof_round7_predictions.csv");old=read(s6/"oof_repaired_predictions.csv")
    assert len(d)==1851 and len(o)==1480 and o.date.nunique()==449
    assert not o.duplicated(["date","code"]).any()
    assert sha(s7/"round7_dataset.csv")==sha(s6/"repaired_round6_dataset.csv")
    manifest=json.loads((s7/"round7_manifest.json").read_text(encoding="utf-8"))
    manifest_hashes={k:sha(s7/k)==v for k,v in manifest["artifacts_sha256"].items()}
    assert all(manifest_hashes.values())
    models=joblib.load(s7/"round7_all_fold_models.joblib")
    meds=json.loads((s7/"feature_medians.json").read_text(encoding="utf-8"))
    features=json.loads((s7/"feature_order.json").read_text(encoding="utf-8"))
    replays=[]
    for fold,(cut,start,end) in enumerate(FOLDS,1):
        raw=d[d.date.ge(start)&d.date.le(end)].sort_values(["date","code"])
        saved=o[o.fold.eq(fold)].sort_values(["date","code"])
        assert list(zip(raw.date,raw.code))==list(zip(saved.date,saved.code))
        for h,(target,dt,score) in HEADS.items():
            key=f"fold_{fold}_{h}";m=meds[key]
            train=d[d.date.le(cut)&d[target].notna()&d[dt].le(cut)]
            assert np.isclose(train.price.median(),m["_head_med_price"])
            assert np.isclose(train.float_cap_log.median(),m["_head_med_cap"])
            trainf=transform(train,{"_head_med_price":m["_head_med_price"],"_head_med_cap":m["_head_med_cap"]},features)
            assert np.allclose(trainf[features].median().values,[m[f] for f in features],equal_nan=True)
            z=transform(raw,m,features)
            delta=float(np.max(np.abs(models[key].predict_proba(z[features])[:,1]-saved[score].values)))
            assert delta<1e-12
            known=saved[target].notna()
            auc=float(roc_auc_score(saved.loc[known,target],saved.loc[known,score]))
            manifest_key={"3to4":"auc_3to4","space":"auc_space","5b":"auc_5b","gain46":"auc_gain46","reb18":"auc_reb18"}[h]
            assert np.isclose(auc,manifest["fold_metrics"][fold-1][manifest_key])
            replays.append({"fold":fold,"head":h,"n_train":len(train),"max_prediction_error":delta,"medians_match_training":True,"auc":auc})
    formula_checks={"arm0":np.allclose(o.score_arm0_r4,o.score_3to4),"arm1":np.allclose(o.score_arm1_full,.45*o.score_3to4+.25*o.score_height_5+.15*o.score_path_gain46+.15*o.score_rebound18),"arm2":np.allclose(o.score_arm2_primary,.6*o.score_space_leader+.4*o.score_height_5),"arm3":np.allclose(o.score_arm3_capacity,.4*o.score_space_leader+.35*o.score_height_5+.25*o.score_3to4)}
    assert all(formula_checks.values())
    # Keep identical policy eligibility across both models, rather than conflate rule/model changes.
    policy=o[["date","code","six_tag_veto"]].rename(columns={"six_tag_veto":"r7_policy_veto"})
    old=old.merge(policy,on=["date","code"],validate="one_to_one")
    o=o.merge(policy,on=["date","code"],validate="one_to_one")
    comparison={};tops={}
    for name,frame,col in [("R6_arm2",old,"score_arm2_primary"),("R7_arm2",o,"score_arm2_primary"),("R7_arm3",o,"score_arm3_capacity")]:
        t=pick(frame,col);tops[name]=t
        touched=t[t.next_open_gap.gt(0)&t.trade_one_word_up.ne(1)&t.execution_status.eq("inferred_daily_limit_touch_fill")]
        old_gate=touched[touched.base_historical_score.ge(.5)&touched.base_cohort_match_status.eq("COMPLETE_MATCH")]
        new_gate=touched[~touched.r7_policy_veto]
        complete=frame.groupby("date").filter(lambda g:len(g)>=2)
        complete=complete[complete.base_cohort_match_status.eq("COMPLETE_MATCH")]
        comparison[name]={"all_top1":stats(t),"not_one_price_not_fill":stats(t[t.trade_one_word_up.ne(1)]),"old_gate_identical_policy":stats(old_gate),"new_gate_identical_policy":stats(new_gate),"complete_competitive100":stats(pick(complete,col))}
        t.to_csv(OUT/(name+"_top1.csv"),index=False,encoding="utf-8-sig")
    paired_results={"R7_arm3_vs_R6_arm2":paired(tops["R7_arm3"],tops["R6_arm2"]),"R7_arm3_vs_R7_arm2":paired(tops["R7_arm3"],tops["R7_arm2"])}
    t=tops["R7_arm3"];s1=t[~t.six_tag_veto];s2=s1[s1.next_open_gap.gt(0)];s3=s2[s2.trade_one_word_up.ne(1)];s4=s3[s3.execution_status.eq("inferred_daily_limit_touch_fill")]
    funnel=[stats(x) for x in [t,s1,s2,s3,s4]]
    saved_eval=json.loads((s7/"round7_comprehensive_evaluation.json").read_text(encoding="utf-8"))
    assert [(x["n"],x["is_3to4"]["hits"]) for x in funnel]==[(x["n"],x["3to4_hits"]) for x in saved_eval["funnel_arm3_capacity_dragon"]]
    s4.to_csv(OUT/"R7_arm3_S4_daily_proxy.csv",index=False,encoding="utf-8-sig")
    # Boundary probes are deliberately synthetic, not quotations of a real stock's historical facts.
    spec=importlib.util.spec_from_file_location("r7_rule_snapshot",s7/"audit_three_board_traps.py")
    auditmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditmod)
    common={"code":"SYNTHETIC","name":"合成边界样本","board_count":3,"is_theme_leader":True,"turnover_rate":20.57,"amount":1e9,"base_score":27.7,"history_had_limit_down_after_top":False,"historical_board_max":6}
    cases=[("early_and_tail_unknown",{}),("leader_early_but_no_late_add",{"first_seal_time":"09:36","late_order_action":"STABLE"}),("nonleader_early_ADD",{"is_theme_leader":False,"first_seal_time":"09:36","late_order_action":"ADD"}),("known_prior_nuclear_but_historical_max6",{"first_seal_time":"09:36","late_order_action":"ADD","history_had_limit_down_after_top":True}),("four_board_applicability",{"board_count":4,"first_seal_time":"09:36","late_order_action":"ADD"}),("late_cancel_veto",{"first_seal_time":"09:36","late_order_action":"CANCEL"})]
    probes={name:auditmod.audit_three_board_candidate(**(common|changes)) for name,changes in cases}
    assert not probes["early_and_tail_unknown"]["veto"]
    assert not probes["leader_early_but_no_late_add"]["veto"]
    assert not probes["four_board_applicability"]["applicable"]
    assert probes["late_cancel_veto"]["veto"]
    (OUT/"synthetic_rule_probes.json").write_text(json.dumps(probes,ensure_ascii=False,indent=2),encoding="utf-8")
    # Data semantics: rank 1 is lowest turnover, not volume leadership.
    low_turnover=o.turnover_cohort_rank.eq(1)&o.is_cohort_volume_leader.ne(1)
    risk_positive=o.is_high_churn_risk.eq(1)
    tag_text=o.six_tag_tags.fillna("")
    feature_semantics={"capacity_flag_rows":int(o.is_capacity_dragon.sum()),"capacity_flag_3to4_hits":int(o.loc[o.is_capacity_dragon.eq(1),"is_3to4"].sum()),"low_turnover_only_mislabeled_volume_leader":int(low_turnover.sum()),"churn_risk_flag_rows":int(risk_positive.sum()),"churn_risk_market_tail_positive_rows":int((risk_positive&o.previous_limit_tail_loss.gt(0)).sum()),"late_ADD_tags_in_saved_oof":int(tag_text.str.contains("尾盘加单").sum()),"late_CANCEL_tags_in_saved_oof":int(tag_text.str.contains("尾盘撤单").sum()),"none_or_placeholder_first_seal_rows":int((o.first_seal_minute.isna()|o.first_seal_minute.eq(0)).sum()),"S4_capacity_flag_count":int(s4.is_capacity_dragon.sum()),"S4_core_dragon_tag_count":int(s4.six_tag_tags.fillna("").str.contains("换手核心真龙").sum()),"S4_BASE_lt50":int(s4.base_historical_score.lt(.5).sum()),"S4_BASE_missing":int(s4.base_historical_score.isna().sum()),"unknown_industry_labeled_sole_leader":int((o.is_industry_unknown.eq(1)&o.is_theme_sole_leader.eq(1)).sum())}
    allbase=o[o.base_historical_score.notna()]
    full=o[o.base_cohort_match_status.eq("COMPLETE_MATCH")]
    comp=full.groupby("date").filter(lambda g:len(g)>=2)
    base_comparison={"partial412_not_same_pool":stats(pick(allbase,"base_historical_score")),"complete185_BASE":stats(pick(full,"base_historical_score")),"complete185_R7":stats(pick(full,"score_arm3_capacity")),"competitive100_BASE":stats(pick(comp,"base_historical_score")),"competitive100_R7":stats(pick(comp,"score_arm3_capacity"))}
    initial=json.loads((ROOT/"research/model_upgrade_validation/dragon_round6_codex_audit_20261027T171732Z/audit.json").read_text(encoding="utf-8"))
    prod_hashes={k:v for k,v in initial["source_sha256"].items() if "/fixed_20261007T112017Z/models/relay/BASE/" in k.replace("\\","/")}
    production_unchanged=all((ROOT/k).exists() and sha(ROOT/k)==v for k,v in prod_hashes.items())
    out={"formed_at_utc":datetime.now(timezone.utc).isoformat(),"fits_performed_by_this_review":0,"new_forward_predictions":0,"orders":0,"dataset_rows":len(d),"oof_rows":len(o),"days":o.date.nunique(),"dataset_identical_to_latest_R6":True,"dataset_max_date":d.date.max(),"production_BASE_bound_files":len(prod_hashes),"production_BASE_unchanged":production_unchanged,"manifest_hashes_match":manifest_hashes,"all20_model_replay":replays,"score_formulas_verified":formula_checks,"comparison_fixed_policies":comparison,"descriptive_paired_comparisons":paired_results,"R7_arm3_funnel":funnel,"feature_and_evidence_semantics":feature_semantics,"BASE_same_pool_comparison":base_comparison,"rule_probe_results":probes,"production_workbench_new_rules_mentioned":any(t in (snap/"relay_review_workbench.py").read_text(encoding="utf-8") for t in ["late_order_action","seal_delta","is_turnover_dragon"]),"source_sha256":before,"limitations":["Reproduction does not certify supplied flags as real historical facts.","Early seal and late order action are explicitly None at every historical evaluator call.","No real contemporary 2026-10-27/28 or four-board performance is tested by the saved all-three-board dataset.","No new threshold or model selection performed in this review; comparisons are already-seen historical diagnostics.","Saved cohorts retain legacy coverage, unknowns and censored-label limitations."]}
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in src},"Source changed during review; snapshot needed for qualification."
    (OUT/"review.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ["source_sha256","all20_model_replay","rule_probe_results"]},ensure_ascii=False,indent=2))
    print("20 model maximum replay error",max(v["max_prediction_error"] for v in replays))

if __name__=="__main__":main()
