"""Read-only follow-up audit of the repaired R7 snapshot; no estimator fitting."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
R7=ROOT/"research/model_upgrade_validation/dragon_round7_capacity_dragon_20261028T094500Z"
PRIOR=ROOT/"research/model_upgrade_validation/dragon_round7_codex_review_20261028T100106Z/review.json"
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


def transform(x,m,features):
    z=x.copy()
    z["is_low_price"]=z.price.lt(m["_head_med_price"]).astype(float)
    z["is_capacity_volume_leader"]=(z.is_cohort_volume_leader.eq(1)|z.amount_cohort_rank.eq(1)|z.signal_amount.ge(8e8)).astype(float)
    z["is_tight_amplitude"]=z.amplitude.lt(5).astype(float)
    z["is_high_churn_risk"]=(z.turnover.gt(.25)&(z.bomb_count.gt(0)|z.amplitude.gt(6))).astype(float)
    z["is_capacity_dragon"]=(z.signal_amount.ge(8e8)&z.turnover.ge(.12)&z.turnover.le(.28)&z.amplitude.lt(5)).astype(float)
    z["firm_seal_strength"]=z.amplitude.lt(3).astype(float)
    z["healthy_turnover"]=(z.turnover.ge(.10)&z.turnover.le(.28)).astype(float)
    z["theme_assisted_dragon"]=(z.theme_first_board_count.ge(2)&z.theme_zt_share.ge(.3)).astype(float)
    z["lone_wolf_penalty"]=z.is_isolated_lone_wolf*(1-z.unique_space)
    z["theme_diffusion_heating"]=(z.theme_first_board_expansion.gt(0)&z.theme_cluster_expansion.gt(0)).astype(float)
    z[features]=z[features].fillna({k:v for k,v in m.items() if not k.startswith("_")})
    return z


def pick(x,c):return x.sort_values(["date",c,"code"],ascending=[True,False,True]).groupby("date").head(1).copy()


def main():
    snap=OUT/"source_snapshot"
    assert not snap.exists(),"Preserve prior checks; use a fresh snapshot directory."
    raw_minute_rel=[f"results/daily_review_20261027T0928Z/minutes/{v}.response" for v in ["sz000678","sh603528","sh603028"]]
    src=[p for p in R7.iterdir() if p.is_file()]+[ROOT/f for f in raw_minute_rel]+[ROOT/"audit_three_board_traps.py",ROOT/"relay_review_workbench.py",ROOT/"AGENTS.md"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in src}
    snap.mkdir()
    for p in src:
        dest=snap/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(dest)==hashes[str(p.relative_to(ROOT))]
    ss=snap/R7.relative_to(ROOT)
    prior=json.loads(PRIOR.read_text(encoding="utf-8"))
    changed=[k for k,v in hashes.items() if k in prior["source_sha256"] and prior["source_sha256"][k]!=v]
    d=read(ss/"round7_dataset.csv");o=read(ss/"oof_round7_predictions.csv")
    features=json.loads((ss/"feature_order.json").read_text(encoding="utf-8"))
    meds=json.loads((ss/"feature_medians.json").read_text(encoding="utf-8"))
    models=joblib.load(ss/"round7_all_fold_models.joblib")
    manifest=json.loads((ss/"round7_manifest.json").read_text(encoding="utf-8"))
    manifest_checks={k:{"declared":v,"observed":sha(ss/k),"match":sha(ss/k)==v} for k,v in manifest["artifacts_sha256"].items()}
    replay=[]
    for fold,(cut,start,end) in enumerate(FOLDS,1):
        raw=d[d.date.ge(start)&d.date.le(end)].sort_values(["date","code"])
        saved=o[o.fold.eq(fold)].sort_values(["date","code"])
        assert list(zip(raw.date,raw.code))==list(zip(saved.date,saved.code))
        for head,(target,dt,score) in HEADS.items():
            key=f"fold_{fold}_{head}";m=meds[key]
            train=d[d.date.le(cut)&d[target].notna()&d[dt].le(cut)]
            assert np.isclose(train.price.median(),m["_head_med_price"])
            assert np.isclose(train.float_cap_log.median(),m["_head_med_cap"])
            mt=transform(train,{"_head_med_price":m["_head_med_price"],"_head_med_cap":m["_head_med_cap"]},features)
            assert np.allclose(mt[features].median().values,[m[k] for k in features],equal_nan=True)
            z=transform(raw,m,features)
            diff=float(np.max(np.abs(models[key].predict_proba(z[features])[:,1]-saved[score].values)))
            assert diff<1e-12
            known=saved[target].notna()
            auc=float(roc_auc_score(saved.loc[known,target],saved.loc[known,score]))
            replay.append({"fold":fold,"head":head,"max_difference":diff,"auc":auc,"training_medians_match":True})
    spec=importlib.util.spec_from_file_location("r7_current_snapshot_rule",ss/"audit_three_board_traps.py")
    rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
    common={"code":"SYNTHETIC","name":"合成边界样本","board_count":4,"is_theme_leader":True,"first_seal_time":"09:36","turnover_rate":20.57,"amount":1e9,"base_score":27.7,"historical_board_max":6,"history_had_limit_down_after_top":False}
    cases=[("missing_early_and_tail",{"first_seal_time":None}),
           ("leader_without_late_add",{}),
           ("valid_numeric_add_and_four_board",{"seal_order_1430":1e8,"seal_order_1500":1.3e8}),
           ("plain_ADD_without_numbers",{"late_order_action":"ADD"}),
           ("zero_denominator_plus_plain_ADD",{"seal_order_1430":0,"seal_order_1500":1e8,"late_order_action":"ADD"}),
           ("infinite_delta",{"seal_order_delta":float("inf")}),
           ("blank_first_time",{"first_seal_time":"","seal_order_delta":1.3}),
           ("prior_limitdown_flag_max6",{"history_had_limit_down_after_top":True,"seal_order_delta":1.3}),
           ("high_BASE_unknown_timing",{"first_seal_time":None,"base_score":70}),
           ("numeric_cancel",{"seal_order_delta":.6,"base_score":70})]
    probes={name:rule.audit_three_board_candidate(**(common|changes)) for name,changes in cases}
    assert probes["missing_early_and_tail"]["veto"] and probes["leader_without_late_add"]["veto"]
    assert probes["valid_numeric_add_and_four_board"]["applicable"] and probes["valid_numeric_add_and_four_board"]["is_turnover_dragon_exempted"]
    assert probes["numeric_cancel"]["veto"]
    # Extract real-name demonstrations as literal inputs, not verified market evidence.
    tree=ast.parse((ss/"evaluate_round7_capacity.py").read_text(encoding="utf-8"))
    demo_inputs=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="cases_20261027" for t in node.targets):
            for call in node.value.elts:demo_inputs.append({kw.arg:ast.literal_eval(kw.value) for kw in call.keywords})
    demos=[{"inputs":x,"output":rule.audit_three_board_candidate(**x),"source_qualification":"Literal example arguments; no raw snapshot path, source hash, acquisition time or outcome label bound."} for x in demo_inputs]
    minute_checks=[]
    for rel in raw_minute_rel:
        p=snap/rel;content=json.loads(p.read_text(encoding="utf-8"));code=p.stem
        x=content["data"][code];qt=x["qt"][code];rows=x["data"]["data"]
        minute_checks.append({"source_file":rel,"sha256":sha(p),"code":code,"records_1430_1500":[s for s in rows if s.startswith(("1430 ","1500 "))],"schema":"minute, price, cumulative volume, cumulative amount; no historical bid queue per minute","quote_time":qt[30],"quote_bid1_price":qt[9],"quote_bid1_lots":qt[10],"quote_bid1_amount_if_100shares_per_lot":float(qt[9])*float(qt[10])*100})
    top=pick(o,"score_arm3_capacity")
    s1=top[~top.six_tag_veto];s2=s1[s1.next_open_gap.gt(0)];s3=s2[s2.trade_one_word_up.ne(1)];s4=s3[s3.execution_status.eq("inferred_daily_limit_touch_fill")]
    funnel=[{"n":len(x),"3to4_hits":int(x.is_3to4.sum()),"space_hits":int(x.is_market_space_leader.sum()),"five_hits":int(x.is_3to5.sum())} for x in [top,s1,s2,s3,s4]]
    comp=o.groupby("date").filter(lambda g:len(g)>=2);comp=comp[comp.base_cohort_match_status.eq("COMPLETE_MATCH")]
    fixed100={name:{"n":len(t),"3to4_hits":int(t.is_3to4.sum())} for name,t in [("R7",pick(comp,"score_arm3_capacity")),("BASE",pick(comp,"base_historical_score"))]}
    eval_text=(ss/"evaluate_round7_capacity.py").read_text(encoding="utf-8")
    notes={"historical_rows":len(o),"all_three_board":bool(o.boards.eq(3).all()),"first_seal_placeholder_zero_mapped_to0930":int(o.first_seal_minute.eq(0).sum()),"first_seal_missing":int(o.first_seal_minute.isna().sum()),"bomb_count_positive":int(o.bomb_count.gt(0).sum()),"bomb_count_missing":int(o.bomb_count.isna().sum()),"saved_late_add_tags":int(o.six_tag_tags.fillna("").str.contains("尾盘加单").sum()),"saved_late_cancel_tags":int(o.six_tag_tags.fillna("").str.contains("尾盘撤单").sum()),"saved_exemptions_from_reasons":int(o.six_tag_reasons.fillna("").str.contains("豁免机械").sum()),"eval_still_passes_late_action_none":"late_order_action=None" in eval_text,"capacity_rank_misused_turnover_field_removed":"turnover_cohort_rank" not in (ss/"train_and_evaluate_round7_capacity.py").read_text(encoding="utf-8").split("def engineer_features_for_head",1)[1].split("def main",1)[0],"production_workbench_no_new_rule_calls":not any(k in (snap/"relay_review_workbench.py").read_text(encoding="utf-8") for k in ["seal_order_delta","late_order_action","audit_three_board_candidate"])}
    src_after={str(p.relative_to(ROOT)):sha(p) for p in src}
    assert src_after==hashes,"Files changed during follow-up."
    out={"formed_at_utc":datetime.now(timezone.utc).isoformat(),"fits_performed":0,"new_forward_predictions":0,"orders":0,"changed_since_prior_review":changed,"source_sha256":hashes,"manifest_checks":manifest_checks,"all20_replay":replay,"rule_boundary_probes":probes,"hardcoded_case_demonstrations":demos,"available_minute_source_qualification":minute_checks,"R7_arm3_funnel":funnel,"competitive100":fixed100,"evidence_and_integration_notes":notes,"limitations":["Handwritten cases test rule behavior but do not establish historical capture, correct values or prospective success.","A queue decrease alone does not distinguish executions from cancellations; a net increase does not identify the account or guarantee locking.","A ratio using 15:00 data cannot be evaluated as an observed 14:55 input.","Snapshot and prediction replay validate file/compute consistency, not trading effectiveness."]}
    # JSON standard has no Infinity; keep malformed-input probe result explicit and serializable.
    def clean(x):
        if isinstance(x,float) and not np.isfinite(x):return str(x)
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    (OUT/"followup.json").write_text(json.dumps(clean(out),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"formed_at_utc":out["formed_at_utc"],"manifest_checks":manifest_checks,"20_model_max_replay_error":max(r["max_difference"] for r in replay),"evidence_notes":notes,"funnel":funnel,"competitive100":fixed100,"synthetic_probes":{k:{f:v.get(f) for f in ["applicable","veto","is_turnover_dragon_exempted"]} for k,v in probes.items()}},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
