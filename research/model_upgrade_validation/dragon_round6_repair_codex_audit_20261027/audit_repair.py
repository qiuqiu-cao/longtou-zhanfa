"""Read-only audit of Gemini repair. No fits and no writes to input artifacts."""
from pathlib import Path
from datetime import datetime, timezone
from bisect import bisect_right
import ast
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z"
OLD_AUDIT = ROOT / "research/model_upgrade_validation/dragon_round6_codex_audit_20261027T171732Z/audit.json"
FOLDS = [("2024-08-31", "2024-09-01", "2024-12-31"), ("2024-12-31", "2025-01-01", "2025-05-31"), ("2025-05-31", "2025-06-01", "2025-11-30"), ("2025-11-30", "2025-12-01", "2026-08-31")]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(1024*1024), b""):
            h.update(x)
    return h.hexdigest()


def read(p):
    d = pd.read_csv(p, dtype={"code": str}, low_memory=False)
    if "code" in d:
        d["code"] = d.code.str.zfill(6)
    return d


def choose(d, score, n=1):
    return d.sort_values(["date", score, "code"], ascending=[True, False, True]).groupby("date").head(n).copy()


def rates(d):
    return {"n": len(d), **{k: {"hits": int(d[k].sum()), "rate": float(d[k].mean()) if len(d) else None} for k in ["is_3to4", "is_3to5", "is_market_space_leader", "is_emotion_dragon_proxy"]}, "whole_day_one_word": int(d.trade_one_word_up.eq(1).sum())}


def paired(a, b):
    x = a.set_index("date").join(b.set_index("date"), lsuffix="_a", rsuffix="_b", how="inner")
    r = {"n": len(x), "different_top1": int(x.code_a.ne(x.code_b).sum())}
    for k in ["is_3to4", "is_3to5", "is_market_space_leader", "is_emotion_dragon_proxy"]:
        wins = int((x[k+"_a"].eq(1) & x[k+"_b"].eq(0)).sum())
        losses = int((x[k+"_a"].eq(0) & x[k+"_b"].eq(1)).sum())
        r[k] = {"a_only_hits": wins, "b_only_hits": losses, "net": wins-losses, "descriptive_p": float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.0}
    return r


def engineer(x, med):
    z = x.copy()
    z["is_low_price"] = z.price.le(med["med_price"]).astype(float)
    z["is_small_cap"] = z.float_cap_log.le(med["med_cap"]).astype(float)
    z["is_tight_amplitude"] = z.amplitude.le(3).astype(float)
    z["is_high_churn_risk"] = (z.turnover.gt(.18) & z.one_word_proxy.eq(0)).astype(float)
    z["is_crowded"] = z.same_height_rivals.ge(4).astype(float)
    z["firm_seal_strength"] = z.one_word_proxy*2 + z.amplitude.le(2).astype(float)-z.amplitude.gt(8).astype(float)
    z["healthy_turnover"] = (z.turnover.ge(.05)&z.turnover.le(.18)).astype(float)
    z["theme_assisted_dragon"] = (z.theme_first_board_count.ge(2)&z.is_theme_sole_leader.eq(1)).astype(float)
    z["lone_wolf_penalty"] = z.is_isolated_lone_wolf.eq(1).astype(float)
    z["theme_diffusion_heating"] = (z.theme_first_board_expansion.gt(0)&z.theme_zt_share.ge(.05)).astype(float)
    return z


def main():
    old_hashes = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))["source_sha256"]
    source_files = [p for p in SRC.iterdir() if p.is_file()]
    source_files += [ROOT/k for k in old_hashes if (ROOT/k).is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    old_changed = [k for k,v in old_hashes.items() if k not in before or before[k]!=v]
    d = read(SRC / "repaired_round6_dataset.csv")
    o = read(SRC / "oof_repaired_predictions.csv")
    ev = json.loads((SRC / "repaired_comprehensive_evaluation.json").read_text(encoding="utf-8"))
    assert len(d)==1851 and len(o)==1480 and o.date.nunique()==449
    assert not d.duplicated(["date", "code"]).any() and not o.duplicated(["date", "code"]).any()
    formula_checks = {
        "arm0": float(np.max(np.abs(o.score_arm0_r4-(.55*o.score_space_leader+.45*o.score_height_5)))),
        "arm1": float(np.max(np.abs(o.score_arm1_full-(.50*o.score_space_leader+.35*o.score_height_5+.15*o.score_path_gain46)))),
        "arm2": float(np.max(np.abs(o.score_arm2_primary-(.60*o.score_space_leader+.40*o.score_height_5)))),
    }
    assert max(formula_checks.values())<1e-12
    tops = {n: choose(o, c) for n,c in [("arm0", "score_arm0_r4"), ("arm1", "score_arm1_full"), ("arm2", "score_arm2_primary")]}
    comparisons = {n: {"all": rates(t), "not_one_word_not_proof_of_fill": rates(t[t.trade_one_word_up.ne(1)])} for n,t in tops.items()}
    json_arm_names = {"arm0": "arm0_repaired_r4_baseline", "arm1": "arm1_repaired_r6_full_horizon", "arm2": "arm2_repaired_dual_track_space_core"}
    for n,t in tops.items():
        s=ev["arms_full_449_days"][json_arm_names[n]]
        for k in ["is_3to4", "is_3to5", "is_market_space_leader"]:
            assert int(t[k].sum())==s[k]["hits"]
        t.to_csv(OUT/(n+"_top1.csv"), index=False, encoding="utf-8-sig")
    r4=read(ROOT/"research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z/oof_round4_predictions.csv")
    r4top=choose(r4, "score_round4_composite")
    comp=o.groupby("date").filter(lambda g:len(g)>=2)
    complete=comp[comp.base_cohort_match_status.eq("COMPLETE_MATCH")]
    ct,bt=choose(complete,"score_arm2_primary"),choose(complete,"base_historical_score")
    pair_results={"arm2_vs_repaired_arm0":paired(tops["arm2"],tops["arm0"]), "arm2_vs_originalR4":paired(tops["arm2"],r4top), "complete100_arm2_vs_BASE":paired(ct,bt), "limitation":"Descriptive paired-day tests; no correction for repeated research, serial dependence or multiple targets."}
    t=tops["arm2"]
    base=t[t.base_historical_score.ge(.5)]
    auc=base[base.next_open_gap.gt(0)]
    liquid=auc[auc.trade_one_word_up.ne(1)]
    touched=liquid[liquid.execution_status.eq("inferred_daily_limit_touch_fill")]
    noncumulative=t[t.next_open_gap.gt(0)&t.trade_one_word_up.ne(1)&t.execution_status.eq("inferred_daily_limit_touch_fill")]
    funnel={"signals":rates(t),"BASE_ge50":rates(base),"then_positive_open":rates(auc),"then_not_one_word":rates(liquid),"then_daily_touch_proxy":rates(touched),"reported_noncumulative_206":rates(noncumulative),"reported206_BASE_lt50":int(noncumulative.base_historical_score.lt(.5).sum()),"reported206_BASE_missing":int(noncumulative.base_historical_score.isna().sum()),"six_tags_not_verified":True,"actual_fills_verified":0}
    # Check persisted estimators on reconstructed raw test rows, for all sixteen heads.
    models=joblib.load(SRC/"repaired_all_fold_models.joblib")
    medians=json.loads((SRC/"feature_medians.json").read_text(encoding="utf-8"))
    features=json.loads((SRC/"feature_order.json").read_text(encoding="utf-8"))
    head_defs={"space":("is_market_space_leader","determined_date_final","score_space_leader"), "5b":("is_3to5","det_date_3to5","score_height_5"), "gain46":("target_path_gain_46pct","det_date_gain46","score_path_gain46"), "reb18":("target_rebound_18pct","det_date_reb18","score_rebound18")}
    persistence=[]; preprocessing=[]; maturity=[]
    for fold,(cut,start,end) in enumerate(FOLDS,1):
        sf=o[o.fold.eq(fold)].sort_values(["date","code"])
        raw=d[d.date.ge(start)&d.date.le(end)].sort_values(["date","code"])
        assert list(zip(sf.date,sf.code))==list(zip(raw.date,raw.code))
        rebuilt=engineer(raw,medians[f"fold_{fold}"])
        assert np.allclose(rebuilt[features],sf[features],equal_nan=True)
        for head,(target,dt,score) in head_defs.items():
            m=models[f"fold_{fold}_{head}"]
            pred=m.predict_proba(rebuilt[features])[:,1]
            diff=float(np.max(np.abs(pred-sf[score].values)))
            assert diff<1e-12
            persistence.append({"fold":fold,"head":head,"type":type(m).__name__,"features":int(m.n_features_in_),"max_score_difference":diff})
            train=d[d.date.le(cut)&d[target].notna()&d[dt].le(cut)]
            own_med={"med_price":float(train.price.median()),"med_cap":float(train.float_cap_log.median())}
            own=engineer(raw,own_med)
            changed=((own.is_low_price.values!=rebuilt.is_low_price.values)|(own.is_small_cap.values!=rebuilt.is_small_cap.values))
            preprocessing.append({"fold":fold,"head":head,"training_medians":own_med,"inference_medians":medians[f"fold_{fold}"],"test_rows_using_different_binary_feature_definition":int(changed.sum())})
            maturity.append({"fold":fold,"head":head,"train_n":len(train),"label_missing":int(train[target].isna().sum()),"maturity_missing":int(train[dt].isna().sum())})
    # Recompute official-session target timing against the OHLC archive.
    sessions=json.loads((ROOT/"data/v279/exchange_calendar_audit_20261002_v1/calendar.json").read_text(encoding="utf-8"))["sessions"]
    mp={v:i for i,v in enumerate(sessions)}
    bars=read(ROOT/"data/v3/imported/daily-bars.csv")[["date","code","high","close"]].sort_values(["code","date"])
    assert not bars.duplicated(["date","code"]).any()
    bycode={c:g for c,g in bars.groupby("code")}
    rows=[]
    for r in d.itertuples():
        i=bisect_right(sessions,r.date); expected=sessions[i:i+10]
        b=bycode.get(r.code)
        sub=b[b.date.isin(expected)] if b is not None else bars.head(0)
        full=len(expected)==10 and sub.date.tolist()==expected
        g_hits=sub[sub.high.ge(r.price*1.464)]
        r_hits=sub[sub.high.ge(r.price*1.18)]
        g_target=1 if r.terminal_board>=7 or len(g_hits) else (0 if full else np.nan)
        g_date=r.determined_date_final if r.terminal_board>=7 else (g_hits.date.iloc[0] if len(g_hits) else (expected[-1] if full else None))
        reb_target=0 if r.terminal_board>4 else (1 if len(r_hits) else (0 if full else np.nan))
        reb_date=r.determined_date_final if r.terminal_board>4 else (r_hits.date.iloc[0] if len(r_hits) else (expected[-1] if full else None))
        y5date=sessions[mp[r.date]+2] if r.is_3to5==1 else r.determined_date_final
        rows.append({"date":r.date,"code":r.code,"official_complete_expected":full,"official_complete_saved":r.official10_complete,"gain_target_expected":g_target,"gain_target_saved":r.target_path_gain_46pct,"gain_date_expected":g_date,"gain_date_saved":r.det_date_gain46,"reb_target_expected":reb_target,"reb_target_saved":r.target_rebound_18pct,"reb_date_expected":reb_date,"reb_date_saved":r.det_date_reb18,"five_date_expected":y5date,"five_date_saved":r.det_date_3to5})
    diag=pd.DataFrame(rows)
    diag.to_csv(OUT/"all1851_label_recheck.csv", index=False,encoding="utf-8-sig")
    def mismatch(a,b):
        return int((~(diag[a].eq(diag[b])|(diag[a].isna()&diag[b].isna()))).sum())
    label_checks={"official_complete_disagreements":mismatch("official_complete_expected","official_complete_saved"),"gain_label_disagreements":mismatch("gain_target_expected","gain_target_saved"),"gain_date_disagreements":mismatch("gain_date_expected","gain_date_saved"),"reb_label_disagreements":mismatch("reb_target_expected","reb_target_saved"),"reb_date_disagreements":mismatch("reb_date_expected","reb_date_saved"),"five_date_disagreements":mismatch("five_date_expected","five_date_saved"),"gain_unknown":int(d.target_path_gain_46pct.isna().sum()),"reb_unknown":int(d.target_rebound_18pct.isna().sum()),"official_incomplete":int((~d.official10_complete).sum()),"warning":"Reproducing implemented proxies does not validate their reported names or true dragon identity."}
    unknown=d[d.is_industry_unknown.eq(1)]
    unknown_info={"n":len(unknown),"positive_first_board_assists":int(unknown.theme_first_board_count.gt(0).sum()),"labeled_sole_theme_leader":int(unknown.is_theme_sole_leader.eq(1).sum()),"labeled_lone_wolf":int(unknown.is_isolated_lone_wolf.eq(1).sum())}
    regime=read(SRC/"repaired_regime_dataset.csv")
    expected_final=d.groupby("date").determined_date_final.max()
    assert regime.set_index("date").determined_date_regime.equals(expected_final)
    reg_maturity=[{"cutoff":cut,"train_days":len(regime[regime.date.le(cut)&regime.determined_date_regime.le(cut)]),"unmatured_days_excluded":len(regime[regime.date.le(cut)&regime.determined_date_regime.gt(cut)])} for cut,_,_ in FOLDS]
    negative_window_info={"top1_n":len(t),"official_incomplete":int((~t.official10_complete).sum()),"missing_return":int(t.forward_10d_close_return.isna().sum()),"negative_recorded_last_close":int(t.forward_10d_close_return.lt(0).sum()),"note":"Recorded last close is not necessarily official T+10 when official10_complete is false; NaN comparisons must not count as nonloss."}
    after={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    assert before==after,"Source changed during read-only audit."
    out={"formed_at_utc":datetime.now(timezone.utc).isoformat(),"fit_count":0,"orders":0,"new_forward_predictions":0,"source_unchanged_during_audit":True,"original_files_changed_since_previous_audit":old_changed,"data_rows":len(d),"oof_rows":len(o),"oof_days":o.date.nunique(),"formula_max_difference":formula_checks,"saved_evaluation_counts_reproduced":True,"comparisons":comparisons,"paired_comparisons":pair_results,"execution_funnel":funnel,"persistence_all16":persistence,"per_head_preprocessing":preprocessing,"training_maturity":maturity,"regime_maturity":reg_maturity,"label_recheck":label_checks,"unknown_industry":unknown_info,"negative_path_reporting":negative_window_info,"model_keys":list(models),"feature_count":len(features),"source_sha256":before}
    (OUT/"audit.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:out[k] for k in ["formed_at_utc","original_files_changed_since_previous_audit","comparisons","paired_comparisons","execution_funnel","label_recheck","unknown_industry","negative_path_reporting","feature_count"]},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
