"""Separate first-to-second and second-to-third EOD research models.

Only adjacent official-session closes define promotion. A second-board recheck
uses newly available second-board facts and never alters the first-board call.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score

import first_board_leader_model as base

VERSION="first-second-stage-20261020"
TARGET="promotion_next_LABEL_ONLY"
EXTRA=["gene_max_streak_60","gene_limits_60","gene_relay_rate_60",
       "turnover_ratio_5","prior_compression_5","two_session_turnover_sum",
       "previous_one_price","opened_after_one_price","cycle_height_change",
       "cycle_height_break_20","cycle_anchor_breaks","cycle_anchor_collapses",
       "cycle_high_survival","cycle_first_expansion_5","cycle_first_relay_ratio",
       "cycle_top_one_price_share","stage_pool_size","stage_one_price_share"]
VARIANTS={"STAGE_BASE":base.FEATURES,"STAGE_CYCLE_GENE":base.FEATURES+EXTRA}
MARKET=[x for x in base.FEATURES if x.startswith("market_")]+["previous_first_promotion","previous_relay_promotion","previous_top_return","previous_top_broken"]
SOURCES=base.ROOT/"research/first_second_stage_20261020_sources"


def build_rows(inputs=base.INPUTS,feature_cutoff=None):
    first,bars,sessions,audit=base.build_dataset(inputs,feature_cutoff=feature_cutoff)
    g=bars.groupby("code",sort=False)
    continuous60=bars.p.sub(g.p.shift(60)).eq(60)
    bars["gene_max_streak_60"]=g.height.transform(lambda x:x.shift().rolling(60,min_periods=60).max()).where(continuous60)
    bars["gene_limits_60"]=g.limit.transform(lambda x:x.shift().rolling(60,min_periods=60).sum()).where(continuous60)
    historical_success=bars.limit&bars.previous_limit
    numerator=historical_success.groupby(bars.code).transform(lambda x:x.shift().rolling(60,min_periods=60).sum())
    denominator=g.previous_limit.transform(lambda x:x.shift().rolling(60,min_periods=60).sum())
    bars["gene_relay_rate_60"]=(numerator/denominator.where(denominator.gt(0))).where(continuous60)
    prior_turnover=g.turnover.transform(lambda x:x.shift().rolling(5,min_periods=5).mean())
    bars["turnover_ratio_5"]=(bars.turnover/prior_turnover.where(prior_turnover.gt(0))).where(bars.p.sub(g.p.shift(5)).eq(5))
    hi=g.high.transform(lambda x:x.shift().rolling(5,min_periods=5).max())
    lo=g.low.transform(lambda x:x.shift().rolling(5,min_periods=5).min())
    bars["prior_compression_5"]=(hi/lo-1).where(bars.p.sub(g.p.shift(5)).eq(5))
    adjacent=bars.p.sub(g.p.shift()).eq(1)
    bars["two_session_turnover_sum"]=(bars.turnover+g.turnover.shift()).where(adjacent)
    bars["previous_one_price"]=g.one_price.shift().where(adjacent)
    bars["opened_after_one_price"]=((bars.previous_one_price.eq(1))&bars.one_price.eq(0)).astype(float).where(bars.height.eq(2))
    # Counts are only complete within the declared archived security universe.
    eligible=bars[bars.eligible&bars.valid_price]
    market=eligible.groupby("date").height.max().to_frame("height")
    market["cycle_height_change"]=market.height.diff()
    cap=market.height.shift().rolling(20,min_periods=20).max()
    market["cycle_height_break_20"]=market.height.gt(cap).astype(float).where(cap.notna())
    anchors=bars.previous_height.ge(4)&bars.previous_limit
    for col,mask in [("cycle_anchor_breaks",anchors&~bars.limit),("cycle_anchor_collapses",anchors&bars.day_return.le(-.07))]:
        market[col]=bars[mask].groupby("date").size().reindex(market.index,fill_value=0)
    denominator=bars[anchors].groupby("date").size()
    numerator=bars[anchors&bars.limit].groupby("date").size()
    market["cycle_high_survival"]=numerator.reindex(market.index,fill_value=0)/denominator.reindex(market.index).replace(0,np.nan)
    first_count=eligible.groupby("date")["first"].sum()
    relay_count=eligible[eligible.height.ge(2)].groupby("date").size().reindex(market.index,fill_value=0)
    market["cycle_first_expansion_5"]=first_count/first_count.shift().rolling(5,min_periods=5).mean().replace(0,np.nan)
    market["cycle_first_relay_ratio"]=first_count/relay_count.replace(0,np.nan)
    top=eligible[eligible.height.eq(eligible.groupby("date").height.transform("max"))&eligible.height.ge(2)]
    market["cycle_top_one_price_share"]=top.groupby("date").one_price.mean()
    second=bars.height.eq(2)&bars.previous_limit&bars.previous_height.eq(1)&g["first"].shift().fillna(False).astype(bool)&adjacent
    rows=bars[bars["first"]|second].copy().rename(columns={"date":"signal_date"})
    rows["stage"]=np.where(rows["first"],1,2)
    rows=rows.merge(first[["signal_date"]+MARKET].drop_duplicates("signal_date"),on="signal_date",how="left",validate="many_to_one")
    rows=rows.merge(market.drop(columns="height"),left_on="signal_date",right_index=True,how="left",validate="many_to_one")
    for out,col in [("turnover_rank","turnover"),("amount_rank","amount"),("volume_ratio_rank","volume_ratio_5"),("pre_momentum_rank","return_20")]:
        rows[out]=rows.groupby(["signal_date","stage"])[col].rank(pct=True)
    rows["stage_pool_size"]=rows.groupby(["signal_date","stage"]).code.transform("size")
    rows["stage_one_price_share"]=rows.groupby(["signal_date","stage"]).one_price.transform("mean")
    lookup=bars.set_index(["p","code"])[["limit","valid_price","regular_price","eligible"]]
    idx=pd.MultiIndex.from_arrays([rows.p.to_numpy()+1,rows.code.to_numpy()],names=["p","code"])
    future=lookup.reindex(idx)
    observed=future.valid_price.fillna(False).to_numpy(dtype=bool)&future.regular_price.fillna(False).to_numpy(dtype=bool)&future.eligible.fillna(False).to_numpy(dtype=bool)
    rows[TARGET]=np.where(observed,future.limit.fillna(False).to_numpy(dtype=float),np.nan)
    rows["label_end"]=rows.p.map(lambda p:sessions[p+1] if p+1<len(sessions) else None)
    rows["baseline_score"]=-rows.turnover
    fields=VARIANTS["STAGE_CYCLE_GENE"]
    rows[fields]=rows[fields].replace([np.inf,-np.inf],np.nan)
    rows=rows.sort_values(["signal_date","stage","code"]).reset_index(drop=True)
    audit.update(stage_rows=rows.groupby("stage").size().to_dict(),stage_known=rows.groupby("stage")[TARGET].count().to_dict(),
                 next_session_label="Exactly next official session; next session missing/ineligible/invalid price remains UNKNOWN",extra_features=EXTRA,
                 stage2_identity="Previous adjacent official session is a verified reconstructed first board, current close is second exact limit")
    return rows,bars,sessions,audit


def metric(rows,score="model_score",k=1):
    take=base.selected(rows,score,k)
    known=take[TARGET].notna();hits=int(take[TARGET].eq(1).sum());n=int(known.sum())
    total_positive=int(rows[TARGET].eq(1).sum())
    if n:
        z=1.959963984540054;p=hits/n;den=1+z*z/n
        center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        interval=[max(0,center-half),min(1,center+half)]
    else:interval=[None,None]
    return dict(selected=len(take),selected_dates=take.signal_date.nunique(),known=n,unknown=int((~known).sum()),hits=hits,
                hit_rate=hits/n if n else None,wilson95=interval,full_positive_recall=hits/total_positive if total_positive else None,
                false_positives=int(take[TARGET].eq(0).sum()),false_negatives=total_positive-hits,positive_pool=total_positive,
                outcome_coverage=n/len(take) if len(take) else None)


def calibration_metrics(rows):
    known=rows[TARGET].notna()&rows.calibrated_score.notna()
    x=rows[known];y=x[TARGET].astype(int);p=x.calibrated_score
    if not len(x):return dict(status="UNAVAILABLE",historical_gate_passed=False)
    groups=pd.cut(p,np.linspace(0,1,11),include_lowest=True)
    bins=x.assign(bin=groups).groupby("bin",observed=True).agg(n=(TARGET,"size"),mean_probability=("calibrated_score","mean"),observed=(TARGET,"mean"))
    ece=float((bins.n*(bins.mean_probability-bins.observed).abs()).sum()/len(x))
    brier=brier_score_loss(y,p);baseline=brier_score_loss(y,x.train_prior)
    top=base.selected(rows,"model_score",1);top=top[top[TARGET].notna()&top.calibrated_score.notna()]
    top_error=abs(float(top.calibrated_score.mean()-top[TARGET].mean())) if len(top) else None
    qualified=bool(len(x)>=500 and len(top)>=200 and brier<baseline and ece<=.08 and top_error<=.10)
    return dict(status="HISTORICAL_TEMPORAL_CALIBRATION_ONLY",historical_gate_passed=qualified,rows=len(x),brier=brier,prior_brier=baseline,ece=ece,
                top1_rows=len(top),top1_forecast_mean=float(top.calibrated_score.mean()),top1_observed=float(top[TARGET].mean()),top1_absolute_gap=top_error,
                bins=[dict(bin=str(idx),**row.to_dict()) for idx,row in bins.iterrows()])


def evaluate(rows):
    known=rows[TARGET].notna()
    y=rows.loc[known,TARGET].astype(int);score=rows.loc[known,"model_score"]
    p,r,f,_=precision_recall_fscore_support(y,score.ge(.5),average="binary",zero_division=0)
    return dict(rows=len(rows),sessions=rows.signal_date.nunique(),known=int(known.sum()),unknown=int((~known).sum()),pool_rate=float(y.mean()),
                top={str(k):metric(rows,k=k) for k in (1,3,5)},baseline_top={str(k):metric(rows,score="baseline_score",k=k) for k in (1,3,5)},
                threshold_0_5_uncalibrated=dict(accuracy=float((score.ge(.5).to_numpy()==y.to_numpy()).mean()),precision=p,recall=r,f1=f,auc=roc_auc_score(y,score)),
                calibration=calibration_metrics(rows))


def fit_core(train,fields,out,name):
    directory=out/"models";directory.mkdir(exist_ok=True)
    receipt=dict(name=name,started_at=base.now(),rows=len(train),positives=int(train[TARGET].sum()),max_label_end=train.label_end.max(),
                 min_signal=train.signal_date.min(),max_signal=train.signal_date.max(),features=fields)
    base.write_json(directory/f"{name}_BEFORE_FIT.json",receipt)
    train[["signal_date","code","stage","label_end",TARGET]].to_csv(directory/f"{name}_members.csv",index=False)
    model=base.make_model("hist_gradient")
    weights=1/train.groupby("signal_date").code.transform("size");weights=weights/weights.mean()
    model.fit(train[fields],train[TARGET].astype(int),sample_weight=weights)
    joblib.dump(model,directory/f"{name}.joblib")
    receipt.update(completed_at=base.now(),sha256=base.sha(directory/f"{name}.joblib"))
    base.write_json(directory/f"{name}_COMPLETED.json",receipt)
    return model


def predict(bundle,score):
    out=score.copy();raw=bundle["model"].predict_proba(out[bundle["features"]])[:,1]
    out["model_score"]=raw
    if bundle["calibrator"] is not None:
        logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
        out["calibrated_score"]=bundle["calibrator"].predict_proba(logits.reshape(-1,1))[:,1]
    else:out["calibrated_score"]=np.nan
    out["train_prior"]=bundle["train_prior"]
    return out


def fit_head(train,variant,out,name):
    fields=VARIANTS[variant];dates=sorted(train.signal_date.unique())
    if len(dates)<100:raise ValueError("insufficient temporal training sessions")
    cutoff=dates[-42];inner=train[train.label_end.lt(cutoff)];cal=train[train.signal_date.ge(cutoff)]
    inner_model=fit_core(inner,fields,out,name+"_calibration_prefix")
    raw=inner_model.predict_proba(cal[fields])[:,1]
    logit=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    calibrator=LogisticRegression(C=1.,max_iter=1000,random_state=42).fit(logit.reshape(-1,1),cal[TARGET].astype(int))
    if calibrator.coef_[0,0]<=0:calibrator=None
    cal_receipt=dict(name=name,cutoff=cutoff,inner_max_label_end=inner.label_end.max(),calibration_rows=len(cal),calibration_range=[cal.signal_date.min(),cal.signal_date.max()],
                     positive_slope=calibrator is not None,formed_at=base.now())
    cal[["signal_date","code","stage","label_end",TARGET]].assign(uncalibrated_oof=raw).to_csv(out/"models"/f"{name}_calibration_rows.csv",index=False)
    base.write_json(out/"models"/f"{name}_calibration_receipt.json",cal_receipt)
    model=fit_core(train,fields,out,name+"_full")
    bundle=dict(version=VERSION,stage=int(train.stage.iloc[0]),model=model,calibrator=calibrator,features=fields,variant=variant,
                train_prior=float(train[TARGET].mean()),train_max_label_end=train.label_end.max(),max_signal=train.signal_date.max(),
                calibration_status="HISTORICAL_QUALIFICATION_PENDING",research_only=True)
    joblib.dump(bundle,out/"models"/f"{name}_bundle.joblib")
    return bundle


def chain_recheck(first_predictions,second_predictions):
    picked=base.selected(first_predictions,"model_score",1).copy()
    picked=picked.rename(columns={"signal_date":"first_signal_date",TARGET:"first_promotion_LABEL_ONLY","model_score":"first_model_score","calibrated_score":"first_calibrated_score","label_end":"second_signal_date"})
    second=second_predictions[["signal_date","code","model_score","calibrated_score",TARGET,"label_end"]].rename(columns={"signal_date":"second_signal_date","model_score":"second_model_score","calibrated_score":"second_calibrated_score",TARGET:"second_promotion_LABEL_ONLY","label_end":"third_target_date"})
    keep=["first_signal_date","code","name","second_signal_date","first_promotion_LABEL_ONLY","first_model_score","first_calibrated_score"]
    merged=picked[keep].merge(second,on=["second_signal_date","code"],how="left",validate="one_to_one")
    # Second-board assessment exists only when the first call actually promoted.
    fields=["second_model_score","second_calibrated_score","second_promotion_LABEL_ONLY","third_target_date"]
    merged.loc[~merged.first_promotion_LABEL_ONLY.eq(1),fields]=np.nan
    matured=merged.first_promotion_LABEL_ONLY.eq(1)&merged.second_promotion_LABEL_ONLY.notna()
    continued=merged.second_promotion_LABEL_ONLY.eq(1)&merged.first_promotion_LABEL_ONLY.eq(1)
    qualified=matured&merged.second_calibrated_score.ge(.5)
    stats=dict(first_selections=len(merged),first_promoted=int(merged.first_promotion_LABEL_ONLY.eq(1).sum()),second_rechecks_available=int(merged.second_model_score.notna().sum()),
               second_outcomes_known=int(matured.sum()),continued_to_third=int(continued.sum()),
               conditional_second_to_third=float(continued.sum()/matured.sum()) if matured.any() else None,
               qualified_rechecks=int(qualified.sum()),qualified_wins=int((qualified&continued).sum()),
               qualified_hit_rate=float((qualified&continued).sum()/qualified.sum()) if qualified.any() else None,
               full_first_to_third_rate=float(continued.sum()/merged.first_promotion_LABEL_ONLY.notna().sum()))
    return merged,stats


def preregistration():
    return dict(version=VERSION,formed_at=base.now(),user_clarification="首板1进2的胜率为百分之60，然后再重新评判该股2进3的胜率",
        previous_goal_turn="PROGRESS: saved and replay-verified prior first-board-to-four-board model; new clarification authorizes next-board targets, no old result overwritten.",
        primary="At each archived first-board close rank the entire observed first-board pool; Top1 next official session exact closing second-board hit rate target .60, every candidate day retained.",
        second_stage="Train independently on complete archived verified second-board pools; recheck the same selected stock only after it becomes second board, using that close's updated inputs; next official session third-board outcome.",
        hypothesis="Stage-specific next-board learning with causally available stock limit-up genetics and cycle-node context improves durable promotion ranking; forum nodes are unverified mechanisms, not labels.",
        source_hashes={str(p.resolve()):base.sha(p) for p in base.INPUTS.values()},base_module_sha256=base.sha(base.__file__),
        source_notes_hash=base.sha(SOURCES/"source_adoption.json"),variants=VARIANTS,
        parameters=dict(core="Same fixed hist_gradient parameters as prior run: 160 trees,15 leaves,leaf60,lr.06,l2=8,balanced,seed42,no early stopping",weights="Equal total signal-day weight"),
        validation="Train signals and label ends strictly before 2025-01-01. Compare STAGE_BASE vs STAGE_CYCLE_GENE once per stage on 2025H1 Top1 hits; STAGE_BASE wins ties. Freeze separately before evaluating.",
        evaluation="2025-07 through 2026-08. Selected variant only per stage, monthly previous252 official signal-date positions; all training label ends strictly before month boundary.",
        calibration="For each training window, fit fixed prefix model before last42 training signal dates, produce temporal held-out scores, fit positive-slope Platt C=1; final core trained full window. Test probabilities qualified only if >=500 pool rows,>=200 Top1 known, Brier below monthly training-prior baseline,ECE10<=.08,Top1 mean probability error<=.10.",
        target_60_gate="Top1 first-to-second observed hit rate >=.60, >=200 known selections, outcome coverage>=.95, both 2025H2 and2026 hit rates>=.55. Report confidence interval and unresolved days; none treated as failures or discarded.",
        baseline="Lowest signal turnover, code ascending ties; original four-board model's saved first-board scores are reference only, not retrained.",
        budget_minutes=30,stop="One fixed comparison with at most2 variants per stage; no retuning on test. 60% unmet remains an active goal.",
        historical_replay_not_pristine_blind=True,full_historical_universe_complete=False,production_unchanged=True,actual_order_count=0)


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError("refuse overwrite")
    out.mkdir(parents=True)
    protocol=preregistration();base.write_json(out/"protocol_BEFORE_ANY_FIT.json",protocol)
    rows,bars,sessions,audit=build_rows()
    rows.to_csv(out/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",index=False,encoding="utf-8-sig")
    base.write_json(out/"data_audit.json",audit)
    validation={};selection={}
    for stage in (1,2):
        pool=rows[rows.stage.eq(stage)];known=pool[pool[TARGET].notna()]
        train=known[known.signal_date.lt("2025-01-01")&known.label_end.lt("2025-01-01")]
        val=pool[pool.signal_date.between("2025-01-01","2025-06-30")]
        validation[stage]={}
        for variant in VARIANTS:
            name=f"validation_S{stage}_{variant}"
            bundle=fit_head(train,variant,out,name)
            scores=predict(bundle,val);scores.to_csv(out/f"{name}_PREDICTIONS.csv",index=False)
            validation[stage][variant]=evaluate(scores)
        selection[stage]=max(VARIANTS,key=lambda v:validation[stage][v]["top"]["1"]["hits"])
    base.write_json(out/"selection_BEFORE_EVALUATION.json",dict(formed_at=base.now(),selected=selection,validation=validation))
    all_predictions={}
    for stage in (1,2):
        pool=rows[rows.stage.eq(stage)];known=pool[pool[TARGET].notna()]
        test=pool[pool.signal_date.ge("2025-07-01")];parts=[]
        for month,score in test.groupby(test.signal_date.str[:7],sort=True):
            boundary=month+"-01";p=int(np.searchsorted(sessions,boundary))
            train=known[known.p.ge(p-252)&known.label_end.lt(boundary)]
            name=f"{month}_S{stage}_{selection[stage]}"
            bundle=fit_head(train,selection[stage],out,name)
            assert bundle["train_max_label_end"]<score.signal_date.min()
            predictions=predict(bundle,score);predictions["model_month"]=month
            predictions.to_csv(out/f"{month}_S{stage}_PREDICTIONS_BEFORE_METRICS.csv",index=False)
            parts.append(predictions)
        all_predictions[stage]=pd.concat(parts,ignore_index=True)
        all_predictions[stage].to_csv(out/f"S{stage}_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
    comparisons={}
    for stage,predictions in all_predictions.items():
        comparisons[stage]={}
        for period,mask in [("all",pd.Series(True,index=predictions.index)),("2025H2",predictions.signal_date.lt("2026-01-01")),("2026",predictions.signal_date.ge("2026-01-01")),("non_one_price",predictions.one_price.eq(0))]:
            comparisons[stage][period]=evaluate(predictions[mask])
    chain,chain_info=chain_recheck(all_predictions[1],all_predictions[2]);chain.to_csv(out/"FIRST_PICK_TO_SECOND_RECHECK_ALL.csv",index=False,encoding="utf-8-sig")
    first_metric=comparisons[1]["all"]["top"]["1"]
    gate=bool(first_metric["hit_rate"]>=.60 and first_metric["known"]>=200 and first_metric["outcome_coverage"]>=.95 and all(comparisons[1][p]["top"]["1"]["hit_rate"]>=.55 for p in ("2025H2","2026")))
    bundles={}
    cutoff=audit["last_source_session"];p=sessions.index(cutoff)
    for stage in (1,2):
        known=rows[rows.stage.eq(stage)&rows[TARGET].notna()]
        train=known[known.p.ge(p-252)&known.label_end.lt(cutoff)]
        bundle=fit_head(train,selection[stage],out,f"latest_S{stage}")
        bundle.update(knowledge_cutoff=cutoff,calibration_status="HISTORICAL_QUALIFIED" if comparisons[stage]["all"]["calibration"]["historical_gate_passed"] else "PROBABILITY_NOT_QUALIFIED")
        bundles[stage]=bundle
    joblib.dump(dict(version=VERSION,heads=bundles,knowledge_cutoff=cutoff,goal_60_passed=gate,research_only=True),out/"first_second_stage_bundle.joblib")
    trades,curve,sim=base.simulation(all_predictions[1],bars,sessions,"model_score")
    trades.to_csv(out/"S1_COST_SCENARIO_TRADES.csv",index=False,encoding="utf-8-sig");curve.to_csv(out/"S1_COST_SCENARIO_EQUITY.csv",index=False)
    report=dict(version=VERSION,completed_at=base.now(),selected_variants=selection,data=audit,evaluation=comparisons,chain=chain_info,
                goal_60_passed=gate,goal_status="TARGET_REACHED_RESEARCH_ONLY" if gate else "ACTIVE_TARGET_NOT_REACHED",scenario=sim,
                core_fit_count=len(list((out/"models").glob("*_COMPLETED.json"))),actual_trades=0,new_forward_results=0,source_hashes=protocol["source_hashes"],
                old_model_not_retrained=True,production_deployment_allowed=False,code_sha256=base.sha(__file__),bundle_sha256=base.sha(out/"first_second_stage_bundle.joblib"))
    for path,value in protocol["source_hashes"].items():assert base.sha(path)==value
    base.write_json(out/"report.json",report);render_report(out,report)
    print(json.dumps(base.clean(dict(output=str(out.resolve()),selected=selection,first_top1=first_metric,second_top1=comparisons[2]["all"]["top"]["1"],chain=chain_info,goal_60_passed=gate,core_fits=report["core_fit_count"])),ensure_ascii=False,indent=2))
    return report


def render_report(out,r):
    def pct(x):return "未知" if x is None else f"{x*100:.2f}%"
    e=r["evaluation"];e={int(k):v for k,v in e.items()}
    lines=["# 首板一进二与二板再评估训练回测", "",f"目标：首板收盘后每日首选下一交易日收盘晋级二板的历史命中率60%；已晋级股票在二板收盘后重新评估二进三。完成时间：{r['completed_at']}。", "",
           "两个阶段独立训练、分别选型。二板收盘信息没有进入原首板预测。全候选、漏报、误报与未知保留；不靠全断板Accuracy或事后减少出选日数达标。", "",
           "|阶段|范围|首选命中/已知|未知|首选命中率|95%区间|低换手基准|", "|---|---|---:|---:|---:|---|---:|"]
    for stage in (1,2):
        for period in ("all","2025H2","2026","non_one_price"):
            a=e[stage][period]["top"]["1"];b=e[stage][period]["baseline_top"]["1"]
            lines.append(f"|{stage}进{stage+1}|{period}|{a['hits']}/{a['known']}|{a['unknown']}|{pct(a['hit_rate'])}|{pct(a['wilson95'][0])}—{pct(a['wilson95'][1])}|{pct(b['hit_rate'])}|")
    c=r["chain"]
    lines += ["",f"首板首选完整{c['first_selections']}条，其中晋级二板{c['first_promoted']}条；二板重新评分覆盖{c['second_rechecks_available']}条，可核下一日结果{c['second_outcomes_known']}条，继续三板{c['continued_to_third']}条，条件二进三率{pct(c['conditional_second_to_third'])}。完整首板到三板比例{pct(c['full_first_to_third_rate'])}，不能混成一进二胜率。", "",
              f"60%目标：{'本批历史门达到；仍非实盘验收' if r['goal_60_passed'] else '尚未达到，目标保持进行中'}。", "", "## 概率校准", ""]
    for stage in (1,2):
        q=e[stage]["all"]["calibration"]
        lines.append(f"{stage}进{stage+1}：样本外Brier={q.get('brier')}，训练先验基准={q.get('prior_brier')}，ECE={q.get('ece')}，首选预测均值={pct(q.get('top1_forecast_mean'))}／实际={pct(q.get('top1_observed'))}；历史校准资格={'通过' if q['historical_gate_passed'] else '未通过，概率未验证'}。")
    lines += ["", "校准只使用每个训练窗尾部42日的时间样本外分数；最终模型使用该月以前的成熟样本。历史归档已被研究看过，不称全新盲测。源码、输入哈希、训练成员、前缀校准、保存权重、逐月完整预测均保留。", "",
              "## 经验取舍与边界", "", "GitHub及股吧仅启发假设，不提供晋级标签或事后题材标签。学习历史涨停股性、周期高度突破／断板分化、同身位竞争；不照抄作者固定权重，不使用实时估值回填历史，不以名字相同确认补涨，不采纳买入当日止损。缺少历史真实题材、首封时刻、公告/盘口与母池完整性的问题保留，当前市场并未获得预测。", "",
              "[GitHub一进二筛选器](https://github.com/lpw821/a-stock-1to2)给出多维经验评分；[GitHub观察系统](https://github.com/dongdonglog/a-share-observer)公开指出部分手工延续评分缺乏预测力。股吧的[节点经验](https://www.tgb.cn/a/2vFbPMvPAVQ)和[低位补涨讨论](https://www.tgb.cn/a/2uwf2vKo0AF)只用作待证伪线索，不认证因果、父龙关系或次日结果。", "",
              f"首选开盘成本情景：组合回报{pct(r['scenario']['total_return'])}、最大回撤{pct(r['scenario']['max_drawdown'])}、情景平仓{r['scenario']['closed_trades']}笔。含涨跌停开盘取消、整手、现金、T+1、延期退出和费用。有效成本价含不利滑点，不是经确认成交报价。实际成交0，生产模型、原冻结与定时任务未改。"]
    (Path(out)/"首板一进二与二进三_训练回测报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def recheck(bundle_path,features_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    bundle=joblib.load(bundle_path)
    rows=pd.read_csv(features_path,dtype={"code":str},float_precision="round_trip")
    if rows.empty or rows.duplicated(["signal_date","code"]).any():raise ValueError("empty or duplicate candidates")
    if not rows.signal_date.gt(bundle["knowledge_cutoff"]).all():raise ValueError("signal must be after training cutoff")
    if not rows.stage.isin([1,2]).all():raise ValueError("stage must be1 or2")
    parts=[]
    for stage,part in rows.groupby("stage"):
        head=bundle["heads"][int(stage)]
        if set(head["features"])-set(part):raise ValueError("incomplete features")
        if not base.true_values(part.scope_known).all() or not base.true_values(part.eligible).all():raise ValueError("unverified candidate scope")
        scored=predict(head,part)
        scored["probability_status"]=head["calibration_status"]
        if head["calibration_status"]!="HISTORICAL_QUALIFIED":scored["calibrated_score"]=np.nan
        parts.append(scored)
    result=pd.concat(parts,ignore_index=True)
    result=result[[c for c in result if "LABEL_ONLY" not in c and c!="label_end"]]
    result.sort_values(["signal_date","stage","model_score","code"],ascending=[True,True,False,True]).to_csv(out,index=False,encoding="utf-8-sig")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train");train.add_argument("--output",required=True)
    score=sub.add_parser("recheck");score.add_argument("--bundle",required=True);score.add_argument("--features",required=True);score.add_argument("--output",required=True)
    args=p.parse_args()
    if args.command=="train":run(args.output)
    else:recheck(args.bundle,args.features,args.output)
