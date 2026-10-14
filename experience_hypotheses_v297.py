"""Fixed empirical-proxy ledger; viewed history, no mysticism-driven upgrade."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from next_session_forecast_lab_v271 import ROOT,sha
from next_session_expectation_contract_v271 import digest

CONTROL=ROOT/"results/v279/daily_path_paired_20261002_v2"
CALENDAR=ROOT/"data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
DEVELOPMENT_END="2026-01-01"
N_BOOT=5000
TESTED=["digital_height","first_session","pre_holiday","friday","month_end","high_height","code_eight"]
HYPOTHESES=[
    ("digital_height","数字板高度压制","当前5板与同日4/6板比较；固定局部代理，不代表所有数字关口"),
    ("first_session","特定日期效应","当月第一个官方交易日与其余交易日比较"),
    ("pre_holiday","节前效应","已知下一交易日距信号日超过3个自然日，与其余日比较"),
    ("friday","周五效应","信号日为周五，与其余交易日比较"),
    ("month_end","月底效应","当月最后一个官方交易日，与其余交易日比较"),
    ("leader_break","龙头断板日","需要事先确认的周期/题材龙头及当日已发生的断板，不能用最高板替代"),
    ("high_height","高标压制","当前5板及以上与2至4板比较；高度与环境混杂需分层"),
    ("symbolic_name","名字玄学","需当时名称、预先固定字典、首次讨论时间及完整同字分母"),
    ("zodiac","生肖玄学","需预先固定生肖字典、当时名称与叙事发生时间；禁止看赢家扩字典"),
    ("code_eight","代码玄学","固定代码末位8与其余末位比较，仅限原连板候选范围"),
    ("stock_character","历史股性","需要此前全股票结果及失败记录，禁止只数后来幸存龙头"),
    ("region","地域炒作","需当时注册地址/业务地区、主题首次出现时间及完整同地域分母"),
    ("controller","同一实控人联动","需当时控制关系与公告可得时间，不能用当前关系回填"),
    ("old_leader_rebound","老龙反抽","需此前已确认身份、反抽与失败统一标签、未反抽对照"),
    ("monster_memory","妖股记忆","需事先定义历史妖股范围并保存全部失败/沉寂样本")]


def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open("x",encoding="utf-8") as f:f.write(payload)


def verify_sources():
    checkpoint=json.loads((CONTROL/"CONTROL_checkpoint.json").read_text(encoding="utf-8"))
    if sha(CONTROL/"CONTROL_predictions_LABEL_ONLY.csv")!=checkpoint["prediction_sha256"]:raise ValueError("original predictions changed")
    for path,wanted in checkpoint["source_hashes"].items():
        if sha(ROOT/path)!=wanted:raise ValueError("original input/model source changed: "+path)
    calendar=json.loads(CALENDAR.read_text(encoding="utf-8"))
    if calendar.get("status")!="verified" or not calendar.get("source"):raise ValueError("official audited calendar required")
    return calendar


def indicators(frame,calendar):
    sessions=calendar["sessions"];dates=pd.to_datetime(sessions)
    by_month={}
    for date in sessions:by_month.setdefault(date[:7],[]).append(date)
    first={v[0] for v in by_month.values()};last={v[-1] for v in by_month.values()}
    successor=dict(zip(sessions[:-1],sessions[1:]))
    if set(frame.date)-set(sessions):raise ValueError("candidate on nontrading date")
    result=frame[["date","code","boards"]].copy()
    result["digital_height"]=np.where(frame.boards.eq(5),1.,np.where(frame.boards.isin([4,6]),0.,np.nan))
    result["first_session"]=frame.date.isin(first).astype(float)
    next_dates=pd.to_datetime(frame.date.map(successor))
    if next_dates.isna().any():raise ValueError("calendar successor coverage ended")
    result["pre_holiday"]=(next_dates-pd.to_datetime(frame.date)).dt.days.gt(3).astype(float)
    result["friday"]=pd.to_datetime(frame.date).dt.weekday.eq(4).astype(float)
    result["month_end"]=frame.date.isin(last).astype(float)
    result["high_height"]=frame.boards.ge(5).astype(float)
    if not frame.code.map(lambda x:isinstance(x,str) and len(x)==6 and x.isdigit()).all():raise ValueError("code must be six-digit string")
    result["code_eight"]=frame.code.str.endswith("8").astype(float)
    return result


def bootstrap(rows,field,target,calendar,seed):
    known=rows.label_valid & rows[target].notna() & rows[field].notna()
    part=rows[known].copy()
    unmatched=0
    if field=="digital_height":
        paired_dates=set(part.loc[part[field].eq(1),"date"])&set(part.loc[part[field].eq(0),"date"])
        unmatched=int((~part.date.isin(paired_dates)).sum())
        part=part[part.date.isin(paired_dates)]
    group1=part[part[field].eq(1)];group0=part[part[field].eq(0)]
    dates=[d for d in calendar["sessions"] if rows.date.min()<=d<=rows.date.max()] if len(rows) else []
    def group(g):return {"rows":len(g),"dates":int(g.date.nunique()),"promotions":int(g[target].sum()),"rate":float(g[target].mean()) if len(g) else None}
    out={"full_candidate_rows":len(rows),"exposure_known_rows":int(rows[field].notna().sum()),"label_or_proxy_unknown_rows":int((~known).sum()),
        "official_sessions":len(dates),"candidate_dates":int(rows.date.nunique()),"missing_entire_dates":sorted(set(dates)-set(rows.date)),
        "unmatched_group_rows_retained_not_scored":unmatched,
        "exposed":group(group1),"control":group(group0),"difference":float(group1[target].mean()-group0[target].mean()) if len(group1) and len(group0) else None}
    if not len(group1) or not len(group0):out.update(block95=None,family_block_interval=None,valid_boot_fraction=None);return out
    aggregates=np.zeros((len(dates),4));index={d:i for i,d in enumerate(dates)}
    for flag,g in [(1,group1),(0,group0)]:
        counts=g.groupby("date")[target].agg(["sum","count"])
        for date,r in counts.iterrows():aggregates[index[date],0 if flag else 2]=r["sum"];aggregates[index[date],1 if flag else 3]=r["count"]
    block=min(5,len(dates));rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(dates),size=(N_BOOT,int(np.ceil(len(dates)/block))))
    indices=((starts[:,:,None]+np.arange(block))%len(dates)).reshape(N_BOOT,-1)[:,:len(dates)]
    sums=aggregates[indices].sum(axis=1);valid=(sums[:,1]>0)&(sums[:,3]>0)
    boot=sums[valid,0]/sums[valid,1]-sums[valid,2]/sums[valid,3]
    alpha=.05/len(TESTED)
    out.update(block95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,
        family_block_interval=np.quantile(boot,[alpha/2,1-alpha/2]).tolist() if len(boot) else None,
        family_tests=len(TESTED),family_interval_nominal_level=1-alpha,valid_boot_fraction=float(valid.mean()),
        inference="exploratory date-block percentile bootstrap, Bonferroni interval levels; not exact simultaneous-coverage guarantee")
    out["support_sufficient"]=bool(len(group1)>=50 and len(group0)>=50 and group1.date.nunique()>=20 and group0.date.nunique()>=20 and valid.mean()>=.95)
    return out


def evaluate(rows,field,target,calendar,seed):
    summary=bootstrap(rows,field,target,calendar,seed)
    summary["by_height"]={tier:bootstrap(g,field,target,calendar,seed+i+1) for i,(tier,g) in enumerate([
        ("2",rows[rows.boards.eq(2)]),("3",rows[rows.boards.eq(3)]),("4",rows[rows.boards.eq(4)]),("5+",rows[rows.boards.ge(5)])])}
    summary["nonword"]=bootstrap(rows[rows.daily_true_one_word.eq(0)],field,target,calendar,seed+20)
    return summary


def classified(development,confirmation,name):
    # A statistical proxy can never automatically become a core rule. In this
    # archive the independent source/forward gate remains explicitly missing.
    supported=bool(confirmation.get("support_sufficient"))
    interval=confirmation.get("family_block_interval")
    adjusted_nonzero=bool(interval and (interval[0]>0 or interval[1]<0))
    consistent=bool(development.get("difference") is not None and confirmation.get("difference") is not None and development["difference"]*confirmation["difference"]>0)
    return {"grade":"B","core_rule":False,"historical_proxy_support_sufficient":supported,
        "family_adjusted_interval_excludes_zero":adjusted_nonzero,"development_confirmation_sign_consistent":consistent,
        "reason":"历史来源/全股票分母与独立前向证据仍未完成，不能升为已证实规律" if supported else "独立暴露日期或股票样本不足；历史来源也未全面认证",
        "candidate_for_future_confirmation":supported and adjusted_nonzero and consistent}


def split_periods(frame):
    development=frame[frame.date.lt(DEVELOPMENT_END)].copy()
    unavailable=~development.target_date.lt(DEVELOPMENT_END)
    # Retain the last signal day's full denominator, but do not use its labels
    # which only realize after the historical development boundary.
    development.loc[unavailable,"label_valid"]=False
    return development,frame[frame.date.ge(DEVELOPMENT_END)].copy(),int(unavailable.sum())


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    calendar=verify_sources()
    protocol={"version":"v297-fixed-experience-proxies-1","recorded_at":now(),"hypotheses":[{"id":key,"name":name,"definition":definition,"testable":key in TESTED} for key,name,definition in HYPOTHESES],
        "development":"2025-07 through 2025-12","confirmation":"2026-01 through 2026-09; previously viewed diagnostic, not pristine blind",
        "outcome":"original exact-next-session closing board promotion","bootstrap_dates":5,"bootstrap_draws":N_BOOT,"family_tests":len(TESTED),
        "minimum_exposed_and_control":{"rows_each":50,"dates_each":20},"all_rules_core":False,"no_definition_or_threshold_search":True}
    write(output/"protocol_BEFORE_LABEL_JOIN.json",protocol)
    csv=CONTROL/"CONTROL_predictions_LABEL_ONLY.csv"
    # Projection only: no next-session labels, returns, names, terminal heights.
    features=pd.read_csv(csv,dtype={"code":str},usecols=["date","code","boards","daily_true_one_word"])
    masks=indicators(features,calendar)
    masks["daily_true_one_word"]=features.daily_true_one_word
    if masks.duplicated(["date","code"]).any():raise ValueError("duplicate original denominator")
    masks.to_csv(output/"hypothesis_masks_NO_NEXT_SESSION_LABELS.csv",index=False,encoding="utf-8-sig")
    labels=pd.read_csv(csv,dtype={"code":str},usecols=["date","code","target_date","label_valid","target"])
    labels=labels.rename(columns={"target":"target_original"});labels["target_pressure_zero"]=labels.target_original
    pressure=json.loads((ROOT/"results/v290/closing_label_audit_20261002_v1/report.json").read_text(encoding="utf-8"))["positive_contradictions"]
    affected=[]
    for row in pressure:
        found=labels.date.eq(row["date"])&labels.code.eq(row["code"])
        if found.sum()!=1 or not labels.loc[found,"target_original"].eq(1).all():raise ValueError("original pressure row changed")
        labels.loc[found,"target_pressure_zero"]=0.;affected.append({"date":row["date"],"code":row["code"]})
    successor=dict(zip(calendar["sessions"][:-1],calendar["sessions"][1:]))
    if not labels.loc[labels.label_valid,"target_date"].eq(labels.loc[labels.label_valid,"date"].map(successor)).all():raise ValueError("known label skips actual next session")
    frame=masks.merge(labels,on=["date","code"],validate="one_to_one")
    frame.to_csv(output/"cohort_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    results=[]
    dev,confirm,boundary_rows=split_periods(frame)
    for i,(key,name,definition) in enumerate(HYPOTHESES):
        if key not in TESTED:
            grade="C" if key in ["symbolic_name","zodiac"] else "D" if key in ["stock_character","monster_memory"] else "B"
            results.append({"id":key,"name":name,"definition":definition,"status":"UNAVAILABLE_POINT_IN_TIME_INPUTS","grade":grade,"core_rule":False,
                "reason":definition,"backtest_completed":False,"null_is_not_no_effect":True});continue
        result={"id":key,"name":name,"definition":definition,"status":"HISTORICAL_PROXY_TEST_COMPLETED","views":{},"backtest_completed":True}
        for target in ["target_original","target_pressure_zero"]:
            development=evaluate(dev,key,target,calendar,297+i*100)
            confirmation=evaluate(confirm,key,target,calendar,297+i*100+50)
            result["views"][target]={"development":development,"confirmation":confirmation}
        result.update(classified(result["views"]["target_original"]["development"],result["views"]["target_original"]["confirmation"],name))
        results.append(result)
        print(json.dumps({"completed_proxy":key,"grade":result["grade"]}),flush=True)
    inputs=[Path(__file__),CALENDAR,csv,CONTROL/"CONTROL_checkpoint.json",ROOT/"results/v290/closing_label_audit_20261002_v1/report.json",ROOT/"股吧经验与玄学研究V273.md",ROOT/"forum_hypotheses.csv"]
    record={"version":"v297-fixed-experience-proxies-1","status":"SEVEN_HISTORICAL_PROXIES_TESTED_EIGHT_INPUT_GAPS_RETAINED","recorded_at":now(),
        "candidate_rows":len(frame),"candidate_dates":int(frame.date.nunique()),"results":results,"pressure_rows":affected,
        "development_boundary_rows_retained_not_scored":boundary_rows,
        "all_original_candidates_preserved":True,"original_predictions_or_labels_rewritten":False,"core_model_upgrade":False,
        "forward_samples_added":0,"stable_60_verified":False,"goal_complete":False,"trade_authorized":False,"actual_fills":0,"max_drawdown":None,"profit_loss_ratio":None,
        "limitations":["Prior viewed history and partially verified provider labels; not pristine OOS or full-market proof",
            "Calendar/code proxies are fixed research definitions, not claims that market folklore is true",
            "Height, mood, price, industry and period confounding remain; stratified summaries are not causal controls",
            "Code effect only within original >=2-board candidates, not all listed securities",
            "Pressure zeros are sensitivity assumptions, not certified corrected truth",
            "Bootstrap interval levels are exploratory multiplicity adjustment, not an exact significance guarantee",
            "Rare-event hypotheses need many more independent calendar episodes"],
        "source_hashes":{str(p.resolve()):sha(p) for p in inputs},
        "artifact_hashes":{p.name:sha(p) for p in [output/"protocol_BEFORE_LABEL_JOIN.json",output/"hypothesis_masks_NO_NEXT_SESSION_LABELS.csv",output/"cohort_LABEL_ONLY.csv"]}}
    write(output/"report.json",record)
    return {"status":record["status"],"candidate_rows":len(frame),"candidate_dates":int(frame.date.nunique()),"grades":{r["name"]:r["grade"] for r in results},"stable_60_verified":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
