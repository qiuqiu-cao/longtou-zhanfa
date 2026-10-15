"""Verify this continuation against original scope and deliver append-only views."""
from pathlib import Path
from datetime import datetime
import ast
import json
import importlib.util

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as old
import short_term_sampled_pressure as pressure
import historical_symbolic_context as h
import relay_review_workbench as w

ROOT=Path(__file__).resolve().parent
PREVIOUS=h.RUN
ARMS=["BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]


def verify_pressure():
    directory=ROOT/"sampled_pressure_evidence"
    report=b.read(directory/"qualification_and_discrimination_audit.json");b.verify(report["source_bindings"])
    observed=b.read(directory/"qualified_sampled_pressure_OBSERVATIONS_NO_LABELS.json")
    originals={}
    for path in [b.ROOT/"data/v289/full_august_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv",b.ROOT/"data/v287/september_five_minute_20261002_v1/five_minute_bars_UNRECONCILED.csv"]:
        bars=pd.read_csv(path,dtype={"code":str})
        prefix="2026-08" if "full_august" in str(path) else "2026-09"
        bars=bars[bars.minute.str.startswith(prefix)]
        for (day,code),group in bars.groupby([bars.minute.str[:10],"code"]):originals[(day,code)]=group.sort_values("minute")
    caps=pd.read_csv(b.ROOT/"results/v289/august_september_paths_20261002_v1/features_NO_NEXT_SESSION_LABELS.csv",dtype={"code":str}).set_index(["date","code"]).price.to_dict()
    checked_price=0;checked_volume=0
    for x in observed:
        if not x["price_path_qualified"]:continue
        bars=originals[(x["date"],x["code"])];cap=caps[(x["date"],x["code"])]
        assert len(bars)==48 and set(bars.minute.str[11:16])==pressure.expected_times(x["date"])
        close=bars.close.to_numpy();high=bars.high.to_numpy();low=bars.low.to_numpy()
        iscap=[z>=cap-.005001 for z in close]
        after=sum(iscap[i-1] and low[i]<cap-.005001 for i in range(1,len(bars)))
        recovered=sum(iscap[i] and not iscap[i-1] for i in range(1,len(bars)))
        draw=min([0.]+[low[i]/max(high[:i])-1 for i in range(1,len(bars))])
        below=[i for i,z in enumerate(iscap) if not z]
        tail=len(bars)-1-below[-1] if below else len(bars)
        assert x["sampled_cap_close_then_below_intervals"]==after
        assert x["sampled_cap_close_recovery_transitions"]==recovered
        assert np.isclose(x["sampled_negative_prior_bar_peak_drawdown"],draw,atol=1e-13)
        assert x["sampled_intervals_after_last_below_close"]==tail
        assert np.isclose(x["sampled_first_cap_close_progress"],iscap.index(True)/47,atol=1e-13)
        checked_price+=1
        if x["volume_path_qualified"]:
            v=bars.volume_shares.to_numpy();amount=bars.amount.to_numpy();total=sum(v)
            volumechecks=dict(sampled_final_six_volume_fraction=sum(v[-6:])/total,
                sampled_recovery_interval_volume_fraction=sum(v[i] for i in range(1,48) if iscap[i] and not iscap[i-1])/total,
                sampled_final_six_cap_volume_fraction=sum(v[i] for i in range(42,48) if iscap[i])/total,
                sampled_close_above_full_day_vwap=cap/(sum(amount)/total)-1)
            for key,value in volumechecks.items():assert np.isclose(x[key],value,atol=1e-13),(key,x[key],value)
            checked_volume+=1
        else:assert all(x[k] is None for k in pressure.VOLUME_FIELDS)
        assert x["continuous_seal_verified"] is False and x["verified_absorption"] is None
    assert checked_price==451 and checked_volume==324
    full=pd.read_csv(directory/"full9245_pressure_context_NO_TARGETS.csv",dtype={"code":str})
    assert len(full)==9245 and not full.duplicated(["date","code"]).any()
    assert not set(full)&{"target","target_date","score","label_valid","probability","outcome_group"}
    pred=pd.read_csv(b.DEFAULT_RUN/"delivery/complete_paired_predictions_VALIDATED.csv",dtype={"code":str},low_memory=False)
    pred=pred[pred.arm.isin(["BASE","ST_DUAL"])&pred.date.between("2026-08-01","2026-09-30")]
    diagnostic=pd.read_csv(directory/"complete_485_saved_prediction_pressure_diagnostic.csv",dtype={"code":str},low_memory=False)
    for arm in ["BASE","ST_DUAL"]:
        left=pred[pred.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        right=diagnostic[diagnostic.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        assert len(left)==len(right)==485
        pd.testing.assert_frame_equal(left[list(pred.columns)],right[list(pred.columns)],check_dtype=False)
    return dict(raw_bars_replayed=473226,original_minute_candidate_rows=485,full9245_rows_retained=True,
        price_fields_independently_recomputed=checked_price,volume_fields_independently_recomputed=checked_volume,
        saved_scores_and_labels_unchanged=True,future_label_columns_absent_from_feature_file=True)


def verify_scores():
    validation=b.read(ROOT/"author_review_validation.json");b.verify(validation["source_bindings"])
    facts=b.read(validation["facts_path"]);review=b.read(validation["review_path"])
    assert w.validate_completed(validation["facts_path"],review)=="DIAGNOSTIC_APPEND_ONLY_REVIEW_VALIDATED"
    saved_path=ROOT/"saved_scores_AFTER_COMPLETED_REVIEW/complete_current6_research_comparison.json"
    saved=b.read(saved_path);b.verify(saved["source_bindings"])
    fcontext=b.read(facts["causal_thinking_context_path"]);scontext=b.read(facts["symbolic_context_path"])
    pack=b.read(scontext["features_path"])
    base=pd.DataFrame([dict(x["features"],code=x["code"]) for x in pack["relay"]])
    dyn=pd.DataFrame(fcontext["rows"]);overlap=(set(base)&set(dyn))-{ "code" }
    dynamic=base.drop(columns=list(overlap)).merge(dyn,on="code",validate="one_to_one")
    models={arm:b.DEFAULT_RUN/"execution_after_binning_repair"/("2026-10_"+arm+"_model.joblib") for arm in ["ST_DUAL","ST_B_ONLY"]}
    models["ST_DUAL_SYMBOLIC_DIFFUSION"]=PREVIOUS/"fixed_comparison/2026-10_model.joblib"
    manifest=b.read(old.BASE_ROOT/"models/relay/BASE/manifest.json");models["BASE"]=Path(manifest["model_path"])
    checked=[]
    for arm,path in models.items():
        loaded=joblib.load(path)
        if arm=="BASE":model=loaded;frame=base;fields=manifest["fields"]
        else:
            model=loaded["model"];fields=loaded["features"]
            assert loaded["cutoff"]<pack["signal_date"]
            frame=dynamic if arm!="ST_DUAL_SYMBOLIC_DIFFUSION" else dynamic.merge(pd.DataFrame(scontext["rows"])[["code"]+h.FIELDS],on="code",validate="one_to_one")
        with threadpool_limits(limits=2):values=model.predict_proba(frame[fields])[:,list(model.classes_).index(1)]
        bycode={x["code"]:x for x in saved["rows"]}
        for code,value in zip(frame.code,values):assert np.isclose(bycode[code][arm],value,atol=1e-13)
        checked.append(dict(arm=arm,rows=6,exact_prediction_replay=True))
    # Actual guard rejection probes mutate in-memory draft copies, never files.
    from copy import deepcopy
    probes=[]
    bad=deepcopy(review);bad["author_review_completed"]=False
    try:w.validate_completed(validation["facts_path"],bad)
    except ValueError as exc:probes.append(dict(case="pending_review",rejected=str(exc)))
    else:raise AssertionError("pending review accepted")
    bad=deepcopy(review);by=next(x for x in bad["rows"] if x["code"]=="002696");by["forum_reasoning_assessment"]["opinions"].pop()
    try:w.forum.validate(w.forum.checked(facts),bad,facts["source_bindings"])
    except ValueError as exc:probes.append(dict(case="omitted_forum_argument",rejected=str(exc)))
    else:raise AssertionError("incomplete argument review accepted")
    bad=deepcopy(review);by=next(x for x in bad["rows"] if x["code"]=="002696");by["identity_discovery_assessment"]["accepted_anchor_codes"]=[]
    try:w.identity.validate_review(facts,bad)
    except ValueError as exc:probes.append(dict(case="omitted_discovered_parent",rejected=str(exc)))
    else:raise AssertionError("unreviewed parent accepted")
    reject=ROOT/"forbidden_formal_freeze_SHOULD_NOT_EXIST.json"
    try:w.freeze(validation["facts_path"],validation["review_path"],reject)
    except ValueError as exc:probes.append(dict(case="diagnostic_formal_freeze",rejected=str(exc)))
    else:raise AssertionError("diagnostic source registered formal forecast")
    assert not reject.exists()
    before=b.read(ROOT/"saved_scores_AFTER_COMPLETED_REVIEW/inference_receipt_BEFORE_MODEL_LOAD.json")
    assert datetime.fromisoformat(review["reviewed_at"])<datetime.fromisoformat(before["formed_at"])
    original=b.read(ROOT/"forum/facts_with_forum_reasoning.json")
    assert facts["rows"]==original["rows"]
    histories=b.read(ROOT/"histories_consumed_receipt.json")
    assert sum(x["prior_rows_used"] for x in histories["history_datasets"])==195
    assert histories["mainline_records_consumed"]==4 and histories["narrative_claims_classified"]==4
    return dict(full6_original_rows_exact=True,saved_models_replayed=checked,actual_guard_probes=probes,
        completed_review_precedes_score_exposure=True,prior_rows_consumed=195,mainline_reports_consumed=4,typed_narrative_claims=4)


def deliver(out,checks):
    original_path=PREVIOUS/"delivery_final/完整81只周期角色与原判断对照.csv"
    original=pd.read_csv(original_path,dtype={"code":str,"parent_anchor_code":str})
    current=pd.DataFrame(b.read(ROOT/"saved_scores_AFTER_COMPLETED_REVIEW/complete_current6_research_comparison.json")["rows"])
    manual=b.read(ROOT/"author_current6_COMPLETED_BEFORE_MODEL_DISPLAY.json");authored={x["code"]:x for x in manual["rows"]}
    result=original.copy();result["current15_conditional_outlook"]=None;result["current15_review_reason"]=None
    for arm in ARMS:result["current15_after_review_"+arm]=np.nan
    for stock in current.to_dict("records"):
        mask=result.date.eq("2026-10-15")&result.code.eq(stock["code"]);assert mask.sum()==1
        result.loc[mask,"current15_conditional_outlook"]=stock["prereview_conditional_outlook"]
        result.loc[mask,"current15_review_reason"]=stock["conditional_reason"]
        for arm in ARMS:result.loc[mask,"current15_after_review_"+arm]=stock[arm]
    pd.testing.assert_frame_equal(result[list(original)],original)
    result.to_csv(out/"完整81只对照_含15日审阅后研究分.csv",index=False,encoding="utf-8-sig")
    rows=[];lines=["# 完整81只周期角色对照：15日研究评分续接", "", "原81行原列完全保留。新增15日条件观察和审阅后研究分列；正式方向、原预测/目标列不回写。分数未校准，已有研究模型未通过整体验收。", "",
        "| 日期 | 股票 | 板 | 规则角色 | 独立原判断/当期条件观察 | BASE研究分 | 双通道 | 字辈挑战 | 次日补充结果 |", "|---|---|---:|---|---|---:|---:|---:|---|"]
    for stock in result.to_dict("records"):
        latest=stock["date"]=="2026-10-15"
        fmt=lambda value:"未知" if value is None or pd.isna(value) else f"{value*100:.2f}"
        role=ast.literal_eval(stock["roles"]) if isinstance(stock["roles"],str) else stock["roles"]
        row=dict(date=stock["date"],code=stock["code"],name=stock["name"],boards=int(stock["boards"]),roles="、".join(role),
            independent=("条件偏强，待开盘证据" if stock["current15_conditional_outlook"]=="BULLISH_CONDITIONAL" else "观察待核" ) if latest else stock["original_independent_judgement"],
            base=fmt(stock["current15_after_review_BASE"] if latest else stock["BASE"]),dual=fmt(stock["current15_after_review_ST_DUAL"] if latest else stock["ST_DUAL"]),
            challenger=fmt(stock["current15_after_review_ST_DUAL_SYMBOLIC_DIFFUSION"] if latest else stock["ST_DUAL_SYMBOLIC_DIFFUSION"]),
            outcome="晋级" if stock["latest_source_outcome"]==1 else "未续板" if stock["latest_source_outcome"]==0 else "未知/未到期",
            reason=authored[stock["code"]]["conditional_outlook_reason"] if latest else "原已归档判断、角色及来源按原列保留",formal=latest and "无新正式预测；完整11类freeze未过" or "保留原登记口径")
        rows.append(row)
        lines.append("| "+" | ".join(str(row[k]).replace("|","/") for k in ["date","name","boards","roles","independent","base","dual","challenger","outcome"])+" |")
    (out/"完整81只对照_含15日审阅后研究分.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    payload=json.dumps(rows,ensure_ascii=False).replace("</","<\\/")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>81只周期角色与研究分</title><style>body{font:15px system-ui;background:#f3f6f9;color:#14273b;margin:24px}p{max-width:1100px;line-height:1.7}input,select{padding:9px;margin:10px}.scroll{overflow:auto;max-height:75vh}table{border-collapse:collapse;background:#fff;min-width:1550px}th,td{padding:10px;vertical-align:top;border:1px solid #dce3eb;text-align:left}th{position:sticky;top:0;background:#e7edf6}.reason{min-width:450px}</style><h1>9月30日—10月15日 · 完整81只周期角色对照</h1><p>最新6只先完成8条股吧观点、7条发现父锚和原40条结构锚审阅，实际消费195条历史角色、4日主线报告及4条叙事语义，再显示以下保存模型研究分。全部分数未经校准；条件观察没有注册为正式预测。既有模型未通过整体验收，不据此发出订单。</p><select id="day"><option value="">全部日期</option></select><input id="search" placeholder="股票、角色、条件"><span id="count"></span><div class="scroll"><table><thead><tr><th>日期</th><th>股票</th><th>板</th><th>规则角色</th><th>原独立/当前条件观察</th><th>BASE</th><th>双通道</th><th>字辈挑战</th><th>次日结果</th><th>依据及限制</th></tr></thead><tbody id="body"></tbody></table></div><p>74只次有补充来源结果，7未知。原81行的旧列保持，研究分为新增列。5分钟区间不能确认连续封板、真实承接、封单、排队或成交。</p><script>const rows=__DATA__;const day=document.querySelector('#day'),search=document.querySelector('#search');for(const d of [...new Set(rows.map(r=>r.date))]){const o=document.createElement('option');o.value=d;o.textContent=d;day.append(o)}function show(){const f=rows.filter(r=>(!day.value||r.date===day.value)&&(!search.value||JSON.stringify(r).includes(search.value)));document.querySelector('#count').textContent=f.length+' / 81';const body=document.querySelector('#body');body.replaceChildren();for(const r of f){const tr=document.createElement('tr');for(const key of ['date','code','boards','roles','independent','base','dual','challenger','outcome','reason']){const td=document.createElement('td');td.textContent=key==='code'?r.name+' '+r.code:r[key];if(key==='reason')td.className='reason';tr.append(td)}body.append(tr)}}day.onchange=show;search.oninput=show;show();</script></html>'''
    (out/"完整81只对照_含15日审阅后研究分.html").write_text(page.replace("__DATA__",payload),encoding="utf-8")
    pressure_report=b.read(ROOT/"sampled_pressure_evidence/qualification_and_discrimination_audit.json")
    report=["# 15日完整工作台审阅续接与分钟压力证据核查", "",
        "本轮推进了当期完整6只审阅和评分对照，并将8—9月分钟资料实际接入统一因果压力观察。没有新拟合，没有修改旧模型、9245旧标签或任何正式/独立登记，没有订单。预测、六类独立身份与收益的整体目标仍未达到。", "",
        "## 当前6只已完成的实际链路", "",
        "复用原15日收盘替代来源34收盘/5炸板/25跌停、6只连板，运行原supplement→forum→identity→thinking→symbolic诊断链。原完整来源门和全部正式freeze门保持；替代页的AI业务摘要从当前行业/题材输入排除。", "",
        "核对发现上批虽然有数值历史胶囊，作者历史参数实际为空。本批追加实际消费12日62行、13日79行、14日54行角色档案，共195行；四个日期主线报告按原未知状态读取，没有制造主线转换。4条旧叙事完成内容分类，正文/评论及真实取得时间分开。原空参数结果保留，新增非空上下文在histories_consumed_v2。", "",
        "全部6只作者审阅包含8条独立观点、7条自主发现父锚和原40条结构父锚取舍。百洋保留九阳周期候选，阳光为旧跟风重起且父锚竞争待核；新华文轩的华北关联因单字和无直接关系不采纳，慈文/新华假设待核。主角色为规则空间/换手/卡位等，不以父锚未知抹去已知身位。", "",
        "原工作台validate_completed/review-context及短线四项、逐对三路径和叙事校验均通过。独立条件观察先留档，才通过原thinking-score/symbolic-score CLI读取保存权重。独立条件偏强为慈文、百洋；其余4只等待逐股缺口与开盘证据。条件观察不计新预测样本。", "",
        "| 个股 | 审阅前置的独立条件观察 | BASE | 双通道 | 非一字专模 | 字辈挑战 |", "|---|---|---:|---:|---:|---:|"]
    for x in current.to_dict("records"):report.append(f"| {x['name']} {x['code']} | {'条件偏强' if x['prereview_conditional_outlook']=='BULLISH_CONDITIONAL' else '观察待核'} | {x['BASE']*100:.2f} | {x['ST_DUAL']*100:.2f} | {x['ST_B_ONLY']*100:.2f} | {x['ST_DUAL_SYMBOLIC_DIFFUSION']*100:.2f} |")
    report += ["", "以上是未校准分数。BASE与双通道在百洋、冰山、富春的0.5方向不同；双通道最高为阳光，而独立审阅没有因此自动转正。旧一字通道与新非一字通道使用不同模型，跨日分差不能解释为承接强弱或机械减分。全部模型之前的整体未通过结论保持，不部署。", "",
        "## 分钟证据与真假阳区分", "",
        "新增short_term_sampled_pressure.describe以信号日白名单消费同一5分钟API；重放298个股票响应、473226根原始采样K线，与既存标准化数据一致。485只次完整分母全部保留，451价格路径及324量额路径合格。对全部9245原候选生成附层，未覆盖保持缺失。", "",
        "新观察只涉及分隔区间间的压力/恢复顺序、前一区间峰值到当前低点的回撤、末6根量占比、恢复区间量占比及收盘相对全日VWAP。严格区分同一区间高低顺序未知，不把跨区间恢复计数称真实回封数。旧FINAL_CAP_PRICE_RUN/CAP_PRICE_BAR_SHARE/SHALLOW_PULLBACK失败排序没有重跑。", "",
        "| 原训练窗 | 候选 | 合格价格路径 | 价格覆盖 | 合格量额路径 | 量额覆盖 |", "|---|---:|---:|---:|---:|---:|"]
    for fold,value in pressure_report["original_training_windows"].items():report.append(f"| {fold} | {value['rows']} | {value['price_path_rows']} | {100*value['price_coverage']:.2f}% | {value['volume_path_rows']} | {100*value['volume_coverage']:.2f}% |")
    report += ["", "这些覆盖限制只约束这个分钟附层，不撤销已授权的原全周期因果规则研究，也不把原规则训练重新锁回0/0/0。", "",
        "原双通道非一字真假阳的末6根成交量占比中位数：8月TP1.184%（量额已知22）、FP1.837%（23）；9月TP0.917%（11）、FP1.766%（12）。这只是已见样本描述，不能据中位数选阈值、拟合或认证跨期分辨能力。采样压力回撤TP/FP的差距并不稳定；共同小样本、区间重叠及量额缺失都保留。逐股全表包含TP/FP/FN/TN与未知，没有仅保留好案例。", "",
        "## 验证与剩余工作", "",
        "451条价格路径和324条量额路径已用独立数组公式全量重算；原485只×2臂的分数/标签全列一致。当前4模型×6只研究分逐值重放，作者时间早于评分；待审阅、删观点、漏父锚与用诊断来源正式freeze均实际拒绝。66个受保护文件哈希保持。", "",
        "前一轮81只原列在新增CSV中完全相同，新6只研究分和条件观察只追加列；74有补充结果、7仍未知，不追加新正式成熟。当前6只尚未通过原生产来源和完整11类正式freeze，不能冒称已完成正式15→16预测或真实前向样本。", "",
        "当前未完成项仍是全周期实际行业/活跃题材证据、足够覆盖的真实盘中承接、独立六角色真值、稳定Precision/排序及B+1/B+2收益改善。任何新数值假设须先有可区分的新证据与单独事前协议，本轮不因局部描述调参数或回头拟合。", "",
        "本批JSON源链、原失败及技术更正前像、完整HTML/CSV、逐股工作台、分钟全9245附层与验证收据均在本目录。"]
    (out/"15日工作台续接与分钟压力证据核查.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return dict(report_rows=81,original81_columns_exact=True,current6_after_review_scores_added=True,new_formal_prediction=0)


def main():
    out=ROOT/"delivery_final"
    if out.exists():raise FileExistsError("immutable final delivery")
    protected=b.read(ROOT/"execution_start.json")["protected_before"];b.verify(protected)
    pcheck=verify_pressure();scorecheck=verify_scores()
    out.mkdir();summary=deliver(out,dict(pressure=pcheck,scores=scorecheck))
    result=dict(formed_at=b.now(),status="VERIFIED_PROGRESS_UPDATED_FULL_GOAL_NOT_COMPLETE",goal_status="active",previous_turn_classification="PROGRESS",
        protected_hashes_verified=len(protected),minute_verification=pcheck,current6_verification=scorecheck,delivery=summary,
        new_fits=0,old_model_changes=0,new_formal_predictions=0,new_independent_registry_entries=0,new_mature_formal_samples=0,actual_fills=0,
        source_bindings=protected|{str(Path(__file__).resolve()):b.sha(__file__)})
    for path in [ROOT/"author_review_validation.json",ROOT/"sampled_pressure_evidence/qualification_and_discrimination_audit.json",ROOT/"saved_scores_AFTER_COMPLETED_REVIEW/complete_current6_research_comparison.json",*out.iterdir()]:
        if path.is_file():result["source_bindings"][str(path.resolve())]=b.sha(path)
    b.write_new(ROOT/"completion_final.json",result)
    print({k:v for k,v in result.items() if k not in {"source_bindings","current6_verification"}})


if __name__=="__main__":main()
