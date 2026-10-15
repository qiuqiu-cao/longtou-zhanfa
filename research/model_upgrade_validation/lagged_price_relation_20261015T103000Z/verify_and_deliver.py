"""Replay models and all denominators, then append a complete81-row delivery."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import html
import json
import xml.etree.ElementTree as ET
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import lagged_price_relation_research as r
import historical_price_relation_workbench as workbench
from historical_symbolic_verify import classification,ranking,assert_values

ROOT=Path(__file__).resolve().parent


def independent_interval(left,right):
    f=left.merge(right[["date","code","score"]],on=["date","code"],suffixes=("","_new"),validate="one_to_one")
    f=f[f.daily_one_word.eq(0)&f.label_valid.fillna(False).astype(bool)&f.target.eq(1)].copy()
    f["old_hit"]=f.score.gt(.5).astype(int);f["new_hit"]=f.score_new.gt(.5).astype(int)
    days=f.groupby("date").agg(n=("code","size"),a=("old_hit","sum"),z=("new_hit","sum")).to_numpy(float)
    n=len(days);width=min(5,n);rng=np.random.default_rng(20261015)
    starts=rng.integers(0,n,(2000,(n+width-1)//width))
    deltas=[]
    for draw in starts:
        positions=[(int(start)+offset)%n for start in draw for offset in range(width)][:n]
        sums=days[positions].sum(axis=0);deltas.append((sums[2]-sums[1])/sums[0])
    return np.quantile(deltas,[.0125,.9875])


def verify():
    protocol=b.read(ROOT/"protocol_BEFORE_ANY_FIT.json");b.verify(protocol["source_bindings"])
    protected=b.read(ROOT/"execution_start.json")["protected_before"];b.verify(protected)
    path=ROOT/"fixed_comparison/complete_paired_predictions.csv"
    pred=pd.read_csv(path,dtype={"code":str},low_memory=False)
    existing=pd.read_csv(r.previous.ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    for arm in ["BASE","ST_DUAL",r.REFERENCE]:
        a=existing[existing.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        z=pred[pred.arm.eq(arm)].sort_values(["date","code"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,z[list(a)])
    assert len(pred)==15084 and not pred.duplicated(["date","code","arm"]).any()
    full,current=r.load();base=pred[pred.arm.eq("BASE")].set_index(["date","code"]).score
    months=[]
    for item in protocol["training_checks"]:
        month=item["month"];train=r.old.history.training_rows(full,item["cutoff"],"relay")
        assert train.date.nunique()==252 and train.target_date.max()<item["cutoff"]
        assert r.old.history._frame_digest(train,r.FEATURES)==item["training_digest"]
        weights=(1/train.groupby("date").date.transform("size")).to_numpy(copy=True)
        for label in [0,1]:
            members=train.target.to_numpy()==label;weights[members]*=.5/weights[members].sum()
        np.testing.assert_allclose(weights*252,r.old.training_weights(train),atol=1e-13,rtol=0)
        model_path=ROOT/"fixed_comparison"/(month+"_model.joblib");saved=joblib.load(model_path)
        started=b.read(ROOT/"fixed_comparison"/(month+"_fit_STARTED.json"))
        assert protocol["formed_at"]<started["formed_at"]<=saved["trained_at"]
        assert saved["training_digest"]==item["training_digest"]==started["training_digest"]
        assert saved["estimator"]==r.old.PARAMS and saved["features"]==r.FEATURES
        test=current if month=="2026-10" else full[full.date.str[:7].eq(month)]
        stored=pd.read_csv(ROOT/"fixed_comparison"/(month+"_predictions_BEFORE_EVALUATION.csv"),dtype={"code":str})
        assert list(zip(stored.date,stored.code))==list(zip(test.date,test.code))
        with threadpool_limits(limits=2):
            raw=saved["model"].predict_proba(test[r.FEATURES])[:,list(saved["model"].classes_).index(1)]
        np.testing.assert_allclose(raw,stored.b_score,atol=1e-13,rtol=0)
        expected=[base.loc[(x.date,x.code)] if x.daily_one_word==1 else raw[i] for i,x in enumerate(test.itertuples())]
        np.testing.assert_allclose(expected,stored.score,atol=1e-13,rtol=0)
        pd.testing.assert_series_equal(test.target.reset_index(drop=True),stored.target.reset_index(drop=True),check_dtype=False)
        months.append(dict(month=month,train_rows=len(train),mature_days=252,prediction_rows=len(test),replay_exact=True))
    result=b.read(ROOT/"fixed_comparison/model_validation_results.json");checks=0;interval_checks=0
    for fold in r.old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        parts={}
        for arm in ["BASE","ST_DUAL",r.REFERENCE,r.ARM]:
            f=pred[pred.arm.eq(arm)&pred.date.between(fold["start"],fold["end"])];parts[arm]=f
            metrics=result["comparisons"][fold["id"]]["metrics"][arm]
            assert_values(metrics["full"],classification(f));assert_values(metrics["nonword"],classification(f[f.daily_one_word.eq(0)]));checks+=2
            for key,k,cohort,height in [("top1",1,False,False),("top2",2,False,False),("same_height_top1",1,True,False),("same_height_top2",2,True,False),("height_baseline_top1",1,False,True)]:
                assert_values(metrics[key],ranking(f,k,cohort,height));checks+=1
        for arm,key in [("BASE","recall_interval_vs_BASE"),(r.REFERENCE,"recall_interval_vs_corrected_reference")]:
            expected=independent_interval(parts[arm],parts[r.ARM]);actual=result["comparisons"][fold["id"]][key]
            np.testing.assert_allclose(expected,[actual["lower"],actual["upper"]],atol=1e-13,rtol=0);interval_checks+=1
    logs=pd.read_csv(ROOT/"fixed_comparison/all_top1_execution_scenarios.csv",dtype={"code":str})
    execution=joblib.load(b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/execution_context.joblib")
    selected=execution["bars"][execution["bars"].code.isin(logs.code)]
    bars={(x.date,x.code):x for x in selected.itertuples()};sessions=execution["sessions"];cashchecks=0
    for x in logs.itertuples():
        if x.status!="INFERRED_OPEN_PROXY_NOT_ACTUAL_FILL":continue
        i=sessions.index(x.date);entry=bars[(sessions[i+1],x.code)]
        assert x.B==sessions[i+1] and np.isclose(x.entry_open,entry.open)
        for horizon in [1,2]:
            net=getattr(x,f"b{horizon}_net");mark=getattr(x,f"b{horizon}_mark")
            if pd.isna(mark):continue
            exitbar=bars[(sessions[i+1+horizon],x.code)]
            assert np.isclose(mark,exitbar.close/(entry.open*1.001)-1,atol=1e-13)
            if pd.notna(net):
                assert np.isclose(net,exitbar.close*.999/(entry.open*1.001)-1-.003,atol=1e-13);cashchecks+=1
            if getattr(x,f"status_b{horizon}",None)=="SELL_BLOCKED_ONE_PRICE_DOWN_LIMIT":assert pd.isna(net)
    assert result["prediction_screen_pass"] is False and result["production_deployment_authorized"] is False
    # Actual duplicate-run probe: stop must precede every possible fit.
    before={str(p.resolve()):b.sha(p) for p in (ROOT/"fixed_comparison").glob("*_model.joblib")}
    with patch.object(Pipeline,"fit",side_effect=AssertionError("duplicate must not fit")):
        try:r.run()
        except FileExistsError:duplicate_rejected=True
        else:raise AssertionError("duplicate was allowed")
    b.verify(before)
    b.write_new(ROOT/"duplicate_run_REJECTED_BEFORE_FIT.json",dict(formed_at=b.now(),rejected_before_fit=duplicate_rejected,new_fits=0,models_unchanged=before))
    # Same causal function in real current workbench and full historical batch.
    context=b.read(ROOT/"current6_price_context_SCORES_MASKED.json")
    artifact=joblib.load(r.CONTEXT);batch=artifact["rows"];batch=batch[batch.date.eq("2026-10-15")].sort_values("code").reset_index(drop=True)
    live=pd.DataFrame(context["rows"]).sort_values("code").reset_index(drop=True)
    for field in r.relation.FIELDS:
        np.testing.assert_allclose(pd.to_numeric(live[field]),batch[field],atol=1e-13,rtol=0,equal_nan=True)
    scores=b.read(ROOT/"current6_saved_model_scores_AFTER_REVIEW.json")
    original_review=b.read(scores["review_path"])
    assert original_review["formed_at"]<context["formed_at"]<scores["formed_at"]
    probes=ROOT/"guard_probes";probes.mkdir(exist_ok=False)
    pending=b.read(scores["review_path"]);pending["author_review_completed"]=False
    pending_path=probes/"pending_review.json";b.write_new(pending_path,pending)
    with patch.object(Pipeline,"predict_proba",side_effect=AssertionError("pending review must not score")):
        try:workbench.score(context["facts_path"],ROOT/"current6_price_context_SCORES_MASKED.json",pending_path,ROOT/"fixed_comparison/2026-10_model.joblib",probes/"must_not_exist.json")
        except ValueError as error:pending_rejection=str(error)
        else:raise AssertionError("pending review allowed")
    assert not (probes/"must_not_exist.json").exists()
    b.verify(protected)
    bindings=protected|protocol["source_bindings"]|{str(p.resolve()):b.sha(p) for p in [Path(__file__),path,ROOT/"fixed_comparison/model_validation_results.json",ROOT/"current6_price_context_SCORES_MASKED.json",ROOT/"current6_saved_model_scores_AFTER_REVIEW.json",*list((ROOT/"fixed_comparison").glob("*_model.joblib")),*list((ROOT/"fixed_comparison").glob("*_fit_STARTED.json"))]}
    verification=dict(formed_at=b.now(),status="PASS_REPLAY_AND_GUARDS_NOT_PREDICTIVE_ACCEPTANCE",model_replays=16,months=months,
        complete_paired_rows=len(pred),metrics_recomputed=checks,recall_intervals_recomputed=interval_checks,
        execution_cashflows_recomputed=cashchecks,protected_files_exact=len(protected),existing_three_arm_rows_exact=True,
        current_batch_vs_workbench_fields_exact=len(r.relation.FIELDS)*6,pending_review_rejected=pending_rejection,
        duplicate_before_fit_rejected=True,new_fits_in_verifier=0,prediction_screen_pass=False,
        inference_not_new_independent_judgement=True,new_formal_predictions=0,new_mature_samples=0,actual_fills=0,source_bindings=bindings)
    b.write_new(ROOT/"independent_verification.json",verification)
    return verification,pred,result


def deliver(verification,pred,result):
    output=ROOT/"delivery_final";output.mkdir(exist_ok=False)
    previous=r.previous.ROOT/"delivery_final/完整81只对照_含更正炸板输入研究分.csv"
    original=pd.read_csv(previous,dtype={"code":str},low_memory=False);table=original.copy()
    oldscores=pred[pred.arm.eq(r.ARM)].set_index(["date","code"]).score.to_dict()
    current=b.read(ROOT/"current6_saved_model_scores_AFTER_REVIEW.json")
    oldscores.update({("2026-10-15",row["code"]):row["channel_b_score"] for row in current["rows"]})
    table["lagged_price_relation_research_score"]=[oldscores[(row.date,row.code)] for row in table.itertuples()]
    table["lagged_price_relation_model_status"]="FAILED_FIXED_SCREEN_UNCALIBRATED_RESEARCH_ONLY"
    features=joblib.load(r.CONTEXT)["rows"]
    table=table.merge(features[["date","code"]+r.relation.FIELDS],on=["date","code"],validate="one_to_one")
    views=b.read(b.ROOT/"results/daily_review_20261015T093600Z/independent_forward_views_BEFORE_MODEL.json")
    bycode={row["code"]:row for row in views["rows"]}
    mapping={"promotion_judgement":"additional_15_author_call","lean":"additional_15_author_lean", "support":"additional_15_author_support", "rebuttal":"additional_15_author_counterevidence", "cancel_condition":"additional_15_author_cancel"}
    for key,column in mapping.items():
        table[column]=[bycode[row.code][key] if row.date=="2026-10-15" else None for row in table.itertuples()]
    table["additional_15_author_exposure"]=[views["model_exposure"] if row.date=="2026-10-15" else None for row in table.itertuples()]
    pd.testing.assert_frame_equal(table[list(original)],original)
    csv_path=output/"完整81只对照_追加价格联动研究与独立观点.csv"
    table.to_csv(csv_path,index=False,encoding="utf-8-sig")
    summary=[]
    for fold,value in result["comparisons"].items():
        for arm,metrics in value["metrics"].items():
            m=metrics["nonword"];summary.append(dict(period=fold,arm=arm,known=m["scored"],positives=m["positives"],
                TP=m["tp"],FP=m["fp"],Recall=m["Recall"],Precision=m["Precision"],F1=m["F1"],
                Top1=metrics["top1"]["candidate_hit_rate"],Top2=metrics["top2"]["candidate_hit_rate"]))
    pd.DataFrame(summary).to_csv(output/"四方案完整指标.csv",index=False,encoding="utf-8-sig")
    display=table[["date","code","name","boards","cycle_regime","extended_primary_role","original_independent_judgement","additional_15_author_call","BASE","current15_after_review_BASE","ST_DUAL","current15_after_review_ST_DUAL","exact_attempt_corrected_model_research_score","lagged_price_relation_research_score","source_supplement_target","latest_source_outcome","latest_source_status"]].copy()
    for field in ["BASE","ST_DUAL"]:
        display[field]=display[field].fillna(display["current15_after_review_"+field])
    display=display.drop(columns=["current15_after_review_BASE","current15_after_review_ST_DUAL"])
    labels={"date":"信号日","code":"代码","name":"股票","boards":"板数","cycle_regime":"周期规则","extended_primary_role":"角色候选（非真值）","original_independent_judgement":"原独立记录","additional_15_author_call":"追加15日独立观点","BASE":"BASE研究分","ST_DUAL":"原双通道研究分","exact_attempt_corrected_model_research_score":"更正炸板研究分","lagged_price_relation_research_score":"本次联动研究分","source_supplement_target":"补充结果","latest_source_outcome":"最新补充结果","latest_source_status":"结果资格"}
    for key in ["BASE","ST_DUAL","exact_attempt_corrected_model_research_score","lagged_price_relation_research_score"]:
        display[key]=display[key].map(lambda x:"未知" if pd.isna(x) else f"{x*100:.2f}")
    display=display.rename(columns=labels).fillna("未知")
    html_table=display.to_html(index=False,escape=True,table_id="cohort")
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整81只短线思维研究对照</title><style>body{font:14px/1.6 system-ui,sans-serif;margin:32px;color:#173042;background:#f5f8fb}h1{font-size:26px}.note{max-width:1050px}input{padding:10px;width:320px;border:1px solid #aab9c6;border-radius:6px}.wrap{overflow:auto;background:white;border-radius:8px;margin-top:18px;max-height:75vh}table{border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #e1e8ed}th{position:sticky;top:0;background:#173042;color:white}tr:nth-child(even){background:#f3f7fa}td:nth-child(7){white-space:normal;min-width:220px;max-width:350px}</style><h1>完整81只短线思维研究对照</h1><p class="note">9月30日至10月15日，81只次全部保留。原字段未改，新增本次研究分及另份15日作者观点。全部分数未校准；本次固定比较失败，未部署。15日六只等待16日结果，其他历史补充结果不倒填正式账本。角色为规则候选，统计联动不证明行业、题材或资金因果。</p><input id="search" placeholder="筛选日期、股票、代码或角色"><p id="count">81 / 81 行</p><div class="wrap">'''+html_table+'''</div><script>const s=document.getElementById('search'),rs=[...document.querySelectorAll('#cohort tbody tr')];s.oninput=()=>{let n=0;const q=s.value.toLowerCase();rs.forEach(r=>{let show=r.textContent.toLowerCase().includes(q);r.hidden=!show;n+=show});document.getElementById('count').textContent=n+' / 81 行'};</script></html>'''
    html_path=output/"完整81只对照_追加价格联动研究与独立观点.html"
    html_path.write_text(document,encoding="utf-8")
    assert document.count("<tr>")==82 and "NaN" not in document and "�" not in document
    audit=b.read(ROOT/"source_qualification_BEFORE_NEW_FIT.json")
    sep=result["comparisons"]["2026Sep"]["metrics"];new=sep[r.ARM];before=sep["ST_DUAL"]
    lines=["# 历史价格联动证据与固定比较", "", "本次完成一个新证据方案的全历史构建、16个月固定训练和完整复算，结果未通过预登记升级筛选。误报减少与召回退步同时存在，未部署。", "", "## 新增证据及来源资格", "", "从截至信号日可见的日线建立统计联动，使用T-25至T-6的20个官方交易日收益率，扣除同日归档股票收益中位数；至少12个共同有效观测才计算相关系数。正相关平方作为连续权重，不搜索相关阈值。统计同群不能认证行业、活跃题材、父子资金传导或龙头身份。", "", "完整9245历史行和70条10月增量均保留，共1465910条父锚/封板/炸板配对。初版可选昨收字段缺失和46条原候选与精确炸板源冲突已在拟合前修正：只用相邻官方交易日真实收盘，不跨缺失交易日；冲突股票保留原候选记录，但不同时计为封板支持与炸板压力。初版产物、失败说明和源码前像保留。", "", "| 训练截止 | 原成员 | 足12日历史收益覆盖 | 收盘联动可观察覆盖 | 所选父锚相关可观察 |", "|---|---:|---:|---:|---:|"]
    for item in audit["training_windows"]:
        lines.append(f"|{item['cutoff']}|{item['rows']}|{item['own_return_history_coverage']:.2%}|{item['observed_closing_relation_coverage']:.2%}|{item['selected_parent_correlation_coverage']:.2%}|")
    lines += ["", "覆盖率不是全部字段已核。15日当前六只中只有4只满足12日历史门槛；慈文、百洋各有11个有效观测，联动保持未知。其余4只的收盘配对可观察比例约57.58%，没有把缺行情的竞争者删出分母。历史原源仍带当前股票池幸存者偏差，标准化日线和晚取得资料不冒充原时点一手行情。", "", "## 固定比较", "", "仅追加12个价格联动字段，使用更正炸板输入，保留原252成熟日、成员、标签、每日/类别权重、HGB参数及严格0.5判断。原BASE、双通道、名称扩散和更正炸板模型均未重拟合。2025年7月至2026年10月各一次，共16次；协议先落盘、每月预测先于指标保存。", "", "| 时段 | 方案 | TP/FP | Recall | Precision | F1 | Top1 | Top2 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    arm_labels={"BASE":"BASE","ST_DUAL":"原双通道",r.REFERENCE:"更正炸板",r.ARM:"本次价格联动"}
    for item in summary:
        lines.append(f"|{item['period']}|{arm_labels[item['arm']]}|{item['TP']}/{item['FP']}|{item['Recall']:.2%}|{item['Precision']:.2%}|{item['F1']:.2%}|{item['Top1']:.2%}|{item['Top2']:.2%}|")
    lines += ["", "9月同池真实非一字143只次，107条结果有效、40个正例，36条未知保留。相对原双通道，FP22→12、TP20→17，Precision47.62%→58.62%，Recall50%→42.50%，F148.78%→49.28%。Top1从8/13升至9/13（完整选16天、3天结果未知）；Top2从17/26降至16/26（完整32条、6未知）。相对更正炸板方案，Top2同为16/26，Recall47.50%→42.50%。前两历史段Recall、F1均低于更正炸板参考，因此整体失败；不按9月局部改善启用，也不继续调参或重复拟合。", "", "9月Recall相对BASE的探索性日期块区间为+17.07至+38.24个百分点；相对更正炸板参考为-30.77至+14.29个百分点。区间已独立复算，但复用的双臂尾部校正没有覆盖此前全部顺序挑战，历史也已看过，不能据此宣称新的确认性显著提升或前向有效。", "", "## 执行与当期衔接", "", "9月本方案Top1完整选16次：8个可入场开盘代理、6个涨幅门取消、2个涨停开盘不成交；B+1有8个退出代理、B+2仅7个。B+2均值7.76%、一例净亏超过7%，仅占完整选择43.75%；原双通道5个B+2场景均值8.04%，分母和股票不同，不能据此判净收益改善。B是入场日，B+1/B+2从B起算；费用、滑点、未知与卖出受阻单独保留，实际及纸面成交均0。", "", "`historical_price_relation_workbench.py context/score`是原工作台的只读研究伴随接口，当前6只已真实运行。完整历史/主线历史/叙事/父锚参数原样传递，仍校验原逐股审阅；12字段×6只与批量完全一致。原工作台和正式freeze门未改。当前新研究分已在原完成审阅后另存；这些是失败模型分数，未校准，不是新独立观点或正式预测。", "", "另已核验results/daily_review_20261015T093600Z中的6份公司公告PDF和正文哈希、真实收件时间及该归档09:41形成的逐股观点。该作者慈文/百洋/富春偏晋级、冰山偏断板、阳光/新华文轩未知；声明未看模型但未独立认证盲审。它与本线程原9:44逐股审阅分列追加，不覆盖旧方向。全部15日结果仍等待16日，未新增正式登记。", "", "## 验证及未完成目标", "", f"独立复算9555条配对、全部9315行的三类支持量；回放16模型、15084完整配对行、{verification['metrics_recomputed']}组分类/排序指标、8份Recall区间及{verification['execution_cashflows_recomputed']}条代理现金流。{verification['protected_files_exact']}个原文件哈希保持；重复实验在拟合前拒绝，未完成审阅在推断前拒绝。", "", "本批拟合16，原模型重拟合0，正式新增预测0、新正式成熟0、独立账本新增0、实际/纸面成交0。总体目标仍未完成：跨期稳定的Recall/F1和排序提升、独立六角色识别正确率、真实行业/主线覆盖、足量可执行B+2收益均未验收。当前已有价格联动证据只支持下一次具体误差审查，不能据本批结果继续换窗口、相关门槛或模型参数。", "", "完整81行保留原字段，只追加新分和另一份15日作者观点；原74条补充已知/7条未知资格不变。", "", "附件：完整81只CSV/HTML、四方案完整指标、原始全配对无标签账本、协议、每月模型/提前预测、独立核验收据。"]
    report_path=output/"历史价格联动证据与固定比较交付.md"
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    interface=output/"接口与复用说明.md"
    interface.write_text("# 接口与复用说明\n\n先按原relay_review_workbench完成股吧、完整历史/主线/叙事和父锚输入及作者审阅。研究上下文用：\n\n```powershell\npython historical_price_relation_workbench.py context --facts <原完整facts.json> --bundle <已更新到信号日的完整engine.joblib> --output <新遮蔽分数context.json>\npython historical_price_relation_workbench.py score --facts <原完整facts.json> --context <新context.json> --review <原已完成审阅.json> --model <2026-10_model.joblib> --output <新研究分.json>\n```\n\n新日期不能复用旧日期池；engine须绑定当日完整收盘/炸板来源及此前日线。缺日线不跳到最近成交日，未知配对不删除。本接口仅追加研究视图，不写原facts或正式登记。当前已运行且批量/单日相同。该模型已失败，不是生产推荐。\n",encoding="utf-8")
    suite=ET.parse(ROOT/"related_tests.xml").getroot()
    tests=sum(int(x.attrib.get("tests",0)) for x in suite.iter("testsuite"))
    failures=sum(int(x.attrib.get("failures",0))+int(x.attrib.get("errors",0)) for x in suite.iter("testsuite"))
    assert tests>=71 and failures==0
    b.verify(verification["source_bindings"])
    proof=dict(formed_at=b.now(),status="DELIVERED_FIXED_COMPARISON_AND_FULL81_APPEND_ONLY",full_objective_complete=False,
        source_phase_fits=0,new_challenger_fits=16,old_model_refits=0,paired_prediction_rows=len(pred),tests_passed=tests,test_failures=failures,
        protected_files_exact=verification["protected_files_exact"],full_current_table_rows=81,original_all_table_columns_exact=True,
        new_formal_predictions=0,new_independent_registry_entries=0,new_formal_maturity=0,actual_fills=0,paper_fills=0,
        prediction_screen_pass=False,model_deployed=False,challenge_retries_allowed=False,
        next_step="Use saved full-pool rows to inspect whether covariance is reducing false positives at the cost of parent-linked true positives; no window/threshold/learner retries absent distinct qualified evidence",
        source_bindings=verification["source_bindings"]|{str(p.resolve()):b.sha(p) for p in [csv_path,html_path,report_path,interface,ROOT/"related_tests.xml",Path(__file__)]})
    b.write_new(ROOT/"completion_final.json",proof)
    print({k:v for k,v in proof.items() if k not in {"source_bindings","next_step"}})


if __name__=="__main__":
    verified,pred,result=verify();deliver(verified,pred,result)
