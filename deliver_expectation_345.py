"""Assemble the completed 3/4/5 comparison and immutable evidence index."""
from pathlib import Path
import html
import json
import xml.etree.ElementTree as ET
import pandas as pd

import short_term_thinking_dataset_builder as b
import expectation_345_research as trial
import expectation_345_review_packets as packets

ROOT=trial.ROOT
ARMS=["BASE","ST_DUAL","CORRECTED_CAUSAL","CORRECTED_CAUSAL_COST1P5"]
NAMES={"BASE":"正式BASE","ST_DUAL":"旧双通道","CORRECTED_CAUSAL":"新校正特征","CORRECTED_CAUSAL_COST1P5":"新特征＋误报成本1.5"}


def pct(value):return "未知" if value is None or pd.isna(value) else f"{value*100:.2f}%"


def link(path,label):return f"[{label}]({Path(path).resolve().as_posix()})"


def markdown_table(header,rows):
    return "\n".join(["|"+"|".join(header)+"|","|"+"|".join(["---"]*len(header))+"|"]+["|"+"|".join(str(v) for v in row)+"|" for row in rows])


def build():
    dest=ROOT/"delivery";dest.mkdir(exist_ok=False)
    validation_path=ROOT/"evaluation_scope_alignment/verification_and_corrected_evaluation.json"
    llm_path=ROOT/"LLM_evaluation/LLM_repeat_verification_and_results.json"
    validation,llm=b.read(validation_path),b.read(llm_path)
    b.verify(validation["source_bindings"]);b.verify(llm["source_bindings"])
    live_path=ROOT/"current6_retrained_research_workbench_AFTER_AUTHOR_REVIEW.json"
    live=b.read(live_path);b.verify(live["source_bindings"])
    assert live["feature_count"]==92 and live["training_and_live_API_feature_values_exact"]
    assert not validation["corrected_screen_pass"] and not validation["cost_screen_pass"]
    test_path=ROOT/"delivery_tests_with_null_fix.xml"
    suites=ET.parse(test_path).getroot();assert sum(int(x.attrib.get("failures",0))+int(x.attrib.get("errors",0)) for x in suites.iter("testsuite"))==0
    tests=sum(int(x.attrib.get("tests",0)) for x in suites.iter("testsuite"));assert tests==27
    table=[]
    for period,part in validation["comparisons"].items():
        for arm in ARMS:
            m=part["metrics"][arm]["nonword"];t=part["metrics"][arm]["top1"]
            table.append([period,NAMES[arm],m["candidates"],m["known"],m["tp"],m["fp"],pct(m["Recall"]),pct(m["Precision"]),pct(m["F1"]),f"{round(t['candidate_hit_rate']*t['known'])}/{t['known']}；另{t['unknown']}未知"])
    all_metrics=markdown_table(["时段","方案","完整非一字池","结果已知","TP","FP","召回率","精确率","F1","Top1已知命中"],table)
    llm_metrics=pd.read_csv(ROOT/"LLM_evaluation/standalone_effect_coverage_abstention_false_alarm.csv")
    lrows=[]
    selected=["BASE","ST_DUAL","CORRECTED_CAUSAL","LLM_REPEAT_1","LLM_REPEAT_2","LLM_REPEAT_3","LLM_MAJORITY_FIXED","LLM_UNANIMOUS_FIXED"]
    for arm in selected:
        x=llm_metrics[llm_metrics.truth.eq("SUPPLEMENTAL_DISPLAY_LABELS")&llm_metrics.period.eq("HISTORICAL75_EXPOSED")&llm_metrics.scope.eq("FULL")&llm_metrics.arm.eq(arm)].iloc[0]
        lrows.append([NAMES.get(arm,arm),int(x.tp),int(x.fp),int(x.abstentions),pct(x.Recall),pct(x.Precision),pct(x.F1),pct(x.decision_coverage),pct(x.full_known_accuracy)])
    ltable=markdown_table(["同池方案","TP","FP","弃判数","召回率","精确率","F1","已知结果判断覆盖","全已知分母正确率"],lrows)
    current=b.read(ROOT/"LLM_evaluation/current15_six_target16_forward_shadow_FROZEN.json")
    current_table=markdown_table(["相对名次","股票","三次方向","固定多数方向"],[[i+1,f"{x['name']} {x['code']}"," / ".join(v["call"] for v in x["three_authored_reviews"]),x["majority_call"]] for i,x in enumerate(current["rows"])])
    pooled=validation["pooled_historical_nonword"];a,z=pooled[trial.ARMS[0]],pooled[trial.ARMS[1]]
    root_report=dest/"第3至5项优化_独立评测重训与误报验收.md"
    text=f"""# 第3至5项优化：独立评测、校正特征重训与误报控制

本轮已完成3个Codex审阅上下文各81份实际判断、92个校正特征的16个月×2方案训练、全部32个保存模型重放，以及完整同池效果核验。无需外部API密钥。**第3项已补上可核验的稳定性和独立输出评测，第4项已实际重训并核对当前工作台；第5项试验降低了误报，但损失过多召回，没有验收通过。不能称三项预测缺陷均已解决。**

两项新数值方案均不通过事先固定的跨期标准，正式BASE继续保留。32个新研究模型保留供诊断；没有将失败方案接管正式排序。历史评测是已经见过的归档数据上的时间滚动研究，不是新盲测；真实收益和角色身份准确率本轮未认证。

## 第3项：3次真实独立输出及其局限

用户明确授权3个Codex子代理。每个只获分配的同源事实包和固定协议，手工审阅完整7日81只，候选顺序分别正序、倒序、轮换；未用代码自动填方向或理由，未按效果挑最佳审阅者。三份共243行，逐股支持、反证、取消条件和完整排名全部保存，事实引用按各自信号日校验。没有外部API推理；此前仅凭据探测返回401，用户选择无API后用本会话子代理完成。

81只中64只三次方向一致（79.01%），17只有分歧；两两一致率85.19%、91.36%、81.48%，对应κ为0.7773、0.8705、0.7234。当前15日6只中5只一致；两两一致6/6、5/6、5/6，三次完整相对排序一致。

**三份都如实声明接触过部分历史结果与旧分数。** fork后的上下文仍自动带入用户AGENTS历史说明，7日事实包的后续信号事实也可间接透露前日结果。因此以下历史效应只能诊断“输出了什么、分歧在哪里”，不能证明LLM独立盲测改善。没有把曝光标记改成false。稳定性也不等于正确率，三个同类模型上下文不是三种不同模型。

以下完整75只历史池，补充展示来源可核结果74只、28正例，1未知；所有方案使用同一74只结果。原冻结标签另表保留，补充展示结果没有混进正式账本。弃判的正例保留在召回率分母，全已知正确率也不剔除弃判。

{ltable}

固定多数决和全票决只作预先固定的比较，并非已通过效果认证的新融合模型。原作者记录完整保存：81只中60只有原审阅记录，21只无原独立记录；无记录和明确UNKNOWN分别保留，没有当作FAIL或正确。三份逐股理由可在下方完整工作台展开。

已补单信号日封闭包生成器，后续评测一次只给T日完整池，排除后续信号日间接泄漏，检查显式分数/目标列及T之后日线，沿原工作台完成前置预读和作者留档后才展示模型。当前单日包示例只是格式整改，没有追加第4次审阅或改写原3份判断。

## 第4项：校正后的92个特征已真正进入训练

使用原始全池252个已成熟信号日，标签到期日严格早于月初截止；16个月覆盖2025年7月至2026年10月，两个固定方案各月拟合一次，总计32次。16个窗的候选成员、到期标签与旧完整训练池逐行一致；不是只拿历史赢家或非一字正例训练。原BASE和此前挑战模型重拟合0次。

92个显式白名单特征包括全前池竞争分母及未知区间、当期炸板范围、多个周期状态、完整活跃父锚生命状态与启动滞后、并存角色假设、严格价型/连续路径和来源完整性。旧一字代理与两列证据不足的换手引用不进入本批训练。角色仍为因果规则假设，没有变成独立身份真值。每窗92列中87—89列有变化，全空列0；部分证据仍缺失，非空不等于资料全核或分类准确。

算法保持固定HistGradientBoosting参数：learning_rate=0.06、max_iter=120、max_leaf_nodes=7、max_depth=3、min_samples_leaf=25、l2_regularization=3、early_stopping=false。首个方案沿原每日权重及类别平衡；第二个仅预先固定负类成本1.5。判断阈值始终严格大于0.5，未按效果再找权重、阈值或算法。所有分数未校准，不能当事件百分比概率。

32份模型、训练前收据、预测先落盘与完成收据全部齐全；独立重放所有模型，分数误差小于1e-15。当前15日6只通过原完整作者审阅门后实际调用同一投影：92特征数值与训练数据端逐格一致，输出12个研究分，原事实和原作者方向未改。

## 第5项：误报成本试验与跨期结论

历史主比较保持原冻结真实非一字池。9月完整143只，已知107只，40正例，36结果未知；下一列Top1为完整16日选择、13已知、3未知。10月列包含15日6只未到期，不将其算成失败；其余未知也全保留。

{all_metrics}

9月新校正模型相对BASE召回15%→37.5%，但相对旧双通道50%退化，FP22→19；Top1从8/13升到10/13，只有13个已知日，且新Top1里3条方向仍是来源不足UNKNOWN，不能等同3条可买正向信号。10月局部8/9召回、Top1 4/4看起来更好，但仅37个非一字已知结果、9正例、另17未知，不覆盖更早阶段退化。

三个历史主段合计：校正模型FP为{a['fp']}，固定成本方案降到{z['fp']}，减少{(1-z['fp']/a['fp'])*100:.2f}%；同时TP由{a['tp']}降到{z['tp']}，召回率{pct(a['Recall'])}→{pct(z['Recall'])}，F1 {pct(a['F1'])}→{pct(z['F1'])}。这属于压低正向数量造成的明显召回损失。9月成本方案FP19→4，却也把TP15→5，召回37.5%→12.5%，因此不能说误报问题已经优化好。

按信号日5日块、1500次抽样、本批24比较校正的区间已保存；不把多次历史研究包装成从未选过的独立统计发现。成本方案相对校正模型的9月召回差区间约−40.21至−6.98个百分点，方向为退化；这仍是历史描述。

**验收结论：新特征方案和成本方案均失败，不部署。** 失败来自F1、跨期召回或Top1条款，未套用已经废除的全池Accuracy+3pp。未修改本批权重/阈值重跑，也未复跑旧失败挑战。

## 口径修复与未覆盖部分

最初统计误用了最新engine的日线身份元数据，与预登记原冻结非一字范围有18只次不同。原首次统计完整保留，另存修正后的18885行，三参考方案方向恢复自其原冻结身份，全部分数和标签未改。上表以修正后范围为准；更严格最新价型0的结果另列敏感性表。

9月新模型在原非一字143只中有35只当前价型资格未知，21只结果已知（8正、13负），保留UNKNOWN方向。若只诊断原始B分数而不进行资格弃判，校正模型为TP21/FP24、Recall52.5%、F1 49.41%；这说明部分下降来自资料资格约束，不能全归因于模型不分辨。该诊断并未更改新方案方向或验收规则，也不能据此把来源不足样本直接转正。原始B分数比较另表保存。

训练前还修复了boards元数据重叠、两处来源状态枚举拼写以及原适配器漏绑定的48条已存10月评价标签；均在0次拟合时保留错误/前像后修复，历史训练标签未改。工作台首次比较把pandas.NA和JSON null当不同值，已局部规范缺失表示并通过回归检查；没有填0、改特征或重训。27项相关测试通过。

本轮未训练LLM权重、未认证新融合权重、未训练新的2/3/5日数值目标，未新增角色身份金标准；旧多日情景预测及待结算66事件保持，不能拿单一晋级目标冒称这些问题已完成。来源晚取得与当前股票池幸存者偏差继续标明。没有新增定时任务、平台目标状态改写、交易、成交或利润。

## 15日6只：仅固定材料的前向影子比较

目标为下一官方交易日2026-10-16收盘涨停。实际结果0；本表在目标日结果出现前固定保存，不新增正式登记，不覆盖原独立预测。排名只表示该固定材料下相对倾向，概率未验证。

{current_table}

慈文、阳光三次均偏晋级；富春虽相对第三，三次均未知，未强凑第三个正向候选。冰山存在FAIL与UNKNOWN分歧，保留完整理由、条件和后续反证；百洋和文轩也保留，不只展示前列赢家。完整原池、原有多日路径和执行计划仍在原工作台。本批未来比较只检验这6个已固定身份与各自目标，不能在16日换一份池再结算。

## 查看与复核

{link(dest/'完整81只_三次审阅与模型比较.html','完整81只可展开比较工作台')}；{link(validation_path,'32模型重放与修正后验收')}；{link(llm_path,'3份评测源与曝光核验')}。

{link(ROOT/'all16_training_memberships_and_fixed_weights.csv','16窗完整训练成员及固定权重')}；{link(ROOT/'all1472_training_field_coverage.csv','1472格训练特征覆盖')}；{link(ROOT/'evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv','18885行完整同池输出')}；{link(ROOT/'raw_B_vs_qualified_route_DIAGNOSTIC.csv','原始B分数与资格方向诊断')}。

{link(ROOT/'LLM_evaluation/all243_actual_manual_reviews.csv','243份逐股真实理由')}；{link(ROOT/'LLM_evaluation/all81_votes_and_fixed_consensus.json','全部81只投票与固定聚合')}；{link(ROOT/'LLM_evaluation/current15_six_target16_forward_shadow_FROZEN.json','15日6只待16日检验的影子记录')}；{link(live_path,'原作者审阅后当前6只新模型接口核对')}；{link(test_path,'27项测试结果')}。
"""
    root_report.write_text(text,encoding="utf-8")
    # A compact local viewer lets the user inspect every reviewer, not just summaries.
    work=b.read(trial.OLD/"workbench/完整81只_全部证据与原审阅.json")["rows"]
    reviews=[b.read(ROOT/f"LLM_repeat_{i}_actual_review.json")["rows"] for i in (1,2,3)]
    review_maps=[{(x["date"],x["code"]):x for x in rows} for rows in reviews]
    predicted=pd.read_csv(ROOT/"evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv",dtype={"code":str},low_memory=False)
    scores={(x.date,x.code,x.arm):x for x in predicted.itertuples()}
    consensus={(x["date"],x["code"]):x for x in b.read(ROOT/"LLM_evaluation/all81_votes_and_fixed_consensus.json")}
    e=lambda value:html.escape(str(value))
    cards=[]
    for item in work:
        key=item["date"],item["code"];vote=consensus[key]
        outcome="待到期/未知" if item["outcome_for_display"] is None else "晋级" if item["outcome_for_display"]==1 else "未晋级"
        details=[]
        for i,mp in enumerate(review_maps,1):
            row=mp[key]
            details.append(f"<section><h4>第{i}次：{e(row['call'])} · 相对第{row['rank']}</h4><p>{e(row['reason'])}</p><p><b>取消/反证：</b>{e(row['cancel'])}</p><small>支持：{e('；'.join(row['support_fact_ids']))}<br>反证：{e('；'.join(row['counter_fact_ids']))}</small></section>")
        sr=[]
        for arm in ARMS:
            row=scores[(key[0],key[1],arm)]
            sr.append(f"<tr><td>{e(NAMES[arm])}</td><td>{row.ranking_score:.6f}</td><td>{e(row.call)}</td></tr>")
        author=item["archived_author_record"]
        cards.append(f"<details class='stock' data-date='{e(key[0])}' data-disagree='{str(not vote['unanimous']).lower()}'><summary>{e(key[0])} · {e(item['name'])} {e(key[1])} · {item['boards']}板 <b>{e(vote['majority_call'])}</b> <span>{'三次一致' if vote['unanimous'] else '方向有分歧'} · {e(outcome)}</span></summary><div class='inside'><p>补充展示结果：{e(outcome)}；历史结果已曝光。原作者：{e(author['original_direction'] or '无原记录')} · {e(author['archive_status'])}</p><table><thead><tr><th>数值方案</th><th>原始B排序分</th><th>资格方向</th></tr></thead><tbody>{''.join(sr)}</tbody></table>{''.join(details)}</div></details>")
    options="".join(f"<option>{e(day)}</option>" for day in sorted({x['date'] for x in work}))
    pcts="、".join(pct(x["agreement"]) for x in llm["pairwise_ALL81"])
    h=f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>第3至5项优化 · 完整81只比较</title>
<style>body{{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#f3f5f7;color:#182734}}main{{max-width:1150px;margin:auto;padding:30px 20px}}h1{{font-size:28px}}p{{line-height:1.7}}.notice{{background:#fff1d8;border-left:5px solid #c28d29;padding:14px 18px}}.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:22px 0}}.stats div{{background:white;flex:1;min-width:150px;padding:16px;border-radius:10px}}.stats strong{{font-size:30px;display:block}}nav{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#f3f5f7;padding:16px 0}}select,button{{padding:8px 12px;border:1px solid #bfcad3;border-radius:6px;background:white}}details{{background:white;border:1px solid #d7e0e6;border-radius:8px;margin:8px 0}}summary{{cursor:pointer;padding:15px;line-height:1.5}}summary b{{margin-left:14px;color:#24587e}}summary span{{float:right;color:#556675}}.inside{{padding:0 20px 20px}}section{{border-top:1px solid #d7e0e6;padding-top:6px}}small{{color:#596c79;line-height:1.7;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{text-align:left;border-bottom:1px solid #d7e0e6;padding:9px}}[hidden]{{display:none!important}}@media(max-width:700px){{summary span{{float:none;display:block}}}}</style>
<main><h1>3次独立输出 · 校正特征重训 · 误报验收</h1><p class='notice'><b>本批两项新数值方案均未通过，正式BASE保留。</b> 历史LLM结论已接触旧结果，不能称盲测增益；模型分数未校准。当前15日6只目标16日，结果尚未到期。此页保留全部81只。</p>
<div class='stats'><div><strong>243</strong>真实逐股审阅</div><div><strong>64 / 81</strong>三次方向一致</div><div><strong>32</strong>新模型已训练并重放</div><div><strong>92</strong>训练/当前接口一致特征</div></div>
<p>两两一致率{pcts}。9月原非一字池新模型Top1为10/13（另3日未知），召回37.5%低于旧双通道50%；固定误报成本使召回降至12.5%。完整跨期与弃判分母见本目录Markdown报告。</p>
<nav><label>信号日期 <select id='day'><option value=''>全部7日</option>{options}</select></label><label><input type='checkbox' id='disagreement'> 只看三次方向分歧</label><button id='expand'>展开可见理由</button><span id='count'>81只</span></nav>
{''.join(cards)}<p>每条引用只指向其信号日材料。名称共现、规则角色和放量均不自动认证资金因果、角色真值或可交易性。正式账本、原多日路径及成交数据未改。</p></main>
<script>const day=document.getElementById('day'),dis=document.getElementById('disagreement'),cards=[...document.querySelectorAll('.stock')];function filter(){{let n=0;cards.forEach(x=>{{x.hidden=(day.value&&x.dataset.date!==day.value)||(dis.checked&&x.dataset.disagree!=='true');if(!x.hidden)n++}});document.getElementById('count').textContent=n+'只'}}day.addEventListener('change',filter);dis.addEventListener('change',filter);document.getElementById('expand').addEventListener('click',()=>cards.filter(x=>!x.hidden).forEach(x=>x.open=true));</script></html>"""
    (dest/"完整81只_三次审阅与模型比较.html").write_text(h,encoding="utf-8")
    example=packets.single_date_packet(b.read(ROOT/"LLM_repeat_1_closed_packet.json"),"2026-10-15",1)
    example.update(formed_at=b.now(),not_rereviewed=True,existing_three_reviews_not_changed=True)
    b.write_new(dest/"单信号日事实包示例_未追加评测.json",example)
    # Reverify previous delivery in full and bind every artifact in this new batch.
    previous=b.read(trial.OLD/"completion_verified.json");b.verify(previous["source_bindings"])
    code=[b.ROOT/x for x in ["expectation_345_features.py","prepare_expectation_345.py","expectation_345_research.py","expectation_345_evaluation.py","expectation_345_verify.py","expectation_345_llm_evaluation.py","expectation_345_workbench.py","expectation_345_review_packets.py","deliver_expectation_345.py","tests/test_expectation_345.py","tests/test_expectation_345_delivery.py"]]
    paths=[p for p in ROOT.rglob("*") if p.is_file()]+code
    bindings=previous["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths}
    b.verify(bindings)
    completion=dict(verified_at=b.now(),requested_items=[3,4,5],actual_manual_LLM_rows=243,LLM_contexts=3,historical_blind=False,
        external_API_inference_requests=0,independent_call_unanimous=64,full_cohort_rows=81,current_future_rows=6,current_future_actuals=0,
        features_actually_fitted=92,new_model_fits=32,previous_model_refits=0,saved_models_replayed=32,paired_predictions=18885,
        training_windows_membership_and_labels_identical=16,training_live_current_feature_values_exact=True,
        corrected_model_pass=False,cost_model_pass=False,false_alarm_control_accepted=False,
        item3_status="MEASURED_WITH_EXPOSURE_LIMITS_FUTURE_ACCURACY_PENDING",item4_status="ACTUAL_RETRAIN_COMPLETED_VALIDATION_FAILED",item5_status="FIXED_COST_REDUCED_FP_BUT_RECALL_AND_F1_FAILED",
        test_count=tests,previous_sources_unchanged=True,previous_bindings_verified=len(previous["source_bindings"]),
        old_formal_ledger_changed=False,new_formal_predictions=0,actual_trades=0,production_replaced=False,all_three_predictive_defects_solved=False,
        source_bindings=bindings)
    b.write_new(ROOT/"completion_345_verified.json",completion)
    print(json.dumps({k:completion[k] for k in ["new_model_fits","actual_manual_LLM_rows","test_count","previous_bindings_verified","corrected_model_pass","cost_model_pass"]}))


if __name__=="__main__":build()
