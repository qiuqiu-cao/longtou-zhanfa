"""Append-only research delivery, no fits, score selection or registration."""
from pathlib import Path
from copy import deepcopy
import ast
import html
import json

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import historical_symbolic_context as h
import historical_symbolic_research as r
import relay_review_workbench as w

ROOT=h.RUN
LATEST=ROOT/"latest15_prepared_v2"


def text(value):
    if isinstance(value,list):return "、".join(map(str,value))
    return "未知" if value is None or isinstance(value,float) and np.isnan(value) else str(value)


def roles(value):
    if isinstance(value,list):return value
    if isinstance(value,str):return ast.literal_eval(value)
    return []


def pct(value):return "未知" if value is None else f"{100*value:.2f}%"


def parent_review(edge,all_edges):
    code,anchor=edge["code"],edge["anchor_code"]
    state="REJECT_CURRENT_SUPPLEMENT_ATTRIBUTION"
    reason="启动时序或价格相近只构成筛选条件，现有资料未提供足够的逐股关系印证；不采纳本条补涨归属，亦不称关系已证伪。"
    if edge["parent_anchor_state"]=="ANCHOR_UNOBSERVED":
        state="PENDING"
        reason="原源缺少父锚当日可比日线；新华传媒15日日线缺失也未被自动写成断板或崩塌。父锚生命状态和带动关系待核。"
    if (code,anchor)==("002696","002242"):
        state="ADOPT_AS_CANDIDATE"
        reason="复用已核10月12日正文/评论的九阳—百洋线索及9日首板、13日九阳断连续板而百洋晋级的先后节点。保留九阳周期补涨候选；15日已并列5板，并不因此成为唯一补涨龙。"
    elif code=="001318" and anchor in {"002242","002696"}:
        state="PENDING"
        reason="阳/洋同音和晚启动时序成立；阳光既有旧跟风重起解释，百洋放量是共同节点而非资金转移证据。两父锚假设并列待核，不自动采纳本次规则的百洋归属。"
    elif code=="002343" and anchor=="600825":
        state="ADOPT_AS_CANDIDATE"
        reason="已核旧文存在传媒伴飞叙事，新华高标周期可作为竞争假设；旧父锚15日状态未知，不能认证仍活跃、资金因果或唯一父龙。"
    elif code=="601811" and anchor in {"603096","002343","600825"}:
        state="PENDING"
        reason="新经典断板与新华文轩2板同日发生，慈文仍5板、新华旧高标周期也构成竞争解释。名称/传媒语义相邻与时序不能认证资金接力，保留三个假设逐一待核。"
    if code=="000530" and anchor=="002696":
        state="REJECT_CURRENT_SUPPLEMENT_ATTRIBUTION"
        reason="算法选中来自2日启动差、父锚放量节点及价格风格；没有名称/行业或已核股吧关系支持。冰山换手活口应独立检验，不能直接归为百洋补涨。"
    if code=="605189" and anchor=="002696":
        state="REJECT_CURRENT_SUPPLEMENT_ATTRIBUTION"
        reason="仅晚启动、节点和价格风格相近，没有染织与百洋的已核叙事连接；不采纳该补涨归属。"
    return dict(anchor_code=anchor,anchor_name=edge["anchor_name"],state=state,reason=reason,
        rule_hypothesis=deepcopy(edge),counterevidence="父锚时序是必要背景而非资金因果；当前跌停扩散、竞争锚共存，逐股关系可能只是共时反应。",
        same_cycle_competing_anchors=[x["anchor_code"] for x in all_edges if x["code"]==code and x["anchor_code"]!=anchor],
        cancellation="若原始时序/名称绑定被推翻、独立公司/叙事来源指向其他主线，撤销该归属；次日未确认时不据此入场。")


def current_review(out):
    context=b.read(LATEST/"causal_context_SCORES_MASKED.json")
    symbol=b.read(LATEST/"symbolic_context_SCORES_MASKED.json")
    pack=b.read(LATEST/"features_BEFORE_MODEL_DISPLAY.json")
    pre=b.read(ROOT/"latest15_forum_preread/cards/report.json")
    edges=pd.read_csv(LATEST/"complete_current6_parent_candidates.csv",dtype={"code":str,"anchor_code":str}).replace({np.nan:None}).to_dict("records")
    snippets={x["code"]:x for x in pre["rows"]}
    # These are actual role reviews; numeric/official promotion forecasts remain
    # unavailable until the original full prepare/freeze chain can be completed.
    judgements={
        "002343":dict(role="并列空间龙、换手核心候选；新华周期传媒竞争假设。",support="与百洋共同5板；当日OHLC非一字，规则记录4次分歧换手，换手约16.17%、相对昨日量1.35倍。",rebuttal="新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",condition="仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。"),
        "002696":dict(role="并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。",support="14日一字后15日真实非一字封板，九阳旧关系有已核正文/评论及先后节点。",rebuttal="量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",condition="不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"),
        "000530":dict(role="3板换手/穿越规则候选；未采纳百洋补涨归属。",support="与阳光共同3板；14日一字后开板仍封板，日线满足规则换手检验。",rebuttal="换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",condition="需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。"),
        "001318":dict(role="旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",support="真实非一字3板，与冰山共同存活，换手约6.33%、量为昨日约0.44倍。",rebuttal="缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",condition="核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"),
        "601811":dict(role="2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。",support="同身位2只成功、2只冲板失败，已观察非一字；当日成交额在同位居前。",rebuttal="振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",condition="需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。"),
        "605189":dict(role="2板卡位规则候选；不采纳百洋补涨归属。",support="与新华文轩共同2板，同位失败2只；当日非一字且振幅约3.08%。",rebuttal="题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",condition="需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。")}
    rows=[]
    for stock in context["rows"]:
        code=stock["code"];manual=judgements[code];parents=[parent_review(e,edges) for e in edges if e["code"]==code]
        opinions=[]
        for e in snippets[code]["forum_evidence"]:
            if code=="002696" and e["source_section"]=="COMMENT":
                decision="采纳为待核叙事线索，否决主力布局及资金因果断言"
            elif code=="002696" and e["source_id"]=="guba_name":
                decision="采纳名称叙事已经出现，结合原先后节点保留九阳周期候选；不确认唯一龙"
            elif code=="002343":
                decision="采纳旧文传媒伴飞及需换手检验的假设；15日非一字是新事实，尚非盘口承接证明"
            else:
                decision="保留作者旧假设；农业避险/主动性未经当期独立业务与资金证据认证"
            opinions.append(dict(source=e,decision=decision,counterevidence=manual["rebuttal"],cancellation=manual["condition"]))
        rows.append(dict(code=code,name=stock["name"],boards=stock["boards"],role_hypothesis=manual["role"],support=manual["support"],
            rebuttal=manual["rebuttal"],cancel_condition=manual["condition"],promotion_judgement=None,
            independent_status="ROLE_REVIEW_COMPLETE_PROMOTION_FORECAST_PENDING_ORIGINAL_FREEZE",
            relative_thesis="共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
            comparison_peers=[x["code"] for x in context["rows"] if x["boards"]==stock["boards"] and x["code"]!=code],
            parent_decisions=parents,forum_opinion_decisions=opinions,forum_absence_is_not_negative=True,
            opening_confirmation={"09:25":"实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。","09:35":"用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"}))
    sources=context["source_bindings"]|symbol["source_bindings"]|pre["source_bindings"]
    for path in [LATEST/"causal_context_SCORES_MASKED.json",LATEST/"symbolic_context_SCORES_MASKED.json",ROOT/"latest15_forum_preread/cards/report.json",Path(__file__)]:sources[str(path.resolve())]=b.sha(path)
    review=dict(formed_at=b.now(),signal_date="2026-10-15",target_date="2026-10-16",status="ACTUAL_FULL6_ROLE_REVIEW_ONLY_NOT_PROMOTION_FORECAST_FREEZE",rows=rows,
        source_bindings=sources,actual_model_exposure="Current15/16 stock model outputs not reviewed; historical aggregate metrics and older-date scores already seen",formal_prepare_freeze_complete=False,
        official_forecasts_added=0,independent_forecast_registry_added=0,all40_parent_edges_reviewed=sum(len(x["parent_decisions"]) for x in rows),new_fits=0)
    b.write_new(out/"current6_actual_role_review_BEFORE_MODEL_DISPLAY.json",review)
    lines=["# 10月15日完整6只独立角色审阅", "", "研究素材审阅；原完整prepare/freeze尚未完成。当前6只模型分数与正式晋级方向均保持隐藏，未登记预测。", "",
        "背景：替代页主板收盘34、炸板5、跌停25；前日15只连板仅4只晋级，算法节点为退潮切换。", "",
        "股吧限定核对：复用2篇10月12日已核原文及分栏评论，15日合格正文0；慈文、百洋有旧背景，其余4只在限定来源未覆盖。", ""]
    for x in rows:
        lines += [f"## {x['name']} {x['code']}（{x['boards']}板）", "",x["role_hypothesis"], "", "支持："+x["support"], "", "反证/替代解释："+x["rebuttal"], "", "取消及开盘条件："+x["cancel_condition"], "", "同位："+"、".join(x["comparison_peers"]), "", "| 父锚 | 取舍 | 理由 |", "|---|---|---|"]
        lines += [f"| {p['anchor_name']} {p['anchor_code']} | {p['state']} | {p['reason']} |" for p in x["parent_decisions"]]
        for opinion in x["forum_opinion_decisions"]:lines += ["",opinion["source"]["source_section"]+"："+opinion["decision"]]
    (out/"10月15日完整6只独立角色审阅.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return review


def main():
    out=ROOT/"delivery_final"
    if out.exists():raise FileExistsError("immutable final research delivery directory")
    out.mkdir()
    b.verify(b.read(ROOT/"independent_verification.json")["source_bindings"])
    review=current_review(out)
    original_path=b.DEFAULT_RUN/"delivery/完整75只总表_含归档结果补充.csv"
    original=pd.read_csv(original_path,dtype={"code":str,"parent_anchor_code":str})
    original_columns=list(original)
    fresh=pd.read_csv(LATEST/"complete_current6_causal_roles.csv",dtype={"code":str,"parent_anchor_code":str})
    fresh["original_independent_judgement"]="正式方向未冻结；本次已完成逐股角色/父锚审阅"
    fresh["target"]=np.nan;fresh["label_valid"]=False
    combined=pd.concat([original,fresh],ignore_index=True)
    pd.testing.assert_frame_equal(combined.iloc[:75][original_columns].reset_index(drop=True),original,check_dtype=False)
    assert len(combined)==81 and not combined.duplicated(["date","code"]).any()
    for col in ["original_fixed_target","source_supplement_target"]:
        if col not in combined:combined[col]=np.nan
    combined["latest_source_outcome"]=combined.source_supplement_target.combine_first(combined.target)
    combined["latest_source_status"]=combined.outcome_source_status.fillna("LATEST15_PENDING_TARGET16")
    outcomes=b.read(LATEST/"previous14_full15_outcomes_APPEND_ONLY.json")
    for x in outcomes["rows"]:
        mask=combined.date.eq(x["signal_date"])&combined.code.eq(x["code"])
        assert mask.sum()==1
        combined.loc[mask,"latest_source_outcome"]=x["promoted"]
        combined.loc[mask,"latest_source_status"]="NEW15_ALTERNATIVE_PAGE_EXACT_DAY_PRICE_APPEND_ONLY"
    symbols=joblib.load(ROOT/"qualified_final/symbolic_context.joblib")["rows"]
    latest=b.read(LATEST/"symbolic_context_SCORES_MASKED.json")
    symbols=pd.concat([symbols,pd.DataFrame(latest["rows"])],ignore_index=True)
    cols=["date","code","historical_name","historical_name_source", "symbolic_roles","extended_rule_roles","extended_primary_role"]+h.FIELDS
    cols=[x for x in cols if x in symbols]
    symbols=symbols[cols]
    combined=combined.merge(symbols,on=["date","code"],how="left",validate="one_to_one")
    scores=pd.read_csv(ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    extra=scores[scores.arm.eq(r.ARM)][["date","code","score"]].rename(columns={"score":"ST_DUAL_SYMBOLIC_DIFFUSION"})
    combined=combined.merge(extra,on=["date","code"],how="left",validate="one_to_one")
    combined["model_display_status"]=["CURRENT6_SCORES_MASKED_PENDING_ORIGINAL_FREEZE" if d=="2026-10-15" else "RETROSPECTIVE_RESEARCH_SCORES_ALREADY_SEEN" for d in combined.date]
    combined=combined.sort_values(["date","boards","code"],ascending=[True,False,True]).reset_index(drop=True)
    assert combined[combined.date.eq("2026-10-15")][["BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]].isna().all().all()
    combined.to_csv(out/"完整81只周期角色与原判断对照.csv",index=False,encoding="utf-8-sig")
    new_outcome_metrics={}
    day=combined[combined.date.eq("2026-10-14")].copy()
    for arm in ["BASE","ST_DUAL","ST_B_ONLY","ST_DUAL_SYMBOLIC_DIFFUSION"]:
        f=day.assign(score=day[arm],target=day.latest_source_outcome,label_valid=day.latest_source_outcome.notna())
        from historical_symbolic_verify import classification,ranking
        new_outcome_metrics[arm]=dict(full=classification(f),nonword=classification(f[f.daily_one_word.eq(0)]),top1=ranking(f),late_source_not_new_fixed_model_validation=True)
    y=day.latest_source_outcome.eq(1);calls=day.original_independent_judgement.map(lambda s:True if str(s).startswith("偏晋级") else False if str(s).startswith("偏断板") else None)
    decided=calls.notna();truepositive=int((calls.eq(True)&y).sum());falsepositive=int((calls.eq(True)&~y).sum())
    new_outcome_metrics["ORIGINAL_INDEPENDENT_REVIEW_ALREADY_SAW_MODEL"]=dict(full_candidates=15,known_outcomes=15,decided=int(decided.sum()),correct=int((calls[decided]==y[decided]).sum()),
        Accuracy=float((calls[decided]==y[decided]).mean()),complete_Recall=truepositive/int(y.sum()),Precision=truepositive/(truepositive+falsepositive),coverage=float(decided.mean()),new_forward_validation=False)
    b.write_new(out/"previous14_full15_comparison_APPEND_ONLY.json",dict(formed_at=b.now(),metrics=new_outcome_metrics,labels_not_added_to_frozen_training=True,original_registry_unchanged=True,new_formal_maturity=0))
    table=["# 9月30日至10月15日完整周期角色与原判断对照", "", "完整81只次。74只次有补充来源结果，7未知（15日6只未到目标16日，另新华传媒12日原结果缺口）。", "",
        "前75行原分数、角色、独立判断及冻结目标列保留。新增15日结果在单独列展示，不写回旧训练标签。15日6只分数与正式方向保持隐藏。", "",
        "规则角色/字辈代理不等于真实资金身份；独立角色准确率仍无独立金标准。", "",
        "| 日期 | 个股 | 板 | 节点 | 规则角色 | 父锚/状态 | 字辈扩展 | 原独立判断 | BASE | 双通道 | 字辈挑战 | 次日补充结果 |", "|---|---|---:|---|---|---|---|---|---:|---:|---:|---|"]
    records=[]
    for x in combined.to_dict("records"):
        record=dict(date=x["date"],code=x["code"],name=x["name"],boards=int(x["boards"]),regime=x["cycle_regime"],
            role="、".join(roles(x["roles"])),parent=text(x["parent_anchor_code"])+" / "+text(x["parent_anchor_state"]),
            symbolic="、".join(roles(x.get("symbolic_roles"))) or "无合格字辈群",independent=x["original_independent_judgement"],
            base=pct(None if pd.isna(x["BASE"]) else x["BASE"]),dual=pct(None if pd.isna(x["ST_DUAL"]) else x["ST_DUAL"]),
            challenger=pct(None if pd.isna(x["ST_DUAL_SYMBOLIC_DIFFUSION"]) else x["ST_DUAL_SYMBOLIC_DIFFUSION"]),
            outcome="晋级" if x["latest_source_outcome"]==1 else "未续板" if x["latest_source_outcome"]==0 else "未知/未到期",source=x["latest_source_status"])
        records.append(record)
        table.append("| "+" | ".join(text(record[k]).replace("|","/") for k in ["date","name","boards","regime","role","parent","symbolic","independent","base","dual","challenger","outcome"])+" |")
    (out/"完整81只周期角色与原判断对照.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    payload=json.dumps(records,ensure_ascii=False).replace("</","<\\/")
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整81只周期角色对照</title><style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172538}h1{font-size:25px}p{max-width:1100px;line-height:1.7}.panel{background:white;padding:16px;border-radius:10px;margin:12px 0}select,input{padding:8px;margin-right:12px}table{border-collapse:collapse;min-width:1800px;background:white}td,th{padding:9px;border:1px solid #d9e0e8;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#e9eff6}.scroll{overflow:auto;max-height:75vh}.unknown{color:#7c5960}small{color:#516176}</style><h1>9月30日—10月15日 · 完整81只周期角色对照</h1><p>自动规则与独立判断分栏；分数未经校准。74只次有补充结果，7未知。15日6只当前分数保持隐藏，原完整prepare/freeze尚未完成；没有新正式预测或成交。名称共现不认证真实题材或资金因果。</p><div class="panel"><label>日期 <select id="day"><option value="">全部日期</option></select></label><input id="query" placeholder="股票 / 角色 / 父锚"><span id="count"></span></div><div class="scroll"><table><thead><tr><th>日期</th><th>个股</th><th>板</th><th>节点</th><th>规则角色</th><th>父锚及状态</th><th>字辈扩展代理</th><th>原独立判断</th><th>BASE</th><th>双通道</th><th>字辈挑战</th><th>次日结果</th><th>结果来源</th></tr></thead><tbody id="body"></tbody></table></div><p><small>数据保留完整分母；筛选仅影响显示。原75行未写回。15日结果是新取得的替代页及精确日线附页，非原V293正式成熟或真实交割。</small></p><script>const rows=__DATA__;const fields=['date','code','boards','regime','role','parent','symbolic','independent','base','dual','challenger','outcome','source'];const day=document.querySelector('#day'),query=document.querySelector('#query');for(const d of [...new Set(rows.map(r=>r.date))]){const o=document.createElement('option');o.value=d;o.textContent=d;day.append(o)}function render(){const selected=rows.filter(r=>(!day.value||r.date===day.value)&&(!query.value||JSON.stringify(r).includes(query.value)));document.querySelector('#count').textContent=selected.length+' / 81只次';const body=document.querySelector('#body');body.replaceChildren();for(const r of selected){const tr=document.createElement('tr');for(const f of fields){const td=document.createElement('td');td.textContent=f==='code'?r.name+' '+r.code:r[f];tr.append(td)}body.append(tr)}}day.onchange=render;query.oninput=render;render();</script></html>'''
    (out/"完整81只周期角色与原判断对照.html").write_text(template.replace("__DATA__",payload),encoding="utf-8")
    validation=b.read(ROOT/"fixed_comparison/model_validation_results.json")
    coverage=b.read(ROOT/"qualified_final/source_qualification_and_coverage.json")
    report=["# 历史名称扩散补层、固定比较与最新收盘续接", "",
        "本批完成新的因果字辈扩散输入、16个月固定增量比较、历史/工作台一致性验证及10月15日完整来源续接。新增字辈方案未通过预测验收；原双通道9月召回提升的误报代价仍未解决，总目标保持active。", "",
        "## 覆盖与身份边界", "",
        "9245条原历史角色和晋级标签均保留；新增17字段在三个原训练窗口的计算覆盖与日期有效名称覆盖均100%。这里包括已知完整池中未出现合格字辈群的零值，不是把缺失值填零。", "",
        "使用3879条历史名称区间，修正V215池626行名称关联，并排除3条区间明确的ST助攻。原较早9063候选名称与有效区间一致；9月182条超出主表8月31日边界，保留归档名称、明确降低来源等级，不延长历史区间。", "",
        f"历史新增THEME_LEADER名称代理{coverage['historical_symbolic_theme_leader_candidates']}、THEME_SEED名称代理{coverage['historical_symbolic_theme_seed_candidates']}只次；实际日期匹配行业证据仅{coverage['actual_historical_industry_candidate_rows']}只次。名称代理、主题资金身份与独立金标准分栏，独立角色识别准确率仍未知。", "",
        "| 训练窗口 | 完整候选 | 17字段计算覆盖 | 日期名称覆盖 | 字辈主线候选 | 字辈先锋候选 |", "|---|---:|---:|---:|---:|---:|"]
    for fold,value in coverage["windows"].items():report.append(f"| {fold} | {value['rows']} | 100% | 100% | {value['symbolic_theme_leader_candidates']} | {value['symbolic_theme_seed_candidates']} |")
    report += ["", "原首批全周期重建、角色分布及32次双模型训练见上一批交付。本批只追加这一个有新资料依据的挑战，不重拟合BASE或原双通道。", "",
        "## 固定比较结果", "", "同原252日成员、成熟标签、日期/类别权重、HGB参数和严格0.5。16次拟合均在事前协议后、约1分钟完成；没有失败后调参或重跑。", "",
        "| 阶段 | 模型 | 非一字Recall | Precision | F1 | Top-1 | Top-2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for fold,value in validation["comparisons"].items():
        for arm,m in value["metrics"].items():report.append(f"| {fold} | {arm} | {pct(m['nonword']['Recall'])} | {pct(m['nonword']['Precision'])} | {pct(m['nonword']['F1'])} | {pct(m['top1']['candidate_hit_rate'])} | {pct(m['top2']['candidate_hit_rate'])} |")
    report += ["", "9月字辈挑战TP19/FP25，原双通道TP20/FP22。Top-1为9/13（已知13、完整选16），原双通道8/13；Top-2退化到16/26。Top-1共同日期区间跨0，不能据单项上升启用。失败来自预登记的非一字Recall/Precision/F1与排名门，不是已废除的全池Accuracy+3pp。", "",
        "## 收益代理与新增来源", "", "只补原始日线可重建范围，不改变任何旧分数或已选择股票。新增144个原缺失评价格，冲突0。B是实际假设入场日，B+1/B+2从B往后数；原开盘0—7%、非涨停开盘、滑点/费用及跌停卖出受阻口径保持。", "",
        "9月BASE的B+2已知从4增到5，均值由4.92%变为6.25%；原双通道仍5个、8.04%。字辈挑战只有3个、12.31%，完整16次选股中仅18.75%形成该项结果。不同取消与缺失率、样本过小，不能认证期望收益改善；全部是日线场景，真实成交0。", "",
        "15日替代页原文36收盘/7炸板/26跌停，主板非ST范围34/5/25；显式连板梯队6只。68次逐股日线请求每股一次，67只返回15日精确记录，新华传媒仍缺15日，不自动当断板。新旧日线515重合响应记录完全一致，合并状态机599个重合OHLC键也一致。来源均收盘后真实取得。网页AI业务摘要不作为公告事实。", "",
        "14日15只完整结果追加：冰山、阳光、慈文、百洋晋级，其余11只未续板；路畅/星帅尔原停牌连续板资格缺口继续保留。原独立审阅已见模型，已判断6只正确4只、完整Recall75%、覆盖40%；BASE全15只正确10只、Recall25%。两个分母不同，不能称独立判断优于BASE。14日真实非一字2正例（慈文、阳光），原双通道仍全漏，作为新反例保留。", "",
        "完整总表现在81只次、74个补充来源结果、7未知。原冻结标签仍只按原口径，没有将新增15日结果写进16次已完成拟合或正式账本。", "",
        "## 当期接入与验证", "",
        "工作台新增prepare --symbolic-context和独立symbolic/symbolic-score入口；原历史、主线历史、叙事、父锚参数完整传递。实际14日15只可用原已完成审阅后展示研究分数；改写/伪造上下文、待审阅先看分数及缺失原因果层均拒绝。", "",
        "17字段及扩展角色在14日完整15只强制走新日期重算后与历史批量精确相同；15日完整6只也实际通过同一因果引擎历史/工作台一致性核对。15日新素材已做全6只角色、全部40条父锚和旧文观点审阅，正式晋级方向/模型分数保持隐藏，原完整prepare/freeze未完成，不冒称正式预测。", "",
        "16保存模型重放、11313配对行、38受保护哈希和620源哈希通过；84组分类/排名检查及745笔代理退出算式复算通过。37项相关单元核验通过。None/NaN比较与只读NumPy验证失败的原件均保留，修复仅统一缺失值表示/复制验证数组，没有新拟合。", "",
        "当前仍缺跨期独立角色真值、广泛同日期行业/实际活跃题材、完整可成交/分钟证据及稳定的Precision和收益增益。本批失败模型不部署、不换阈值再跑。下一步只审查独立新增题材/盘口证据的覆盖及真假阳区分能力，再按单独事前协议决定是否形成新假设。正式来源可用时沿原prepare/freeze/compare/execution，不修改旧登记、任务或交易。", "",
        "文件：完整81只HTML/CSV、当前6只角色审阅、14→15完整追加结果、固定模型比较、独立重放收据均在同批目录。"]
    (out/"历史短线补层与10月15日续接交付.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    summary=dict(formed_at=b.now(),status="STAGED_RESEARCH_COMPLETE_GOAL_NOT_ACCEPTED",goal_status="active",original_historical_rows=9245,
        new_challenger_fits=16,old_models_refitted=0,prediction_screen_pass=False,full_current_report_rows=len(combined),
        outcomes_with_supplementary_sources=int(combined.latest_source_outcome.notna().sum()),unknown_outcomes=int(combined.latest_source_outcome.isna().sum()),
        current6_model_scores_hidden=True,current6_role_reviewed=True,current6_parent_edges_reviewed=review["all40_parent_edges_reviewed"],
        new_formal_predictions=0,new_formal_maturity=0,actual_fills=0,protected_files_changed=0,independent_role_accuracy=None,
        source_bindings={str(original_path.resolve()):b.sha(original_path),str(Path(__file__).resolve()):b.sha(__file__),str((ROOT/"independent_verification.json").resolve()):b.sha(ROOT/"independent_verification.json")})
    assert summary["outcomes_with_supplementary_sources"]==74 and summary["unknown_outcomes"]==7
    for path in out.iterdir():
        if path.is_file():summary["source_bindings"][str(path.resolve())]=b.sha(path)
    b.write_new(ROOT/"completion_final.json",summary)
    print({k:v for k,v in summary.items() if k!="source_bindings"})


if __name__=="__main__":main()
