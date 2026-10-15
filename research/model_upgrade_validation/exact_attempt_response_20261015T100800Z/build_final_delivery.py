"""Full-scope evidence audit and append-only corrected-input result delivery."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import exact_attempt_response_research as r
import short_term_thinking_dataset_builder as b
import short_term_thinking_workbench as t

ROOT=r.ROOT
WORK=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z"


def percent(value):return "未知" if value is None else f"{100*value:.2f}%"


def main():
    verification=b.read(ROOT/"independent_verification.json");b.verify(verification["source_bindings"])
    result=b.read(ROOT/"fixed_comparison/model_validation_results.json")
    correction=b.read(r.SOURCE/"correction_and_impact_report.json")
    out=ROOT/"delivery_final"
    if out.exists():raise FileExistsError("immutable corrected-input final delivery exists")
    out.mkdir()
    original_path=WORK/"delivery_final/完整81只对照_含15日审阅后研究分.csv"
    original=pd.read_csv(original_path,dtype={"code":str,"parent_anchor_code":str})
    pred=pd.read_csv(ROOT/"fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    arm=pred[pred.arm.eq(r.ARM)].set_index(["date","code"]).score
    current=b.read(ROOT/"current6_after_original_completed_review/corrected_model_scores_RESEARCH_ONLY.json")
    current_scores={x["code"]:x["channel_b_score"] for x in current["rows"]}
    combined=original.copy()
    combined["exact_attempt_corrected_model_research_score"]=[current_scores[x.code] if x.date=="2026-10-15" else arm.loc[(x.date,x.code)] for x in combined.itertuples()]
    combined["corrected_model_status"]="FAILED_FIXED_SCREEN_RESEARCH_ONLY_NOT_PRODUCTION"
    pd.testing.assert_frame_equal(combined[list(original)],original)
    assert len(combined)==81 and combined.exact_attempt_corrected_model_research_score.notna().all()
    combined.to_csv(out/"完整81只对照_含更正炸板输入研究分.csv",index=False,encoding="utf-8-sig")
    # Use the original, now fully qualified workbench card layout. Extend its
    # payload with a new column; no original HTML or old row is overwritten.
    oldhtml=WORK/"delivery_final/完整81只对照_含15日审阅后研究分.html"
    soup=BeautifulSoup(oldhtml.read_text("utf-8"),"html.parser");script=soup.find("script").get_text()
    data=json.loads(script.partition("const rows=")[2].partition(";const day=")[0])
    values=combined.set_index(["date","code"]).exact_attempt_corrected_model_research_score.to_dict()
    for row in data:row["corrected"]=f"{100*values[(row['date'],row['code'])]:.2f}"
    html=oldhtml.read_text("utf-8")
    payload=json.dumps(data,ensure_ascii=False).replace("</","<\\/")
    start=html.index("const rows=")+len("const rows=");end=html.index(";const day=",start)
    html=html[:start]+payload+html[end:]
    html=html.replace("<th>次日结果</th>","<th>炸板更正研究分</th><th>次日结果</th>")
    html=html.replace("'challenger','outcome','reason'","'challenger','corrected','outcome','reason'")
    html=html.replace("<h1>","<p>新增列使用恢复362条真实触板失败后的固定模型；该模型依然未通过验收，所有分数均仅研究观察。</p><h1>",1)
    (out/"完整81只对照_含更正炸板输入研究分.html").write_text(html,encoding="utf-8")
    full,current_frame=r.load()
    role_rates=[]
    for fold in r.old.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]:
        f=(current_frame if fold["id"]=="2026Oct" else full)
        f=f[f.date.between(fold["start"],fold["end"])]
        for role in b.ROLES:
            selected=f[f["role_"+role].eq(1)];known=selected[selected.target.notna()&selected.label_valid]
            role_rates.append(dict(period=fold["id"],rule_role=role,candidates=len(selected),known_labels=len(known),promoted=int(known.target.sum()),
                conditional_promotion=float(known.target.mean()) if len(known) else None,independent_identity_accuracy=None,
                rule_label_not_independent_identity_gold=True))
    pd.DataFrame(role_rates).to_csv(out/"six_role_conditional_promotion_NOT_IDENTITY_ACCURACY.csv",index=False,encoding="utf-8-sig")
    # Requirement audit keeps the full objective, including what cannot yet be
    # substantiated by these reconstructions, viewed-history tests or proxies.
    requirements=[
        dict(id="four_groups_9245_causal_features",state="PARTLY_PROVEN",evidence="Original and corrected full9245 artifacts, signal-field allowlists, causal tests; complete role decisions100%",remaining="Actual theme/industry and some price/path fields still missing; row coverage is not100% verified fact coverage"),
        dict(id="V215_full58772_attempts",state="PROVEN_RECONSTRUCTION_WITH_SCOPE_LIMIT",evidence="All58772 independently checked;56008 exact-price observations,362restored failed attempts,2763no touch and1unresolved cap regime",remaining="Current security universe survivor bias and exchange-wide eligibility remain unverified"),
        dict(id="three_training_windows_95_to100_coverage",state="PARTLY_PROVEN",evidence="4062/3243/3059 training rows, operational role100%; dynamic shape above99%; name proxy fields100%",remaining="Third window13same-height rows unknown at over-cap date; minute price0/0/8.43%; independent actual-theme fields incomplete"),
        dict(id="six_roles_across_periods",state="RULE_CANDIDATES_PROVEN_EFFECTIVENESS_UNPROVEN",evidence="Nonempty primary role for all9245, supplement/space/usurp/turnover and separate name-theme proxies; full conditional-promotion table",remaining="Independent role correctness and money-flow causality are unverified; conditional promotion is not identity accuracy"),
        dict(id="two_channels_252_monthly_training",state="PROVEN_EXECUTED_NOT_UPGRADE",evidence="32original models +16symbolic models +16corrected-input models; eachmonth252prior mature signal days; originalBASE retained for one-word channel",remaining="All challenger packages failed fixed predictive screen; no deployment"),
        dict(id="nonword_recall_f1_top1_top2_improvement",state="NOT_ACHIEVED",evidence="Original dual SeptemberR50vsBASE15 butP47.62vs60; corrected R47.5/P45.24/F46.34 andTop2decline",remaining="Precision, F1 stability and ranking across all original periods not accepted"),
        dict(id="Bplus1_tail_Bplus2_net",state="PROXY_EVALUATED_NOT_ACHIEVED",evidence="DailyOPEN scenarios with cost/slippage/blocked sells; complete cancellations and missingness",remaining="Small selected/matured samples, unavailable order queue/auction and no actual fills; no credible net-profit improvement"),
        dict(id="same_history_and_current_code_workbench",state="PROVEN_WITH_FORMAL_SOURCE_LIMIT",evidence="Same causal function, corrected capsule through15; full6 original-row equality; actual195history rows+4mainline reports+4narrative types consumed; same current6 after correction",remaining="Original V293 production admission and complete11role formalfreeze unavailable; diagnostic review is not a registered forecast"),
        dict(id="Sept30_to_latest_full_report",state="PROVEN_RESEARCH_REPORT",evidence="81full candidate observations over7dates;74supplementary results and7unknown; current6allscore columns after prior review",remaining="No certified new formal forecast;16day not yet matured; unknowns remain visible"),
        dict(id="six_cases_and_forum_operators",state="SOURCE_AND_RULE_REVIEW_PROVEN_EFFECT_UNPROVEN",evidence="Six original case-bank cycles remain original; full8forum arguments,7discovered parents and40structural parent decisions preserved",remaining="Case narratives do not prove training causality, current consensus or actual execution; no automatic author-copying"),
        dict(id="immutability_and_fidelity",state="PROVEN_THIS_BATCH",evidence="66protected hashes, fulloriginalBASE/ST_DUALprediction rows, originaltargets preserved; no original receipts rewritten; source-only0fit thennewfixed16fit",remaining=None)]
    audit=dict(formed_at=b.now(),objective_file=r"C:\Users\Administrator\.codex\attachments\c84f9d98-65db-439c-be78-35c4eb490852\goal-objective.md",
        requirements=requirements,goal_achieved=False,goal_status="active",success_not_redefined_to_passing_tests=True)
    b.write_new(out/"full_objective_requirement_audit.json",audit)
    lines=["# 更正炸板输入固定比较与完整交付", "", "恢复362条可核炸板后完成了一个新的16个月固定比较，结果仍不通过。数据更正保留，旧模型及旧失败保留，新模型不部署；整个用户目标仍active。", "",
        "## 实际数据更正", "", "原V21558772行中362条触及精确涨停而收低1分，应归为炸板；旧重建器因分类冲突把它们整行删除。本版用Decimal十进制半入计算涨停价，并按相邻官方交易日重新计算冲板身位，有效55646→56008。2763条未触及精确上限排除，另1条超出假设10%上限保持资格未知。", "",
        "684候选同位上下文变化，原有效55646行分类/身位不变。全部9245候选目标、其他特征及冻结模型分数未改写；三个原训练窗受到影响454/72/110只次。8月7日13个候选同位淘汰率保持未知，不能为凑100%填0。原数值市场总炸板率来自独立归档，本次没有混改其语义。", "",
        "## 固定数值结果", "", "单一新臂ST_DUAL_EXACT_FAILED_ATTEMPTS，其他特征、HGB参数、原252日成熟成员、日期/类别权重及严格0.5都不变。事前协议10:06:26UTC形成，16次各月一次拟合及提前预测保存完成；没有按结果调参/换阈值或重复挑战。", "",
        "| 阶段 | 模型 | 非一字Recall | Precision | F1 | Top1 | Top2 |", "|---|---|---:|---:|---:|---:|---:|"]
    for fold,z in result["comparisons"].items():
        for arm,m in z["metrics"].items():lines.append(f"| {fold} | {arm} | {percent(m['nonword']['Recall'])} | {percent(m['nonword']['Precision'])} | {percent(m['nonword']['F1'])} | {percent(m['top1']['candidate_hit_rate'])} | {percent(m['top2']['candidate_hit_rate'])} |")
    lines += ["", "2025H2的Recall/Precision/F1小幅改善；1—8月退化，9月TP20→19、FP22→23，Top1同为8/13（完整16天、3结果未知），Top2为16/26（完整32、6未知）。失败仍按非一字与排名门判断，不使用全池Accuracy+3pp否决。", "",
        "9月B+2日线场景均值为7.59%（6个）对旧双通道8.04%（5个），亏损至少7%的比例16.67%对20%；分母及选择不同，样本太小，不能确认风险或期望收益改善。各期全部取消、未知与卖出受阻保留于原执行日志；真实成交0。", "",
        "## 完整当期对照", "", "当前15→16完整6只在原诊断链完成8条论坛观点、7条发现父锚及40条结构锚取舍，实际消费195历史角色/4日主线/4条叙事，再读取模型。独立条件偏强仍是慈文、百洋；其他4只观察待核。新输入修正不改变当前6只因果事实，仅新权重输出另列。", "",
        "| 股票 | 原BASE | 原双通道 | 更正输入模型 |", "|---|---:|---:|---:|"]
    latest=combined[combined.date.eq("2026-10-15")]
    for x in latest.itertuples():lines.append(f"| {x.name} {x.code} | {x.current15_after_review_BASE*100:.2f} | {x.current15_after_review_ST_DUAL*100:.2f} | {x.exact_attempt_corrected_model_research_score*100:.2f} |")
    lines += ["", "分数未经校准；原正式来源与完整11角色freeze未通过，仍无新正式或独立登记。所有研究模型均未被启用。旧81行/列完全保留，新增更正模型分列全81只次非空；74补充来源结果、7未知不变，旧目标与原预测不回填。", "",
        "## 全目标验收与后续边界", "", "全目标逐项审计见full_objective_requirement_audit.json：全9245规则角色、历史/工作台同码及完整报告已有实证；实际题材/行业、独立角色有效识别、稳定Precision/排序、可成交净收益仍没有达到用户目标。分钟附层只覆盖485只次，其中451价格路径、324量额合格，不能把离散采样叫连续封板或真实承接。", "",
        "16保存模型重放、每月成员/标签/权重/截止时间、11313完整配对行及84组指标独立复算通过；66受保护哈希保持。更正后失败不再以原输入、不同阈值或权重重复拟合。下一步须寻找有独立源依据的题材或承接证据，并先检查新增字段的跨期覆盖与区分能力；当前不足不是把所有历史研究重新封锁。", "",
        "来源与当期审阅详见上一批workbench_latest_20261015T091200Z；此前数字包与失败收据完整保留。本批新模型16次、原BASE/旧双通道重拟合0、新正式成熟0、成交0。"]
    (out/"更正炸板输入固定比较与完整交付.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    script=BeautifulSoup((out/"完整81只对照_含更正炸板输入研究分.html").read_text("utf-8"),"html.parser").find("script").get_text()
    embedded=json.loads(script.partition("const rows=")[2].partition(";const day=")[0]);assert len(embedded)==81 and all(x["corrected"] for x in embedded)
    completion=dict(formed_at=b.now(),status="CORRECTED_SOURCE_FIXED_COMPARISON_COMPLETE_PREDICTIVE_GOAL_NOT_ACHIEVED",goal_status="active",new_fits=16,old_BASE_or_ST_DUAL_refits=0,
        prediction_screen_pass=False,source_failed_rows_restored=362,qualified_attempt_rows=56008,protected_files_exact=66,complete_report_rows=81,
        current6_original_completed_review_retained=True,new_formal_predictions=0,new_independent_registry_entries=0,new_mature_samples=0,actual_fills=0,
        requirement_audit_path=str((out/"full_objective_requirement_audit.json").resolve()),source_bindings=verification["source_bindings"]|{str(original_path.resolve()):b.sha(original_path),str(Path(__file__).resolve()):b.sha(__file__)})
    for path in out.iterdir():
        if path.is_file():completion["source_bindings"][str(path.resolve())]=b.sha(path)
    b.write_new(ROOT/"completion_final.json",completion);print({k:v for k,v in completion.items() if k!="source_bindings"})


if __name__=="__main__":main()
