"""Append final requirement evidence after actual browser and source inspection.

This does not rerun any estimator, modify frozen output, or settle future labels.
"""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
import hashlib
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
import short_term_thinking_dataset_builder as b


def main():
    completion = b.read(ROOT / "completion_final.json")
    verification = b.read(ROOT / "full_scope_verification.json")
    b.verify(completion["source_bindings"])
    b.verify(verification["source_bindings"])
    objective = Path(r"C:\Users\Administrator\.codex\attachments\fd62f217-e468-4f83-aa83-13b485abe89f\goal-objective.md")
    assert b.sha(objective) == "45deb2088b46719760c64eafe0b5d8af9fa69546d2d1b7d426da1d658050613f"
    assert completion["new_fits"] == completion["new_inferences"] == 0
    assert verification["test_failures"] == 0

    table = pd.read_csv(ROOT / "delivery_final/完整9315只次_事实代理未知与全部冻结特征.csv", dtype={"code": str}, low_memory=False)
    coverage = pd.read_csv(ROOT / "delivery_final/逐字段覆盖_不等于事实认证.csv")
    assert len(table) == 9315 and len(table[table.date.le("2026-09-30")]) == 9245
    assert not table.duplicated(["date", "code"]).any()
    assert not set(["target", "target_date", "label_valid"]) & set(table)
    assert len(coverage) == 2 * len(table.columns) == 338
    for row in coverage.itertuples():
        part = table[table.date.le("2026-09-30")] if row.scope == "HISTORY9245" else table[table.date.gt("2026-09-30")]
        assert row.rows == len(part) and row.populated == part[row.field].notna().sum() and row.null == part[row.field].isna().sum()
        assert row.independent_role_identity_gold_n == 0
    trials = b.read(ROOT / "inventory/relevant_completed_fixed_trials_80_saved_models.json")
    model_hashes = {}
    for trial in trials["trials"]:
        assert trial["new_fits_this_batch"] == 0 and trial["retry_authorized"] is False
        assert trial["saved_models"] == len(trial["model_hashes"])
        b.verify(trial["model_hashes"])
        model_hashes.update(trial["model_hashes"])
        protocol = b.read(trial["protocol_path"])
        assert b.sha(trial["protocol_path"]) == trial["protocol_sha256"]
        for check in protocol.get("training_checks", []):
            day_counts = [check[k] for k in ["days", "train_days", "training_days"] if k in check]
            assert day_counts and all(value == 252 for value in day_counts)
            assert check["max_target"] < check["cutoff"]
    assert len(model_hashes) == 80

    record = b.read(ROOT / "current_forecast/15日收盘_16日优先与2_3_5日情景_已看模型.json")
    previous_comparison = b.read(ROOT / "current_forecast/current66_events_SNAPSHOT_COMPARISON_PENDING.json")
    b.verify(previous_comparison["source_bindings"])
    assert len(record["rows"]) == 6 and len(previous_comparison["rows"]) == 66
    assert all(x["actual"] is None and x["status"] == "PENDING_OFFICIAL_WINDOW" for x in previous_comparison["rows"])
    now = datetime.now(timezone.utc)
    assert pd.Timestamp(now).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") < "2026-10-16", "recheck actual matured state before finalization"
    assert set(x["code"] for x in record["rows"]) == set(table[table.date.eq("2026-10-15")].code)
    test_root = ET.parse(ROOT / "related_tests.xml").getroot()
    assert sum(int(x.attrib.get("tests", 0)) for x in test_root.iter("testsuite")) == 98
    assert sum(int(x.attrib.get("failures", 0)) + int(x.attrib.get("errors", 0)) for x in test_root.iter("testsuite")) == 0

    old_html = ROOT / "workbench/完整81只_可展开证据工作台.html"
    display_html = ROOT / "workbench/完整81只_可展开证据工作台_显示修正版.html"
    css = '<style>.line>span,.head>span{min-width:0;overflow-wrap:anywhere;word-break:normal}</style>'
    assert display_html.read_text(encoding="utf-8").replace(css, "", 1) == old_html.read_text(encoding="utf-8")
    qa = dict(observed_at=now.isoformat(), method="Actual Codex in-app browser, accessibility, screenshot and read-only rendered DOM",
        artifact=str(display_html), artifact_sha256=b.sha(display_html), total_rows=81, filtered_signal_date="2026-10-15", filtered_rows=6,
        badge="6 / 81", expanded_rows=1, expanded_code="002343", full_previous_members_visible=True,
        all_parent_relations_visible=True, original_author_hash_visible=True, original_conditions_visible=True,
        pending_target16_visible=True, adjacent_summary_columns_do_not_overlap=True,
        original_html_preserved=True, page_returned_http200=True, temporary_browser_closed=True,
        local_one_request_preview_server_exit_code=0, no_persistent_server_or_automation_created=True,
        note="First screenshot exposed text overlap; appended CSS-only display revision was loaded and visually rechecked. Browser telemetry 403 messages did not prevent the local page loading.")
    qa_path = ROOT / "workbench/browser_ui_qa.json"
    b.write_new(qa_path, qa)

    specs = [
        ("I.1", "研究归档9245条与当前70条，区分四层且不声称大模型微调、全市场或盈利", "delivery_final/完整9315只次_事实代理未知与全部冻结特征.csv", "9315个唯一日期股票，原9245保留；来源/代理/身份真值与结果分列，真实身份和利润仍未知。"),
        ("I.2", "允许因果规则研究，未知不填0，角色标签不是独立真值", "inventory/field_meanings_FACT_PROXY_UNKNOWN.json", "新结构角色及旧冻结角色分栏；每字段覆盖与独立身份真值0明示。"),
        ("II.1", "先读台账、当前说明、协议、收据和失败，不重跑已结束试验", "inventory/relevant_completed_fixed_trials_80_saved_models.json", "4个协议、80份原模型哈希当前逐一相同；本修复批拟合及数值推断0。README与最新AGENTS已读。"),
        ("II.2", "顺序清点、核引擎；无新增合格假设则落实具体实现修复", "construction_receipt.json", "新完整前池、周期/角色语义、全部父锚、未知通道与工作台均有实际代码和全量产物；未开新数值批次。"),
        ("II.3", "保留原预测、评分、排名、来源、失败、登记、窗口与生产BASE", "completion_final.json", "929项绑定当前复核无变化；旧说明文档仅用已核原字节快照显式映射，非数值文件替代。未改定时任务、下单或外发。"),
        ("III.1", "事件/公开/取得/截止时钟及历史生效版本分别记录，不倒填", "inventory/bound_source_manifest_WITH_EXPLICIT_UNKNOWN_CLOCKS.json", "原时钟未知保持未知，归档生成时间不是HTTP收件；晚取得及当前名单幸存者范围明确，旧说明恢复不冒称原取得。"),
        ("III.2", "收盘仅进入收盘后决策，竞价/盘中仅追加；未知非负例", "delivery_final/现有入口与到期比较.md", "原freeze/compare/execution及事实校验不变，09:25/09:35不回填收盘事实；当前旧日期曝光明示。"),
        ("III.3", "预处理及训练标签截止成熟、滚动量不含未来结果", "full_scope_verification.json", "原协议64个训练窗检查均252整日且max_target早于cutoff；当前因果函数信号字段白名单，不读target，当前318字段与批量相同。"),
        ("III.4", "完整候选和逐字段事实/代理/冲突/未知/不适用分别报覆盖", "delivery_final/逐字段覆盖_不等于事实认证.csv", "169字段×历史/10月共338行已逐项复算；填充数不认证事实，22条触板适配缺项隔离保留。"),
        ("IV.1", "周期使用高度、完整晋级、广度、溢价、损失效应；固定重叠/混合/缺失规则", "context/all_causal_node_contexts.csv", "classify_node用截至当日的完整/部分前池、炸板、跌停及昨日溢价，低高度不能单独破局；跨状态至少两压力日，角色非因果真值。"),
        ("IV.2", "父锚自主发现并保留全部合格范围、多锚及未知，不限手工赢家", "context/all_active_parent_relations_not_only_lag_matches.csv", "88521条过去5官方日>=4报板已观察锚关系（排除自己），全量独立重建；旧滞后2至6仅兼容子集，缺范围另标。"),
        ("IV.3", "当前连板起点、较早首板、重启/首次断板/后续负反馈按官方相邻日分开", "context/all9315_original_fields_and_review_semantics.csv", "当前起点须连续价格加前一节点边界；缺价/停牌不当断板，锚life-state和启动滞后明示，初始化修复前像保留。"),
        ("IV.4", "关联分层及替代解释，弱名称、同音、价量不得直接认证身份或资金迁移", "workbench/完整81只_全部证据与原审阅.json", "全部锚弱关系与作者采纳分列；历史真实题材/行业无完整资格保持UNKNOWN，旧冻结名称代理未当业务题材。"),
        ("IV.5", "完整前日k-1池晋级率含未触板失败，与当日触板失败率分开", "context/all_previous_members_including_no_touch_failures.csv", "37949成员、3827组逐项复算，含无幸存组/全部成功/分化/单成员/未知；有成员未知不出完整比率，保留界。"),
        ("IV.6", "精确日期/涨停价/资格、缺炸板不填0，相对排名不能认证主动领先", "context/all_height_groups_including_no_survivors.csv", "10%精确分价仅归档范围代理，交易所机制资格未认证；未知范围不补完整。同位量价分位为原冻结字段，语义修正另列。"),
        ("IV.7", "OHLC单价与旧低换手代理分栏，不用O不等于L、不从日线还原分时", "inventory/field_meanings_FACT_PROXY_UNKNOWN.json", "price_state核完整OHLC，严格新层再核精确cap/正量/相邻参考；真实开板次数/承接及分钟路径不足仍未知。"),
        ("IV.8", "累计非单价、首次打开、换手变化与参考日分别记录，不以爆量认证承接", "delivery_final/完整9315只次_事实代理未知与全部冻结特征.csv", "保留observed_nonword_boards与first_open_after_one_word；旧换手参照明确上次入池，原参考日未知不补；新相邻量比另标是volume非turnover。"),
        ("V.1", "非互斥六类角色候选、固定主角色优先、普通与UNKNOWN合法", "context/all9315_original_fields_and_review_semantics.csv", "统一POLICY角色优先序固定；空间/穿越/破局/作者支持补涨/相对卡位/非单价路径各有条件，真实换手核心/题材主线缺证据不认证。"),
        ("V.2", "规则角色条件晋级不冒称身份准确率，不用目标涨停代替金标准", "delivery_final/逐字段覆盖_不等于事实认证.csv", "独立身份真值0与角色准确率null；允许合法晋级训练，未声称补涨/主线身份识别已通过。"),
        ("VI.1", "完整池先行，限定股吧2至3原文及独立公告行情核验，失败有界保留", "workbench/完整81只_全部证据与原审阅.json", "复用2篇原12日正文与独立评论时钟，15日当日正文覆盖缺口明确；3PDF补核失败不算新增原文，不无限扩采。"),
        ("VI.2", "逐观点命题/推理/预期/采纳/反证/替代/取消；讨论非资金因果", "workbench/完整81只_全部证据与原审阅.json", "原审阅论坛opinion及处置完整绑定，支持/反证/竞争和取消保留；名气/自报收益不认证准确率，未覆盖不自动否或是。"),
        ("VI.3", "独立留档后展示模型，已曝光日期披露且不倒填盲审", "current6_saved_score_view_EXPLICIT_CHANNELS.json", "真实原作者稿完成后才显示保存分，先validate_completed再读分；当前追加预测明确已看模型，60原审阅/21缺记录，不冒称独立认证盲测。"),
        ("VII.1", "新增证据/未结束假设才训练，基准至多2挑战/30分钟/固定次数停止", "inventory/relevant_completed_fixed_trials_80_saved_models.json", "旧协议预算1800秒与一月一次收据保留，全部结束失败且不重试；本批仅数据/实现修复，不形成调参或新挑战。"),
        ("VII.2", "先完整252成熟信号日再分B，A/B/未知分栏，目标和全池一致", "evaluation/all15084_frozen_predictions_with_explicit_channels.csv", "3771行×4臂原字段相同；每臂A731/B3028/未知12，未知不输出B方向；逐窗协议检查先252再B。"),
        ("VII.3", "B按Recall/Precision/F1/误报漏报弃判、按日及同位Top1/2评估", "evaluation/all_daily_sameheight_score_height_Top1_Top2.csv", "372分层矩阵与128排序全量复算，完整选择/可核/未知/不足k/同分代码序保留，不套旧全池Accuracy+3pp否决。"),
        ("VII.4", "BASE/高度基线、板位/真实形态/情绪/来源分层、完整反例", "evaluation/all_fixed_strata_classification_WITH_UNKNOWN.csv", "含简单高度Top1/2和全部预定分层；全7542误差配对保留，9月减少FP亦丢TP，前两段和10月退化明确。"),
        ("VII.5", "时间块不确定性、多重比较与反复查看限制，分数未校准", "delivery_final/全周期短线研究_修复比较与验收.md", "旧固定时间块区间保留；尾部双臂校正未涵盖全部顺序试验明确作探索，补充分层不作新显著性，未报校准概率。"),
        ("VII.6", "取舍全部呈现，部署满足原适用资格，未永久保护BASE", "delivery_final/全周期短线研究_修复比较与验收.md", "已完成挑战未通过；本次修正语义尚未拟合，不能移植为原模型表现。当前未部署，未来满足授权资格仍可择优/回退。"),
        ("VIII.1", "B为入场日、B+1/+2官方日，完整信号至成交退出漏斗", "evaluation/complete_execution_funnel_FULL_DENOMINATORS.json", "4臂4段从3771完整信号分层；实际确认未知，代理开盘入场限制/缺价/未成交/退出流失保留，没有09:35倒回开盘成交。"),
        ("VIII.2", "实际/代理、费用滑点、未平仓浮亏、回撤、卖阻与净亏<=-7%分开", "evaluation/complete_execution_funnel_FULL_DENOMINATORS.json", "原代理逐笔与成本口径保存；9月联动B+2仅7/16代理且1例<=-7%，真实账户/仓位/收益/回撤未知，缺成交不报空仓或0收益。"),
        ("VIII.3", "现有prepare/freeze/compare/execution保持，全历史/主线/叙事/股吧/身份/父锚接入", "current6_workbench_context.json", "原接口及受保护源码字节不变；只读伴随CLI已实跑6只，全历史初始化+318重合字段一致，未知通道和未完审阅/重复行门保留。"),
        ("VIII.4", "历史当前同代码、范围/缺失/初始化一致，新上下文不改事实或自动方向", "full_scope_verification.json", "全9315原字段与当前6逐字段对照；价格初始化及停牌保护前像和失败保留，原事实/判断/账本未改。"),
        ("IX.1", "D1台账范围与排除、D2全特征/角色/失败者/替代锚", "delivery_final/最新目标逐项完成审查.json", "台账/来源manifest、9315×169表、338覆盖、37949成员和88521锚关系全部存在并绑定。"),
        ("IX.2", "D3固定比较与D4逐日工作台", "workbench/browser_ui_qa.json", "完整指标排序/分层/不确定性和81只逐股证据可打开；当期6只实际筛选展开，原21缺审阅不补。CSS显示修正版保留原件。"),
        ("IX.3", "D5执行验收及最终明确修复、指标改善退化、角色收益、部署结论", "delivery_final/全周期短线研究_修复比较与验收.md", "5类交付完成；研究结论允许失败，身份及真实盈利未验证、未部署与下一证据明确。没有把未来结果验证宣称完成。"),
        ("AG.1", "最新Top3最多2观察并逐股未来2/3/5上涨与分歧路径", "current_forecast/16日优先候选与2_3_5日独立情景.md", "慈文/百洋/富春前三、阳光/新华文轩观察，冰山偏否；完整6只支持/反证/同位/取消分列，概率未验证。"),
        ("AG.2", "次日收盘/触板、累计/逐日/连续涨停区分，到期全池留分母", "current_forecast/current66_events_SNAPSHOT_COMPARISON_PENDING.json", "6×11事件，16/19/20/22官方日期已固定；当前全66待到期而非零。评价器四项有意义测试且实际CLI运行，没有提前成熟或伪造成交。"),
    ]
    acceptance_dir = ROOT / "final_acceptance"
    acceptance_dir.mkdir(exist_ok=False)
    requirements = []
    for identifier, requirement, relative, evidence in specs:
        path = ROOT / relative
        assert path.exists(), str(path)
        requirements.append(dict(id=identifier, requirement=requirement, status="PROVEN_FOR_CURRENT_RESEARCH_SCOPE",
            evidence_path=str(path.resolve()), evidence_sha256=b.sha(path), inspected_finding=evidence))
    audit = dict(audited_at=now.isoformat(), objective_path=str(objective), objective_sha256=b.sha(objective),
        prior_goal_turn_classification="PROGRESS", prior_turn_evidence="Actual code, full-scope artifacts, 98 passing tests, completed comparison and forecast artifacts; verified current bindings",
        scope="Full fd62 research objective, all nine sections and latest local forecast clarification; research completion is not strategy efficacy or future outcome validation",
        requirements=requirements, requirements_proven=len(requirements), missing_current_deliverable_requirements=[],
        future_outcomes_not_claimed_complete=dict(events=66, first_target="2026-10-16", later_targets=["2026-10-19", "2026-10-20", "2026-10-22"], status="PENDING_OFFICIAL_WINDOWS_NOT_MATURED_AT_ACTUAL_AUDIT_TIME"),
        not_achieved_as_explicit_research_results=["Independent role identity accuracy", "Verified actual net profitability and portfolio drawdown", "Production eligibility", "Calibrated event probabilities"],
        current_research_deliverables_complete=True, strategy_or_profitability_proven=False)
    audit_path = acceptance_dir / "目标逐条证据与结论.json"
    b.write_new(audit_path, audit)

    index = ROOT / "delivery_final/交付索引与最终验收.md"
    def link(label, path):
        return f"[{label}](<{Path(path).resolve().as_posix()}>)"
    prior_reports = [
        ("首批B_ONLY/DUAL固定比较（32模型）", ROOT.parent / "short_term_thinking_20261015T071000Z/delivery/全周期短线思维重建与模型验证交付_定稿.md"),
        ("名称扩散固定比较（16模型）", ROOT.parent / "historical_symbolic_context_20261015T081200Z/delivery_final/历史短线补层与10月15日续接交付.md"),
        ("精确炸板修正固定比较（16模型）", ROOT.parent / "exact_attempt_response_20261015T100800Z/delivery_final/更正炸板输入固定比较与完整交付.md"),
        ("价格联动固定比较（16模型）", ROOT.parent / "lagged_price_relation_20261015T103000Z/delivery_final/历史价格联动证据与固定比较交付.md"),
    ]
    for _, p in prior_reports:
        assert p.exists()
    text = ["# 交付入口与最终验收", "", "研究、实现修复和完整交付已完成；预测增益有改善也有退化，角色准确率及真实盈利尚未验证，目前未部署。原80份相关模型未重训，旧失败未改写。本次最终检查仅修正工作台长字段换行，原页面与全部数据字节保留。", "",
        "- " + link("16日优先候选及2/3/5日逐股情景", ROOT / "delivery_final/16日优先候选与2_3_5日独立情景.md"),
        "- " + link("主报告：数据/实现修复、完整比较与执行边界", ROOT / "delivery_final/全周期短线研究_修复比较与验收.md"),
        "- " + link("81只可展开工作台（显示修正版）", display_html),
        "- " + link("9315只次完整冻结特征和新证据层", ROOT / "delivery_final/完整9315只次_事实代理未知与全部冻结特征.csv"),
        "- " + link("逐字段覆盖", ROOT / "delivery_final/逐字段覆盖_不等于事实认证.csv"),
        "- " + link("逐条目标证据", audit_path),
        "- " + link("现有入口与到期比较命令", ROOT / "delivery_final/现有入口与到期比较.md"), "",
        "全链固定比较原件（各自保留全部指标、来源、失败、区间和反例）：", ""]
    text += ["- " + link(label, p) for label, p in prior_reports]
    text += ["", "98项相关测试通过。最终逐项复核929个交付绑定、80份保存模型、338项覆盖统计；UI实际显示81/81，筛选15日显示6/81，展开慈文可见前池成员/全部锚/原作者哈希/取消条件。当前66个未来事件全部未到期，16/19/20/22日结果尚未发生，不能把研究交付完成称为多日预测已通过。", ""]
    assert not index.exists()
    index.write_text("\n".join(text), encoding="utf-8")
    additions = [Path(__file__), objective, ROOT / "completion_final.json", ROOT / "full_scope_verification.json", qa_path,
        ROOT / "final_acceptance_protocol_schema_failure_preserved.json", ROOT / "source_preimages/finalize_acceptance_BEFORE_PROTOCOL_DAY_SCHEMA_FIX.py",
        ROOT / "workbench/显示修正记录.json", display_html, audit_path, index,
        ROOT / "current_forecast/current66_events_SNAPSHOT_COMPARISON_PENDING.json", *[p for _, p in prior_reports]]
    bindings = completion["source_bindings"] | verification["source_bindings"] | model_hashes | {str(p.resolve()): b.sha(p) for p in additions}
    b.verify(bindings)
    final = dict(verified_at=datetime.now(timezone.utc).isoformat(), status="FULL_CURRENT_RESEARCH_OBJECTIVE_VERIFIED_COMPLETE_STRATEGY_UNPROVEN",
        objective_sha256=b.sha(objective), requirements_proven=len(requirements), current_requirements_missing=0,
        new_fits=0, new_inferences=0, saved_models_verified=80, source_bindings_verified=len(bindings),
        full_rows=9315, historical_rows=9245, field_coverage_cells_recomputed=338, workbench_rows=81,
        actual_browser_filter_rows=6, actual_browser_expansion_verified=True, tests_passed=98,
        future_events_pending=66, new_mature_forecast_events=0, new_formal_forecasts=0, actual_orders=0,
        role_accuracy_verified=False, actual_profitability_verified=False, production_deployed=False,
        all_original_numerical_sources_unchanged=True, goal_status_change_not_yet_claimed=True, source_bindings=bindings)
    b.write_new(ROOT / "completion_verified.json", final)
    print(json.dumps({k: v for k, v in final.items() if k != "source_bindings"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
