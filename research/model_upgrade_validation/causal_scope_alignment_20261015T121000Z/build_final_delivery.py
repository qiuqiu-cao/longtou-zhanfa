from pathlib import Path
import sys,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import historical_symbolic_context as symbolic
from current_model_diagnostics import RELAY_FIELDS
import preserved_bindings as preservation

ROOT=Path(__file__).resolve().parent


def main():
    verified=b.read(ROOT/"full_scope_verification.json");b.verify(verified["source_bindings"])
    output=ROOT/"delivery_final";output.mkdir(exist_ok=False)
    artifact=joblib.load(ROOT/"context/review_scope_context.joblib");engine=joblib.load(artifact["engine_path"])
    rows=artifact["rows"].copy()
    raw=engine["prepared"]["relay"][["date","code"]+RELAY_FIELDS].rename(columns={x:"frozen_raw_"+x for x in RELAY_FIELDS})
    table=rows.merge(raw,on=["date","code"],validate="one_to_one")
    symbol_path=symbolic.RUN/"qualified_final/symbolic_context.joblib"
    bank=joblib.load(symbol_path)
    current_path=symbolic.RUN/"latest15_prepared_v2/symbolic_context_SCORES_MASKED.json"
    current=pd.DataFrame(b.read(current_path)["rows"])
    symbols=pd.concat([bank["rows"][["date","code"]+symbolic.FIELDS],current[["date","code"]+symbolic.FIELDS]],ignore_index=True)
    assert len(symbols)==9315 and not symbols.duplicated(["date","code"]).any()
    symbols=symbols.rename(columns={x:"frozen_symbolic_"+x for x in symbolic.FIELDS})
    table=table.merge(symbols,on=["date","code"],validate="one_to_one")
    table["frozen_symbolic_layer_status"]="PREVIOUS_FIXED_NAME_PROXY_VERSION_NOT_ACTIVE_BUSINESS_THEME_OR_RECOMPUTED_CHALLENGER"
    table["legacy_turnover_reference_semantics"]="LAST_OBSERVED_RELAY_POOL_NOT_NECESSARILY_PREVIOUS_SESSION"
    table["legacy_turnover_reference_date"]=None
    assert len(table)==9315 and not any(x in table for x in ["target","target_date","label_valid"])
    pd.testing.assert_frame_equal(table[list(rows)],rows)
    table.to_csv(output/"完整9315只次_事实代理未知与全部冻结特征.csv",index=False,encoding="utf-8-sig")
    coverage=[]
    for part,frame in [("HISTORY9245",table[table.date.le("2026-09-30")]),("OCTOBER70",table[table.date.gt("2026-09-30")])]:
        for field in table:
            coverage.append(dict(scope=part,field=field,rows=len(frame),populated=int(frame[field].notna().sum()),null=int(frame[field].isna().sum()),
                cell_population_not_fact_certification=True,independent_role_identity_gold_n=0))
    pd.DataFrame(coverage).to_csv(output/"逐字段覆盖_不等于事实认证.csv",index=False,encoding="utf-8-sig")
    # Freeze exact current instruction bytes without changing the existing forecast.
    forecast_dir=ROOT/"current_forecast";snap=[]
    for live in [b.ROOT/"AGENTS.md",b.ROOT/"templates/复盘_涨停优先与多日持续上涨.md"]:
        target=forecast_dir/("frozen_"+live.name);shutil.copy2(live,target)
        assert b.sha(target)==b.sha(live)
        snap.append(dict(original_path=str(live.resolve()),sha256=b.sha(live),snapshot_path=str(target.resolve())))
    b.write_new(forecast_dir/"instruction_snapshots.json",dict(captured_at=b.now(),documents=snap,
        exact_existing_forecast_bound_bytes=True,not_original_public_or_received_time=True,no_prediction_rewrite=True))
    forecast=b.read(ROOT/"forecast_completion.json")
    first=forecast_dir/"16日优先候选与2_3_5日独立情景.md"
    shutil.copy2(first,output/first.name)
    prior=ROOT.parent/"lagged_price_relation_20261015T103000Z/fixed_comparison/model_validation_results.json"
    result=b.read(prior);periods=[]
    for period,value in result["comparisons"].items():
        for arm in ["BASE","ST_DUAL","ST_DUAL_EXACT_FAILED_ATTEMPTS","ST_DUAL_LAGGED_PRICE_RELATION"]:
            m=value["metrics"][arm];n=m["nonword"]
            periods.append(f"|{period}|{arm}|{n['scored']}/{n['candidates']}|{n['tp']}/{n['fp']}|{n['Recall']:.2%}|{n['Precision']:.2%}|{n['F1']:.2%}|{m['top1']['known']}/{m['top1']['selected']}；{m['top1']['candidate_hit_rate']:.2%}|{m['top2']['known']}/{m['top2']['selected']}；{m['top2']['candidate_hit_rate']:.2%}|")
    counts=rows[rows.date.le("2026-09-30")].review_primary_role.value_counts().to_dict()
    lines=["# 全周期短线研究：实现修复、固定结果与当前候选", "", "**在项目已归档的15日6只连板候选中，16日相对优先为慈文传媒、百洋股份、富春染织；阳光乳业、新华文轩观察，冰山冷热偏不晋级。概率未验证。** 当前排序和2/3/5日情景已另行留档，已看模型，不冒称盲审；不代表可买或已成交。", "", "按最新fd62目标完成的是研究、实现修复、完整同池检验与如实交付。原固定挑战失败仍是有效结果；角色独立真值0、真实收益未知及未部署均明确保留，不能据此宣称策略有效，也不把它们扩大为禁止完成研究。", "", "## 本轮实际修复", "", "本轮拟合0、数值推断0、正式预测/成熟/订单0。新增完整前池转移、校正周期与角色语义、全部已观察父锚、明确未知通道、完整分层与执行漏斗、逐股原条件来源及多日事件比较器。没有重开任何已结束挑战、重选阈值或改写原评分/标签/登记。", "", "旧same_height_elimination_rate只是旧候选与触板失败的近似比率，不是完整前池淘汰率。新层逐组列前一官方日k−1池→T日k池，包含未触板失败及缺价/停牌/冲突未知；同时另列精确触板失败率。3827组、37949成员全量复算，2349个可比较组中1580组与旧近似口径不同。这是口径修复证据，不是预测增益。", "", "9315行全部保留：9245历史、70条10月观察。8549行所在组可完整算率，648行有成员未知、111行前池范围未知、7行空组不适用。严格价格适配另隔离22条现有资料缺字段的触板记录，缺日线不当断板、缺炸板不填0。原V21558772输入中56008精确合格、362炸板恢复、2763未精确触板和1条涨停机制未决的排除原件继续保留。", "", "低高度只作为观察，不能单独触发破局；节点同时保留正常、混合与缺资料状态。穿越候选要求连续价格路径、至少两个压力日及不同状态；单次分歧只作较弱幸存线索。卡位候选还要求实际竞争者失败和相对位置改善，日线换手结构不再认证真实承接核心。", "", "过去5官方日达到4板的全部已观察锚逐股保留（排除自己），共88521条关系，范围缺失另列。旧价格联动实际只取启动滞后2–6子集：原9245行有7848行的锚范围更窄，旧metadata的all措辞不准确；未据此重跑模型或把它当性能变化原因。首次起板、较早首板经历、重启、断板及后续负反馈分列；起点须有价格路径和前一节点边界，不能只倒推报板。", "", "历史主角色分布（研究候选而非金标准）："+json.dumps(counts,ensure_ascii=False)+"。题材核心无法认证的保留UNKNOWN；冻结名称代理、实际行业、股吧叙事及真实题材分别展示，不从常见单字或一项放量认定父龙/资金迁移。", "", "## 完整比较与误报代价", "", "下表全部来自已完成、已冻结的80份本目标相关模型链，本轮没有再训练。四方案15084配对原字段完全相同；原训练为252完整成熟信号日后划分通道，月初只读此前已成熟标签。不同初始方案同时改变了估计器、特征和训练权重，不能单因子归因。", "", "|区间|方案|已核/非一字完整池|TP/FP|Recall|Precision|F1|Top1可核/完整；命中率|Top2可核/完整；命中率|", "|---|---|---:|---:|---:|---:|---:|---:|---:|",*periods,"", "价格联动对9月BASE：Recall15%→42.5%、F124%→49.28%，Precision60%→58.62%。相对原双通道，FP22→12但TP20→17，Recall50%→42.5%；Top1为9/13对8/13，完整各选16天；Top2为16/26对17/26，完整各选32条。前两历史段相对更正炸板参考的Recall/F1退步；10月原已核37条非一字中5TP/8FP也差于原6TP/7FP。不能只报9月进步。原全部方案均未过各自固定门，未部署。", "", "独立变化审查显示9月相对原双通道：去掉12个FP同时新增2个FP，漏掉6个旧TP同时救回3个FN，另6条方向改变仍无合格目标标签。误报减少不是只发生在无父锚或资料不足组，关联中位数也不能作因果解释。原全部7542对照、未知与预定维度均保留。", "", "补充372行板位/情绪/来源/形态/联动覆盖分层及128组按日/同位、分数/高度的Top1/Top2。分层属于事后描述，不产生新确认性显著性或阈值优化。每臂3771条分为原形态A731、B3028、未知12；旧后两方案缺失的周期/父锚元数据已从原因果表另列补回，非数值评分Bug。未知形态不输出B方向。", "", "原全部时间块区间继续引用固定结果文件：9月Recall相对BASE探索区间约+17.07至+38.24pp；相对更正炸板参考约−30.77至+14.29pp。旧双臂尾部校正未覆盖全部顺序挑战；历史已看过，不称新盲测或新的确认性显著提升。", "", "## 工作台、当前预测与执行", "", "完整81只工作台包含全部原字段、逐股周期/锚/完整同位与前池、题材证据层、原独立方向及模型分歧、实际归档的支持/反证/开盘确认/取消条件和原文路径哈希。60行有原审阅；9/30与10/8共21行未定位原独立记录，保留未知而不补事前观点。74个补充结果、7未知与原资格完全不变，补充结果不写回冻结训练或正式账本。", "", "原relay_review_workbench.py的prepare/freeze/compare/execution未改。新relay_causal_review_workbench.py只读消费原绑定事实和完整历史/主线/叙事/父锚，已实际运行当前6只CLI；318个重合字段与批量相同，先验证原完成审阅才显示已有研究分。待审阅、重复股票和未知形态边界已测试；原事实行、作者方向及原11类冻结门保留。", "", "当前新追加6只未来情景，主次和2/3/5日路径见同目录16日优先候选文件。未来2/3/5日分别对应19/20/22日；期末累计上涨、逐日上涨、连续涨停和16日触板/收盘涨停分别定义。66项目前全待窗口成熟，不当0或新成熟样本。该记录已看模型、未正式freeze；没有校准概率或多日数值头。新3个PDF网址补核工具不可读，没有把失败当新正文；依旧只使用已绑定公司文件，未认证当前新增公告。", "", "执行漏斗现在从每臂完整3771信号开始，逐层列形态、Top1选择、入场取消/未知/代理、B+1/B+2退出。实际同刻竞价确认、券商成交和账户仓位仍未知，不报空仓或零收益。B是代理入场日，B+1/B+2再按官方日历后移，费用与滑点及卖出受阻分列；没有用09:35观察倒回开盘成交。9月价格联动Top1有8个开盘代理入场、B+2仅7个退出，完整选择16次中覆盖43.75%，均值7.76%、一例净亏≤−7%。原双通道5例均值8.04%，不同股票和覆盖不能证明期望收益改善，也不能生成组合回撤。", "", "## 来源、验证与结论", "", "865个原绑定输入中，864个非说明文件保持原字节。工作期间另一项用户澄清更新AGENTS.md；旧版本按已保存SHA精确恢复并另存，新说明实际读取执行，没有回退用户要求。唯有这项说明更新按显式映射换到旧字节快照；任何行情、模型、评分、标签或登记变化仍拒绝。所有原始失败、初始化修复前像、Pandas空日期校验修复及初版输出保留。", "", "98项相关测试通过。全量复算37949成员、3827转移组、88521锚关系、372分层、128排序，并核对15084冻结预测原字段、81完整工作台和当前318字段。初始化与明确停牌保护修复后的现存9315行未变。字段覆盖表是填充/缺失统计，不是事实认证；原始时钟、公开时钟未知的保持未知，晚取得资料仅作当前知识下回顾。", "", "明确回答：数据适配、分母、周期/角色语义、元数据/未知通道、分层/漏斗/来源条件与多日事件评价已修复或补齐；部分晋级指标提高、部分退化，原失败结论未改；角色身份准确率与真实收益尚未验证；当前不满足部署条件。研究交付已完成并不等于预测能力、资金因果或盈利已经验证。", "", "下一项具体证据：16日完整6只及其官方交易资格、公告和有真实时钟的确认/日线；19/20/22日到期逐事件补齐。新增资料须与原记录同池比较，不重新选择赢家或修改旧窗口；若未来形成可区分新证据，另立固定协议才能开新的数值批次。"]
    report=output/"全周期短线研究_修复比较与验收.md";report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    interface=output/"现有入口与到期比较.md"
    interface.write_text("# 现有入口与到期比较\n\n原relay_review_workbench.py prepare完整传history-dataset/mainline-history/narrative-reviews/identity-evidence/thinking-bundle/symbolic-context，先遮蔽分数、作者审阅，再原freeze/compare/execution。旧已见日期只作诊断附层。\n\n新证据层：\n\n```powershell\npython relay_causal_review_workbench.py context --facts <原已绑定完整facts> --bundle <已更新到当日完整engine> --output <全新context.json>\n# 已有原完整审阅后可加 --review，标明实际曝光。\npython relay_causal_review_workbench.py score-view --facts <原facts> --context <上述context> --review <原完整作者稿> --scores <同池已保存研究分> --output <新独立视图>\n```\n\n新日期必须有自己的完整来源，不能把15日6池换日期；前池/价量缺失保留未知。此伴随视图不改变原事实/注册，不重训。\n\n未来窗口到期：\n\n```powershell\npython research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z/compare_current_forecast.py --record <本批原6只情景JSON> --bundle <含到期原始日线来源的新engine> --asof <真实带时区时刻> --output <全新比较JSON>\n```\n\n只对官方窗口到期事件结算，缺交易资格/价格仍未知；累计、逐日、连续涨停分开，保留完整66行。说明文件后来更新时只允许按已认证字节快照核验，数据/代码/模型/原预测变化不能绕过。实际资金收益仍由原execution来源另算。\n",encoding="utf-8")
    requirements=[]
    for ident,requirement,evidence,limits in [
        ("D1","台账与范围",["inventory_completion.json","inventory/all_research_protocol_fit_failure_records_NOT_FIT_COUNTS.csv","inventory/bound_source_manifest_WITH_EXPLICIT_UNKNOWN_CLOCKS.json"],"当前股票池幸存者范围、未决涨停机制/字段、未知原始时钟明示；文件数不等于拟合数"),
        ("D2","完整事实/代理/角色表",["delivery_final/完整9315只次_事实代理未知与全部冻结特征.csv","context/all_height_groups_including_no_survivors.csv","context/all_previous_members_including_no_touch_failures.csv","context/all_active_parent_relations_not_only_lag_matches.csv"],"规则候选不是身份真值；实际题材及部分交易资格未知，名称代理仅冻结旧版"),
        ("D3","完整固定比较",["evaluation_completion.json","evaluation/all_fixed_strata_classification_WITH_UNKNOWN.csv","evaluation/all_daily_sameheight_score_height_Top1_Top2.csv"],"旧比较失败保持；补充分层仅描述；原时间块区间不作全部顺序试验确认性显著性"),
        ("D4","逐日工作台",["workbench_completion.json","workbench/完整81只_可展开证据工作台.html","current_forecast/15日收盘_16日优先与2_3_5日情景_已看模型.json"],"21个原独立记录未知，15日已看模型披露；未来窗口尚未到期，不倒填"),
        ("D5","执行与验收状态",["evaluation/complete_execution_funnel_FULL_DENOMINATORS.json","forecast_completion.json","compare_current_forecast.py"],"无券商与同刻确认，真实收益/仓位/回撤未知；代理小样本未证盈利，未部署")]:
        for path in evidence:assert (ROOT/path).exists(),path
        requirements.append(dict(id=ident,requirement=requirement,status="PROVEN_AS_RESEARCH_DELIVERABLE_WITH_EXPLICIT_LIMITATIONS",evidence=evidence,limitations=limits))
    audit=dict(audited_at=b.now(),objective_path=r"C:\Users\Administrator\.codex\attachments\fd62f217-e468-4f83-aa83-13b485abe89f\goal-objective.md",
        objective_sha256=b.sha(Path(r"C:\Users\Administrator\.codex\attachments\fd62f217-e468-4f83-aa83-13b485abe89f\goal-objective.md")),requirements=requirements,
        crosscutting={"four_layers":"fact/proxy/candidate+author/outcome+execution separated","causality":"signal allowlists, official adjacency, no target inputs, new history/current318-field replay",
            "training":"80 previously saved relevant models remain byte-identical; no repeats, no current fits","forum":"original checked bodies/opinions and failures bound; no popularity/fund causality inference",
            "new_dates":"original author-before-model flow retained; current old15 exposure disclosed","source_time":"unknown original/public/receipt clocks not synthesized; recovered instruction bytes do not assert old receipt",
            "deployment":"not eligible; original source/forward gates and windows preserved, no permanent proof BASE optimal",
            "next_windows":"66 future events pending; actual calendar comparator tested, no elapsed current event left unsettled"},
        latest_local_instruction="Top3+up to2observations and future2/3/5 paths delivered, probabilities unvalidated and exposure disclosed",
        full_research_goal_achieved=True,strategy_effectiveness_or_profitability_achieved=False,model_deployed=False)
    b.write_new(output/"最新目标逐项完成审查.json",audit)
    paths=[Path(__file__),symbol_path,current_path,ROOT/"compare_current_forecast.py",*list(output.glob("*")),*list(forecast_dir.glob("frozen_*")),forecast_dir/"instruction_snapshots.json"]
    bindings=verified["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths};b.verify(bindings)
    proof=dict(formed_at=b.now(),status="RESEARCH_DELIVERABLES_COMPLETE_STRATEGY_UNPROVEN_NOT_DEPLOYED",full_research_goal_achieved=True,
        new_fits=0,new_inferences=0,new_formal_predictions=0,new_formal_mature_samples=0,actual_orders=0,actual_fill_records_added=0,
        research_fixed_models_reused=80,full_historical_rows=9245,full_rows_with_current=9315,complete_daily_workbench_rows=81,
        tests_passed=98,old_numerical_sources_unchanged=True,external_instruction_doc_revision_preserved=True,
        independent_role_accuracy=None,real_profitability=None,production_deployment=False,future_unregistered_scenario_events_pending=66,
        source_bindings=bindings)
    b.write_new(ROOT/"completion_final.json",proof)
    print({k:v for k,v in proof.items() if k!="source_bindings"})


if __name__=="__main__":main()
