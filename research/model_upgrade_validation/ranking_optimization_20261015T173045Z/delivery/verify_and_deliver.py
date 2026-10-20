"""Seal the fixed research batch without changing a model or a prediction."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import short_term_thinking_dataset_builder as b

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path(__file__).resolve().parent
REPORT = DELIVERY / "重新调试_新旧模型与评分系统完整交付.md"


def run():
    if REPORT.exists() or (DELIVERY / "batch_verified_NOT_MODEL_ACCEPTANCE.json").exists():
        raise FileExistsError("immutable delivery already exists")
    receipts = [
        REPO / "research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z/completion_verified.json",
        REPO / "research/model_upgrade_validation/expectation_345_20261015T163226Z/completion_345_verified.json",
        ROOT / "protocol_BEFORE_ANY_FIT.json",
        ROOT / "fixed_comparison/ranking_validation_results.json",
        ROOT / "verification/独立核验收据.json",
        ROOT / "source_feature_training/protocol_BEFORE_ANY_FIT.json",
        ROOT / "source_feature_training/same_source_route_alignment/same_source_aligned_fixed_result.json",
        ROOT / "source_feature_training/same_source_route_alignment/current15_six_SHARED_FEATURE_API_receipt.json",
        ROOT / "source_feature_training/independent_verification/最终同资格独立核验收据.json",
        ROOT / "source_feature_training/independent_verification/三段增益归因独立核验收据.json",
        ROOT / "source_feature_training/independent_verification/实际API无目标泄漏与A守恒收尾核验收据.json",
        ROOT / "source_feature_training/metric_applicability_final/metric_applicability_receipt.json",
        ROOT / "source_feature_training/delivery_frozen_receipt.json",
        ROOT / "scorecard/final_verified26439_research_view/manifest.json",
        ROOT / "scorecard/final_verified26439_independent_verification.json",
        ROOT / "score_reliability/final26439/read_only_diagnostic_receipt.json",
    ]
    combined = {}
    sizes = {}
    for path in receipts:
        data = b.read(path)
        sources = data.get("source_bindings", data.get("sources", {})).copy()
        sources.update(data.get("output_bindings", {}))
        sources.update(data.get("implementation_bindings", {}))
        for field in ["source", "implementation", "output"]:
            if field + "_path" in data and field + "_sha256" in data:
                sources[data[field + "_path"]] = data[field + "_sha256"]
        b.verify(sources)
        sizes[str(path.resolve())] = len(sources)
        for name, digest in sources.items():
            if name in combined and combined[name] != digest:
                raise ValueError("conflicting frozen versions must be resolved explicitly: " + name)
            combined[name] = digest
        combined[str(path.resolve())] = b.sha(path)

    ranking = b.read(ROOT / "fixed_comparison/ranking_validation_results.json")
    source = b.read(ROOT / "source_feature_training/same_source_route_alignment/same_source_aligned_fixed_result.json")
    source_check = b.read(ROOT / "source_feature_training/independent_verification/最终同资格独立核验收据.json")
    decomposition = b.read(ROOT / "source_feature_training/independent_verification/三段增益归因独立核验收据.json")
    scoring = b.read(ROOT / "scorecard/final_verified26439_independent_verification.json")
    if not source_check["passed"] or not decomposition["passed"]:
        raise ValueError("the independent source and attribution checks must pass")
    if source["screen_pass"] or ranking["historical_screen_pass"]:
        raise ValueError("this report describes the recorded failed screens")
    if ranking["new_fits"] != 16 or ranking["old_model_refits"] != 0:
        raise ValueError("unexpected fixed ranking fit counts")
    if source["original_experiment_new_fits"] != 9 or source["original_experiment_reused_windows"] != 7:
        raise ValueError("unexpected source experiment fits")
    if source_check["final_rows"] != 26439 or not source_check["original22662_all_fields_unchanged"]:
        raise ValueError("complete comparison or preservation failed")
    if scoring.get("actual_browser_visual_verified", False):
        raise ValueError("this environment did not complete browser visual verification")

    page_dir = ROOT / "scorecard/final_verified26439_research_view"
    manifest = b.read(page_dir / "manifest.json")
    if manifest["complete_output_rows"] != 26439 or len(manifest["arms"]) != 7:
        raise ValueError("complete final scorecard required")
    for name, digest in manifest["artifact_hashes"].items():
        path = page_dir / name
        if b.sha(path) != digest:
            raise ValueError("final scorecard artifact differs: " + name)
        combined[str(path.resolve())] = digest

    metrics_path = ROOT / "source_feature_training/metric_applicability_final/same_pool_classification_metrics_APPLICABILITY_CLEAN.csv"
    rank_path = ROOT / "source_feature_training/same_source_route_alignment/complete_full_slot_Top1_Top2_Top3.csv"
    metrics = pd.read_csv(metrics_path)
    ranks = pd.read_csv(rank_path)
    sep = metrics[metrics.period.eq("2026Sep") & metrics.scope.eq("ORIGINAL_NONWORD")].set_index("arm")
    pooled = metrics[metrics.period.eq("HISTORICAL_POOLED") & metrics.scope.eq("ORIGINAL_NONWORD")].set_index("arm")
    old, new = pooled.loc["CORRECTED_CAUSAL"], pooled.loc["SOURCE_PATCHED_FEATURE_CLASSIFIER"]
    gap = (new.F1 - old.F1) * 100
    if abs(gap - 0.1200530524) > 1e-7:
        raise ValueError("unexpected fixed pooled F1 difference")
    current_path = ROOT / "source_feature_training/same_source_route_alignment/current15_six_SHARED_FEATURE_API_target16_UNMATURED.csv"
    current = pd.read_csv(current_path, dtype={"code": str})
    if len(current) != 6 or current.target.notna().any() or current.label_valid.fillna(False).any():
        raise ValueError("current saved research cohort must retain six unknown outcomes")

    test_path = ROOT / "all_integration_frozen.xml"
    suite = ET.parse(test_path).getroot()
    suites = [suite] if suite.tag == "testsuite" else list(suite.iter("testsuite"))
    tests = sum(int(x.attrib.get("tests", 0)) for x in suites)
    failures = sum(int(x.attrib.get("failures", 0)) + int(x.attrib.get("errors", 0)) for x in suites)
    if tests != 75 or failures:
        raise ValueError("the final meaningful integration checks must pass")

    def pct(x):
        return f"{100 * x:.2f}%"

    table = []
    for arm, name in [("BASE", "正式BASE"), ("ST_DUAL", "原双通道研究模型"), ("CORRECTED_CAUSAL", "修正特征模型＋相同资料资格"), ("SOURCE_PATCHED_FEATURE_CLASSIFIER", "单字段补全后重训研究头")]:
        row = sep.loc[arm]
        table.append(f"| {name} | {int(row.tp)} | {int(row.fp)} | {pct(row.Recall)} | {pct(row.Precision)} | {pct(row.F1)} |")
    lines = []
    for arm, name in [("ST_DUAL", "原双通道"), ("PAIRWISE_DAY_RANK", "新成对排序"), ("CORRECTED_CAUSAL", "原修正特征头"), ("SOURCE_PATCHED_FEATURE_CLASSIFIER", "单字段重训头")]:
        rows = ranks[ranks.period.eq("HISTORICAL_POOLED") & ranks.arm.eq(arm)].set_index("k")
        one, three = rows.loc[1], rows.loc[3]
        lines.append(f"| {name} | {int(one.hits)}/{int(one.full_slots)} | {int(three.hits)}/{int(three.full_slots)} |")

    root = ROOT.resolve().as_posix()
    repo = REPO.as_posix()
    report = f"""# 重新调试、新旧模型与评分系统交付

本批完成25次新拟合：成对排序16次，来源补全单字段分类9次；另7个训练窗因输入完全相同而精确复用旧模型。旧BASE、旧双通道、原预测与正式账本保持原值。两项新挑战均未通过固定跨期标准，整体预测升级仍未验收。

## 已实际修复的内容

从原已绑定原始响应中核对同日及相邻官方交易日价格、前收和正成交证据，补出47条价格资格。原244条资格缺口中197条仍是不合格或未知，保留具体原因；没有把缺失当0、把停牌当断板、把原生手数混入历史成交量比。9月35条资格缺口均有可用附层。原资料取得时钟保留为晚到，不宣称原时点已知或新盲测。

来源修复是新增附层，没有重写原预测。9月修正特征模型原冻结方向TP15/FP19、Recall37.5%，补资料资格后TP21/FP24、Recall52.5%；这6个新增真阳性和5个误报属于资料资格变化。旧保存模型使用补全推断输入，仅5个原始分数微变，方向及Top-k均不变。随后同输入重学权重才多抓1个真阳性，误报不变。

## 同池9月非一字比较

原冻结非一字完整143条，107条结果已知、40条正例、36条结果未知。所有方案保留相同分母；未知结果不作失败，弃判的正例仍计入漏判。

| 模型及资料口径 | 抓到晋级TP | 误报FP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

不能把BASE15%到新头55%的全部差值归为本批重训。原双通道已经达到50%；相同资料资格下，本批重训只是52.5%到55%。单字段重训的跨期F1为{pct(old.F1)}到{pct(new.F1)}，提高{gap:.2f}个百分点，未达预登记1个百分点；Top3还少2个命中，因此失败且未部署。

## 排序是否改善

以下是2025H2、2026年1—8月及9月合并的完整固定槽口径。Top1共302槽、其中4未知；Top3共906槽、其中12未知。先从完整池选择再查结果，未知保留在槽内。表内分母包含未知槽，并非把未知判成错误。

| 排序方案 | Top1已知命中／完整槽 | Top3已知命中／完整槽 |
|---|---:|---:|
{chr(10).join(lines)}

新成对排序Top1比原双通道多6个命中，但Top3少15个，固定筛选失败。它只有相对偏好分，没有绝对涨停事件头；不计算其绝对Recall、F1、误报率或收益。原V282线性排序失败记录保留，本次是不同资料和固定结构的研究，未重跑原参数搜索。

## 评分系统实际交付

[完整评分页]({root}/scorecard/final_verified26439_research_view/complete_scorecard.html)包含7组×同一3777候选观察，共26439条模型输出；默认显示归档最新2026-10-15的42条，可分页、筛选并展开完整证据。原6组的原始及派生字段逐值不变。

原BASE分数／方向／排名、100分显示、研究分类分、来源补全后方向、完整池／同位／非一字／非一字同位排名、并列与分差分别显示。100分只是原分×100，仍非事件概率；相对第一不改原FAIL或UNKNOWN。新排序与单字段研究头标记失败未采用。

只读可靠性诊断新增35组汇总和300个固定十分位箱。原6组已有24条诊断汇总不变，未训练校准器、未改阈值或分数。概率仍未验证，不报涨停百分比概率。

实际验证包括75项针对检查、16个保存排序模型重放、单字段16窗（9新／7复用）重放、旧16模型patched输入归因，以及全量目标／未来字段扰动下排名守恒。HTML通过本地Node脚本功能验证；真实浏览器视觉核验受file协议策略限制，状态为未完成，未绕过策略。

## 可调用入口与边界

- [评分模块]({repo}/relay_research_scorecard.py)：`build_scorecard(frame)`／`write_scorecard(...)`；只读导出，不生成预测或训练。
- [单字段共享构造器]({repo}/relay_source_feature_research.py)：`patched_feature_copy(...)`，训练与当前6只真实推断均调用同一函数。
- [相同资格实际推断包装]({repo}/relay_source_feature_alignment.py)：`predict_with_same_source_routes(...)`；已核非一字用自身B分严格>0.5，其余继承同资格旧头方向，避免将一字资格变化算作训练增益。
- [审阅后排序入口]({repo}/relay_pairwise_workbench.py)：先验证原完整作者审阅，再加载本批固定模型及训练／当期一致的特征投影。它拒绝把未绑定新日期伪装为本批当期。

新日期仍走完整池、股吧预读、独立逐股判断留档、模型冲突、原执行与结算门。原始训练输出中9条一字路由差异及dtype守卫技术失败均保留；本批追加真实可调用包装修正，没有因此重拟合或覆盖原件。

归档10月15日完整6只已使用同一特征API和保存10月模型实际推断，特征补值0格；其目标为官方10月16日收盘，保存结果仍6只未知。本批没有读取其目标结果作调试，没有新增正式预测、成熟样本、成交或定时任务。原66条多日预测事件保留，尚未训练新的2/3/5日上涨事件头；角色准确率、真实盘口、成交及扣费收益也未证明改善。

本批工程与研究交付完成，目标继续active。下一步优先解决有新增证据支持的输入缺口并检验新日期；现有历史的反复研究不能替代前向验证。本批两项失败方案不调阈值、权重重跑，不以未通过的研究头替换旧模式。

验证收据：[batch_verified_NOT_MODEL_ACCEPTANCE.json]({root}/delivery/batch_verified_NOT_MODEL_ACCEPTANCE.json)。完整比较、原失败、协议、模型及独立核验均随收据绑定。
"""
    REPORT.write_text(report, encoding="utf-8")
    sources = [Path(__file__), REPORT, test_path, metrics_path, rank_path, current_path,
        ROOT / "source_feature_training/independent_verification/来源资格_推断输入_重学权重三段增量.csv"]
    for path in sources:
        combined[str(path.resolve())] = b.sha(path)
    b.verify(combined)
    receipt = dict(
        verified_at=b.now(), status="BATCH_VERIFIED_OVERALL_MODEL_UPGRADE_NOT_ACCEPTED",
        old_959_bindings_verified=True, previous_1165_bindings_verified=True,
        exact_original22662_cells_preserved=source_check["original_cells_unchanged"],
        complete_paired_outputs=26439, candidates_per_arm=3777, arms=7,
        actual_new_fits=25, new_pairwise_fits=16, new_source_feature_fits=9,
        unchanged_training_windows_reusing_original_models=7, old_BASE_refits=0,
        source_price_qualifications_added=47, unresolved_or_ineligible_qualifications=197,
        qualification_input_and_weight_effects_separated=True,
        independent_source_and_ranking_verification=True, meaningful_tests_passed=tests,
        pairwise_fixed_screen_pass=False, source_feature_fixed_screen_pass=False,
        event_probability_validated=False, role_accuracy_validated=False,
        actual_net_profit_improvement_validated=False, production_replaced=False,
        new_formal_forecasts=0, new_matured_events=0, actual_orders=0,
        saved_current_six_unknown_outcomes=6, archived_multiday_events_preserved=66,
        actual_browser_visual_verified=False, local_HTML_functional_validation=True,
        goal_remains_active=True, goal_completed=False, schedules_changed=False,
        checked_receipts_and_binding_counts=sizes, source_bindings=combined,
    )
    b.write_new(DELIVERY / "batch_verified_NOT_MODEL_ACCEPTANCE.json", receipt)
    print({"status": receipt["status"], "source_bindings_verified": len(combined), "tests": tests, "new_fits": 25})


if __name__ == "__main__":
    run()
