"""Read-only audit of already-saved predictions; no training or policy changes."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SOURCE = ROOT.parent / "lagged_price_relation_20261015T103000Z"
DATA = ROOT.parent / "workbench_latest_20261015T091200Z/attempt_reclassification_correction/corrected_dataset_SCHEMA_COMPLETE_RESEARCH_NOT_FITTED.joblib"
ENGINE = ROOT.parent / "workbench_latest_20261015T091200Z/attempt_reclassification_correction/corrected_engine_context_THROUGH_OCT15_RESEARCH_NOT_FITTED.joblib"
ARM = "ST_DUAL_LAGGED_PRICE_RELATION"
REFERENCES = ["ST_DUAL", "ST_DUAL_EXACT_FAILED_ATTEMPTS"]
PERIODS = [("2025H2", "2025-07-01", "2025-12-31"), ("2026JanAug", "2026-01-01", "2026-08-31"), ("2026Sep", "2026-09-01", "2026-09-30"), ("2026Oct", "2026-10-01", "2026-10-31")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def metrics(frame, field):
    valid = frame.label_valid.fillna(False) & frame.target.isin([0., 1.])
    known = frame[valid]
    prediction = known[field].gt(.5)
    positive = known.target.eq(1)
    tp, fp = int((prediction & positive).sum()), int((prediction & ~positive).sum())
    fn, tn = int((~prediction & positive).sum()), int((~prediction & ~positive).sum())
    return dict(rows=len(frame), known=len(known), invalid_or_unknown=int((~valid).sum()), positives=int(positive.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None, precision=tp/(tp+fp) if tp+fp else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    inputs = [SOURCE / "fixed_comparison/complete_paired_predictions.csv", SOURCE / "causal_context_v2/complete_price_relation_NO_LABELS.csv", SOURCE / "causal_context_v2/all_pool_and_parent_pairs_NO_LABELS.csv", SOURCE / "protocol_BEFORE_ANY_FIT.json", WORKSPACE / "historical_price_relation.py", WORKSPACE / "short_term_thinking_dataset_builder.py", DATA, ENGINE]
    before = {str(p): sha(p) for p in inputs}
    pred = pd.read_csv(inputs[0], dtype={"code": str})
    assert len(pred) == 15084 and not pred.duplicated(["date", "code", "arm"]).any()
    assert set(pred.arm) == {"BASE", ARM, *REFERENCES}
    relation = pd.read_csv(inputs[1], dtype={"code": str})
    pairs = pd.read_csv(inputs[2], dtype={"code": str, "peer_code": str})
    assert len(relation) == 9315 and len(pairs) == 1465910
    original_features = joblib.load(DATA)["features"]
    assert len(original_features) == 9245
    engine = joblib.load(ENGINE)
    features = engine["features"]
    assert len(features) == 9315
    historical_check = features[features.date.le("2026-09-30")].sort_values(["date", "code"]).reset_index(drop=True)
    missing_engine_columns = sorted(set(original_features) - set(historical_check))
    assert missing_engine_columns == ["price_identity_source_tier"]
    common_fields = [field for field in original_features if field in historical_check]
    pd.testing.assert_frame_equal(historical_check[common_fields].reset_index(drop=True), original_features.sort_values(["date", "code"])[common_fields].reset_index(drop=True))
    additional = features[["date", "code", "active_parent_count", "parent_candidates", "relation_channel_match", "supplement_coupling_score", "cycle_regime", "parent_anchor_state", "parent_anchor_code"]].copy()
    additional = additional.rename(columns={key: "causal_" + key for key in ["cycle_regime", "parent_anchor_state", "parent_anchor_code"]})
    additional["code"] = additional.code.astype(str).str.zfill(6)
    def nparents(v):
        if isinstance(v, str):
            v = json.loads(v)
        return len(v) if isinstance(v, (tuple, list)) else 0
    additional["timing_compatible_parent_count"] = additional.parent_candidates.map(nparents)
    additional = additional.drop(columns="parent_candidates")
    parents = pairs[pairs.is_active_parent].copy()
    parents["observed_parent_pair"] = parents.correlation.notna().astype(int)
    parents["missing_parent_under12"] = (parents.correlation.isna() & parents.common_sessions.lt(12)).astype(int)
    parents["missing_parent_constant_or_numerical"] = (parents.correlation.isna() & parents.common_sessions.ge(12)).astype(int)
    parent_audit = parents.groupby(["date", "code"]).agg(observed_parent_pairs=("observed_parent_pair", "sum"), missing_parent_under12=("missing_parent_under12", "sum"), missing_parent_constant_or_numerical=("missing_parent_constant_or_numerical", "sum"))
    relation = relation.merge(parent_audit, left_on=["date", "code"], right_index=True, how="left", validate="one_to_one")
    for key in parent_audit.columns:
        relation[key] = relation[key].fillna(0).astype(int)
    relation = relation.merge(additional, on=["date", "code"], how="left", validate="one_to_one")
    relation["active_parent_scope_gap"] = relation.active_parent_count.sub(relation.parent_peers)
    meta = [c for c in relation.columns if c not in ["name", "boards"]]
    source = pred[pred.arm.eq(ARM)].drop(columns="arm").rename(columns={"score": "new_score"})
    source = source.merge(relation[meta], on=["date", "code"], how="left", validate="one_to_one", indicator=True)
    assert source._merge.eq("both").all()
    source = source.drop(columns="_merge")
    known = source.label_valid.fillna(False) & source.target.isin([0., 1.])
    source["outcome_status"] = np.select([~source.label_valid.fillna(False), source.target.isna()], ["INVALID_LABEL_RETAINED", "UNKNOWN_LABEL_RETAINED"], default="VALID_BINARY")
    source["coverage_group"] = np.select([source.price_relation_history_fraction.lt(.6), ~source.scope_complete, source.price_relation_closing_pair_fraction.lt(1)], ["OWN_RETURNS_UNDER12", "INCOMPLETE_CURRENT_SCOPE", "PARTIAL_CLOSING_PAIR_COVERAGE"], default="FULL_CLOSING_PAIR_COVERAGE")
    source["parent_coverage_group"] = np.select([source.parent_peers.eq(0), source.price_relation_history_fraction.lt(.6), source.observed_parent_pairs.eq(0), source.observed_parent_pairs.lt(source.parent_peers)], ["NO_TIMING_COMPATIBLE_PARENT", "OWN_RETURNS_UNDER12", "NO_OBSERVABLE_PARENT_PAIR", "PARTIAL_PARENT_PAIR_COVERAGE"], default="ALL_TIMING_COMPATIBLE_PARENT_PAIRS_OBSERVED")
    summary, strata, continuous, all_rows = [], [], [], []
    relation_fields = [c for c in relation if c.startswith("price_relation_")] + ["closing_peers", "failed_peers", "parent_peers", "active_parent_count", "active_parent_scope_gap", "return_reference_conflict_sessions", "relation_channel_match", "supplement_coupling_score"]
    for reference in REFERENCES:
        old = pred[pred.arm.eq(reference)]
        frame = source.merge(old[["date", "code", "score", "target", "label_valid", "daily_one_word"]], on=["date", "code"], suffixes=("", "_reference"), validate="one_to_one")
        for field in ["target", "label_valid", "daily_one_word"]:
            pd.testing.assert_series_equal(frame[field], frame[field + "_reference"], check_names=False)
            frame = frame.drop(columns=field + "_reference")
        frame = frame.rename(columns={"score": "reference_score"})
        frame["reference_arm"] = reference
        a, z = frame.reference_score.gt(.5), frame.new_score.gt(.5)
        positive, negative = known & frame.target.eq(1), known & frame.target.eq(0)
        frame["transition"] = np.select([positive & a & ~z, positive & ~a & z, negative & a & ~z, negative & ~a & z, ~known & a.ne(z)], ["TP_LOST", "FN_RESCUED", "FP_REMOVED", "TN_TO_FP", "UNKNOWN_OUTCOME_DIRECTION_CHANGED"], default="DIRECTION_UNCHANGED")
        frame["period"] = None
        for period, start, end in PERIODS:
            select = frame.date.between(start, end)
            frame.loc[select, "period"] = period
            for shape, shaped in [("ALL_ROWS", frame[select]), ("TRUE_NONWORD", frame[select & frame.daily_one_word.eq(0)]), ("ONEWORD", frame[select & frame.daily_one_word.eq(1)]), ("UNKNOWN_SHAPE", frame[select & frame.daily_one_word.isna()])]:
                summary.append(dict(period=period, reference=reference, shape=shape, reference_metrics=metrics(shaped, "reference_score"), new_metrics=metrics(shaped, "new_score"), all_direction_changes=int(shaped.reference_score.gt(.5).ne(shaped.new_score.gt(.5)).sum()), transitions=shaped.transition.value_counts(dropna=False).to_dict(), invalid_or_unknown_direction_changes=int((~(shaped.label_valid.fillna(False) & shaped.target.isin([0., 1.])) & shaped.reference_score.gt(.5).ne(shaped.new_score.gt(.5))).sum())))
            nonword = frame[select & frame.daily_one_word.eq(0)]
            for category in ["boards", "causal_cycle_regime", "coverage_group", "parent_coverage_group", "causal_parent_anchor_state", "primary_role"]:
                for value, group in nonword.groupby(category, dropna=False):
                    strata.append(dict(period=period, reference=reference, category=category, value=value, rows=len(group), valid_known=int((group.label_valid.fillna(False)&group.target.isin([0.,1.])).sum()), **group.transition.value_counts(dropna=False).to_dict()))
            for transition, group in nonword.groupby("transition"):
                continuous.append(dict(period=period, reference=reference, transition=transition, rows=len(group), observations={field: dict(known=int(group[field].notna().sum()), median=float(group[field].median()) if group[field].notna().any() else None) for field in relation_fields}))
        all_rows.append(frame)
    complete = pd.concat(all_rows, ignore_index=True)
    assert len(complete) == 7542 and complete.period.notna().all()
    complete.to_csv(ROOT / "all3771_x2_reference_rows_WITH_UNKNOWN.csv", index=False, encoding="utf-8-sig")
    complete[complete.daily_one_word.eq(0) & complete.transition.ne("DIRECTION_UNCHANGED")].to_csv(ROOT / "all_nonword_direction_changes_WITH_UNKNOWN.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strata).fillna({key: 0 for key in ["TP_LOST", "FN_RESCUED", "FP_REMOVED", "TN_TO_FP", "UNKNOWN_OUTCOME_DIRECTION_CHANGED", "DIRECTION_UNCHANGED"]}).to_csv(ROOT / "all_nonword_changes_by_fixed_strata.csv", index=False, encoding="utf-8-sig")
    historical = relation[relation.date.le("2026-09-30")]
    assert len(historical) == 9245
    scope_audit = dict(historical_rows=len(historical), historical_rows_active_parent_scope_smaller=int(historical.active_parent_scope_gap.gt(0).sum()), total_active_parent_relations=int(historical.active_parent_count.sum()), total_temporally_compatible_parent_relations=int(historical.parent_peers.sum()), omitted_active_parent_relations=int(historical.active_parent_scope_gap.sum()), rows_negative_scope_gap=int(historical.active_parent_scope_gap.lt(0).sum()), parent_correlations_are_lag2to6_subset_not_all_active_anchors=True, outcome_filter=False, names_filter=False)
    metadata_missing = {arm: {field: int(part[field].isna().sum()) for field in ["cycle_regime", "parent_anchor_state", "parent_anchor_code", "primary_role"]} for arm, part in pred.groupby("arm")}
    restored_metadata = {field: int(source[field].notna().sum()) for field in ["causal_cycle_regime", "causal_parent_anchor_state", "causal_parent_anchor_code"]}
    result = dict(formed_at=datetime.now(timezone.utc).isoformat(), status="READ_ONLY_SAVED_MODEL_ERROR_AUDIT_NO_NEW_FIT", actual_new_fits=0, threshold_changed=False, original_score_threshold="strict >0.5 unchanged", original_labels_preserved=True, paired_original_candidates=3771, four_arm_prediction_rows=15084, pairwise_reference_audit_rows=len(complete), summary=summary, relation_statistics_by_transition=continuous, parent_scope_semantic_discrepancy=scope_audit, original_prediction_missing_metadata=metadata_missing, separately_bound_causal_metadata_nonmissing=restored_metadata, engine_missing_display_metadata=missing_engine_columns, common_historical_input_fields_exact=True, limitations=["Descriptive retrospective associations cannot isolate attribution to any correlated feature.", "All already-seen periods remain retrospective diagnostics; no optimized threshold, window, or refit.", "Missing or invalid labels are retained outside metric denominators; missing shape is not assigned to nonword.", "Parent price similarity is a statistical proxy, never independent role or fund-flow truth.", "Parent correlation consumes the engine's launch-lag 2–6 compatible subset despite metadata wording claiming all active anchors.", "Saved exact-attempt and price-relation prediction CSV metadata loses cycle_regime and parent_anchor_state; causal_* columns are separately bound to preserved original feature inputs, not a rewrite of saved predictions."], source_bindings=before)
    assert all(sha(path) == value for path, value in before.items())
    result["all_input_hashes_unchanged"] = True
    (ROOT / "audit_result.json").write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [s for s in summary if s["shape"] == "TRUE_NONWORD"]
    lines = ["# 保存模型变化独立审查", "", "本批只读既存预测，拟合0；严格0.5、成员、标签和全部未知保持。", "", "|区间|对照|非一字行/已核|TP旧→新|FP旧→新|漏掉旧TP|救回FN|去掉FP|新增FP|未知方向改变|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rows:
        a,z,t = s["reference_metrics"],s["new_metrics"],s["transitions"]
        lines.append(f"|{s['period']}|{s['reference']}|{a['rows']}/{a['known']}|{a['tp']}→{z['tp']}|{a['fp']}→{z['fp']}|{t.get('TP_LOST',0)}|{t.get('FN_RESCUED',0)}|{t.get('FP_REMOVED',0)}|{t.get('TN_TO_FP',0)}|{t.get('UNKNOWN_OUTCOME_DIRECTION_CHANGED',0)}|")
    lines += ["", "历史父锚口径发现：", "", f"原9245行中{scope_audit['historical_rows_active_parent_scope_smaller']}行的父锚相关性仅取启动滞后2–6的兼容锚，活跃父锚关系共{scope_audit['total_active_parent_relations']}，相关性纳入{scope_audit['total_temporally_compatible_parent_relations']}，其差{scope_audit['omitted_active_parent_relations']}。这与‘all prior active high anchors’元数据不一致；不是结果筛选或名称筛选，也不单独解释性能变化。旧模型与协议原件未改。", "", "输出元数据发现：原更正炸板与价格联动两臂保存CSV的cycle_regime、parent_anchor_state各3771行全部空白，而原因果输入对应字段完整。审查表以causal_*独立列绑定原输入，保存CSV的空字段完全保留。模型使用的regime_*数值特征不因此被证明有错；本批不改权重、不重算预测。", "", "完整7542对照行、未知状态、全部固定分层及各变化组关联覆盖和中位数分别在同目录CSV/JSON。统计描述不能证明某字段导致误报减少或漏报增加，不据此开启重训。"]
    (ROOT / "保存模型变化独立审查.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(clean(dict(status=result["status"], parent_scope=scope_audit, nonword=rows)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
