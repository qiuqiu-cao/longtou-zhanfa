from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
import short_term_thinking_dataset_builder as b
import preserved_bindings as preservation

ROOT=Path(__file__).resolve().parent


def main():
    start=b.read(ROOT/"execution_start.json");protected=preservation.verify(start["protected_before"])
    # Preserve the exact earlier companion source, including its original hash.
    old_context=b.read(ROOT/"initial_before_price_history_fix/current6_workbench_context.json")
    companion=b.ROOT/"relay_causal_review_workbench.py"
    old_source=companion.read_text(encoding="utf-8").replace(' or len(scores["rows"])!=len(context["rows"]) or len({x["code"] for x in scores["rows"]})!=len(scores["rows"])','')
    expected=old_context["source_bindings"][str(companion)]
    assert hashlib.sha256(old_source.encode()).hexdigest()==expected
    preimage=ROOT/"source_preimages/relay_causal_review_workbench_BEFORE_DUPLICATE_GUARD.py"
    preimage.write_text(old_source,encoding="utf-8")
    inventory=[]
    for path in (b.ROOT/"research/model_upgrade_validation").rglob("*"):
        if not path.is_file() or ROOT in path.parents:continue
        name=path.name.lower()
        if not (name.endswith("fit_started.json") or ("protocol" in name and name.endswith(".json")) or ("fail" in name and name.endswith(".json"))):continue
        copy=any(x in str(path).lower() for x in ["preimage","source_preimages","initial_","backup"])
        row=dict(path=str(path.resolve()),kind="FIT_START" if name.endswith("fit_started.json") else "PROTOCOL" if "protocol" in name else "FAILURE_RECORD",
            archive_copy=copy,sha256=b.sha(path),success_not_inferred_from_file_existence=True)
        try:
            obj=b.read(path)
            for k in ["formed_at","arm","month","cutoff","fit_calls","fits_max","fit_budget_seconds","status","error","refit_allowed"]:
                if k in obj:row[k]=obj[k]
        except (ValueError,TypeError):row["parse_status"]="UNAVAILABLE"
        inventory.append(row)
    out=ROOT/"inventory";out.mkdir(exist_ok=False)
    pd.DataFrame(inventory).to_csv(out/"all_research_protocol_fit_failure_records_NOT_FIT_COUNTS.csv",index=False,encoding="utf-8-sig")
    trials=[]
    paths=[("ST_B_ONLY_ST_DUAL",b.DEFAULT_RUN/"protocol_BEFORE_ANY_FIT.json",b.DEFAULT_RUN/"execution_after_binning_repair"),
        ("ST_DUAL_SYMBOLIC_DIFFUSION",ROOT.parent/"historical_symbolic_context_20261015T081200Z/protocol_BEFORE_ANY_FIT.json",ROOT.parent/"historical_symbolic_context_20261015T081200Z/fixed_comparison"),
        ("ST_DUAL_EXACT_FAILED_ATTEMPTS",ROOT.parent/"exact_attempt_response_20261015T100800Z/protocol_BEFORE_ANY_FIT.json",ROOT.parent/"exact_attempt_response_20261015T100800Z/fixed_comparison"),
        ("ST_DUAL_LAGGED_PRICE_RELATION",ROOT.parent/"lagged_price_relation_20261015T103000Z/protocol_BEFORE_ANY_FIT.json",ROOT.parent/"lagged_price_relation_20261015T103000Z/fixed_comparison")]
    for arm,protocol,directory in paths:
        obj=b.read(protocol);models=list(directory.glob("*_model.joblib"));receipts=list(directory.glob("*_fit_STARTED.json"))
        assert len(models)==len(receipts)==(32 if arm=="ST_B_ONLY_ST_DUAL" else 16)
        trials.append(dict(arm=arm,protocol_path=str(protocol.resolve()),protocol_sha256=b.sha(protocol),protocol_formed_at=obj.get("formed_at"),
            saved_models=len(models),fit_start_receipts=len(receipts),model_hashes={str(p.resolve()):b.sha(p) for p in models},
            status="COMPLETED_FAILED_ORIGINAL_FIXED_SCREEN_NOT_DEPLOYED",new_fits_this_batch=0,retry_authorized=False))
    b.write_new(out/"relevant_completed_fixed_trials_80_saved_models.json",dict(formed_at=b.now(),trials=trials,previous_saved_models=80,new_fits=0,
        original_preoptimization_technical_failure_not_counted_as_saved_model=True,
        scope="Relevant causal-thinking/model chain only, not a claim of total project optimizer calls; complete file-level registry separate"))
    manifest=[]
    receipts_keywords=["received_at","requested_at","retrieved_at","retrieval_time","published_at","publication_time","knowledge_cutoff"]
    for path,digest in protected.items():
        p=Path(path);record=dict(path=path,sha256=digest,bytes=p.stat().st_size,
            event_time="PER_ROW_SIGNAL_DATE_OR_ORIGINAL_RECEIPT",public_time=None,actual_received_time=None,decision_cutoff=None,
            file_mtime_not_used_as_receipt=True,source_grade="SEE_BOUND_ORIGINAL_SOURCE_AND_FIELD_QUALIFICATION")
        if p.suffix.lower()==".json" and p.stat().st_size<2_000_000:
            try:
                obj=b.read(p)
                if isinstance(obj,dict):
                    record["declared_top_level_clocks"]={k:obj[k] for k in receipts_keywords if k in obj}
                    record["public_time"]=obj.get("publication_time",obj.get("published_at"))
                    record["actual_received_time"]=obj.get("received_at",obj.get("retrieved_at"))
                    record["decision_cutoff"]=obj.get("knowledge_cutoff")
                    record["record_generated_at_not_public_or_received_time"]=obj.get("formed_at",obj.get("generated_at"))
            except (ValueError,TypeError):pass
        manifest.append(record)
    b.write_new(out/"bound_source_manifest_WITH_EXPLICIT_UNKNOWN_CLOCKS.json",dict(formed_at=b.now(),sources=manifest,
        individual_request_clocks_remain_in_original_raw_manifests=True,no_original_time_backfill=True,new_public_requests=0,
        current_universe_survivorship_bias=True,exchange_eligibility_not_certified=True))
    fields=dict(formed_at=b.now(),legacy_frozen_columns={
        "cycle_regime":"旧模型输入原件；低高度等旧规则，不能独自认证真实周期",
        "same_height_elimination_rate":"旧候选与触板失败近似比率，非完整前池淘汰率，亦非新精确触板池比率",
        "verified_turnover_boards":"旧日线形态/量价规则代理，不是真实回封、抛压或承接证明",
        "parent_candidates":"旧启动滞后2–6兼容锚子集，非全部活跃父锚",
        "primary_role":"旧因果规则角色，非独立身份真值",
        "daily_one_word":"原固定评估日线形态标签，与新价格/连续路径资格分开"},
        current_review_fields={
        "full_cohort_promotion_rate":"前一官方日完整k-1池所有成员在T日晋级k的比例；任何成员未知则null",
        "touch_failure_rate":"T日精确价格触板池中未封住占比；无合格完整范围则null",
        "review_cycle_state":"多字段规则状态；低高度单列、重叠为MIXED、资料不足UNKNOWN",
        "cross_state_continuation_candidate":"连续价格路径+至少两个压力日及不同状态的延续候选，不是完整周期穿越真值",
        "all_active_parent_relations":"过去5官方日≥4报板的全部已观察锚，排除自己；资格与缺失范围逐项保留",
        "current_streak_start_observed":"已核连续10%价格路径及前一节点非涨停边界；否则未知",
        "channel_review":"当日精确10%价格/形态范围内的A/B代理；不确定为UNRESOLVED，B方向为空",
        "review_primary_role":"固定优先序结构候选，普通RELAY_CONTENDER及UNKNOWN合法",
        "theme_core_status":"无完整已分类题材成员时保持UNKNOWN，名称/行业/叙事分层"},
        clocks="信号日是事件索引，不是公开/真实收件时间。晚取得用于当前知识下回顾，不称原时点已取得。",
        model_scores="未校准排序分，不称可信概率。新语义层未拟合、未宣称改善。")
    b.write_new(out/"field_meanings_FACT_PROXY_UNKNOWN.json",fields)
    old_attempt=ROOT.parent/"workbench_latest_20261015T091200Z/attempt_reclassification_correction/correction_and_impact_report.json"
    result=dict(formed_at=b.now(),new_fits=0,new_public_requests=0,existing_sources=len(manifest),
        existing_protocol_or_fit_or_failure_files=len(inventory),related_completed_saved_models=80,
        fit_file_count_not_actual_fit_count=True,raw_attempts=58772,exact_kept=56008,restored_failed_attempts=362,
        exclusions=dict(no_exact_cap_touch=2763,unresolved_over_cap_regime=1),
        newly_normalized_current_attempt_rows=434,strict_adapter_quarantine_rows=22,
        source_scope_not_assumed_full_exchange=True,
        source_bindings=protected|{str(p.resolve()):b.sha(p) for p in [Path(__file__),old_attempt,preimage,*list(out.glob("*"))]})
    b.write_new(ROOT/"inventory_completion.json",result)
    print({k:result[k] for k in ["new_fits","existing_sources","existing_protocol_or_fit_or_failure_files","related_completed_saved_models"]})


if __name__=="__main__":main()
