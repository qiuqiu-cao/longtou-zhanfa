from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import joblib
import pandas as pd

import short_term_thinking_dataset_builder as b
import relay_full_cohort_transition as transition
import relay_causal_review_context as semantic

ROOT=Path(__file__).resolve().parent
ENGINE=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/attempt_reclassification_correction/corrected_engine_context_THROUGH_OCT15_RESEARCH_NOT_FITTED.joblib"
REVIEW=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/author_current6_COMPLETED_BEFORE_MODEL_DISPLAY.json"


def main():
    proof=b.read(ROOT/"execution_start.json");b.verify(proof["protected_before"])
    engine=joblib.load(ENGINE);b.verify(engine["sources"])
    c=engine["features"];a=semantic.normalize_attempts(engine["attempts"],engine["bars"],engine["sessions"])
    print("bound sources and normalized absent adapter fields",len(c),len(a),flush=True)
    trans=transition.build_transition_context(c,a,engine["bars"],engine["sessions"])
    print("complete cohort transitions reconstructed",trans["summary"]["candidate_status_counts"],flush=True)
    review=b.read(REVIEW);edges=[]
    for row in review["rows"]:
        for parent in row.get("identity_discovery_assessment",{}).get("accepted_anchor_codes",[]):
            edges.append(dict(date=review["signal_date"],code=row["code"],parent_code=parent,accepted=True,
                hypothesis_public_date="2026-10-12",source="ALREADY_ARCHIVED_AUTHOR_HYPOTHESIS_NOT_FACT_OR_IDENTITY_GOLD",review_path=str(REVIEW)))
    context=semantic.build_causal(c,engine["prepared"]["market"],a,engine["bars"],engine["sessions"],trans,author_parent_edges=edges)
    out=ROOT/"context";out.mkdir(exist_ok=False)
    trans["rows"].to_csv(out/"all9315_full_previous_cohort_context.csv",index=False,encoding="utf-8-sig")
    trans["groups"].to_csv(out/"all_height_groups_including_no_survivors.csv",index=False,encoding="utf-8-sig")
    trans["members"].to_csv(out/"all_previous_members_including_no_touch_failures.csv",index=False,encoding="utf-8-sig")
    trans["attempt_quarantine"].to_csv(out/"source_attempt_quarantine.csv",index=False,encoding="utf-8-sig")
    context["all_active_parent_relations"].to_csv(out/"all_active_parent_relations_not_only_lag_matches.csv",index=False,encoding="utf-8-sig")
    context["cycle_days"].to_csv(out/"all_causal_node_contexts.csv",index=False,encoding="utf-8-sig")
    combined=c.merge(trans["rows"].drop(columns=["boards"]),on=["date","code"],validate="one_to_one")
    combined=combined.merge(context["rows"].drop(columns=["name"]),on=["date","code"],validate="one_to_one")
    pd.testing.assert_frame_equal(combined[list(c)],c.reset_index(drop=True))
    assert len(combined)==9315
    combined.to_csv(out/"all9315_original_fields_and_review_semantics.csv",index=False,encoding="utf-8-sig")
    sources=proof["protected_before"]|engine["sources"]|{str(p.resolve()):b.sha(p) for p in [ENGINE,REVIEW,Path(__file__),Path(transition.__file__),Path(semantic.__file__)]}
    artifact=dict(version=semantic.VERSION,policy=semantic.POLICY,rows=combined,transitions=trans,
        cycles=context["cycle_days"],parents=context["all_active_parent_relations"],engine_path=str(ENGINE),author_edges=edges,source_bindings=sources)
    joblib.dump(artifact,out/"review_scope_context.joblib")
    report=dict(formed_at=b.now(),status="ADDITIVE_CAUSAL_INTERPRETATION_REPAIR_NO_FIT",rows=len(combined),historical_rows=int(combined.date.le("2026-09-30").sum()),
        transition_summary=trans["summary"],price_path_status=combined.current_price_path_status.value_counts().to_dict(),
        review_primary_roles=combined.review_primary_role.value_counts().to_dict(),review_cycle_states=combined.review_cycle_state.value_counts().to_dict(),
        all_active_parent_relations=len(context["all_active_parent_relations"]),
        prior_frozen_fields_preserved=True,old_scores_or_labels_changed=False,new_fits=0,new_formal_predictions=0,
        source_bindings=sources|{str(p.resolve()):b.sha(p) for p in out.glob("*") if p.is_file()})
    b.write_new(ROOT/"construction_receipt.json",report)
    print({k:report[k] for k in ["rows","historical_rows","all_active_parent_relations","new_fits","price_path_status"]})


if __name__=="__main__":main()
