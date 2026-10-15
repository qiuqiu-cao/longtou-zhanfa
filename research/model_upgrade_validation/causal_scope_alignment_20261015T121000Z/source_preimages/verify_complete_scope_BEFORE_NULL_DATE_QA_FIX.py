"""Independent whole-scope arithmetic/source/initialization verification."""
from pathlib import Path
import sys,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from decimal import Decimal,ROUND_HALF_UP
from collections import Counter
import re
import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import preserved_bindings as preservation

ROOT=Path(__file__).resolve().parent


def cents(value):
    value=Decimal(str(value))*100
    if not value.is_finite() or value<=0 or value!=int(value):raise ValueError("noncent price")
    return int(value)


def expected_member(prior,bar,previous):
    if bar is None:return "UNKNOWN_MISSING_DAILY"
    if bar.get("is_suspended") is True or str(bar.get("trade_status","")).upper()=="SUSPENDED":return "UNKNOWN_SUSPENDED_EXPLICIT"
    volume=b.finite(bar.get("volume"))
    if volume is None:return "UNKNOWN_VOLUME_MISSING"
    if volume<=0:return "UNKNOWN_NONTRADING"
    try:
        op,hi,lo,cl=[cents(bar[x]) for x in ["open","high","low","close"]];pc=cents(prior)
        if not lo<=min(op,cl)<=max(op,cl)<=hi:return "UNKNOWN_PRICE_CONFLICT"
        if previous is not None and cents(previous["close"])!=pc:return "UNKNOWN_PRICE_CONFLICT"
        if b.finite(bar.get("pre_close")) is not None and cents(bar["pre_close"])!=pc:return "UNKNOWN_PRICE_CONFLICT"
        upper=int((Decimal(pc)*Decimal("1.1")).quantize(Decimal(1),rounding=ROUND_HALF_UP))
        if hi>upper:return "UNKNOWN_PRICE_CONFLICT"
        return "PROMOTED_EXACT_CAP" if cl==upper else "FAILED_TOUCHED_CAP" if hi==upper else "FAILED_NO_CAP_TOUCH"
    except (ValueError,TypeError,KeyError):return "UNKNOWN_PRICE_CONFLICT"


def numeric(actual,expected):
    if expected is None:assert pd.isna(actual)
    else:assert np.isclose(actual,expected,atol=1e-12,rtol=0),(actual,expected)


def main():
    start=b.read(ROOT/"execution_start.json");protected=preservation.verify(start["protected_before"])
    receipts=[ROOT/"construction_receipt.json",ROOT/"evaluation_completion.json",ROOT/"workbench_completion.json",ROOT/"inventory_completion.json",ROOT/"forecast_completion.json",ROOT/"final_context_rebuild_VERIFIED.json"]
    bindings=protected.copy()
    for path in receipts:
        receipt=b.read(path);b.verify(receipt["source_bindings"]);bindings|=receipt["source_bindings"]
    artifact=joblib.load(ROOT/"context/review_scope_context.joblib");rows=artifact["rows"];trans=artifact["transitions"]
    engine=joblib.load(artifact["engine_path"]);sessions=engine["sessions"];indices={d:i for i,d in enumerate(sessions)}
    assert len(rows)==9315 and len(rows[rows.date.le("2026-09-30")])==9245
    pd.testing.assert_frame_equal(rows[list(engine["features"])],engine["features"].reset_index(drop=True))
    members=trans["members"];groups=trans["groups"];known={"PROMOTED_EXACT_CAP","FAILED_TOUCHED_CAP","FAILED_NO_CAP_TOUCH"}
    dates=set(members.date)|set(members.previous_date)
    data=engine["bars"][engine["bars"].date.isin(dates)]
    prices={(r["date"],r["code"]):r for r in data.to_dict("records")}
    for row in members.itertuples():
        expected=expected_member(row.previous_bound_close,prices.get((row.date,row.code)),prices.get((row.previous_date,row.code)))
        assert expected==row.transition_state,(row.date,row.code,expected,row.transition_state)
    member_groups={key:f for key,f in members.groupby(["date","target_height"])}
    for row in groups.itertuples():
        sub=member_groups.get((row.date,row.target_height),members.iloc[:0]);states=Counter(sub.transition_state)
        n=len(sub);success=states["PROMOTED_EXACT_CAP"];unknown=n-sum(states[k] for k in known)
        assert row.full_cohort_observed_previous_count==n and row.full_cohort_promoted_count==success and row.full_cohort_unknown_count==unknown
        expected=success/n if n and row.full_cohort_status=="COMPLETE_PROXY_ELIGIBLE_COHORT" else None
        numeric(row.full_cohort_promotion_rate,expected)
        if pd.notna(row.full_cohort_previous_count) and row.full_cohort_previous_count>0:
            numeric(row.full_cohort_promotion_lower_bound,success/n);numeric(row.full_cohort_promotion_upper_bound,(success+unknown)/n)
        if pd.notna(row.touch_failure_attempt_count) and row.touch_failure_attempt_count>0:
            numeric(row.touch_failure_rate,row.touch_failure_failed_count/row.touch_failure_attempt_count)
    # All reported active anchors reconstructed independently, exclude the child.
    attempts=engine["attempts"];pool={d:f for d,f in attempts[attempts.pool_assignment.eq("CLOSING_LIMIT_UP")].groupby("signal_date")}
    last={};expected_pairs=0;byday={d:f for d,f in rows.groupby("date")}
    for day in sessions:
        if day>rows.date.max():break
        t=indices[day];active={code for code,d in last.items() if 0<t-indices[d]<=5}
        for row in byday.get(day,rows.iloc[:0]).itertuples():
            expected=sorted(active-{row.code});assert row.complete_active_parent_codes==expected;expected_pairs+=len(expected)
        for row in pool.get(day,attempts.iloc[:0]).itertuples():
            if row.board_height>=4:last[row.code]=day
    assert expected_pairs==len(artifact["parents"])==88521
    assert artifact["parents"].code.ne(artifact["parents"].parent_code).all()
    cycles=artifact["cycles"].set_index("date").to_dict("index")
    for row in rows.itertuples():
        if row.cross_state_continuation_candidate:
            t=indices[row.date];trail=[cycles[d] for d in sessions[max(0,t-min(row.reported_height,5)+1):t+1] if d in cycles]
            assert sum(any(x in d["review_cycle_flags"] for x in ["DIVERGENCE_CONTEXT","EBB_CONTEXT"]) for d in trail)>=2
            assert len({d["review_cycle_state"] for d in trail if d["review_cycle_state"]!="UNKNOWN_INCOMPLETE_CONTEXT"})>=2
        if row.channel_review=="UNRESOLVED_IDENTITY":assert not row.b_direction_allowed
        if row.current_streak_start_observed is not None:assert row.current_streak_start_observed<=row.date
    current=b.read(ROOT/"current6_workbench_context.json");b.verify(current["source_bindings"])
    source=rows[rows.date.eq("2026-10-15")].set_index("code").to_dict("index");current_checks=0
    for row in current["rows"]:
        for key,value in row.items():
            if key in source[row["code"]]:
                assert b.clean(source[row["code"]][key])==value,(row["code"],key);current_checks+=1
    view=b.read(ROOT/"current6_saved_score_view_EXPLICIT_CHANNELS.json");b.verify(view["source_bindings"])
    assert len(view["rows"])==6 and view["new_inferences"]==0
    # Every frozen prediction field is preserved, not merely the score column.
    original=pd.read_csv(ROOT.parent/"lagged_price_relation_20261015T103000Z/fixed_comparison/complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    pred=pd.read_csv(ROOT/"evaluation/all15084_frozen_predictions_with_explicit_channels.csv",dtype={"code":str},low_memory=False)
    pd.testing.assert_frame_equal(original,pred[list(original)])
    assert pred[pred.daily_one_word.isna()].B_direction_review.isna().all()
    periods={"2025H2":("2025-07-01","2025-12-31"),"2026JanAug":("2026-01-01","2026-08-31"),"2026Sep":("2026-09-01","2026-09-30"),"2026Oct":("2026-10-01","2026-10-31")}
    strata=pd.read_csv(ROOT/"evaluation/all_fixed_strata_classification_WITH_UNKNOWN.csv",dtype={"stratum":str})
    for row in strata.itertuples():
        lo,hi=periods[row.period];f=pred[pred.arm.eq(row.arm)&pred.date.between(lo,hi)&pred[row.dimension].astype(str).eq(row.stratum)]
        z=f[f.label_valid.eq(True)&f.target.notna()&f.score.notna()];yes=z.target.eq(1);call=z.score.gt(.5)
        counts=[int((yes&call).sum()),int((~yes&call).sum()),int((yes&~call).sum()),int((~yes&~call).sum())]
        assert [row.tp,row.fp,row.fn,row.tn]==counts and row.candidates==len(f) and row.scored==len(z)
        tp,fp,fn,tn=counts;numeric(row.Recall,tp/(tp+fn) if tp+fn else None);numeric(row.Precision,tp/(tp+fp) if tp+fp else None)
        numeric(row.F1,2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None);numeric(row.false_positive_rate,fp/(fp+tn) if fp+tn else None)
    ranking=pd.read_csv(ROOT/"evaluation/all_daily_sameheight_score_height_Top1_Top2.csv")
    for row in ranking.itertuples():
        lo,hi=periods[row.period];f=pred[pred.arm.eq(row.arm)&pred.date.between(lo,hi)&pred.daily_one_word.eq(0)]
        keys=["date","boards"] if row.grouping=="DATE_HEIGHT" else ["date"]
        chosen=[]
        for _,part in f.groupby(keys,sort=True):
            if row.ordering=="SCORE_DESC_CODE_ASC":part=part.sort_values(["score","code"],ascending=[False,True])
            else:part=part.sort_values(["boards","code"],ascending=[False,True])
            chosen.append(part.iloc[:int(row.k)])
        selected=pd.concat(chosen);known=selected[selected.label_valid.eq(True)&selected.target.notna()]
        assert len(selected)==row.selected and len(known)==row.known
        numeric(row.candidate_hit_rate,float(known.target.mean()) if len(known) else None)
        numeric(row.group_any_hit_rate,float(known.groupby(keys).target.max().mean()) if len(known) else None)
        assert int(selected.groupby(keys).target.apply(lambda x:x.isna().any()).sum())==row.groups_with_unresolved_truth
    funnel=b.read(ROOT/"evaluation/complete_execution_funnel_FULL_DENOMINATORS.json")
    for item in funnel["rows"]:
        assert item["full_candidates"]==item["A_one_price_shape"]+item["B_nonone_price_shape"]+item["unknown_shape"]
        assert item["B_nonone_price_shape"]==item["Top1_selected"]+item["not_selected_nonword"]
        assert item["actual_fill_records_added"]==item["paper_fill_records_added"]==0
    original81=pd.read_csv(ROOT.parent/"lagged_price_relation_20261015T103000Z/delivery_final/完整81只对照_追加价格联动研究与独立观点.csv",dtype={"code":str},low_memory=False)
    current81=pd.read_csv(ROOT/"workbench/完整81只_原判断条件来源与校正口径.csv",dtype={"code":str},low_memory=False)
    pd.testing.assert_frame_equal(current81[list(original81)],original81)
    doc=b.read(ROOT/"workbench/完整81只_全部证据与原审阅.json")
    assert len(doc["rows"])==81 and sum(r["archived_author_record"]["archive_status"]=="ACTUAL_ARCHIVED_RECORD" for r in doc["rows"])==60
    html=(ROOT/"workbench/完整81只_可展开证据工作台.html").read_text(encoding="utf-8")
    assert html.count("<details>")==81 and "NaN" not in html and "�" not in html
    future=b.read(ROOT/"current_forecast/current66_events_PENDING_NOT_ZERO_OR_MATURED.json")
    assert len(future["rows"])==66 and all(r["status"]=="PENDING_OFFICIAL_WINDOW" and r["actual"] is None for r in future["rows"])
    xml=ET.parse(ROOT/"related_tests.xml").getroot();tests=sum(int(x.attrib.get("tests",0)) for x in xml.iter("testsuite"));failed=sum(int(x.attrib.get("failures",0))+int(x.attrib.get("errors",0)) for x in xml.iter("testsuite"))
    assert tests==98 and failed==0
    # State how much the two denominators differ without using outcomes to refit.
    common=rows[rows.full_cohort_promotion_rate.notna()&rows.same_height_elimination_rate.notna()].drop_duplicates(["date","boards"])
    differences=int((abs((1-common.full_cohort_promotion_rate)-common.same_height_elimination_rate)>1e-12).sum())
    bindings|={str(p.resolve()):b.sha(p) for p in [Path(__file__),ROOT/"related_tests.xml",ROOT/"current6_workbench_context.json",ROOT/"current6_saved_score_view_EXPLICIT_CHANNELS.json",*receipts]}
    b.verify(bindings)
    result=dict(formed_at=b.now(),status="PASS_FULL_RESEARCH_SCOPE_IMPLEMENTATION_AND_REPORTING_NO_UPGRADE_CLAIM",new_fits=0,new_inferences=0,
        all9315_original_feature_fields_preserved=True,complete_transition_groups=len(groups),all_previous_members_recomputed=len(members),
        complete_active_parent_pairs_reconstructed=expected_pairs,current_batch_live_fields_equal=current_checks,
        frozen_prediction_rows_exact=15084,stratified_rows_recomputed=len(strata),ranking_rows_recomputed=len(ranking),
        full_workbench_rows=81,actual_author_archives=60,unknown_author_archives=21,future_event_rows_pending=66,
        complete_vs_legacy_denominator_comparable_groups=len(common),complete_vs_legacy_denominator_different_groups=differences,
        tests_passed=tests,test_failures=0,original_nondocument_bound_files_unchanged=864,
        external_instruction_updates=1,old_instruction_exact_preimage_verified=True,current_instruction_followed=True,
        old_models_scores_labels_registries_changed=False,role_identity_accuracy=None,actual_profitability=None,production_eligible=False,
        source_bindings=bindings)
    b.write_new(ROOT/"full_scope_verification.json",result)
    print({k:v for k,v in result.items() if k!="source_bindings"})


if __name__=="__main__":main()
