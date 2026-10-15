"""Complete frozen-score reporting; no new choice, threshold, model, or fit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as original
from historical_symbolic_verify import classification,ranking,selection,assert_values
import preserved_bindings as preservation

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/"lagged_price_relation_20261015T103000Z"
PERIODS=original.history.FOLDS+[dict(id="2026Oct",start="2026-10-01",end="2026-10-31")]


def descriptive_metrics(frame):
    out=classification(frame)
    known=frame[frame.target.notna()&frame.label_valid.fillna(False).astype(bool)]
    positives=known.target.eq(1);calls=known.score.gt(.5)
    fp=int((calls&~positives).sum());tn=int((~calls&~positives).sum())
    out.update(false_positive_rate=fp/(fp+tn) if fp+tn else None,
        false_positive_per_known_candidate=fp/len(known) if len(known) else None,
        unresolved_labels=len(frame)-len(known),unknown_shape_abstentions=int(frame.daily_one_word.isna().sum()),
        descriptive_only=True,threshold="strict >0.5 unchanged")
    return out


def main():
    before=preservation.verify(b.read(ROOT/"execution_start.json")["protected_before"])
    source=PREVIOUS/"fixed_comparison/complete_paired_predictions.csv"
    pred=pd.read_csv(source,dtype={"code":str},low_memory=False)
    original_columns=list(pred)
    baseline=pred[pred.arm.eq("BASE")]
    metadata=joblib.load(ROOT/"context/review_scope_context.joblib")["rows"]
    context=metadata[["date","code","cycle_regime","parent_anchor_state","current_price_path_status","channel_review","review_cycle_state"]].rename(
        columns={"cycle_regime":"canonical_frozen_cycle_regime","parent_anchor_state":"canonical_frozen_parent_state","channel_review":"current_qualification_channel"})
    pred=pred.merge(context,on=["date","code"],how="left",validate="many_to_one")
    prices=pd.read_csv(PREVIOUS/"causal_context_v2/complete_price_relation_NO_LABELS.csv",dtype={"code":str},low_memory=False)
    pred=pred.merge(prices[["date","code","price_relation_history_fraction","price_relation_closing_pair_fraction"]],on=["date","code"],how="left",validate="many_to_one")
    tier=baseline.set_index(["date","code"]).price_identity_source_tier.to_dict()
    pred["source_tier_review"]=[tier.get((r.date,r.code)) or "UNAVAILABLE_OR_SEPARATE_CURRENT_ADAPTER" for r in pred.itertuples()]
    pred["source_tier_review"]=pred.source_tier_review.fillna("UNAVAILABLE_OR_SEPARATE_CURRENT_ADAPTER")
    pred["frozen_evaluation_channel"]=np.select([pred.daily_one_word.eq(1),pred.daily_one_word.eq(0)],["A_ONE_PRICE_SHAPE","B_NONONE_PRICE_SHAPE"],default="UNRESOLVED_IDENTITY")
    pred["B_direction_review"]=[("PROMOTE" if row.score>.5 else "FAIL") if row.daily_one_word==0 else None for row in pred.itertuples()]
    pred["price_history_coverage_band"]=np.select([pred.price_relation_history_fraction.lt(.6),pred.price_relation_history_fraction.eq(1)],["UNDER12_COMMON_RETURN_INPUT","FULL20_RETURN_INPUT"],default="AT_LEAST12_PARTIAL20_OR_MISSING")
    # The original rows/columns are untouched; additional metadata is explicitly canonical.
    original_frame=pd.read_csv(source,dtype={"code":str},low_memory=False)
    pd.testing.assert_frame_equal(pred[original_columns],original_frame)
    assert len(pred)==15084
    output=ROOT/"evaluation";output.mkdir(exist_ok=False)
    pred.to_csv(output/"all15084_frozen_predictions_with_explicit_channels.csv",index=False,encoding="utf-8-sig")
    stratified=[];rankings=[];selected_rows=[];funnel=[];overall_checks=0
    logs=pd.read_csv(PREVIOUS/"fixed_comparison/all_top1_execution_scenarios.csv",dtype={"code":str},low_memory=False)
    report=b.read(PREVIOUS/"fixed_comparison/model_validation_results.json")
    for period in PERIODS:
        group=pred[pred.date.between(period["start"],period["end"])]
        for arm,frame in group.groupby("arm",sort=True):
            reference=report["comparisons"][period["id"]]["metrics"][arm]
            assert_values(reference["full"],classification(frame));assert_values(reference["nonword"],classification(frame[frame.daily_one_word.eq(0)]));overall_checks+=2
            for dimension in ["frozen_evaluation_channel","boards","canonical_frozen_cycle_regime","source_tier_review","price_history_coverage_band","current_qualification_channel"]:
                for value,part in frame.groupby(dimension,dropna=False,sort=True):
                    stratified.append(dict(period=period["id"],arm=arm,dimension=dimension,stratum=str(value),**descriptive_metrics(part)))
            for cohort in [False,True]:
                for height in [False,True]:
                    for k in [1,2]:
                        selected=selection(frame,k,cohort,height)
                        eligible=frame[frame.daily_one_word.eq(0)];groups=["date","boards"] if cohort else ["date"]
                        sizes=eligible.groupby(groups).size()
                        metric=ranking(frame,k,cohort,height)
                        row=dict(period=period["id"],arm=arm,grouping="DATE_HEIGHT" if cohort else "DATE",ordering="HEIGHT_DESC_CODE_ASC" if height else "SCORE_DESC_CODE_ASC",k=k,
                            groups_fewer_than_k=int(sizes.lt(k).sum()),available_groups=len(sizes),**metric)
                        rankings.append(row)
                        selected=selected.copy();selected["period"]=period["id"];selected["selection_grouping"]=row["grouping"];selected["selection_ordering"]=row["ordering"];selected["k"]=k
                        selected_rows.append(selected)
                        key=("same_height_" if cohort else "")+"top"+str(k)
                        if not height:assert_values(reference[key],metric);overall_checks+=1
            selected=selection(frame,1,False,False)
            log=logs[logs.fold.eq(period["id"])&logs.arm.eq(arm)]
            assert list(zip(selected.date,selected.code))==list(zip(log.date,log.code))
            B=int(frame.daily_one_word.eq(0).sum());A=int(frame.daily_one_word.eq(1).sum());unknown=len(frame)-A-B
            entry=int(log.status.eq("INFERRED_OPEN_PROXY_NOT_ACTUAL_FILL").sum())
            row=dict(period=period["id"],arm=arm,full_candidates=len(frame),A_one_price_shape=A,B_nonone_price_shape=B,unknown_shape=unknown,
                Top1_selected=len(selected),not_selected_nonword=B-len(selected),entry_proxy=entry,
                no_live_confirmation_evidence_status="UNKNOWN_NOT_A_CONFIRMED_TRIGGER",actual_fill_records_added=0,paper_fill_records_added=0,
                actual_account_position="UNKNOWN_NO_BROKER_EVIDENCE_NOT_VERIFIED_FLAT",entry_status_counts=log.status.value_counts().to_dict())
            assert len(frame)==A+B+unknown and B==len(selected)+row["not_selected_nonword"]
            for h in [1,2]:
                net=pd.to_numeric(log["b"+str(h)+"_net"],errors="coerce");valid=net.notna()
                row.update({f"Bplus{h}_net_known":int(valid.sum()),f"Bplus{h}_coverage_of_full_candidates":float(valid.sum()/len(frame)),
                    f"Bplus{h}_coverage_of_Top1":float(valid.sum()/len(selected)) if len(selected) else None,
                    f"Bplus{h}_mean_net_proxy":float(net[valid].mean()) if valid.any() else None,
                    f"Bplus{h}_loss_le_minus7_count":int(net.le(-.07).sum()),
                    f"Bplus{h}_exit_status_counts":log.get(f"status_b{h}",pd.Series(dtype=object)).value_counts(dropna=False).to_dict()})
            funnel.append(row)
    pd.DataFrame(stratified).to_csv(output/"all_fixed_strata_classification_WITH_UNKNOWN.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rankings).to_csv(output/"all_daily_sameheight_score_height_Top1_Top2.csv",index=False,encoding="utf-8-sig")
    pd.concat(selected_rows,ignore_index=True).to_csv(output/"all_selected_rows_including_unknown_labels.csv",index=False,encoding="utf-8-sig")
    b.write_new(output/"complete_execution_funnel_FULL_DENOMINATORS.json",dict(formed_at=b.now(),rows=funnel,actual_portfolio_return=None,actual_drawdown=None,
        independent_actual_trades=None,fee_and_slippage_policy="EntryOPEN*1.001,ExitCLOSE*.999,roundtripfee.003; daily proxy, no queue or timed trigger certification",
        terminal_unknowns_not_zero_returns=True))
    b.verify(before)
    receipt=dict(formed_at=b.now(),status="COMPLETE_FROZEN_SCORE_REPORTING_NO_REFIT",new_fits=0,old_scores_labels_exact=True,
        frozen_predictions=len(pred),stratified_rows=len(stratified),ranking_rows=len(rankings),ranking_selection_rows=sum(len(f) for f in selected_rows),
        existing_summary_checks=overall_checks,full_universe_per_arm=len(baseline),channel_counts_per_arm=pred[pred.arm.eq("BASE")].frozen_evaluation_channel.value_counts().to_dict(),
        unknown_channel_never_B_direction=True,old_missing_metadata_not_a_numeric_model_defect=True,
        strata_status="POSTHOC_DESCRIPTIVE_FIXED_DIMENSIONS_NO_THRESHOLD_SELECTION_OR_NEW_CONFIRMATORY_P_VALUES",
        original_intervals_preserved=str(PREVIOUS/"fixed_comparison/model_validation_results.json"),
        original_interval_limitation="Prior two-arm tail correction does not adjust all sequential historical experiments; exploratory only",
        source_bindings=before|{str(p.resolve()):b.sha(p) for p in [source,ROOT/"context/review_scope_context.joblib",Path(__file__),*list(output.glob("*"))]})
    b.write_new(ROOT/"evaluation_completion.json",receipt)
    print({k:receipt[k] for k in ["frozen_predictions","stratified_rows","ranking_rows","existing_summary_checks","channel_counts_per_arm","new_fits"]})


if __name__=="__main__":main()
