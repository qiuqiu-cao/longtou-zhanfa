import copy
import numpy as np
import pandas as pd
import pytest

import expectation_345_features as features
import expectation_345_evaluation as evaluation
import expectation_345_research as research
import short_term_thinking_research as previous


def feature_fixture():
    dates=["2024-01-02","2024-01-03","2024-01-04"]
    row=dict(date=dates[1],code="000001",review_cycle_state="UNKNOWN_INCOMPLETE_CONTEXT",review_roles=["UNKNOWN"],
        current_price_shape=np.nan,cycle_context_complete=False,current_price_path_status="UNKNOWN_INCOMPLETE_CONTINUOUS_PATH",
        active_anchor_discovery_scope="PARTIAL_DATES_UNKNOWN_ADDITIONAL_ANCHORS",complete_active_parent_count=0,
        full_cohort_status="UNKNOWN_PREVIOUS_POOL_SCOPE",target=1)
    row.update({k:np.nan for k in features.COHORT_FIELDS})
    relay=pd.DataFrame([dict(date=dates[1],code="000001",target=0,next_open_gap=.10,base_score=.99,**{k:1. for k in features.STATIC})])
    cycles=pd.DataFrame([dict(date=d,**{k:float(i+1) for k in features.CYCLE_FIELDS}) for i,d in enumerate(dates)])
    parents=pd.DataFrame(columns=["date","code","anchor_current_state","observed_launch_lag","shared_characters","handover_evidence_dates_in_child_streak"])
    return dict(rows=pd.DataFrame([row]),cycles=cycles,parents=parents),relay,dates


def test_explicit_projection_ignores_target_scores_and_future_cycles():
    artifact,relay,dates=feature_fixture()
    a,fields=features.project(artifact,relay,dates)
    changed=copy.deepcopy(artifact);changed["rows"]["target"]=0
    changed["cycles"].loc[changed["cycles"].date.eq(dates[2]),"height"]=999
    other=relay.copy();other["target"]=1;other["base_score"]=.01;other["next_open_gap"]=-.10
    z,second=features.project(changed,other,dates)
    pd.testing.assert_frame_equal(a,z);assert fields==second
    assert not {"target","base_score","next_open_gap"}&set(fields)


def test_partial_anchor_scope_and_missing_cohort_remain_unknown():
    artifact,relay,dates=feature_fixture();out,_=features.project(artifact,relay,dates)
    assert np.isnan(out.causal_parent_observed_fraction.iloc[0])
    assert np.isnan(out.causal_full_cohort_promotion_rate.iloc[0])
    assert out.causal_parent_observed_lag_count.iloc[0]==0


def test_reference_cost_one_exactly_reproduces_original_weight_formula():
    train=pd.DataFrame(dict(date=["a","a","b","b","b"],target=[0,1,0,0,1]))
    np.testing.assert_allclose(research.weights(train,1.),previous.training_weights(train),atol=1e-12)


def test_negative_cost_is_fixed_class_mass_and_total_mass_unchanged():
    train=pd.DataFrame(dict(date=["a","a","b","b","b"],target=[0,1,0,0,1]))
    weight=research.weights(train,1.5);mass=weight.groupby(train.target).sum()
    assert np.isclose(mass[0]/mass[1],1.5) and np.isclose(weight.sum(),2.)
    with pytest.raises(ValueError):research.weights(train[train.target.eq(0)],1.5)


def metric_frame():
    return pd.DataFrame(dict(date=["a","a","b","c"],code=["000002","000001","000003","000004"],boards=[2,2,3,2],
        daily_one_word=[0.,0.,0.,0.],target=[1.,0.,1.,np.nan],label_valid=[True,True,True,False],
        call=["PROMOTE","UNKNOWN","UNKNOWN","UNKNOWN"],ranking_score=[.9,.9,.8,.7]))


def test_abstention_never_creates_fake_recall_or_accuracy():
    out=evaluation.classification(metric_frame())
    assert out["Recall"]==.5 and out["Precision"]==1 and out["F1"]==pytest.approx(2/3)
    assert out["abstained_positives"]==1 and out["missed_positives"]==1
    assert out["full_known_accuracy"]==pytest.approx(1/3) and out["called_accuracy"]==1
    assert out["unknown_outcomes"]==1


def test_top_selection_preserves_ties_unknown_outcomes_and_unknown_routes():
    frame=metric_frame();selected=evaluation.selection(frame)
    assert selected.code.tolist()==["000001","000003","000004"]
    out=evaluation.ranking(frame)
    assert out["selected"]==3 and out["known"]==2 and out["unknown"]==1
    assert out["unknown_or_ineligible_selected"]==3


def test_same_height_top2_preserves_groups_with_fewer_than_two():
    result=evaluation.ranking(metric_frame(),2,True)
    assert result["groups_fewer_than_k"]==2 and result["selected"]==4


def test_block_intervals_compare_identical_calls_to_zero():
    a=metric_frame();out=evaluation.block_intervals(a,a,draws=100)
    for v in out.values():assert v["lower"]==0 and v["upper"]==0 and v["family_tests"]==24


def test_three_way_disagreement_remains_unknown_not_failure():
    reviews=[pd.DataFrame([dict(date="a",code="000001",call=c,repeat_id=i+1)]) for i,c in enumerate(["PROMOTE","FAIL","UNKNOWN"])]
    pairs,consensus=evaluation.agreement(reviews)
    assert consensus.majority_call.iloc[0]=="UNKNOWN" and not consensus.unanimous.iloc[0]
    assert len(pairs)==3 and all(x["agreement"]==0 for x in pairs)


def test_complete_agreement_does_not_claim_predictive_accuracy():
    reviews=[pd.DataFrame([dict(date="a",code="000001",call="PROMOTE",repeat_id=i+1)]) for i in range(3)]
    pairs,consensus=evaluation.agreement(reviews)
    assert consensus.unanimous.iloc[0] and consensus.majority_call.iloc[0]=="PROMOTE"
    assert "accuracy" not in consensus and all(x["kappa"] is None for x in pairs)


def test_archived_evaluation_label_binding_preserves_original_and_future_unknown():
    full=pd.DataFrame(dict(date=["2026-10-08","2026-10-15"],code=["000001","000001"],target=[np.nan,np.nan],label_valid=[False,False]))
    saved=pd.DataFrame([dict(date="2026-10-08",code="000001",target=1.,label_valid=True)])
    out=research.bind_evaluation_labels(full,saved)
    assert out.target.iloc[0]==1 and np.isnan(out.target.iloc[1])
    assert out.engine_target_original.isna().all() and full.target.isna().all()


def test_known_label_disagreement_rejected_instead_of_overwritten():
    full=pd.DataFrame([dict(date="2026-10-08",code="000001",target=0.,label_valid=True)])
    saved=full.copy();saved["target"]=1.
    with pytest.raises(ValueError,match="known target disagreement"):research.bind_evaluation_labels(full,saved)
