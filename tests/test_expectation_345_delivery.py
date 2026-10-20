import copy
import json
import pandas as pd
import pytest
from pathlib import Path

import expectation_345_llm_evaluation as llm
import expectation_345_workbench as workbench
import expectation_345_verify as verifier
import expectation_345_review_packets as packets
from test_expectation_345 import feature_fixture


def specimen():
    path=llm.ROOT
    packet=json.loads((path/"LLM_repeat_1_closed_packet.json").read_text(encoding="utf-8"))
    review=json.loads((path/"LLM_repeat_1_actual_review.json").read_text(encoding="utf-8"))
    return packet,review


def test_actual_full_manual_review_accepts_honest_exposure():
    packet,review=specimen()
    assert len(llm.validate_review(packet,review))==81
    assert review["outcomes_seen"] and review["model_scores_seen"]


@pytest.mark.parametrize("corruption",["future_fact","missing_row","duplicate_rank","empty_cancel","other_hash"])
def test_closed_packet_validator_rejects_real_integrity_failures(corruption):
    packet,review=specimen()
    if corruption=="future_fact":review["rows"][0]["support_fact_ids"]=["market:2026-10-16"]
    elif corruption=="missing_row":review["rows"].pop()
    elif corruption=="duplicate_rank":review["rows"][0]["rank"]=review["rows"][1]["rank"]
    elif corruption=="empty_cancel":review["rows"][0]["cancel"]=""
    else:review["canonical_facts_sha256"]="wrong"
    with pytest.raises(ValueError):llm.validate_review(packet,review)


def test_qualified_projector_mirrors_source_enums_and_preserves_missing():
    artifact,relay,dates=feature_fixture()
    artifact["rows"]["full_cohort_status"]="PARTIAL_MEMBERS_UNKNOWN"
    frame,fields=workbench.qualified_projection(artifact,relay,dates)
    assert frame.causal_cohort_PARTIAL_MEMBERS_UNKNOWN.iloc[0]==1
    assert "causal_cohort_PARTIAL_MEMBER_OUTCOMES" not in fields
    assert pd.isna(frame.causal_full_cohort_promotion_rate.iloc[0])


def test_research_score_view_rejects_missing_author_before_loading_models(tmp_path):
    facts=tmp_path/"facts.json";review=tmp_path/"review.json"
    facts.write_text('{"rows":[]}',encoding="utf-8");review.write_text('{"rows":[],"author_review_completed":false}',encoding="utf-8")
    with pytest.raises((ValueError,KeyError)):
        workbench.score(facts,"missing_context",review,"missing_artifact",tmp_path/"out.json")
    assert not (tmp_path/"out.json").exists()


def test_independent_metric_counts_keep_abstained_positive_in_recall():
    data=pd.DataFrame(dict(target=[1.,1.,0.],label_valid=[True]*3,call=["PROMOTE","UNKNOWN","FAIL"]))
    value=verifier.scalar_metrics(data)
    assert value["Recall"]==.5 and value["abstained_positives"]==1
    assert value["full_known_accuracy"]==pytest.approx(2/3)


def test_single_date_packets_remove_later_signal_leakage_and_keep_same_facts():
    source,_=specimen()
    a=packets.single_date_packet(source,"2026-10-13",1)
    z=packets.single_date_packet(source,"2026-10-13",3)
    assert len(a["days"])==1 and a["days"][0]["signal_date"]=="2026-10-13"
    assert a["canonical_facts_sha256"]==z["canonical_facts_sha256"]
    assert len(a["days"][0]["candidates"])==10


def test_single_date_packets_reject_direct_target_and_future_price():
    source,_=specimen()
    source["days"][-1]["candidates"][0]["target"]=1
    with pytest.raises(ValueError,match="score or outcome"):
        packets.single_date_packet(source,"2026-10-15",1)
    del source["days"][-1]["candidates"][0]["target"]
    source["days"][-1]["candidates"][0]["daily_bars"][0]["date"]="2026-10-16"
    with pytest.raises(ValueError,match="future daily price"):
        packets.single_date_packet(source,"2026-10-15",1)


def test_live_historical_unknown_representations_compare_without_filling_zero():
    assert workbench.comparable({"a":pd.NA,"b":[pd.NaT,float("nan")]})=={"a":None,"b":[None,None]}
    assert workbench.comparable(pd.NA)!=0
