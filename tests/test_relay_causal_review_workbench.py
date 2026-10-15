from pathlib import Path
import pytest
import relay_causal_review_workbench as m
import relay_review_workbench as original


def setup(monkeypatch):
    facts={"signal_date":"2026-10-15"}
    review={"author_review_completed":True,"facts_sha256":"exact"}
    context={"signal_date":"2026-10-15","facts_sha256":"exact","rows":[{"code":"000001","channel_review":"UNRESOLVED_IDENTITY","b_direction_allowed":False}],"source_bindings":{}}
    scores={"signal_date":"2026-10-15","rows":[{"code":"000001","name":"甲","channel_b_score":.9,"direction":"PROMOTE"}],"source_bindings":{}}
    inputs={"facts":facts,"review":review,"scores":scores}
    monkeypatch.setattr(m.b,"read",lambda path:inputs[str(path)])
    monkeypatch.setattr(m.b,"sha",lambda path:"exact")
    monkeypatch.setattr(m,"checked",lambda path:context)
    monkeypatch.setattr(original,"validate_completed",lambda *args:None)
    return inputs,context


def test_unknown_shape_preserves_score_but_has_no_B_direction(monkeypatch,tmp_path):
    setup(monkeypatch)
    result=m.saved_score_view("facts","context","review","scores",tmp_path/"result.json")
    assert result["rows"][0]["frozen_research_score"]==.9
    assert result["rows"][0]["legacy_direction"]=="PROMOTE"
    assert result["rows"][0]["B_direction"] is None
    assert result["new_inferences"]==result["new_fits"]==0


def test_duplicate_score_rows_rejected(monkeypatch,tmp_path):
    inputs,_=setup(monkeypatch);inputs["scores"]["rows"]*=2
    with pytest.raises(ValueError,match="same complete dated scores"):
        m.saved_score_view("facts","context","review","scores",tmp_path/"must_not_exist.json")
    assert not (tmp_path/"must_not_exist.json").exists()


def test_pending_review_rejected_before_reading_score_file(monkeypatch,tmp_path):
    setup(monkeypatch)
    def reject(*args):raise ValueError("author still pending")
    monkeypatch.setattr(original,"validate_completed",reject)
    monkeypatch.setattr(m,"checked",lambda *args:pytest.fail("must not prepare score context"))
    with pytest.raises(ValueError,match="author still pending"):
        m.saved_score_view("facts","context","review","scores",tmp_path/"must_not_exist.json")
