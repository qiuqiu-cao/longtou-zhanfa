import pytest
import relay_pairwise_workbench as w


def test_pending_original_review_blocks_model_or_feature_read(monkeypatch,tmp_path):
    monkeypatch.setattr(w.b,"read",lambda p:{"rows":[],"author_review_completed":False})
    def pending(*args):raise ValueError("author review pending")
    monkeypatch.setattr(w.original,"validate_completed",pending)
    monkeypatch.setattr(w.joblib,"load",lambda p:pytest.fail("must not load model before review"))
    with pytest.raises(ValueError,match="author review pending"):
        w.score_after_review("facts","review",tmp_path/"out.json")
    assert not (tmp_path/"out.json").exists()
