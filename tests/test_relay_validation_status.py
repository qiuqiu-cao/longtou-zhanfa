import copy
import json

import pytest

import relay_validation_status as status


def result(registration="original", truth=True, call=False):
    return {"registration_hash": registration, "rows": [{"layer": "promotion", "key": "000001",
        "call": call, "truth": truth, "forecast_made": call is not None, "outcome_known": truth is not None},
        {"layer": "leader", "key": "000001", "call": True, "truth": True}]}


def test_revisions_use_latest_full_pool_once_without_erasing_failure():
    old = result(); new = result(truth=False)
    events = [{"revision": 1, "result": old}, {"revision": 2, "result": new}]
    latest = status.latest_events(events)
    assert len(latest) == 1
    assert events[0]["result"]["rows"][0]["truth"] is True
    m = status.promotion_metrics([event["result"] for event in latest.values()])
    assert (m["registered"], m["TN"], m["known_labels"]) == (1, 1, 1)


def test_unknown_labels_and_abstentions_have_distinct_full_denominators():
    a = result("unknown", truth=None, call=True)
    b = result("abstain", truth=True, call=None)
    m = status.promotion_metrics([a, b])
    assert (m["registered"], m["known_labels"], m["unknown_labels"], m["positive_abstentions"]) == (2, 1, 1, 1)
    assert m["complete_positive_Recall"] == 0
    assert m["Accuracy"] is None
    assert m["Brier"] is None


def test_duplicate_candidate_and_out_of_order_revision_rejected():
    with pytest.raises(ValueError, match="duplicate candidate"):
        status.promotion_metrics([result(), result()])
    with pytest.raises(ValueError, match="revision order"):
        status.latest_events([{"revision": 2, "result": result()}])


def test_invalid_chain_does_not_silently_emit_scored_aggregate(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"; ledger.write_text("changed")
    monkeypatch.setattr(status.runner, "read_chain", lambda p: (_ for _ in ()).throw(ValueError("changed source")))
    r = status.formal_status(tmp_path / "registry", ledger)
    assert r["aggregate_withheld"] is True
    assert r["status"] == "INVALID_ORIGINAL_LEDGER"
    assert "reverified_promotion_metrics" not in r


def pending_report(tmp_path):
    source = tmp_path / "source.json"; source.write_text("{}")
    row = {"code": "000001", "independent_call": None, "BASE_call": None, "actual_promoted": None,
        "independent_error_flags": None, "BASE_error_flags": None, "outcome_pending": True,
        "error_evaluation_status": "NOT_EVALUATED", "opinions": []}
    document = {"status": "PENDING_FULL_POOL_ARGUMENT_FEEDBACK_NO_FORECAST", "outcome_pending": True,
        "forecast_registered": False, "original_comparison_metrics": None, "signal_date": "2026-10-14",
        "target_date": "2026-10-15", "pool_size": 1, "opinion_count": 0, "rows": [row],
        "source_bindings": {str(source): status.judgement.sha(source)}}
    return document


def test_pending_forum_does_not_add_an_abstention_or_mature_sample(tmp_path):
    document = pending_report(tmp_path)
    path = tmp_path / "pending.json"; path.write_text(json.dumps(document))
    r = status.pending_forum_status(path)
    assert r["new_forecast_abstentions"] == r["new_formal_mature_samples"] == 0
    assert r["included_in_formal_or_independent_metrics"] is False
    document["rows"][0]["actual_promoted"] = False
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="contains a forecast"):
        status.pending_forum_status(path)
