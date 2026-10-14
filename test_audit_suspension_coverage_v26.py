import pandas as pd
from audit_suspension_coverage_v26 import assess_gap


def fixture():
    bars = pd.DataFrame([
        dict(code="603580", date="2026-07-06", close=10., pre_close=10., status_verified=True, st=False, new_listing=False),
        dict(code="603580", date="2026-07-09", close=11., pre_close=10., status_verified=True, st=False, new_listing=False)])
    calendar = ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
    evidence = dict(code="603580", suspension_start="2026-07-07", resumption_date="2026-07-09", publication_date="2026-07-08", source="fixture", evidence_status="verified")
    return bars, calendar, evidence


def test_available_and_no_relabel():
    result = assess_gap(*fixture(), "2026-07-09T09:25")
    assert result["status"] == "GAP_EXPLAINED_NOT_RELABELLED"
    assert result["labels_changed"] is False


def test_same_day_evidence_not_assumed_premarket():
    assert assess_gap(*fixture(), "2026-07-08T09:25")["reason"] == "EVIDENCE_NOT_YET_AVAILABLE"


def test_partial_gap_not_bridged():
    bars, calendar, evidence = fixture()
    evidence["suspension_start"] = "2026-07-08"
    assert assess_gap(bars, calendar, evidence, "2026-07-10")["reason"] == "GAP_NOT_FULLY_COVERED"


def test_prior_limit_not_reset():
    bars, calendar, evidence = fixture()
    bars.loc[0, "pre_close"] = 9.09
    assert assess_gap(bars, calendar, evidence, "2026-07-10")["reason"] == "PRE_SUSPENSION_LIMIT_REQUIRES_COUNT_POLICY"


def test_mirror_retained():
    bars, calendar, evidence = fixture()
    evidence["evidence_status"] = "mirror_observed"
    assert assess_gap(bars, calendar, evidence, "2026-07-10")["source_status"] == "mirror_observed"
