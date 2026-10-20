import numpy as np
import pandas as pd
import pytest

import first_board_capital_risk_model as cap
import first_second_stage_model as stage


def sample():
    return pd.DataFrame(dict(signal_date=["2025-01-02"] * 4, code=["000001", "000002", "000003", "000004"],
        float_cap=[2e9, 5e9, 20e9, np.nan], turnover=[1., 10., 2., 5.], one_price=[1., 0., 1., 0.],
        volume_ratio_5=[.5, 2., .2, 1.], model_score=[.9, .7, .8, .6], promotion_next_LABEL_ONLY=[1., 0., 1., np.nan]))


def test_complete_denominator_and_discarded_winners_are_retained():
    rows = cap.capital_context(sample(), 10e9)
    assert len(rows) == 4
    assert rows.research_selection_allowed.tolist() == [False, True, False, False]
    assert rows.threeboard_evidence_status.eq("INSUFFICIENT_EVIDENCE").all()
    m = cap.screen_metric(rows)
    assert m["hits"] == 0 and m["false_negatives"] == 2
    assert m["excluded_known_promoters"] == 2 and m["excluded_unknown"] == 1
    assert m["full_signal_dates"] == 1 and m["abstention_dates"] == 0
    rows["research_selection_allowed"] = False
    m = cap.screen_metric(rows)
    assert m["selected"] == 0 and m["abstention_dates"] == 1
    assert m["full_positive_pool"] == 2


def test_future_outcomes_and_later_cohorts_never_affect_first_close():
    r = sample()
    a = cap.capital_context(r, 10e9)
    r[stage.TARGET] = 1 - r[stage.TARGET]
    later = r.copy(); later["signal_date"] = "2025-01-03"; later["float_cap"] *= 100
    b = cap.capital_context(pd.concat([r, later]), 10e9).iloc[:4]
    pd.testing.assert_frame_equal(a[cap.CAP_FIELDS], b[cap.CAP_FIELDS])
    assert a.research_selection_allowed.tolist() == b.research_selection_allowed.tolist()


def test_postclose_or_single_seat_claims_cannot_bypass_filter(tmp_path):
    raw = tmp_path / "source.txt"; raw.write_text("synthetic original evidence")
    item = dict(signal_date="2025-01-02", code="000003", published_at="2025-01-02T15:30:00+08:00",
        acquired_at="2025-01-02T16:00:00+08:00", raw_path=str(raw), raw_sha256=cap.base.sha(raw),
        source_url="https://www.cninfo.com.cn/test", review_status="AUTHOR_VERIFIED", rationale="test", kind="RESTRUCTURING_DISCLOSURE")
    with pytest.raises(ValueError, match="signal close"):
        cap.capital_context(sample(), 10e9, [item])
    item["published_at"] = "2025-01-02T12:00:00+08:00"
    result = cap.capital_context(sample(), 10e9, [item])
    assert result.loc[2, "research_selection_allowed"]
    with pytest.raises(ValueError, match="signal close"):
        cap.capital_context(sample(), 10e9, [item], strict_acquisition=True)
    item["kind"] = "SUSPECTED_THREEBOARD_STRUCTURE"
    item["independently_verified_channels"] = ["coordinated_seats"]
    with pytest.raises(ValueError, match="Single-seat"):
        cap.capital_context(sample(), 10e9, [item])
    r = sample(); r["verified_restructuring_preclose"] = 1.
    with pytest.raises(ValueError, match="Evidence flags"):
        cap.capital_context(r, 10e9)
