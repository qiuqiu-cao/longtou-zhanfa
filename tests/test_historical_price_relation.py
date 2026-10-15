import numpy as np
import pandas as pd
import pytest

import historical_price_relation as p


def fixture():
    dates = pd.bdate_range("2024-01-01", periods=42).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(45)
    values = rng.normal(0, .02, (42, 5))
    values[:, 1] = values[:, 0] * .7 + rng.normal(0, .006, 42)
    bars = pd.DataFrame([dict(date=d, code=f"{j+1:06}", close=10*(1+values[i,j]),
        pre_close=10*(1+values[i-1,j]) if i else 10)
        for i,d in enumerate(dates) for j in range(5)])
    candidates = pd.DataFrame([dict(date=d, code="000001", name="甲", boards=3,
        parent_candidates=["000002", "000005"], parent_anchor_code="000002") for d in dates[31:34]])
    attempts = pd.DataFrame([dict(signal_date=d, code=f"{j+1:06}", board_height=[3,4,1,2,1][j],
        pool_assignment="FAILED_BOARD_OBSERVATION" if j==4 else "CLOSING_LIMIT_UP")
        for d in dates[31:34] for j in range(5)])
    return dates,bars,candidates,attempts


def test_pairwise_complete_matches_numpy_and_constants_unknown():
    x = np.array([[1.,2.,5.],[2.,np.nan,5.],[3.,1.,5.],[4.,0.,5.]])
    corr,n = p.pairwise_complete_correlation(x,3)
    assert n[0,1] == 3
    assert corr[0,1] == pytest.approx(np.corrcoef([1,3,4],[2,1,0])[0,1])
    assert np.isnan(corr[0,2])


def test_future_and_recent_excluded_prices_and_labels_do_not_change_result():
    dates,bars,candidates,attempts = fixture()
    one = candidates.iloc[:1].copy()
    expected,pairs = p.build_causal(one, attempts, bars, dates, dates)
    altered = bars.copy()
    altered.loc[altered.date.ge(dates[26]), "close"] *= 50
    candidates["target"] = [1,0,1]
    actual,actual_pairs = p.build_causal(candidates, attempts, altered, dates, dates)
    pd.testing.assert_frame_equal(expected,actual.iloc[:1].reset_index(drop=True))
    pd.testing.assert_frame_equal(pairs,actual_pairs[actual_pairs.date.eq(dates[31])].reset_index(drop=True))
    assert expected.estimate_end.iloc[0] == dates[25]


def test_batch_and_single_date_have_identical_features_and_pairs():
    dates,bars,candidates,attempts = fixture()
    all_rows,all_pairs = p.build_causal(candidates,attempts,bars,dates,dates)
    for day in candidates.date:
        rows,pairs = p.build_causal(candidates[candidates.date.eq(day)],attempts,bars,dates,dates)
        pd.testing.assert_frame_equal(rows,all_rows[all_rows.date.eq(day)].reset_index(drop=True))
        pd.testing.assert_frame_equal(pairs,all_pairs[all_pairs.date.eq(day)].reset_index(drop=True))


def test_missing_source_scope_not_zero_and_full_pairs_retained():
    dates,bars,candidates,attempts = fixture()
    rows,pairs = p.build_causal(candidates,attempts,bars,dates,[])
    assert rows[p.FIELDS[1:8]].isna().all().all()
    assert rows.price_relation_parent_max_corr.notna().all()
    assert len(pairs)==12
    assert pairs.groupby("date").peer_code.nunique().eq(4).all()


def test_missing_parent_prices_not_negative_or_fake_zero():
    dates,bars,candidates,attempts = fixture()
    bars = bars[~bars.code.isin(["000002","000005"])]
    rows,pairs = p.build_causal(candidates,attempts,bars,dates,dates)
    assert rows.price_relation_parent_max_corr.isna().all()
    assert rows.price_relation_selected_parent_corr.isna().all()
    assert rows.price_relation_parent_pair_fraction.eq(0).all()
    assert pairs[pairs.is_active_parent].correlation.isna().all()


def test_duplicate_and_unqualified_attempt_rejected():
    dates,bars,candidates,attempts = fixture()
    with pytest.raises(ValueError,match="duplicate"):
        p.build_causal(candidates,attempts,pd.concat([bars,bars.iloc[:1]]),dates,dates)
    attempts.loc[0,"pool_assignment"] = "UNKNOWN"
    with pytest.raises(ValueError,match="unqualified"):
        p.build_causal(candidates,attempts,bars,dates,dates)


def test_optional_previous_reference_does_not_erase_observed_returns():
    dates,bars,candidates,attempts = fixture()
    original,_ = p.build_causal(candidates,attempts,bars,dates,dates)
    bars["pre_close"] = np.nan
    absent,_ = p.build_causal(candidates,attempts,bars,dates,dates)
    pd.testing.assert_frame_equal(original,absent)
    # A genuinely absent adjacent session is never bridged to a prior trade.
    bars = bars[~(bars.date.eq(dates[20]) & bars.code.eq("000001"))]
    missing,_ = p.build_causal(candidates,attempts,bars,dates,dates)
    assert missing.price_relation_history_fraction.iloc[0] == .9


def test_original_candidate_failed_conflict_not_double_counted():
    dates,bars,candidates,attempts = fixture()
    other = candidates.copy()
    other["code"] = "000005"
    c = pd.concat([candidates,other],ignore_index=True)
    rows,pairs = p.build_causal(c,attempts,bars,dates,dates)
    assert rows[rows.code.eq("000005")].candidate_failed_source_conflict.all()
    for row in rows[rows.code.eq("000001")].itertuples():
        assert row.closing_peers == 3
        assert row.failed_peers == 1
    assert pairs[pairs.peer_code.eq("000005")].peer_pool.eq("FAILED").all()
