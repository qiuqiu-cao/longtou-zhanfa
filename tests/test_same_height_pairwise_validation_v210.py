import pandas as pd

from same_height_pairwise_validation_v210 import FACTORS, build


def candidate(code, complete=False, value=1.0):
    row = {"signal_date": "2026-01-05", "code": code, "name": code, "boards": 1,
           "net_return_B_plus_2": .02 if code == "A" else -.01,
           "big_loss_target": 0, "five_factor_point_in_time_complete": complete}
    row.update({factor: value for factor in FACTORS})
    return row


def test_pair_rows_are_grouped_and_not_counted_as_independent_validation():
    pairs, groups, report = build(pd.DataFrame([
        candidate("A"), candidate("B"), candidate("C")]))
    assert len(pairs) == 3
    assert len(groups) == 1
    assert report["distinct_signal_dates"] == 1
    assert report["governance"]["pairs_count_as_independent_samples"] is False
    assert report["winner_claimed"] is False


def test_missing_five_factor_blocks_evaluable_group():
    _, _, report = build(pd.DataFrame([candidate("A", True), candidate("B", False)]))
    assert report["five_factor_evaluable_groups"] == 0
    assert report["same_height_five_factor_uplift_validated"] is False
