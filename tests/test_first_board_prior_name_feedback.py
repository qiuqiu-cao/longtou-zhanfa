import numpy as np
import pandas as pd
import pytest
import first_board_prior_name_feedback as f


def example():
    sessions = ["2024-01-02", "2024-01-03", "2024-01-04"]
    candidates = pd.DataFrame([dict(signal_date=sessions[1], code="000005", historical_name="百洋股份", name_ok=True)])
    prior = pd.DataFrame([dict(signal_date=sessions[0], code=code, p=0, height=height,
            historical_name=name, name_ok=True) for code, name, height in
            [("000001", "九阳股份", 4), ("000002", "海洋王", 3), ("000003", "圣阳股份", 1),
             ("000004", "澳洋健康", 2), ("000005", "百洋股份", 1)]])
    states = []
    for p in (0, 1):
        for code, ret, limit in [("000001", .1, True), ("000002", -.08, False),
                                ("000003", -.08, False), ("000004", .03, False), ("000005", .1, True)]:
            states.append(dict(p=p, code=code, eligible=True, valid_price=True, regular_price=True,
                               limit=limit, day_return=ret))
    return candidates, prior, pd.DataFrame(states), sessions


def test_prior_cohort_includes_today_nonlimit_losses_and_deduplicates_relations():
    row = f.build(*example()).iloc[0]
    assert row.prior_name_anchor_present == 1
    assert row.anchor_members_DIAGNOSTIC == 2  # exact-character + phonetic overlap counted once
    assert row.follower_members_DIAGNOSTIC == 2  # candidate itself is left out
    assert row.name_anchor_current_mean_return == pytest.approx(.01)
    assert row.name_anchor_current_limit_survival == .5
    assert row.name_prev_followers_current_mean_return == pytest.approx(-.025)
    assert row.name_prev_followers_current_loss7_share == .5


def test_missing_current_member_is_unknown_not_a_failed_or_removed_stock():
    c, n, s, days = example()
    s = s[~(s.p.eq(1) & s.code.eq("000003"))]
    row = f.build(c, n, s, days).iloc[0]
    assert row.follower_members_DIAGNOSTIC == 2 and row.follower_unknown_DIAGNOSTIC == 1
    assert pd.isna(row.name_prev_followers_current_mean_return)
    assert pd.isna(row.name_prev_followers_current_loss7_share)
    assert row.name_anchor_current_limit_survival == .5


def test_unknown_previous_name_does_not_become_no_anchor():
    c, n, s, days = example()
    n.loc[0, "name_ok"] = False
    row = f.build(c, n, s, days).iloc[0]
    assert row[f.FIELDS].isna().all()
    assert row.previous_pool_name_unknown_DIAGNOSTIC == 1


def test_only_previous_official_day_can_select_anchors_and_future_changes_cannot_affect():
    c, n, s, days = example()
    n["height"] = 2
    baseline = f.build(c, n, s, days)
    assert baseline.prior_name_anchor_present.iloc[0] == 0
    assert baseline[f.FIELDS[1:]].isna().all().all()
    future = n.copy()
    future["p"] = 2
    future["signal_date"] = days[2]
    future["height"] = 100
    n = pd.concat([n, future], ignore_index=True)
    future_s = s[s.p.eq(1)].copy()
    future_s["p"] = 2
    future_s["day_return"] = 999
    pd.testing.assert_frame_equal(baseline, f.build(c, n, pd.concat([s, future_s], ignore_index=True), days))


def test_targets_are_rejected_and_ineligible_current_prices_are_not_losses():
    c, n, s, days = example()
    c["promotion_next_LABEL_ONLY"] = 1
    with pytest.raises(ValueError, match="Future target"):
        f.build(c, n, s, days)
    c = c.drop(columns="promotion_next_LABEL_ONLY")
    s.loc[s.p.eq(1) & s.code.eq("000002"), "eligible"] = False
    row = f.build(c, n, s, days).iloc[0]
    assert row.anchor_unknown_DIAGNOSTIC == 1
    assert pd.isna(row.name_anchor_current_mean_return)
