import pandas as pd
import pytest

from lane_attention_competition_audit_v190 import _prepare, build


def _fixtures(session_count=220):
    dates = pd.bdate_range("2024-01-02", periods=session_count).strftime("%Y-%m-%d")
    candidates, outcomes = [], []
    for day_index, date in enumerate(dates):
        first_count = 2 if day_index % 2 else 6
        relay_count = 4 if day_index % 3 else 1
        for lane, count, board in (("FIRST_BOARD", first_count, 1),
                                   ("CONSECUTIVE_BOARD", relay_count, 2)):
            for index in range(count):
                code = f"{day_index * 20 + index + (0 if lane == 'FIRST_BOARD' else 10):06d}"
                candidates.append({"code": code, "signal_date": date,
                                   "boards": board, "board_group": lane})
                promoted = index == 0
                outcomes.append({
                    "code": code, "signal_date": date,
                    "promotion_LABEL_ONLY": promoted,
                    "next_session_touched_limit_LABEL_ONLY": promoted,
                    "next_session_one_price_limit_LABEL_ONLY": False,
                    "b1_big_loss_mark_LABEL_ONLY": False if promoted else pd.NA,
                    "b2_net_close_mark_LABEL_ONLY": .02 if promoted else pd.NA,
                    "b3_net_close_mark_LABEL_ONLY": .01 if promoted else pd.NA,
                    "actual_entry_verified": False,
                    "complete_trade_profit_LABEL_ONLY": pd.NA,
                })
    return pd.DataFrame(candidates), pd.DataFrame(outcomes)


def test_prepare_uses_same_day_lane_supply_and_keeps_no_touch_negative():
    candidates, outcomes = _fixtures(2)
    frame = _prepare(candidates, outcomes,
                     {"target_contract": {"no_touch_rows_retained_as_negative": True}})
    assert frame.groupby("signal_date")["first_supply"].first().tolist() == [6, 2]
    assert frame.groupby("signal_date")["relay_supply"].first().tolist() == [1, 4]
    no_touch = frame[~frame["touchable"]]
    assert not no_touch.empty
    assert not no_touch["b2_positive"].any()


def test_build_is_diagnostic_and_has_separate_lane_comparisons():
    candidates, outcomes = _fixtures()
    report, observations = build(
        candidates, outcomes,
        {"target_contract": {"no_touch_rows_retained_as_negative": True}},
        bootstrap_draws=100,
    )
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["scope"]["both_lane_sessions"] == 220
    assert set(report["date_clustered_high_minus_low"]["development"]) == {
        "FIRST_BOARD_UNDER_RELAY_PRESSURE", "RELAY_UNDER_FIRST_BOARD_SUPPLY"}
    assert report["governance"]["common_first_relay_winner_score_allowed"] is False
    assert report["governance"]["entry_authorized"] is False
    assert len(observations) == len(candidates)


def test_build_rejects_old_outcome_contract():
    candidates, outcomes = _fixtures()
    with pytest.raises(ValueError, match="no-touch-negative"):
        build(candidates, outcomes, {}, bootstrap_draws=100)
