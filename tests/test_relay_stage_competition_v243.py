import pandas as pd

from relay_stage_competition_v243 import evaluate


def _matrix():
    return pd.DataFrame([
        {"code": "603230", "name": "内蒙新华", "boards": 4,
         "winner_hierarchy_rank": 1, "expectation_type": "DIVERGENCE_ABSORPTION_CONTINUATION",
         "normal_gap_low": 0, "normal_gap_high": 5, "trade_expectation_allowed": True},
        {"code": "002285", "name": "世联行", "boards": 4,
         "winner_hierarchy_rank": 2, "expectation_type": "STRONG_TO_STRONG_CONTINUATION",
         "normal_gap_low": 3, "normal_gap_high": 7, "trade_expectation_allowed": True},
        {"code": "002453", "name": "华软科技", "boards": 3,
         "winner_hierarchy_rank": 3, "expectation_type": "STRONG_TO_STRONG_CONTINUATION",
         "normal_gap_low": 3, "normal_gap_high": 7, "trade_expectation_allowed": True},
        {"code": "000504", "name": "南华生物", "boards": 3,
         "winner_hierarchy_rank": float("nan"), "expectation_type": "FIRST_OPEN_PRICE_DISCOVERY",
         "normal_gap_low": float("nan"), "normal_gap_high": float("nan"),
         "trade_expectation_allowed": False},
    ])


def _report():
    return {"signal_date": "2026-09-21", "primary_close_expectation": {"code": "603230"}}


def _observations():
    rows = []
    for code in _matrix().code:
        rows.append({
            "signal_date": "2026-09-21", "trade_date": "2026-09-22", "code": code,
            "available_at": "2026-09-22T10:00:30+08:00", "source_status": "VERIFIED",
            "auction_gap_pct": 4, "auction_amount_ratio_pct": 3,
            "match_amount_decay_proxy": .1, "reclaimed_previous_close_0935": True,
            "above_vwap_0935": True, "vwap_slope_positive_0935": True,
            "first_limit_touch_time": "09:32", "first_seal_time": "09:34",
            "sealed_at_0935": True, "sealed_at_1000": True,
            "failed_seal_episodes_before_1000": 0, "theme_sync_ratio_1000": .8,
            "theme_coverage_ratio_1000": .8, "theme_peer_count_1000": 3,
            "candidate_led_theme_by_time_order": True,
            "theme_strengthened_after_candidate": True,
        })
    return pd.DataFrame(rows)


def _market(systemic=False):
    return {"trade_date": "2026-09-22", "available_at": "2026-09-22T10:00:30+08:00",
            "source_status": "VERIFIED", "height_anchor_severe_negative": systemic,
            "middle_board_batch_negative": systemic, "previous_core_batch_negative": False}


def test_same_height_confirmations_are_co_winners_not_forced_unique():
    observations = _observations()
    observations.loc[observations.code.isin(["002453", "000504"]), "above_vwap_0935"] = False
    _, report = evaluate(_matrix(), _report(), observations, _market(), "2026-09-22")
    assert report["decision"] == "SAME_HEIGHT_CO_WINNERS_NO_UNIQUE_LEADER"
    assert [x["code"] for x in report["highest_identity_winners"]] == ["603230", "002285"]


def test_three_board_takes_over_only_after_every_four_board_fails():
    observations = _observations()
    observations.loc[observations.code.isin(["603230", "002285", "000504"]),
                     "above_vwap_0935"] = False
    _, report = evaluate(_matrix(), _report(), observations, _market(), "2026-09-22")
    assert report["decision"] == "UNIQUE_3_TO_4_CROSS_HEIGHT_TAKEOVER"
    assert report["next_winner_after_primary_failure"]["code"] == "002453"


def test_systemic_failure_makes_survivor_retreat_survivor_not_core_switch():
    observations = _observations()
    observations.loc[~observations.code.eq("002285"), "above_vwap_0935"] = False
    _, report = evaluate(_matrix(), _report(), observations, _market(True), "2026-09-22")
    assert report["decision"] == "RETREAT_SURVIVOR_NOT_PROVEN_CORE_SWITCH"
    assert report["tradable_winner"] is None


def test_locked_board_low_open_can_recover_only_through_full_open_path():
    observations = _observations()
    observations.loc[observations.code.eq("002285"), "auction_gap_pct"] = 1
    observations.loc[~observations.code.eq("002285"), "above_vwap_0935"] = False
    evaluated, report = evaluate(_matrix(), _report(), observations, _market(), "2026-09-22")
    row = evaluated[evaluated.code.eq("002285")].iloc[0]
    assert row.auction_class == "UNDER_EXPECTATION"
    assert bool(row.recovered_from_under_expectation)
    assert report["decision"] == "UNIQUE_SAME_HEIGHT_RIVAL_WINNER"


def test_incomplete_full_denominator_refuses_winner():
    evaluated, report = evaluate(_matrix(), _report(), _observations().iloc[:-1],
                                 _market(), "2026-09-22")
    assert evaluated.empty
    assert report["decision"] == "DATA_UNAVAILABLE_INCOMPLETE_3_4_BOARD_DENOMINATOR"


def test_unverified_market_context_cannot_name_tradable_core():
    observations = _observations()
    observations.loc[~observations.code.eq("603230"), "above_vwap_0935"] = False
    market = _market(); market["source_status"] = "UNAVAILABLE"
    _, report = evaluate(_matrix(), _report(), observations, market, "2026-09-22")
    assert report["decision"] == "MARKET_CONTEXT_UNVERIFIED_IDENTITY_ONLY"
    assert report["tradable_winner"] is None
