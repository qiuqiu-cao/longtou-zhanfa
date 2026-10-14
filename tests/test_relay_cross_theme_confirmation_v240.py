import pandas as pd

from relay_cross_theme_confirmation_v240 import evaluate


def _pool():
    return pd.DataFrame([
        {"code": "603230", "name": "内蒙新华", "boards": 4,
         "predeclared_role": "PRIMARY_EXPECTED_WINNER", "winner_hierarchy_rank": 1,
         "tradable_expectation_eligible": True},
        {"code": "002285", "name": "世联行", "boards": 4,
         "predeclared_role": "SAME_HEIGHT_FALLBACK_WINNER", "winner_hierarchy_rank": 2,
         "tradable_expectation_eligible": True},
        {"code": "002453", "name": "华软科技", "boards": 3,
         "predeclared_role": "LOWER_HEIGHT_FALLBACK_WINNER", "winner_hierarchy_rank": 3,
         "tradable_expectation_eligible": True},
        {"code": "000504", "name": "南华生物", "boards": 3,
         "predeclared_role": "FULL_DENOMINATOR_WATCH_ONLY", "winner_hierarchy_rank": float("nan"),
         "tradable_expectation_eligible": False},
    ])


def _tree():
    return _pool().dropna(subset=["winner_hierarchy_rank"])[["code"]]


def _report():
    return {"signal_date": "2026-09-21", "primary_expected_winner": {"code": "603230"}}


def _observations():
    rows = []
    for code in _pool().code:
        rows.append({
            "signal_date": "2026-09-21", "trade_date": "2026-09-22", "code": code,
            "available_at": "2026-09-22T09:35:30+08:00", "source_status": "VERIFIED",
            "auction_gap_pct": 2, "auction_amount_ratio_pct": 3,
            "match_amount_decay_proxy": .1, "above_vwap_0935": True,
            "vwap_slope_positive": True, "failed_seal_episodes": 0,
            "sealed_next_board_by_0935": True, "reseal_time": "09:33",
            "theme_sync_ratio": .8, "theme_coverage_ratio": .8, "theme_peer_count": 3,
            "evidence_lane": "VERIFIED_REAL_THEME",
        })
    return pd.DataFrame(rows)


def test_primary_wins_when_it_meets_frozen_expectation():
    observations = _observations()
    observations.loc[~observations.code.eq("603230"), "above_vwap_0935"] = False
    _, report = evaluate(_pool(), _tree(), _report(), observations, "2026-09-22")
    assert report["decision"] == "PRIMARY_EXPECTATION_CONFIRMED"
    assert report["conditional_winner"]["code"] == "603230"
    assert not report["entry_authorized"]


def test_two_same_height_promotions_are_co_winners_not_fake_unique_leader():
    _, report = evaluate(_pool(), _tree(), _report(), _observations(), "2026-09-22")
    assert report["decision"] == "SAME_HEIGHT_CO_WINNERS_NO_UNIQUE_LEADER"
    assert report["highest_confirmed_identity_codes"] == ["603230", "002285"]
    assert not report["conditional_winner"]["unique_highest_confirmed"]


def test_first_passing_fallback_wins_observation_but_is_not_replacement_trade():
    observations = _observations()
    observations.loc[observations.code.eq("603230"), "above_vwap_0935"] = False
    _, report = evaluate(_pool(), _tree(), _report(), observations, "2026-09-22")
    assert report["decision"] == "CONDITIONAL_FALLBACK_WINNER_OBSERVED_NO_REPLACEMENT_TRADE"
    assert report["conditional_winner"]["code"] == "002285"
    assert not report["conditional_winner"]["trade_replacement_allowed"]


def test_incomplete_denominator_refuses_to_name_winner():
    observations = _observations().iloc[:-1]
    evaluated, report = evaluate(_pool(), _tree(), _report(), observations, "2026-09-22")
    assert evaluated.empty
    assert report["decision"] == "DATA_UNAVAILABLE_INCOMPLETE_3_4_BOARD_DENOMINATOR"
    assert report["missing_codes"] == ["000504"]


def test_excluded_identity_winner_is_model_miss_even_if_no_tradable_choice_confirms():
    observations = _observations()
    observations["sealed_next_board_by_0935"] = False
    observations.loc[observations.code.eq("000504"), "sealed_next_board_by_0935"] = True
    _, report = evaluate(_pool(), _tree(), _report(), observations, "2026-09-22")
    assert report["decision"] == "NO_WINNER_RETREAT_CASH"
    assert report["unexpected_winners"][0]["code"] == "000504"
    assert report["model_miss_flag"]
