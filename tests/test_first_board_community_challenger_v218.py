import pandas as pd

from first_board_community_challenger_v218 import (
    attach_daily_bar_features,
    attach_current_theme_map,
    daily_top1,
    derive_features,
    score_current,
)


def _rows():
    common = {
        "signal_date": "2026-09-18", "industry_proxy": "电力",
        "last_minus_first_minute": 5, "failed_seal_count": 1,
        "turnover": 10, "amount_to_float_cap": .08,
        "seal_fund_to_float_cap": .01, "one_price_proxy": 0,
        "thin_control_risk": 0, "industry_first_board_count": 3,
        "industry_all_board_count": 4, "industry_higher_board_count": 1,
        "market_first_board_count": 60, "market_relay_count": 10,
    }
    return pd.DataFrame([
        dict(common, code="600001", name="甲", first_seal_minute=20),
        dict(common, code="600002", name="乙", first_seal_minute=40),
        dict(common, code="600003", name="丙", first_seal_minute=100),
    ])


def test_pioneer_and_close_known_features_are_derived_without_labels():
    rows = derive_features(_rows())
    assert rows.industry_pioneer_rank.tolist() == [1, 2, 3]
    assert rows.supported_pioneer.tolist() == [1, 1, 0]
    assert rows.sealed_by_1030.tolist() == [1, 1, 0]
    assert rows.close_quality_gate.tolist() == [True, True, False]
    assert "promotion_label" not in rows.columns


def test_daily_top1_does_not_replace_failed_or_unbuyable_selection():
    rows = _rows()
    rows["promotion_label"] = [0, 1, 1]
    rows["score"] = [0.9, 0.8, 0.7]
    rows.loc[0, "one_price_proxy"] = 1
    selected = daily_top1(rows, "score")
    assert len(selected) == 1
    assert selected.iloc[0].code == "600001"
    assert selected.iloc[0].promotion_label == 0


def test_current_one_price_is_watch_only_even_if_ranked_first():
    history = derive_features(pd.concat([_rows().assign(
        signal_date=f"2026-08-{day:02d}", promotion_label=day % 2)
        for day in range(1, 15)], ignore_index=True))
    current = _rows()
    current.loc[0, "one_price_proxy"] = 1
    current.loc[0, "thin_control_risk"] = 1
    scored = score_current(history, current)
    one_price = scored[scored.code.eq("600001")].iloc[0]
    assert one_price.execution_status == "WATCH_UNBUYABLE_ONE_PRICE"
    assert not bool(one_price.entry_authorized)


def test_daily_bar_features_use_prior_window_for_volume_and_breakout():
    dates = pd.date_range("2026-07-01", periods=25, freq="B")
    bars = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "code": "600001",
        "high": list(range(10, 34)) + [40],
        "close": list(range(9, 33)) + [40],
        "pre_close": list(range(8, 32)) + [32],
        "volume": [100] * 24 + [200], "amount": [1e8] * 25,
        "float_cap": [40e8] * 25,
    })
    rows = pd.DataFrame({"signal_date": [dates[-1].strftime("%Y-%m-%d")], "code": ["600001"]})
    enriched = attach_daily_bar_features(rows, bars)
    assert enriched.iloc[0].volume_ratio_5 == 2
    assert enriched.iloc[0].platform_breakout_20 == 1
    assert enriched.iloc[0].amount_billion == 1


def test_theme_map_prefers_observed_roster_over_industry_or_symbolic_lane():
    current = _rows().iloc[:1]
    theme_map = pd.DataFrame([
        {"signal_date": "2026-09-18", "code": "600001", "theme_name": "华字辈",
         "evidence_lane": "SYMBOLIC_NAME_CLUSTER", "theme_stage": "FULL_LADDER",
         "rank_within_lane": 1, "theme_pioneer_rank": 1},
        {"signal_date": "2026-09-18", "code": "600001", "theme_name": "医药",
         "evidence_lane": "OBSERVED_THEME_ROSTER", "theme_stage": "FULL_LADDER",
         "rank_within_lane": 1, "theme_pioneer_rank": 2},
    ])
    enriched = attach_current_theme_map(current, theme_map)
    assert enriched.iloc[0].best_theme_name == "医药"
    assert "SYMBOLIC_NAME_CLUSTER:华字辈" in enriched.iloc[0].all_theme_tags
