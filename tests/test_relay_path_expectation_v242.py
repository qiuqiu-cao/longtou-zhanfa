import pandas as pd

from relay_path_expectation_v242 import build


def _pool():
    common = {"signal_date": "2026-09-21", "theme_group_id": "x", "theme_name": "题材",
              "theme_evidence_status": "INFERRED", "theme_known": True,
              "theme_support_ex_self": 2, "theme_ladder_observed": True,
              "amount": 100, "announcement_risk_tier": "NO_ACTIVE",
              "announcement_hard_veto": False, "symbolic_tags": "",
              "research_score_NOT_PROBABILITY": 1, "closing_path_tradeable": True,
              "turnover_not_blowoff": True, "in_full_identity_denominator": True,
              "security_eligibility_status": "VERIFIED", "entry_authorized": False}
    return pd.DataFrame([
        {**common, "code": "603230", "name": "内蒙新华", "boards": 4, "turnover": 10,
         "board_path_status": "QUICK_REPAIR", "tradable_expectation_eligible": True,
         "predeclared_role": "PRIMARY_EXPECTED_WINNER", "winner_hierarchy_rank": 1},
        {**common, "code": "002285", "name": "世联行", "boards": 4, "turnover": 14,
         "board_path_status": "LOCKED_NO_REOPEN", "tradable_expectation_eligible": True,
         "predeclared_role": "SAME_HEIGHT_FALLBACK_WINNER", "winner_hierarchy_rank": 2},
        {**common, "code": "600127", "name": "金健米业", "boards": 3, "turnover": 28,
         "board_path_status": "QUICK_REPAIR", "tradable_expectation_eligible": True,
         "predeclared_role": "LOWER_HEIGHT_FALLBACK_WINNER", "winner_hierarchy_rank": 3},
        {**common, "code": "000504", "name": "南华生物", "boards": 3, "turnover": 1,
         "board_path_status": "ONE_PRICE_BUYABILITY_UNPROVEN", "tradable_expectation_eligible": False,
         "predeclared_role": "FULL_DENOMINATOR_WATCH_ONLY", "winner_hierarchy_rank": float("nan")},
    ])


def test_expectation_is_conditioned_on_prior_board_path_not_one_gap_band():
    pool = _pool(); tree = pool.dropna(subset=["winner_hierarchy_rank"])[["code"]]
    matrix, scenarios, report = build(
        pool, tree, {"signal_date": "2026-09-21", "full_3_4_board_denominator": 4,
                     "primary_expected_winner": {"code": "603230"}},
        {"emotion_dimensions": {}})
    types = dict(zip(matrix.code, matrix.expectation_type))
    assert types["603230"] == "DIVERGENCE_ABSORPTION_CONTINUATION"
    assert types["002285"] == "STRONG_TO_STRONG_CONTINUATION"
    assert types["600127"] == "HIGH_TURNOVER_DIVERGENCE_CONTINUATION"
    assert types["000504"] == "FIRST_OPEN_PRICE_DISCOVERY"
    assert not bool(matrix.loc[matrix.code.eq("000504"), "trade_expectation_allowed"].iloc[0])
    assert "THREE_BOARD_CANNOT_BEAT_A_CONFIRMED_FOUR_BOARD" in report["improvements_over_v239"]
    assert len(scenarios) == 5
