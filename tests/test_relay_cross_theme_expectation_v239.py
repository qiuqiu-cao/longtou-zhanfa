import pandas as pd

from relay_cross_theme_expectation_v239 import build


def _scores():
    rows = []
    specs = [
        ("603230", "内蒙新华", 4, 16, "QUICK_REPAIR", 2),
        ("002285", "世联行", 4, 14, "LOCKED_NO_REOPEN", 2),
        ("002453", "华软科技", 3, 18, "LOCKED_NO_REOPEN", 2),
        ("000504", "南华生物", 3, 10, "ONE_PRICE_BUYABILITY_UNPROVEN", 0),
        ("600127", "金健米业", 3, 7, "QUICK_REPAIR", 0),
        ("600448", "华纺股份", 3, -5, "PROLONGED_CONTEST", -4),
    ]
    for code, name, boards, score, path, turnover_factor in specs:
        rows.append({
            "signal_date": "2026-09-21", "code": code, "name": name,
            "boards": boards, "research_score_NOT_PROBABILITY": score,
            "board_path_status": path, "announcement_risk_tier": "NO_ACTIVE",
            "announcement_hard_veto": False, "symbolic_tags": "",
            "factor_turnover_quality": turnover_factor,
        })
    return pd.DataFrame(rows)


def _review():
    themes = {
        "603230": ("publishing", "传媒", 10.6), "002285": ("real_estate", "房地产", 14.7),
        "002453": ("chip", "国产芯片", 6.7), "000504": ("medicine", "医药", 1.6),
        "600127": ("agri", "大农业", 28.4), "600448": ("", "", 39.3),
    }
    return {"signal_date": "2026-09-21", "relay_candidates": [
        {"code": code, "turnover": turnover, "amount": 100,
         "eligibility_status": "VERIFIED_ELIGIBLE",
         "theme_evidence": {"theme_group_id": theme_id, "theme": theme,
                            "status": "INFERRED_OR_LIMITED", "support_ex_self": 2,
                            "ladder_verified": True}}
        for code, (theme_id, theme, turnover) in themes.items()
    ]}


def test_primary_same_height_then_lower_height_hierarchy_is_frozen():
    scores = _scores()
    roles = scores[["code"]].copy()
    pool, tree, report = build(
        scores, roles, {"close_relay_challenger": {"code": "603230"}}, _review())
    assert tree.code.tolist()[:4] == ["603230", "002285", "002453", "600127"]
    assert report["primary_expected_winner"]["code"] == "603230"
    assert report["conditional_fallback_winners"][0]["code"] == "002285"
    assert int(report["full_3_4_board_denominator"]) == 6
    assert bool(pool.in_full_identity_denominator.all())


def test_one_price_and_unknown_theme_remain_in_denominator_but_not_tradeable_fallbacks():
    scores = _scores(); roles = scores[["code"]].copy()
    pool, tree, report = build(
        scores, roles, {"close_relay_challenger": {"code": "603230"}}, _review())
    assert "000504" not in tree.code.tolist()
    assert "600448" not in tree.code.tolist()
    assert not bool(pool.loc[pool.code.eq("000504"), "tradable_expectation_eligible"].iloc[0])
    assert int(report["unknown_theme_rows_retained"]) == 1


def test_no_winner_is_an_explicit_outcome_not_forced_selection():
    scores = _scores(); roles = scores[["code"]].copy()
    _, _, report = build(
        scores, roles, {"close_relay_challenger": {"code": "603230"}}, _review())
    assert report["no_winner_scenario"]["action"] == "CASH_NO_REPLACEMENT_TRADE"
    assert not report["governance"]["higher_board_win_rate_claimed"]
