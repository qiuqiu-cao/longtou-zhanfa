import pandas as pd

from same_height_group_comparison_v205 import build


def row(code, boards=2, complete=True):
    status = "VERIFIED" if complete else "UNAVAILABLE"
    return {
        "signal_date": "2026-09-17", "code": code, "name": code,
        "boards": boards, "first_seal_minute": 570, "failed_seal_count": 0,
        "full_market_complete": True, "b2_positive_probability": .6,
        "b1_big_loss_probability": .1, "dual_target_status": "TIME_ISOLATED_DIAGNOSTIC",
        "evidence_available_at": "2026-09-17T15:10:00+08:00",
        "turnover_absorption_quality": .8, "turnover_absorption_quality_evidence_status": status,
        "theme_purity": .9, "theme_purity_evidence_status": status,
        "sector_leadership": .7, "sector_leadership_evidence_status": status,
        "competitor_elimination": .6, "competitor_elimination_evidence_status": status,
        "announcement_soft_risk": .2, "announcement_soft_risk_evidence_status": status,
    }


def test_same_height_is_forced_into_one_group_and_never_official_champion():
    rows, report = build(pd.DataFrame([row("600001"), row("000001")]),
                         "2026-09-17T15:20:00+08:00")
    assert rows.height_group_id.nunique() == 1
    assert rows.same_height_group_size.eq(2).all()
    assert rows.group_model_eligible.all()
    assert rows.official_champion.eq(False).all()
    assert report["policy"]["seal_speed_or_zero_failed_seals_can_select_champion"] is False


def test_one_missing_factor_blocks_the_entire_height_group():
    rows, _ = build(pd.DataFrame([row("600001"), row("000001", complete=False)]),
                    "2026-09-17T15:20:00+08:00")
    assert not rows.group_model_eligible.any()
    assert rows.group_research_score_NOT_PROBABILITY.isna().all()


def test_future_evidence_blocks_group_rank():
    frame = pd.DataFrame([row("600001"), row("000001")])
    frame["evidence_available_at"] = "2026-09-18T09:00:00+08:00"
    rows, _ = build(frame, "2026-09-17T15:20:00+08:00")
    assert not rows.group_model_eligible.any()
