import pandas as pd

from first_board_multilane_identity_v229 import build


def _watch():
    return pd.DataFrame([
        {"signal_date": "2026-09-21", "code": "000001", "name": "华甲",
         "close_quality_gate": True, "one_price_proxy": False,
         "thin_control_risk": False, "first_seal_minute": 20,
         "community_evidence_count": 6,
         "community_probability_not_calibrated": .20, "theme_research_gate": True},
        {"signal_date": "2026-09-21", "code": "000002", "name": "华乙",
         "close_quality_gate": True, "one_price_proxy": False,
         "thin_control_risk": False, "first_seal_minute": 18,
         "community_evidence_count": 5,
         "community_probability_not_calibrated": .25, "theme_research_gate": False},
        {"signal_date": "2026-09-21", "code": "000003", "name": "医药甲",
         "close_quality_gate": True, "one_price_proxy": False,
         "thin_control_risk": False, "first_seal_minute": 16,
         "community_evidence_count": 7,
         "community_probability_not_calibrated": .30, "theme_research_gate": True},
    ])


def _themes():
    rows = [
        ("000001", "drug", "医药", "OBSERVED_THEME_ROSTER", "FULL_LADDER", 1, 2),
        ("000001", "name_ngram_1_华", "华字辈", "SYMBOLIC_NAME_CLUSTER", "FULL_LADDER", 1, 2),
        ("000002", "name_ngram_1_华", "华字辈", "SYMBOLIC_NAME_CLUSTER", "FULL_LADDER", 1, 1),
        ("000003", "drug", "医药", "OBSERVED_THEME_ROSTER", "FULL_LADDER", 1, 1),
    ]
    return pd.DataFrame([
        {"signal_date": "2026-09-21", "code": code, "theme_id": theme_id,
         "theme_name": name, "evidence_lane": lane, "theme_stage": stage,
         "rank_within_lane": rank, "theme_pioneer_rank": pioneer}
        for code, theme_id, name, lane, stage, rank, pioneer in rows
    ])


def _symbolic():
    return {
        "version": "v217-symbolic-name-cluster-1", "signal_date": "2026-09-21",
        "data_quality": {"name_match_status": "VERIFIED_WITHIN_ARCHIVED_PROVIDER_POOL",
                         "availability_status": "VERIFIED_AVAILABLE_BY_ASOF",
                         "full_market_completeness": "verified"},
        "clusters": [{
            "cluster_id": "name_ngram_1_华", "token": "华",
            "classification": "SYMBOLIC_LADDER_CANDIDATE", "member_count": 5,
            "max_board": 4, "failed_board_member_count": 0,
            "ladder_contiguous": True,
            "previous_session_comparison": {"member_count_change": 2, "max_board_change": 1},
            "members": [
                {"code": "600001", "name": "华高", "boards": 4},
                {"code": "000001", "name": "华甲", "boards": 1},
                {"code": "000002", "name": "华乙", "boards": 1},
            ],
        }],
    }


def _claims():
    return pd.DataFrame([
        {"signal_date": "2026-09-21", "code": "000001",
         "driver_claim": "公司拟回购股份",
         "driver_claim_status": "OBSERVED_SECONDARY_DRIVER_CLAIM",
         "driver_claim_category": "CONTROL_OR_CAPITAL_ACTION",
         "negative_or_limited_exposure": False,
         "recency_language_status": "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED",
         "catalyst_first_public_time_status": "UNAVAILABLE"},
        {"signal_date": "2026-09-21", "code": "000002",
         "driver_claim": "公司不生产相关产品",
         "driver_claim_status": "OBSERVED_SECONDARY_DRIVER_CLAIM",
         "driver_claim_category": "NEGATIVE_OR_LIMITED_EXPOSURE",
         "negative_or_limited_exposure": True,
         "recency_language_status": "STATIC_OR_UNDATED_DESCRIPTION",
         "catalyst_first_public_time_status": "UNAVAILABLE"},
    ])


def test_secondary_best_theme_does_not_hide_symbolic_lane():
    candidates, mappings, report = build(_watch(), _themes(), _symbolic())
    row = candidates[candidates.code.eq("000001")].iloc[0]
    assert row.primary_business_theme == "医药"
    assert row.symbolic_theme_name == "华字辈"
    assert row.multilane_role == "SYMBOLIC_DUAL_IDENTITY_CHALLENGER"
    assert row.symbolic_anchor_names == "华高"
    assert row.symbolic_relay_ladder == "华高(4板)"
    assert not bool(row.symbolic_is_real_theme)
    assert len(mappings[mappings.low_code.eq("000001")]) == 1
    assert report["official_first_board_primary"] is None


def test_pure_symbolic_pioneer_and_dual_candidate_are_separate():
    _, _, report = build(_watch(), _themes(), _symbolic())
    assert report["symbolic_pure_buyable_pioneer"]["code"] == "000002"
    assert report["symbolic_dual_identity_candidate"]["code"] == "000001"
    assert report["narrative_lane_candidate"]["code"] == "000003"


def test_signal_date_mismatch_fails_closed():
    symbolic = _symbolic()
    symbolic["signal_date"] = "2026-09-20"
    try:
        build(_watch(), _themes(), symbolic)
    except ValueError as exc:
        assert "signal dates differ" in str(exc)
    else:
        raise AssertionError("date mismatch must fail")


def test_secondary_driver_claim_is_visible_but_not_promoted_to_fact():
    candidates, mappings, report = build(_watch(), _themes(), _symbolic(), _claims())
    first = candidates[candidates.code.eq("000001")].iloc[0]
    second = candidates[candidates.code.eq("000002")].iloc[0]
    assert first.secondary_driver_claim == "公司拟回购股份"
    assert "SECONDARY_DRIVER_CLAIM_NOT_VERIFIED_FACT" in first.contradictions
    assert "NEGATIVE_OR_LIMITED_EXPOSURE_CLAIM" in second.contradictions
    assert report["data_quality"]["secondary_driver_claim_overlay"] == "ATTACHED"
    assert mappings[mappings.low_code.eq("000001")].iloc[0].high_anchor_secondary_driver_category == "UNAVAILABLE"


def test_official_driver_time_distinguishes_ongoing_from_new_catalyst():
    official = pd.DataFrame([{
        "signal_date": "2026-09-21", "code": "000001",
        "official_driver_evidence_status": "OFFICIAL_CORE_CLAIM_MATCHED",
        "first_public_time_status": "VERIFIED_AT_LATEST_DATE",
        "official_driver_age_lower_bound_days": 31,
        "signal_date_novelty_status": "PREEXISTING_AT_LEAST_31_CALENDAR_DAYS",
        "driver_freshness_readthrough": "ONGOING_ACTION_BUT_NOT_NEW_ON_SIGNAL_DATE",
        "lifecycle_status": "ONGOING_ACTION_VERIFIED",
        "driver_is_new_on_signal_date": False,
    }])
    candidates, _, report = build(
        _watch(), _themes(), _symbolic(), _claims(), official)
    row = candidates[candidates.code.eq("000001")].iloc[0]
    assert row.official_driver_age_lower_bound_days == 31
    assert row.official_driver_lifecycle_status == "ONGOING_ACTION_VERIFIED"
    assert "ONGOING_ACTION_NOT_NEW_CATALYST" in row.contradictions
    assert "CATALYST_FIRST_PUBLIC_TIME_UNAVAILABLE" not in row.contradictions
    assert report["data_quality"]["official_driver_time_overlay"] == "PARTIAL_ATTACHED"
