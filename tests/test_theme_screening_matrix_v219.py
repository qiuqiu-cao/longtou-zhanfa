import pandas as pd

from theme_screening_matrix_v219 import (
    adapt_external_memberships,
    build_matrix,
    industry_memberships,
    normalize_pool,
)


def _pool():
    return pd.DataFrame([
        {"signal_date": "2026-09-21", "code": "600001", "name": "甲", "boards": 2,
         "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
         "industry_provider": "电力", "first_seal_time": 93000},
        {"signal_date": "2026-09-21", "code": "600002", "name": "乙", "boards": 1,
         "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
         "industry_provider": "电力", "first_seal_time": 94500},
        {"signal_date": "2026-09-21", "code": "600003", "name": "丙", "boards": 1,
         "pool_assignment": "FAILED_BOARD", "is_mainboard": True,
         "industry_provider": "电力", "first_seal_time": 95000},
    ])


def test_industry_lane_never_becomes_verified_real_theme():
    pool = normalize_pool(_pool())
    matrix, mapping = build_matrix(pool, industry_memberships(pool))
    assert matrix.iloc[0].evidence_lane == "INDUSTRY_PROXY"
    assert matrix.iloc[0].theme_stage == "RELAY_WITH_SUPPORT"
    assert not bool(matrix.iloc[0].entry_authorized)
    assert len(mapping) == 1


def test_verified_and_symbolic_lanes_are_ranked_separately():
    pool = normalize_pool(_pool())
    base = industry_memberships(pool)
    verified = base.copy()
    verified["theme_id"] = "real:power"
    verified["theme_name"] = "电力改革"
    verified["theme_type"] = "VERIFIED_REAL_THEME"
    verified["membership_status"] = "POINT_IN_TIME_VERIFIED"
    verified["catalyst_status"] = "ACTIVE_DRIVER_VERIFIED"
    symbolic = base.copy()
    symbolic["theme_id"] = "symbol:x"
    symbolic["theme_name"] = "某字辈"
    symbolic["theme_type"] = "SYMBOLIC_NAME_CLUSTER"
    memberships = pd.concat([verified, symbolic], ignore_index=True)
    matrix, _ = build_matrix(pool, memberships)
    assert set(matrix.evidence_lane) == {"VERIFIED_REAL_THEME", "SYMBOLIC_NAME_CLUSTER"}
    assert matrix.groupby("evidence_lane").rank_within_lane.min().eq(1).all()


def test_existing_observed_role_contract_is_adapted_without_inventing_catalyst():
    roles = pd.DataFrame([{
        "signal_date": "2026-09-21", "code": "600001",
        "theme_group_id": "power", "theme": "电力改革",
        "theme_status": "observed_explicit_secondary_roster",
        "evidence_status": "source_page_explicitly_names_security_and_theme",
        "evidence_available_at": "2026-09-21T20:00:00+08:00",
    }])
    adapted = adapt_external_memberships(roles)
    assert adapted.iloc[0].theme_type == "OBSERVED_THEME_ROSTER"
    assert adapted.iloc[0].catalyst_status == "UNAVAILABLE"
