import pandas as pd
import pytest

from abnormal_path_role_v171 import build as build_path_roles
from current_abnormal_distance_v191 import _rounded_ten_percent_returns, _v171_compat, build


def _fixture(missing_stock_date=False):
    dates = pd.bdate_range("2026-08-05", periods=31).strftime("%Y-%m-%d")
    pool = pd.DataFrame([
        {"signal_date": dates[-1], "code": "600001", "name": "沪样本",
         "boards": 2, "board_group": "CONSECUTIVE_BOARD",
         "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
         "last_price": 10 * 1.01 ** 30, "eligibility_status": "unavailable"},
        {"signal_date": dates[-1], "code": "000001", "name": "深样本",
         "boards": 1, "board_group": "FIRST_BOARD",
         "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
         "last_price": 10 * 1.01 ** 30, "eligibility_status": "unavailable"},
    ])
    history = []
    for code in ("600001", "000001"):
        for index, date in enumerate(dates):
            if missing_stock_date and code == "600001" and index == 10:
                continue
            history.append({
                "date": date, "code": code, "close": 10 * 1.01 ** index,
                "provider": "fixture", "price_basis": "provider_unadjusted",
                "data_status": "provider_observed_not_official_reconciled",
            })
    benchmark = pd.DataFrame({
        "date": dates, "close": [100.0] * 31,
        "data_status": ["provider_observed_not_official_reconciled"] * 31,
    })
    pool_manifest = {"signal_date": dates[-1], "mainboard_closing_limit_rows": 2,
                     "point_in_time_security_scope_complete": False}
    input_manifest = {"signal_date": dates[-1], "candidate_count": 2}
    benchmark_manifest = {"requested_end": dates[-1], "official_reconciled": False}
    return pool, pool_manifest, pd.DataFrame(history), input_manifest, {
        "sh000002": benchmark.copy(), "sz399107": benchmark.copy()}, benchmark_manifest


def test_rounded_limit_returns_use_cent_rounding():
    values = _rounded_ten_percent_returns(7.21, 2)
    assert values[0] == pytest.approx((7.93 / 7.21 - 1) * 100)
    assert values[1] == pytest.approx((8.72 / 7.93 - 1) * 100)


def test_build_covers_both_correct_benchmarks_but_never_claims_official_distance():
    candidates, scenarios, report = build(*_fixture())
    by_code = candidates.set_index("code")
    assert by_code.loc["600001", "benchmark_code"] == "sh000002"
    assert by_code.loc["000001", "benchmark_code"] == "sz399107"
    assert candidates["research_indicator_status"].eq("inferred_unreconciled").all()
    assert candidates["regulatory_distance_status"].eq("unavailable").all()
    assert not candidates["ranking_allowed"].any()
    assert report["research_distance_count"] == 2
    assert report["official_regulatory_distance_count"] == 0
    assert report["hundred_percent_abnormal_path_potential"] is None
    assert len(scenarios) == 10


def test_missing_exchange_session_does_not_bridge_returns():
    candidates, _, report = build(*_fixture(missing_stock_date=True))
    row = candidates[candidates["code"].eq("600001")].iloc[0]
    assert row["research_indicator_status"] == "unavailable"
    assert row["unavailable_reason"] == "PROVIDER_CALENDAR_OR_CLOSE_COVERAGE_INCOMPLETE"
    assert report["research_distance_count"] == 1


def test_v171_export_attaches_all_rows_but_cannot_verify_or_select_a_role():
    pool, pool_manifest, history, input_manifest, benchmarks, benchmark_manifest = _fixture()
    candidates, _, _ = build(
        pool, pool_manifest, history, input_manifest, benchmarks, benchmark_manifest)
    evidence = _v171_compat(candidates, "2026-09-17T08:30:00+08:00")
    path_report = build_path_roles(
        pool, evidence, "2026-09-17T08:31:00+08:00")
    assert path_report["evidence_attached_count"] == 2
    assert path_report["verified_distance_count"] == 0
    assert path_report["path_eligible_count"] == 0
    assert path_report["rule_distance_closest"] is None
    assert path_report["hundred_percent_abnormal_path_potential"] is None
