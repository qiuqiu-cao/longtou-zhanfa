import numpy as np
import pandas as pd

from build_first_board_five_factor_inputs_v209 import build


def test_unavailable_competition_and_announcements_do_not_become_zero():
    features = pd.DataFrame([{
        "signal_date": "2026-01-05", "target_date": "2026-01-06",
        "code": "1", "turnover": 10,
        "amount_to_float_cap": .1, "last_minus_first_minute": 5,
        "failed_seal_count": 1, "one_price_proxy": 0, "thin_control_risk": 0,
        "feature_available_at": "2026-01-05T15:05:00+08:00",
    }])
    theme = pd.DataFrame([{
        "signal_date": "2026-01-05", "code": "1", "theme_group_id": "x",
        "theme_support_ex_self": 2, "point_in_time_available_for_next_session": True,
        "active_driver_verified": True, "driver_theme_granularity": "specific",
        "available_at": "2026-01-05T16:00:00+08:00",
    }])
    rows, report = build(features, theme)
    assert rows.loc[0, "turnover_absorption_quality_evidence_status"] == "POINT_IN_TIME_VERIFIED"
    assert rows.loc[0, "theme_purity"] == 1
    assert np.isnan(rows.loc[0, "competitor_elimination"])
    assert np.isnan(rows.loc[0, "announcement_soft_risk"])
    assert not rows.loc[0, "five_factor_point_in_time_complete"]
    assert report["five_factor_complete_rows"] == 0
