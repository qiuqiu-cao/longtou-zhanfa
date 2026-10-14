import numpy as np
import pandas as pd

from first_board_dual_target_v206 import (
    EMOTION_FEATURES, FIVE_FACTORS, STATUS_FIELDS, attach_targets,
)


def feature(code, evidence_at="2026-01-05T15:00:00+08:00"):
    row = {"signal_date": "2026-01-05", "code": code, "name": code,
           "same_height_peer_count": 1, "evidence_available_at": evidence_at}
    row.update({name: 1.0 for name in EMOTION_FEATURES})
    row.update({name: .5 for name in FIVE_FACTORS})
    row.update({name: "VERIFIED" for name in STATUS_FIELDS})
    return row


def outcome(code, b2=.01, big=0):
    return {"signal_date": "2026-01-05", "code": code,
            "fill_status": "FILLED_OPEN_PROXY",
            "exit_date_B_plus_1": "2026-01-06", "exit_date_B_plus_2": "2026-01-07",
            "net_return_B_plus_1": -.08 if big else .01,
            "net_return_B_plus_2": b2, "big_loss_B_plus_1": big}


def test_targets_are_distinct_and_five_factor_is_point_in_time():
    rows = attach_targets(pd.DataFrame([feature("A")]), pd.DataFrame([outcome("A", -.01, 1)]))
    assert rows.loc[0, "executable_positive_B_plus_2"] == 0
    assert rows.loc[0, "big_loss_target"] == 1
    assert rows.loc[0, "five_factor_point_in_time_complete"]
    assert not rows.loc[0, "missing_five_factor_is_zero"]


def test_missing_or_future_factor_evidence_does_not_become_zero():
    features = pd.DataFrame([feature("A", "2026-01-06T09:00:00+08:00")])
    features.loc[0, FIVE_FACTORS[0]] = np.nan
    rows = attach_targets(features, pd.DataFrame([outcome("A")]))
    assert not rows.loc[0, "five_factor_point_in_time_complete"]
    assert np.isnan(rows.loc[0, FIVE_FACTORS[0]])
