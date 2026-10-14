import pandas as pd

from relay_promotion_emotion_overlay_v260 import _check_date_state, _selection


def rows():
    return pd.DataFrame({
        "date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
        "code": ["000002", "000001", "000004", "000003"],
        "boards": [2, 2, 2, 2],
        "state_divergence": [0, 0, 1, 1],
        "score_identity_emotion_price_volume": [0.5, 0.9, 0.9, 0.1],
        "promoted_next_day": [1, 0, 0, 1],
    })


def test_overlay_keeps_model_selection_and_only_changes_status():
    baseline, model = _selection(rows())
    assert model.set_index("date").loc["2026-01-05", "code"] == "000001"
    assert model.set_index("date").loc["2026-01-06", "code"] == "000004"
    assert model.set_index("date").loc["2026-01-05", "identity_status"] == "PRIMARY_RELAY_IDENTITY"
    assert model.set_index("date").loc["2026-01-06", "identity_status"] == "EMOTION_OBSERVATION_ONLY"
    assert baseline.set_index("date").loc["2026-01-06", "code"] == "000003"


def test_state_must_be_constant_within_date():
    x = rows()
    x.loc[1, "state_divergence"] = 1
    try:
        _check_date_state(x)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed state must fail")
