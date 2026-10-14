import pandas as pd

from relay_promotion_regime_gate_v258 import _check_date_state, _select


def _rows():
    return pd.DataFrame({
        "date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
        "code": ["000002", "000001", "000004", "000003"],
        "boards": [2, 2, 2, 2],
        "state_divergence": [0, 0, 1, 1],
        "score_identity_emotion_price_volume": [0.5, 0.9, 0.9, 0.1],
        "promoted_next_day": [1, 0, 0, 1],
    })


def test_gate_uses_model_outside_divergence_and_baseline_inside():
    selected = _select(_rows())
    assert selected["model"].set_index("date").loc["2026-01-05", "code"] == "000001"
    assert selected["regime_gated"].set_index("date").loc["2026-01-05", "code"] == "000001"
    # The neutral baseline is code-ordered on the divergence date.
    assert selected["regime_gated"].set_index("date").loc["2026-01-06", "code"] == "000003"


def test_date_state_must_be_constant():
    rows = _rows()
    rows.loc[1, "state_divergence"] = 1
    try:
        _check_date_state(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("mixed state within a date must fail")
