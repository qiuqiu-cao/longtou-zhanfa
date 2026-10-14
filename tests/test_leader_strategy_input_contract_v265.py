import pandas as pd

from leader_strategy_input_contract_v265 import audit


def test_input_contract_allows_identity_but_not_relay_when_microstructure_is_missing():
    frame = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "signal_date": ["2025-01-02", "2025-01-02"],
            "boards": [2, 3],
        }
    )
    result = audit(frame, "2025-01-02T15:00:00+08:00")
    assert result["identity_review"]["allowed"] is True
    assert result["conditional_relay_review"]["allowed"] is False
    assert result["trade_authorization"]["allowed"] is False
    assert "auction_gap" in result["conditional_relay_review"]["missing_fields"]


def test_input_contract_rejects_future_source_timestamp():
    frame = pd.DataFrame(
        {
            "code": ["000001"],
            "signal_date": ["2025-01-02"],
            "boards": [2],
            "source_timestamp": ["2025-01-03T09:30:00+08:00"],
        }
    )
    result = audit(frame, "2025-01-02T15:00:00+08:00")
    assert result["future_timestamp_rows"] == 1
    assert result["identity_review"]["allowed"] is False
