import numpy as np
import pandas as pd

from early_leader_topk_walkforward_v261 import (
    FEATURES,
    FORBIDDEN_FEATURES,
    _clean,
    _forward_bar_metrics,
    _select,
)


def test_v261_feature_list_excludes_future_labels():
    assert set(FEATURES).isdisjoint(FORBIDDEN_FEATURES)
    assert "future_promotion" not in FEATURES
    assert "future_market_leader" not in FEATURES


def test_v261_clean_converts_scalar_missing_values_without_dataframe_ambiguity():
    cleaned = _clean({"missing": np.nan, "ok": np.int64(3), "nested": [np.float64(1.25)]})
    assert cleaned == {"missing": None, "ok": 3, "nested": [1.25]}


def test_v261_topk_selection_is_date_grouped_and_deterministic():
    frame = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * 4 + ["2025-01-03"] * 2,
            "code": ["000003", "000001", "000002", "000004", "000005", "000006"],
            "boards": [2, 2, 1, 3, 1, 1],
            "score": [0.8, 0.8, 0.9, 0.1, 0.4, 0.2],
        }
    )
    selected = _select(frame, "score", 2)
    assert selected.groupby("signal_date").size().to_dict() == {
        "2025-01-02": 2,
        "2025-01-03": 2,
    }
    first_day = selected[selected.signal_date == "2025-01-02"]
    assert first_day.code.tolist() == ["000002", "000001"]
    assert first_day.selection_rank.tolist() == [1, 2]


def test_v263_forward_proxy_reports_path_and_risk_metrics():
    selected = pd.DataFrame(
        {
            "signal_date": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "forward_return_1d_proxy": [0.02, -0.01, 0.03],
            "forward_return_3d_proxy": [0.04, -0.02, 0.01],
            "forward_return_5d_proxy": [0.05, -0.03, 0.02],
            "forward_max_gain_5d_proxy": [0.10, 0.05, 0.08],
            "forward_max_drawdown_5d_proxy": [-0.04, -0.08, -0.03],
        }
    )
    result = _forward_bar_metrics(selected)
    assert result["status"] == "INFERRED_CLOSE_TO_CLOSE_OHLC_PROXY"
    assert result["1d"]["known_rows"] == 3
    assert result["1d"]["longest_loss_streak"] == 1
    assert result["five_day_excursion"]["worst_max_drawdown"] == -0.08
