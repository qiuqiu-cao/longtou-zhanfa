import numpy as np
import pandas as pd
import pytest

import first_board_co_limit_network as model


def inputs():
    rows = pd.DataFrame({"signal_date": ["2025-01-02", "2025-01-02"], "code": ["000001", "000002"],
                         **{field: [1., np.nan] for field in model.ORIGINAL}, model.stage.TARGET: [1., np.nan]})
    network = rows[model.KEYS].copy()
    for field in model.NETWORK_FIELDS:
        network[field] = [np.nan, 0.]
    return rows, network


def test_unchanged_full_cohort_and_unknown_fields_remain_unknown():
    rows, network = inputs()
    result = model.attach_network(rows, network.iloc[::-1])
    assert result.code.tolist() == ["000001", "000002"]
    assert result.loc[0, model.NETWORK_FIELDS].isna().all()
    assert np.isnan(result.loc[1, model.stage.TARGET])
    assert len(model.FIELDS) == 70 and not set(model.risk.CAP_FIELDS) & set(model.FIELDS)


def test_missing_and_duplicate_peer_keys_rejected():
    rows, network = inputs()
    with pytest.raises(ValueError, match="unchanged full"):
        model.attach_network(rows, network.iloc[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        model.attach_network(rows, pd.concat([network, network.iloc[:1]]))


def test_future_label_cannot_be_smuggled_in_network_schema():
    rows, network = inputs()
    network[model.stage.TARGET] = 1.
    with pytest.raises(ValueError, match="no labels"):
        model.attach_network(rows, network)


def test_baseline_wins_ties_and_both_additional_gates_required():
    def primary(hits, coverage=1.):
        return {"hits": hits, "decision_date_coverage": coverage}
    assert not model.validation_passes(primary(52), {"hits": 48})
    assert not model.validation_passes(primary(53), {"hits": 47})
    assert not model.validation_passes(primary(53, .94), {"hits": 48})
    assert model.validation_passes(primary(53), {"hits": 48})


def test_membership_check_preserves_labels_and_maturity_dates():
    rows = pd.DataFrame({"signal_date": ["2024-12-27"], "code": ["000001"], "stage": [1],
                         "label_end": ["2024-12-30"], model.stage.TARGET: [1.]})
    model.same_members(rows, rows.copy())
    for column, value in [(model.stage.TARGET, 0.), ("label_end", "2025-01-02")]:
        changed = rows.copy()
        changed[column] = value
        with pytest.raises(AssertionError):
            model.same_members(rows, changed)
