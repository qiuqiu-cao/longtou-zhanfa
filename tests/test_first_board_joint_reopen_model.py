import numpy as np
import pandas as pd
import pytest
import first_board_joint_reopen_model as model


def sources():
    original = pd.DataFrame({"signal_date": ["2025-01-02"] * 3, "code": ["000001", "000002", "000003"],
                             model.stage.TARGET: [0., 1., np.nan]})
    qualified = original[model.KEYS].copy()
    qualified[model.FIELD] = [0., 1., np.nan]
    return original, qualified


def test_true_zero_unknown_and_full_label_pool_remain_distinct():
    original, qualified = sources()
    actual = model.attach_flag(original, qualified)
    pd.testing.assert_frame_equal(actual[original.columns], original)
    assert actual.loc[0, model.FIELD] == 0
    assert actual.loc[1, model.FIELD] == 1
    assert np.isnan(actual.loc[2, model.FIELD])


def test_missing_keys_duplicates_unknown_codes_and_nonbinary_rejected():
    original, qualified = sources()
    with pytest.raises(ValueError, match="every original"):
        model.attach_flag(original, qualified.iloc[:2])
    with pytest.raises(ValueError, match="Duplicate"):
        model.attach_flag(original, pd.concat([qualified, qualified.iloc[:1]]))
    qualified.loc[0, model.FIELD] = 2
    with pytest.raises(ValueError, match="binary"):
        model.attach_flag(original, qualified)


def test_future_outcomes_never_admitted_as_feature_source():
    original, qualified = sources()
    qualified[model.stage.TARGET] = original[model.stage.TARGET]
    with pytest.raises(ValueError, match="future targets"):
        model.attach_flag(original, qualified)
    for arm, fields in model.ARMS.items():
        assert len(fields) == int(arm.rsplit("_", 1)[1])
        assert len(fields) == len(set(fields))
        assert not any("LABEL_ONLY" in field or "theme" in field or "label_end" in field for field in fields)


def test_appending_future_rows_cannot_change_prior_reopening_observation():
    original, qualified = sources()
    future = original.copy()
    future["signal_date"] = "2025-01-03"
    future[model.stage.TARGET] = [1., 0., 1.]
    future_qualified = qualified.copy()
    future_qualified["signal_date"] = "2025-01-03"
    actual = model.attach_flag(pd.concat([original, future], ignore_index=True),
                               pd.concat([qualified, future_qualified], ignore_index=True))
    pd.testing.assert_frame_equal(actual.iloc[:3].reset_index(drop=True), model.attach_flag(original, qualified))
