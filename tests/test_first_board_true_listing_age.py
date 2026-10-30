import numpy as np
import pandas as pd
import pytest

import first_board_true_listing_age as model


def test_ipo_date_support_has_no_dependency_on_other_security_state_intervals():
    assert model.listing_date_supported(1, True, "1999-05-27", {"1999-05-27"})
    assert not model.listing_date_supported(2, True, "1999-05-27", {"1999-05-27"})
    assert not model.listing_date_supported(1, False, "1999-05-27", {"1999-05-27"})
    assert not model.listing_date_supported(1, True, "1999-05-27", {"2000-01-01"})


def test_calendar_age_includes_pre_archive_history_and_leap_day():
    keys = pd.DataFrame({"signal_date": ["2024-03-01", "2024-03-01"], "code": ["000001", "000002"]})
    identity = pd.DataFrame({"code": ["000001", "000002"], "listing_date": ["1991-04-03", "2024-02-28"], "identity_qualified": [True, True]})
    result = model.derive_age(keys, identity)
    assert result.loc[0, model.NEW] > 12000
    assert result.loc[1, model.NEW] == 2
    assert model.OLD not in model.FIELDS and len(model.FIELDS) == 62
    assert set(model.FIELDS) - set(model.ORIGINAL_FIELDS) == {model.NEW}


def test_missing_conflicting_or_future_listing_date_remains_unknown():
    keys = pd.DataFrame({"signal_date": ["2024-01-02"] * 3, "code": ["000001", "000002", "000003"]})
    identity = pd.DataFrame({"code": ["000001", "000002"], "listing_date": ["2024-01-03", "2000-01-01"], "identity_qualified": [True, False]})
    assert model.derive_age(keys, identity)[model.NEW].isna().all()


def test_duplicate_identity_rejected_and_future_rows_do_not_change_past_age():
    keys = pd.DataFrame({"signal_date": ["2024-01-02", "2025-01-02"], "code": ["000001", "000001"]})
    identity = pd.DataFrame({"code": ["000001"], "listing_date": ["2000-01-01"], "identity_qualified": [True]})
    assert model.derive_age(keys, identity).loc[0, model.NEW] == model.derive_age(keys.iloc[:1], identity).loc[0, model.NEW]
    with pytest.raises(ValueError, match="Duplicate"):
        model.derive_age(keys, pd.concat([identity, identity]))


def test_old_feature_and_unknown_label_are_not_replaced_in_fact_table():
    rows = pd.DataFrame({"signal_date": ["2024-01-02"], "code": ["000001"],
                         **{field: [1.] for field in model.ORIGINAL_FIELDS}, model.stage.TARGET: [np.nan]})
    extra = pd.DataFrame({"signal_date": ["2024-01-02"], "code": ["000001"], model.NEW: [10000.]})
    result = model.attach_qualified_age(rows, extra)
    assert result.loc[0, model.OLD] == 1.
    assert np.isnan(result.loc[0, model.stage.TARGET])
    with pytest.raises(ValueError, match="Incomplete"):
        model.attach_qualified_age(rows, extra.iloc[:0])
