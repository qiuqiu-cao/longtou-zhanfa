import pandas as pd

from relay_path_history_audit_v245 import audit


def _data():
    rows = []
    for index, (date, code, boards, one_price, gap, promoted) in enumerate([
        ("2026-01-01", "000001", 3, 0, .04, 1),
        ("2026-01-02", "000002", 4, 0, .06, 0),
        ("2026-01-03", "000003", 3, 1, .10, 1),
        ("2026-01-04", "000004", 4, 1, .09, 1),
    ]):
        rows.append({"date": date, "code": code, "boards": boards,
            "promoted_next_day": promoted,
            "promotion_label_status": "verified_from_next_daily_limit_close",
            "next_open_gap": gap, "one_word_proxy": one_price,
            "first_seal_minute": 0, "last_seal_minute": 0, "bomb_count": 0,
            "turnover": 10 + index, "execution_status": "inferred_daily_limit_touch_fill",
            "net_return": .01 * (index - 1), "large_loss": 0})
    return pd.DataFrame(rows)


def test_zero_intraday_placeholders_block_path_validation_without_deleting_labels():
    coverage, metrics, comparisons, report = audit(
        _data(), {"version": "v242-relay-path-expectation-1"})
    assert report["global_zero_intraday_placeholder_detected"]
    assert report["usable_full_path_rows"] == 0
    assert report["promotion_and_open_labeled_rows"] == 4
    assert not report["conclusions"]["v242_path_thresholds_validated"]
    assert not report["conclusions"]["same_open_execution_allowed"]
    assert "V242_LOCKED_3_TO_7_LABEL_ONLY" in set(metrics["slice"])
    assert set(comparisons.boards) == {3, 4}
    assert coverage.loc[coverage.field_group.eq(
        "USABLE_FIRST_LAST_SEAL_AND_FAILED_SEAL_PATH"), "status"].iloc[0].startswith("UNAVAILABLE")
