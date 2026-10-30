import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "research/first_board_top3_40_20261030/analyze_top3_gap.py"
SPEC = importlib.util.spec_from_file_location("top3_target", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pool():
    return pd.DataFrame({"signal_date": ["D1"]*3+["D2"]*3,
                         "code": ["1", "2", "3"]*2,
                         MODULE.P2: [1., 0., 0., 0., 0., float("nan")]})


def test_unknown_and_fixed_slots_remain_visible():
    d = pool()
    result = MODULE.evaluate_selection(d, d[MODULE.KEYS], ["D1", "D2"])
    assert result["known_precision"] == 1/5
    assert result["confirmed_hits_per_all_slots"] == 1/6
    assert result["possible_hits_per_all_slots"] == 2/6
    assert result["unknown"] == 1


def test_only_picking_winner_cannot_pass_daily_top3_target():
    d = pool()
    result = MODULE.evaluate_selection(d, d.iloc[:1], ["D1", "D2"])
    assert result["known_precision"] == 1
    assert result["unfilled_slots"] == 5
    assert not result["observed_target_met"]


def test_selection_cannot_overwrite_labels():
    d = pool()
    fake = d.assign(**{MODULE.P2: 1.})
    assert MODULE.evaluate_selection(d, fake, ["D1", "D2"])["hits"] == 1


@pytest.mark.parametrize("case", ["duplicate", "outside", "missing_date", "invalid_label"])
def test_invalid_comparison_is_rejected(case):
    d = pool()
    picks = d[MODULE.KEYS].copy()
    if case == "duplicate":
        picks = pd.concat([picks, picks.iloc[:1]])
    elif case == "outside":
        picks.loc[0, "code"] = "excluded_large_cap"
    elif case == "missing_date":
        d = d[d.signal_date.eq("D1")]
    else:
        d.loc[0, MODULE.P2] = -1
    with pytest.raises(ValueError):
        MODULE.evaluate_selection(d, picks, ["D1", "D2"])
