"""Tests for the audit's independent accounting helpers; no training."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("independent_pairwise_audit", Path(__file__).with_name("verify_pairwise_independent.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def candidates():
    return pd.DataFrame(dict(date=["2026-10-15"]*4, code=["000001", "000002", "000003", "000004"],
        boards=[2., 3., 2., 4.], ranking_score=[.9, .8, .7, .6],
        daily_one_word=[0., 0., 0., 1.], target=[np.nan, 0., 1., 1.],
        label_valid=[False, True, True, True], call=["UNKNOWN", "FAIL", "PROMOTE", "PROMOTE"]))


def test_unknown_top_slot_is_kept_without_reselection():
    summary, picked = audit.select_before_truth(candidates(), 1)
    assert picked.code.tolist() == ["000001"]
    assert summary["selected"] == summary["unknown"] == 1
    assert summary["hits"] == summary["known"] == 0
    assert summary["hits_per_all_slots"] == 0.
    assert summary["hits_per_known_slots"] is None
    assert summary["unknown_best_case"] == 1.


def test_selection_does_not_consult_labels():
    frame = candidates()
    _, first = audit.select_before_truth(frame, 2)
    frame["target"] = [1., 1., 0., 0.]
    frame["label_valid"] = True
    _, second = audit.select_before_truth(frame, 2)
    assert first.code.tolist() == second.code.tolist()


def test_null_singleton_score_retains_slot_and_height_scope():
    frame = candidates().iloc[:1].copy()
    frame["ranking_score"] = np.nan
    summary, picked = audit.select_before_truth(frame, 3, same_height=True)
    assert picked.code.tolist() == ["000001"]
    assert summary["selected_missing_score"] == 1
    assert summary["groups_fewer_than_k"] == 1


def test_ties_use_code_and_height_baseline_is_independent_of_model_score():
    frame = candidates().iloc[:3].copy().iloc[::-1]
    frame["ranking_score"] = .5
    _, score_picks = audit.select_before_truth(frame, 2)
    _, height_picks = audit.select_before_truth(frame, 2, height_baseline=True)
    assert score_picks.code.tolist() == ["000001", "000002"]
    assert height_picks.code.tolist() == ["000002", "000001"]


def test_pair_dispositions_keep_unknown_and_unit_mixed_day_weight():
    frame = candidates().iloc[:3].copy()
    g = audit.independently_count_groups(frame, "2026-10")
    assert len(g) == 1
    assert g.iloc[0].complete_nonword_candidates == 3
    assert g.iloc[0].known == 2 and g.iloc[0].unknown_labels == 1
    assert g.iloc[0].directed_pairs == 2
    assert g.iloc[0].independently_computed_day_weight == 1.
    frame["target"] = 0.
    frame["label_valid"] = True
    z = audit.independently_count_groups(frame, "2026-10")
    assert z.iloc[0].training_group_status == "ALL_KNOWN_NEGATIVE"
    assert z.iloc[0].directed_pairs == z.iloc[0].independently_computed_day_weight == 0


def test_exact_cell_comparison_detects_change_and_preserves_nulls():
    left = candidates()
    assert all(x["different"] == 0 for x in audit.compare_frame(left, left.copy(), ["date", "code"], ["ranking_score", "target"]))
    changed = left.copy()
    changed.loc[0, "ranking_score"] += 1e-10
    with pytest.raises(AssertionError):
        audit.compare_frame(left, changed, ["date", "code"], ["ranking_score"])
