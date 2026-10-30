import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "research/first_board_sparse_residual_oot_20261030"


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


C = module("residual_core")
M = module("joint_top3_milp")


def sample():
    return pd.DataFrame(dict(signal_date=["2026-01-29"]*4, code=["000004", "000003", "000002", "000001"],
        s1_score=[.4, .3, .2, .1], height_score=[.1, .2, .3, .4],
        reported_close_seal_log_amount_ratio=[np.nan, 1., 2., 3.],
        reported_first_seal_trading_minutes=[np.nan, 10., 20., 30.],
        breakout_20=[np.nan, .1, .2, .3]))


def test_missing_residuals_are_neutral_and_labels_cannot_change_features():
    x = sample()
    f = C.features(x.assign(**{C.P2: 1.}))
    g = C.features(x.assign(**{C.P2: np.nan}))
    pd.testing.assert_frame_equal(f, g)
    assert f.loc[0, C.Z[:3]].eq(0.).all()
    assert all(f.loc[0, z+"_missing"] for z in C.Z[:3])
    assert x.loc[0, "breakout_20"] != x.loc[0, "breakout_20"]


def test_zero_residual_and_ties_use_fixed_code_order_not_input_order():
    f = C.features(sample().assign(s1_score=.2))
    a = C.select(f, np.zeros(4))
    b = C.select(f.sample(frac=1, random_state=9), np.zeros(4))
    pd.testing.assert_frame_equal(a, b)
    assert a.code.to_list() == ["000001", "000002", "000003"]


def test_maturity_purges_entire_future_horizon_day_without_dropping_unknown_labels():
    pool = pd.DataFrame(dict(signal_date=["2026-01-29"]*3 + ["2026-01-30"]*3,
        code=["001", "002", "003"]*2, promotion_next_LABEL_ONLY=[1, 0, np.nan, 1, 1, 1]))
    maturity = pool[C.KEYS].assign(label_end=["2026-01-30"]*3 + ["2026-02-02"]*3)
    train, excluded = C.mature_training(pool, maturity, "2026-01-30")
    assert len(train) == len(excluded) == 3
    assert train[C.P2].isna().sum() == 1
    assert len(pool) == 6


def solver_data(dates, features_count=1):
    f = pd.DataFrame(dict(signal_date=np.repeat(dates, 4), code=["001", "002", "003", "004"]*len(dates),
        s1_score=[.4, .3, .2, .19]*len(dates)))
    for j in range(features_count):
        f[f"z{j}"] = 0.
    return f


def solve(f, labels, **overrides):
    args = dict(l1_bound=.1, max_active_features=1, boundary_margin=1e-5,
                robust_lambda=0., time_limit=5., feature_columns=[c for c in f if c.startswith("z")])
    args.update(overrides)
    return M.solve_joint_top3(f, f[C.KEYS].assign(**{C.P2: labels}), **args)


def selected_hits(result, f, labels):
    return int(result["selections"].merge(f[C.KEYS].assign(**{C.P2: labels}), on=C.KEYS)[C.P2].eq(1).sum())


def test_integer_activation_enforces_real_sparsity_and_joint_top3():
    f = solver_data(["2026-01-05", "2026-01-06"], 2)
    f.loc[3, "z0"] = 1.; f.loc[7, "z1"] = 1.
    labels = [1, 1, 0, 1]*2
    sparse = solve(f, labels)
    both = solve(f, labels, max_active_features=2)
    assert selected_hits(sparse, f, labels) == 5
    assert selected_hits(both, f, labels) == 6
    assert np.count_nonzero(np.abs(sparse["theta"]) > 1e-7) <= 1
    assert both["metadata"]["verified_against_actual_score_ranking"]
    assert both["metadata"]["proven_optimal"]


def test_squared_monthly_penalty_can_reject_gain_concentrated_in_one_month():
    f = solver_data(["2026-01-05", "2026-01-06", "2026-01-07", "2026-02-02"])
    f.loc[f.code.eq("004"), "z0"] = 1.
    labels = [1, 0, 0, 1]*3 + [1, 1, 1, 0]
    a = solve(f, labels)
    b = solve(f, labels, robust_lambda=5.)
    assert selected_hits(a, f, labels) == 8
    assert selected_hits(b, f, labels) == 6
    assert b["metadata"]["minimized_objective"] == pytest.approx(-2/3)
    assert [r["slots"] for r in b["metadata"]["monthly_baseline"]] == [9, 3]


def test_equal_scores_cannot_be_resolved_using_future_winners():
    f = solver_data(["2026-01-05"]).assign(s1_score=.2)
    result = solve(f, [0, 1, 1, 1])
    assert result["theta"] is None
    assert result["metadata"]["status"] == 2
    assert not result["metadata"]["proven_optimal"]


def test_unknown_outcomes_remain_candidates_and_slots_are_not_shrunk():
    f = solver_data(["2026-01-05"])
    result = solve(f, [1, 1, np.nan, 0])
    assert result["metadata"]["training_rows"] == 4
    assert result["metadata"]["unknown_training_outcomes"] == 1
    assert result["metadata"]["monthly_baseline"][0]["slots"] == 3
    assert len(result["selections"]) == 3


def test_timeout_without_incumbent_is_reported_not_called_optimal(monkeypatch):
    monkeypatch.setattr(M, "milp", lambda *a, **k: SimpleNamespace(status=1, message="Time limit", success=False,
        x=None, fun=None, mip_gap=None, mip_dual_bound=None))
    result = solve(solver_data(["2026-01-05"]), [1, 0, 0, 1])
    assert result["theta"] is None
    assert not result["metadata"]["proven_optimal"]
