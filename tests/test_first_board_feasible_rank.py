import numpy as np
import pandas as pd
import pytest

import first_board_feasible_rank as r


def test_contiguous_queries_and_equal_total_day_weights_preserve_constant_groups():
    rows=pd.DataFrame(dict(signal_date=["2024-01-03","2024-01-02","2024-01-03","2024-01-02","2024-01-03"],
        code=["000003","000002","000001","000001","000002"],
        promotion_next_LABEL_ONLY=[0.,1.,0.,1.,0.],research_selection_allowed=True))
    ordered,groups,weights=r.daily_queries(rows)
    assert groups.tolist()==[2,3]
    assert ordered.code.tolist()==["000001","000002","000001","000002","000003"]
    assert ordered.signal_date.tolist()==["2024-01-02"]*2+["2024-01-03"]*3
    # All-positive and all-negative queries stay in the membership ledger.
    assert ordered.groupby("signal_date").promotion_next_LABEL_ONLY.sum().tolist()==[2.,0.]
    assert weights[:2].sum()==pytest.approx(weights[2:].sum())
    assert np.all(weights>0) and weights.mean()==pytest.approx(1)


def test_label_maturity_signal_boundary_window_and_screen_are_all_applied():
    rows=pd.DataFrame(dict(signal_date=["2024-12-18","2024-12-19","2024-12-18","2024-12-18","2024-12-18"],
        code=["000001","000002","000003","000004","000005"],label_end=["2024-12-19","2024-12-20","2024-12-19","2024-12-19","2024-12-19"],
        p=[30,31,30,30,25],promotion_next_LABEL_ONLY=[1.,0.,1.,np.nan,1.],research_selection_allowed=[True,True,False,True,True]))
    selected=r.training_rows(rows,"2024-12-20",start_p=29)
    assert selected.code.tolist()==["000001"]
    # The boundary-day label cannot enter merely because its signal is earlier.
    assert "000002" not in selected.code.tolist()


def test_unknown_labels_excluded_stocks_and_duplicate_members_cannot_form_queries():
    rows=pd.DataFrame(dict(signal_date=["2024-01-02"],code=["000001"],promotion_next_LABEL_ONLY=[1.],research_selection_allowed=True))
    with pytest.raises(ValueError,match="known binary"):
        r.daily_queries(rows.assign(promotion_next_LABEL_ONLY=np.nan))
    with pytest.raises(ValueError,match="Excluded"):
        r.daily_queries(rows.assign(research_selection_allowed=False))
    with pytest.raises(ValueError,match="Duplicate"):
        r.daily_queries(pd.concat([rows,rows],ignore_index=True))
