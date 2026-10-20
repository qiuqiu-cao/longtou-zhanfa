import numpy as np
import pandas as pd
import pytest
import first_board_regularized_ordered as learning


def fixture():
    return pd.DataFrame(dict(signal_date=["2024-01-03","2024-01-02","2024-01-03","2024-01-02","2024-01-03"],
        label_end=["2024-01-04","2024-01-03","2024-01-04","2024-01-03","2024-01-04"],
        code=["000003","000002","000001","000001","000002"],stage=1,promotion_next_LABEL_ONLY=[1.,0.,0.,0.,1.]))


def test_training_order_and_day_class_weights_use_only_current_members():
    rows,weights,classes=learning.prepare_train(fixture(),"2024-01-05")
    assert rows.code.tolist()==["000001","000002","000001","000002","000003"]
    assert rows.signal_date.tolist()==["2024-01-02"]*2+["2024-01-03"]*3
    assert weights[:2].sum()==pytest.approx(weights[2:].sum())
    assert weights.mean()==pytest.approx(1) and (weights>0).all()
    np.testing.assert_allclose(classes,[5/6,5/4])


def test_boundary_day_target_cannot_be_used_as_matured_training():
    with pytest.raises(ValueError,match="strictly mature"):
        learning.prepare_train(fixture(),"2024-01-04")


def test_unknown_labels_other_stage_and_duplicate_keys_are_rejected():
    rows=fixture()
    unknown=rows.copy();unknown.loc[0,"promotion_next_LABEL_ONLY"]=np.nan
    with pytest.raises(ValueError,match="known binary"):
        learning.prepare_train(unknown,"2024-01-05")
    with pytest.raises(ValueError,match="stage1"):
        learning.prepare_train(rows.assign(stage=2),"2024-01-05")
    with pytest.raises(ValueError,match="Duplicate"):
        learning.prepare_train(pd.concat([rows,rows.iloc[:1]],ignore_index=True),"2024-01-05")
