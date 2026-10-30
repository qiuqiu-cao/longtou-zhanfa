from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest
import first_board_joint_seal_model as model

def stamp(time,date="2025-01-02"):
    return int(datetime.fromisoformat(date+"T"+time).replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())

def test_first_clock_auction_lunch_and_close_are_distinct():
    assert model.first_minutes(stamp("09:25:00"),"2025-01-02")==-5
    assert model.first_minutes(stamp("11:30:00"),"2025-01-02")==120
    assert model.first_minutes(stamp("13:00:00"),"2025-01-02")==120
    assert model.first_minutes(stamp("15:00:00"),"2025-01-02")==240
    assert np.isnan(model.first_minutes(stamp("12:00:00"),"2025-01-02"))
    assert np.isnan(model.first_minutes(stamp("15:00:01"),"2025-01-02"))
    assert np.isnan(model.first_minutes(stamp("09:35:00"),"2025-01-03"))

def test_penny_conflicts_and_noncent_prices_are_not_tolerated():
    assert model.exact_cents(11.84)==1184
    assert model.exact_cents(11.85)!=model.exact_cents(11.84)
    assert model.exact_cents(11.841) is None
    assert model.exact_cents(None) is None

def source():
    return pd.DataFrame({"signal_date":["2025-01-02"]*4,"code":["000001","000002","000003","000004"],model.TIME[0]:[5.,5.,15.,np.nan],"reported_close_seal_yuan":[100.,100.,np.nan,100.],"amount":[1000.,0.,1000.,1000.],"float_cap":[10000.]*4})

def test_full_pool_rank_preserves_ties_and_unknowns_without_invented_quotes():
    out=model.derive_fields(source())
    assert out.loc[0,model.TIME[1]]==.5 and out.loc[1,model.TIME[1]]==.5
    assert out.loc[2,model.TIME[1]]==1.
    assert np.isnan(out.loc[3,model.TIME[1]])
    assert np.isnan(out.loc[1,model.FUND[0]]) and np.isnan(out.loc[2,model.FUND[0]])

def test_future_dates_and_future_labels_cannot_change_past_fields():
    old=source();future=old.copy();future["signal_date"]="2025-01-03";future["code"]=future.code+"x"
    all_rows=pd.concat([old,future],ignore_index=True);all_rows["promotion_next_LABEL_ONLY"]=np.arange(8)
    a=model.derive_fields(old)[model.TIME+model.FUND]
    b=model.derive_fields(all_rows).loc[:3,model.TIME+model.FUND]
    pd.testing.assert_frame_equal(a,b)

def test_sparse_observed_pool_keeps_absolute_time_but_withholds_relative_rank():
    rows=source().iloc[:2].copy()
    out=model.derive_fields(rows)
    assert out[model.TIME[0]].notna().all()
    assert out[model.TIME[1]].isna().all()

def test_duplicate_keys_rejected_and_new_arms_have_no_labels_or_unverified_fields():
    with pytest.raises(ValueError,match="Duplicate"):
        model.derive_fields(pd.concat([source(),source().iloc[:1]]))
    assert len(model.ARMS["SEAL_TIME_64"])==64 and len(model.ARMS["SEAL_TIME_FUND_66"])==66
    for fields in model.ARMS.values():
        assert len(fields)==len(set(fields))
        assert all("LABEL_ONLY" not in f and "max_seal" not in f and "theme" not in f for f in fields)
