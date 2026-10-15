import copy
import pandas as pd
import pytest
import relay_multiday_scenarios as m

DATES=pd.bdate_range("2024-01-01",periods=7).strftime("%Y-%m-%d").tolist()


def fixture():
    row=dict(code="000001",name="甲",next_close_limit_up_call="YES",next_intraday_touch_call="YES",horizons={str(h):{event+"_call":"YES" for event in ["cumulative_up","every_day_up","consecutive_limit_ups"]} for h in [2,3,5]})
    record=dict(signal_date=DATES[0],rows=[row])
    closes=[10.,11.,10.9,11.9,11.8,12.1]
    bars=pd.DataFrame([dict(date=d,code="000001",open=c,high=c,low=c,close=c,volume=100.) for d,c in zip(DATES,closes)])
    return record,bars


def test_cumulative_rise_not_everyday_rise_or_consecutive_caps():
    record,bars=fixture();out=m.evaluate(record,bars,DATES,DATES[5]+"T08:00:00Z")
    rows={(r["horizon"],r["event"]):r for r in out["rows"]}
    assert rows[(5,"cumulative_up")]["actual"] is True
    assert rows[(5,"every_day_up")]["actual"] is False
    assert rows[(5,"consecutive_limit_ups")]["actual"] is False


def test_future_and_preclose_windows_stay_pending_even_if_future_bars_supplied():
    record,bars=fixture();out=m.evaluate(record,bars,DATES,DATES[0]+"T08:00:00Z")
    assert len(out["rows"])==11 and out["new_mature_labels"]==0
    assert all(row["status"]=="PENDING_OFFICIAL_WINDOW" for row in out["rows"])
    out=m.evaluate(record,bars,DATES,DATES[1]+"T06:00:00Z")
    assert out["new_mature_labels"]==0


def test_elapsed_missing_intermediate_only_blocks_path_not_endpoint_price():
    record,bars=fixture();bars=bars[bars.date.ne(DATES[2])]
    out=m.evaluate(record,bars,DATES,DATES[5]+"T08:00:00Z")
    rows={(r["horizon"],r["event"]):r for r in out["rows"]}
    assert rows[(5,"cumulative_up")]["actual"] is True
    assert rows[(5,"every_day_up")]["actual"] is None
    assert rows[(5,"consecutive_limit_ups")]["actual"] is None


def test_touch_and_close_are_distinct_and_zero_volume_not_failure():
    record,bars=fixture();bars.loc[bars.date.eq(DATES[1]),["open","low","close"]]=10.8
    out=m.evaluate(record,bars,DATES,DATES[5]+"T08:00:00Z")
    assert out["rows"][0]["actual"] is False and out["rows"][1]["actual"] is True
    bars.loc[bars.date.eq(DATES[1]),"volume"]=0
    out=m.evaluate(record,bars,DATES,DATES[5]+"T08:00:00Z")
    assert out["rows"][0]["actual"] is None
    with pytest.raises(ValueError,match="timezone-aware"):m.evaluate(record,bars,DATES,DATES[5])
