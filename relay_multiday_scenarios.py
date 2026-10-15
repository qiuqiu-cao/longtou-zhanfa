"""Evaluate archived qualitative calls as distinct, official-session events.

No probability model, calibration, actual fill, or portfolio return is claimed.
Future windows stay pending; absent elapsed-window facts remain unknown.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse

import pandas as pd

import short_term_thinking_dataset_builder as b

EVENTS={"next_close_limit_up":"T+1 close equals qualified cap",
    "next_intraday_touch":"T+1 high reaches qualified cap; closing is separate",
    "cumulative_up":"T+h unadjusted cash close > T close",
    "every_day_up":"Every official session T+1..T+h closes above its immediately preceding close",
    "consecutive_limit_ups":"Every official session T+1..T+h closes at the qualified cap"}


def evaluate(record,bars,sessions,asof):
    sessions=list(sessions)
    if sessions!=sorted(set(sessions)):raise ValueError("unique sorted official sessions required")
    now=pd.Timestamp(asof)
    if now.tzinfo is None:raise ValueError("actual timezone-aware asof required")
    if now.to_pydatetime()>datetime.now(timezone.utc):raise ValueError("future asof cannot manufacture mature results")
    local=now.tz_convert("Asia/Shanghai");day=local.strftime("%Y-%m-%d")
    t=sessions.index(record["signal_date"])
    if t+5>=len(sessions):raise ValueError("complete official five-session future calendar required")
    relevant=set(sessions[t:t+6]);x=bars[bars.date.isin(relevant)&bars.date.le(day)].copy()
    x["code"]=x.code.astype(str).str.zfill(6)
    if x.duplicated(["date","code"]).any():raise ValueError("duplicate dated price")
    prices={(r["date"],r["code"]):r for r in x.to_dict("records")}
    def traded(row):
        if row is None:return False
        if row.get("is_suspended") is True or str(row.get("trade_status","")).upper()=="SUSPENDED":return False
        return b.finite(row.get("close")) is not None and row["close"]>0 and b.finite(row.get("volume")) is not None and row["volume"]>0
    def cap_event(previous,current,touch=False):
        if not traded(previous) or not traded(current):return None
        shape=b.price_state(current)
        if shape["one_word"] is None:return None
        if b.finite(current.get("pre_close")) is not None and b.cents(current["pre_close"])!=b.cents(previous["close"]):return None
        upper=b.cap_cents(previous["close"])
        if b.cents(current["high"])>upper:return None
        return b.cents(current["high"] if touch else current["close"])==upper
    def maturity(h):
        target=sessions[t+h]
        return target<day or (target==day and local.hour>=15)
    result=[]
    for stock in record["rows"]:
        code=stock["code"];signal=prices.get((record["signal_date"],code))
        cases=[(1,"next_close_limit_up",stock["next_close_limit_up_call"]),(1,"next_intraday_touch",stock["next_intraday_touch_call"])]
        for h in [2,3,5]:
            for event in ["cumulative_up","every_day_up","consecutive_limit_ups"]:cases.append((h,event,stock["horizons"][str(h)][event+"_call"]))
        for h,event,call in cases:
            label=None;status="PENDING_OFFICIAL_WINDOW"
            if maturity(h):
                window=[prices.get((sessions[t+i],code)) for i in range(h+1)]
                status="UNKNOWN_MISSING_OR_UNQUALIFIED_PRICE"
                if event.startswith("next_"):
                    label=cap_event(window[0],window[1],event=="next_intraday_touch")
                elif event=="cumulative_up":
                    if traded(window[0]) and traded(window[-1]):label=window[-1]["close"]>window[0]["close"]
                elif event=="every_day_up":
                    if all(traded(row) for row in window):label=all(now["close"]>prior["close"] for prior,now in zip(window,window[1:]))
                else:
                    values=[cap_event(prior,now) for prior,now in zip(window,window[1:])]
                    if all(value is not None for value in values):label=all(values)
                if label is not None:status="KNOWN_ARCHIVE_10PCT_PRICE_PROXY_NOT_EXCHANGE_ATTESTATION"
            predicted=True if call=="YES" else False if call=="NO" else None
            result.append(dict(signal_date=record["signal_date"],code=code,name=stock["name"],horizon=h,target_date=sessions[t+h],event=event,
                original_call=call,predicted=predicted,actual=label,status=status,correct=(predicted==label) if predicted is not None and label is not None else None))
    return dict(formed_at=b.now(),asof=asof,rows=result,events=EVENTS,
        source_scope="BOUND_UNADJUSTED_PRICE_AND_ASSUMED_MAINBOARD10_NOT_EXCHANGE_CERTIFIED",
        probabilities_calibrated=False,new_mature_labels=sum(x["actual"] is not None for x in result),new_fits=0,actual_fills=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--record",required=True);parser.add_argument("--bundle",required=True);parser.add_argument("--asof",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    import joblib
    record=b.read(args.record);b.verify(record.get("source_bindings",{}));bundle=joblib.load(args.bundle);b.verify(bundle["sources"])
    result=evaluate(record,bundle["bars"],bundle["sessions"],args.asof)
    result["source_bindings"]={str(Path(p).resolve()):b.sha(p) for p in [args.record,args.bundle,__file__]}
    b.write_new(args.output,result);print(dict(events=len(result["rows"]),known=result["new_mature_labels"],fits=0))


if __name__=="__main__":main()
