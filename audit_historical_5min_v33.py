"""Check archived 5-minute coverage and daily OHLC consistency, no fill inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inspect(raw, daily):
    frame = pd.DataFrame(raw["rows"], columns=raw["fields"])
    day = raw["requested_date"]
    expected = ([day.replace("-", "")+t.strftime("%H%M%S")+"000"
                 for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                 for t in pd.date_range(day+" "+start, day+" "+end, freq="5min")])
    if frame.empty:
        return dict(code=raw["code"], date=day, status="EMPTY", fill_verified=False)
    if (raw["error_code"] != "0" or not frame.code.eq(raw["code"]).all()
            or not frame.date.eq(day).all() or not frame.adjustflag.eq("3").all()
            or frame.time.tolist() != expected):
        return dict(code=raw["code"], date=day, status="INVALID_IDENTITY_OR_COVERAGE", fill_verified=False)
    for field in ("open", "high", "low", "close", "volume", "amount"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    values = frame[["open", "high", "low", "close", "volume", "amount"]]
    valid = (np.isfinite(values).all().all() and frame[["open", "high", "low", "close"]].gt(0).all().all()
             and frame[["volume", "amount"]].ge(0).all().all()
             and frame.high.ge(frame[["open", "close", "low"]].max(axis=1)).all()
             and frame.low.le(frame[["open", "close", "high"]].min(axis=1)).all())
    if not valid:
        return dict(code=raw["code"], date=day, status="INVALID_OHLCV", fill_verified=False)
    aggregate = dict(open=float(frame.iloc[0].open), high=float(frame.high.max()),
                     low=float(frame.low.min()), close=float(frame.iloc[-1].close))
    differences = {field: dict(minute=value, daily=float(daily[field])) for field, value in aggregate.items()
                   if abs(value-float(daily[field])) > .000001}
    # Daily OHLC can legitimately have a wider high/low range because the
    # archived 5-minute series starts at 09:35 and does not prove auction
    # coverage.  Equal open/close plus a minute range contained by the daily
    # range is consistent evidence, not a contradiction.
    endpoints_match = (abs(aggregate["open"] - float(daily["open"])) <= .000001
                       and abs(aggregate["close"] - float(daily["close"])) <= .000001)
    range_contained = (aggregate["high"] <= float(daily["high"]) + .000001
                       and aggregate["low"] >= float(daily["low"]) - .000001)
    if not differences:
        status = "DAILY_OHLC_MATCH_ONLY"
    elif endpoints_match and range_contained:
        status = "DAILY_OHLC_RANGE_CONSISTENT"
    else:
        status = "DAILY_OHLC_CONTRADICTION"
    return dict(code=raw["code"], date=day, rows=len(frame),
                status=status,
                aggregate=aggregate, differences=differences,
                opening_bars=frame.head(6).to_dict("records"),
                exact_seal_time=None, auction_verified=False, fill_verified=False,
                timestamp_note="provider 5-minute stamps from 09:35 to 15:00; no intrabar event ordering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bars = pd.read_csv(args.daily, dtype={"code": str}, usecols=["code", "date", "open", "high", "low", "close"])
    results = []
    for path in sorted(list(args.archive.glob("sz.*.json")) + list(args.archive.glob("sh.*.json"))):
        raw = json.loads(path.read_text(encoding="utf-8"))
        daily = bars[(bars.code == raw["code"].split(".")[1]) & (bars.date == raw["requested_date"])]
        if len(daily) != 1:
            raise ValueError("daily comparator absent or duplicated")
        results.append(inspect(raw, daily.iloc[0]))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(dict(status="DIAGNOSTIC_ONLY", cases=results, verified_fills=0), handle,
                  ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps([{k:v for k,v in r.items() if k != "opening_bars"} for r in results], indent=2))


if __name__ == "__main__":
    main()
