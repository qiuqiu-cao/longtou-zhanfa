"""Strict historical minute CSV import; explicit units and timestamp semantics."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

ALIASES = {"code":["code","代码","股票代码"],"date":["date","日期"],"time":["time","时间"],
           "open":["open","开盘"],"high":["high","最高"],"low":["low","最低"],
           "close":["close","收盘"],"volume":["volume","成交量"],"amount":["amount","成交额"]}


def load_requested_cases(path):
    """Load the immutable provider-code/date request list used by V48/V49."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("requests must be a JSON list")
    cases = []
    for item in payload:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or not re.fullmatch(r"(?:sz|sh)\.\d{6}", str(item[0]))
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item[1]))):
            raise ValueError("invalid requested case")
        cases.append((str(item[0])[3:], str(item[1])))
    if len(cases) != len(set(cases)):
        raise ValueError("duplicate requested cases")
    return set(cases)


def audit_requested_coverage(frame, requested, interval):
    """Bind an import to a predeclared request set without inventing missing bars."""
    expected = 240 // interval
    observed = {(str(code).zfill(6), str(day)): len(group)
                for (code, day), group in frame.groupby(["code", "date"], sort=True)}
    unexpected = sorted(set(observed) - set(requested))
    if unexpected:
        raise ValueError("import contains groups outside the declared request set")
    complete = sorted(key for key, rows in observed.items() if rows == expected)
    incomplete = sorted(key for key in requested if key in observed and observed[key] != expected)
    missing = sorted(set(requested) - set(observed))
    return {
        "requested_groups": len(requested),
        "observed_groups": len(observed),
        "complete_groups": len(complete),
        "incomplete_groups": len(incomplete),
        "missing_groups": len(missing),
        "complete": [{"code": code, "date": day} for code, day in complete],
        "incomplete": [{"code": code, "date": day, "bars": observed[(code, day)],
                        "expected": expected} for code, day in incomplete],
        "missing": [{"code": code, "date": day} for code, day in missing],
    }


def normalize(frame, interval, timestamp_side, volume_unit, amount_unit):
    if interval not in (1,5) or timestamp_side not in ("start","end"):
        raise ValueError("explicit supported interval and timestamp side required")
    result = pd.DataFrame(index=frame.index)
    for key,aliases in ALIASES.items():
        matches = [a for a in aliases if a in frame]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous column: "+key)
        result[key] = frame[matches[0]]
    if result.empty:
        raise ValueError("empty import")
    def code(value):
        value = str(value).strip()
        if not re.fullmatch(r"\d{1,6}",value):
            raise ValueError("plain numeric stock code required")
        return value.zfill(6)
    result["code"] = result.code.map(code)
    if not result.code.str.match(r"(?:00|60)\d{4}$").all():
        raise ValueError("mainboard codes only; historical ST eligibility not inferred")
    ends = pd.to_datetime(result.date.astype(str)+" "+result.time.astype(str),errors="raise")
    if ends.isna().any() or ends.dt.tz is not None:
        raise ValueError("local Shanghai date/time columns required")
    if timestamp_side == "start":
        ends += pd.Timedelta(minutes=interval)
    minute = ends.dt.hour*60+ends.dt.minute
    allowed = (((minute >= 570+interval)&(minute <= 690))|((minute >= 780+interval)&(minute <= 900)))
    if not (allowed & minute.mod(interval).eq(0) & ends.dt.second.eq(0) & ends.dt.microsecond.eq(0)).all():
        raise ValueError("bar crosses session boundary, auction, or unsupported timestamp")
    result["date"] = ends.dt.strftime("%Y-%m-%d")
    result["time"] = ends.dt.strftime("%Y%m%d%H%M%S")+"000"
    if result.duplicated(["code","time"]).any():
        raise ValueError("duplicate minute bar")
    for key in ("open","high","low","close","volume","amount"):
        result[key] = pd.to_numeric(result[key],errors="raise")
    if volume_unit not in ("shares","lots") or amount_unit not in ("yuan","wan_yuan"):
        raise ValueError("explicit volume and amount units required")
    result["volume"] *= 100 if volume_unit == "lots" else 1
    result["amount"] *= 10000 if amount_unit == "wan_yuan" else 1
    values = result[["open","high","low","close","volume","amount"]]
    if not np.isfinite(values).all().all() or not result[["open","high","low","close"]].gt(0).all().all() or not result[["volume","amount"]].ge(0).all().all():
        raise ValueError("missing or invalid numeric data")
    if not (result.high.ge(result[["open","close","low"]].max(axis=1)) & result.low.le(result[["open","close","high"]].min(axis=1))).all():
        raise ValueError("inconsistent OHLC")
    if (result.volume.eq(0)&result.amount.ne(0)).any():
        raise ValueError("amount without volume")
    result["bar_vwap"] = (result.amount/result.volume).where(result.volume.gt(0))
    traded = result.volume.gt(0)
    result["amount_volume_consistent"] = (~traded)|result.bar_vwap.between(result.low-0.01,result.high+0.01)
    return result.sort_values(["code","time"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("--encoding",default="utf-8-sig")
    parser.add_argument("--interval",type=int,choices=[1,5],required=True)
    parser.add_argument("--timestamp-side",choices=["start","end"],required=True)
    parser.add_argument("--volume-unit",choices=["shares","lots"],required=True)
    parser.add_argument("--amount-unit",choices=["yuan","wan_yuan"],required=True)
    parser.add_argument("--adjustment",choices=["unadjusted"],required=True)
    parser.add_argument("--requests",type=Path,
                        help="optional immutable JSON list of [provider_code,date]; rejects extra groups")
    args = parser.parse_args()
    frame = normalize(pd.read_csv(args.input,dtype=str,encoding=args.encoding),args.interval,args.timestamp_side,args.volume_unit,args.amount_unit)
    coverage = [{"code":code,"date":day,"bars":len(group),"expected_full_session":240//args.interval,
                 "row_count_complete_only":len(group)==240//args.interval} for (code,day),group in frame.groupby(["code","date"])]
    request_audit = None
    if args.requests is not None:
        request_audit = audit_requested_coverage(frame, load_requested_cases(args.requests), args.interval)
    args.output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(args.output/"normalized.csv",index=False)
    manifest = dict(status="DIAGNOSTIC_ONLY",data_status="USER_IMPORTED_NOT_INDEPENDENTLY_VERIFIED",
                    source=args.source,imported_at=datetime.now().astimezone().isoformat(),
                    source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    normalized_sha256=hashlib.sha256((args.output/"normalized.csv").read_bytes()).hexdigest(),
                    interval=args.interval,input_timestamp_side=args.timestamp_side,volume_unit=args.volume_unit,amount_unit=args.amount_unit,
                    adjustment_declared=args.adjustment,coverage=coverage,
                    request_file_sha256=None if args.requests is None else hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                    request_audit=request_audit,
                    unit_consistency_failures=int((~frame.amount_volume_consistent).sum()),
                    observed_receipt_times_available=False,verified_fills=0,
                    note="historical reconstruction only; no auction, cancellation, queue or live-data certification")
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=True))


if __name__ == "__main__":
    main()
