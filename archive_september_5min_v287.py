"""Collect every September relay symbol's recent five-minute public history.

Requests are derived from the full already-locked cohort without outcome
selection. Per-signal OHLC/volume checks do not certify tick paths or fills.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import public_minute_capture_v1 as minute


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_times(day):
    return set(pd.date_range(day+" 09:35",day+" 11:30",freq="5min").strftime("%H:%M")) | \
        set(pd.date_range(day+" 13:05",day+" 15:00",freq="5min").strftime("%H:%M"))


def check_signal(frame, reference, code, day):
    rows=frame[frame.minute.str[:10].eq(day)].sort_values("minute").copy() if len(frame) else frame
    symbol=("sz" if code.startswith("00") else "sh")+code
    daily=reference[reference.symbol.eq(symbol)&reference.date.eq(day)]
    result=dict(code=code,date=day,interval_minutes=5,observed_bars=len(rows),
        status="UNAVAILABLE",price_reconciled=False,grid_complete=False,
        missing_times=sorted(expected_times(day)),volume_reconciled=False,
        vwap_observed_bars=None,eligible_for_original_forward_forecast=False,
        auction_verified=False,tick_path_verified=False,actual_fill_verified=False)
    if rows.empty or len(daily)!=1:return result
    daily=daily.iloc[0]
    actual_times=set(rows.minute.str[11:16]);missing=sorted(expected_times(day)-actual_times)
    extra=sorted(actual_times-expected_times(day))
    duplicate=bool(rows.minute.duplicated().any())
    bounds=rows.low.gt(0)&rows.high.ge(rows[["open","low","close"]].max(axis=1)) \
        &rows.low.le(rows[["open","high","close"]].min(axis=1)) \
        &np.isfinite(rows[["open","high","low","close"]]).all(axis=1)
    observed=dict(open=float(rows.open.iloc[0]),close=float(rows.close.iloc[-1]),
                  high=float(rows.high.max()),low=float(rows.low.min()))
    agreement={key:abs(value-float(daily[key]))<=.005001 for key,value in observed.items()}
    price_ok=all(agreement.values()) and bool(bounds.all()) and not extra and not duplicate
    volume_known=bool(rows.volume_shares.notna().all()&rows.volume_shares.ge(0).all())
    volume=float(rows.volume_shares.sum())
    volume_ok=volume_known and daily.volume>0 and abs(volume-float(daily.volume)*100)<=100
    traded=rows[rows.volume_shares.gt(0)]
    implied=traded.amount/traded.volume_shares
    unit_ok=bool(len(traded) and rows.amount.notna().all() and rows.amount.ge(0).all()
                 and (implied.ge(traded.low-.020001)&implied.le(traded.high+.020001)).all())
    result.update(status="PRICE_RECONCILED" if price_ok else "RECONCILIATION_FAILED",
        price_reconciled=price_ok,grid_complete=not missing and not extra and not duplicate,
        missing_times=missing,unexpected_times=extra,duplicate_times=duplicate,
        ohlc_agreement=agreement,observed_ohlc=observed,all_bar_bounds_valid=bool(bounds.all()),
        volume_reconciled=bool(volume_ok),observed_volume=volume,
        raw_volume_to_daily_scale_inferred=volume/float(daily.volume) if daily.volume>0 else None,
        amount_volume_unit_consistent=unit_ok,
        vwap_observed_bars=float(rows.amount.sum()/volume) if price_ok and volume_ok and unit_ok and volume>0 else None)
    return result


def fetch(code,target):
    started=datetime.now().astimezone().isoformat()
    item=dict(code=code,requested_at=started,interval_minutes=5,requested_bar_count=1023)
    empty=pd.DataFrame(columns=["code","minute","open","high","low","close","volume_shares","amount"])
    try:
        response=requests.get(minute.SINA_MINUTE_URL,
            params=dict(symbol=minute.live.sina_symbol(code),scale="5",ma="no",datalen="1023"),
            headers=minute.SINA_HEADERS,timeout=8)
        path=target/(code+".response");path.write_bytes(response.content)
        item.update(url=response.url,http_status=response.status_code,raw_path=str(path.resolve()),
            raw_sha256=sha(path),available_at=datetime.now().astimezone().isoformat())
        response.raise_for_status()
        raw=minute._extract_json_array(response.text)
        timestamps=pd.Series([r.get("day") for r in raw],dtype="object")
        item.update(raw_rows=len(raw),duplicates_before_parser=int(timestamps.duplicated().sum()),
                    invalid_raw_times=int(pd.to_datetime(timestamps,errors="coerce").isna().sum()))
        if item["duplicates_before_parser"] or item["invalid_raw_times"]:
            raise ValueError("raw duplicate or invalid timestamps")
        frame=minute.parse_minute_response(response.text,code)
        item.update(parsed_rows=len(frame),first_bar=frame.minute.min() if len(frame) else None,
                    last_bar=frame.minute.max() if len(frame) else None,
                    status="RAW_PARSED" if len(frame) else "EMPTY_NOT_COMPLETE")
        return item,frame if len(frame) else empty
    except Exception as exc:
        item.update(status="UNAVAILABLE",error=str(exc)[:250])
        return item,empty


def run(cohort_path,daily_path,output):
    cohort=pd.read_csv(cohort_path,dtype={"code":str},low_memory=False)
    if cohort.duplicated(["date","code"]).any() or not cohort.date.str.startswith("2026-09-").all():
        raise ValueError("unique full September candidate cohort required")
    if not cohort.code.str.fullmatch(r"(?:00|60)\d{4}").all() or not cohort.boards.ge(2).all():
        raise ValueError("mainboard relay candidates required")
    reference=pd.read_csv(daily_path)
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    codes=sorted(cohort.code.unique())
    (target/"request_scope.json").write_text(json.dumps(dict(candidate_rows=len(cohort),
        dates=int(cohort.date.nunique()),codes=codes,interval_minutes=5,bar_limit=1023,
        cohort_sha256=sha(cohort_path),outcomes_used_to_select_codes=False),ensure_ascii=False,indent=2),encoding="utf-8")
    audits,frames=[],[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs=[executor.submit(fetch,code,target) for code in codes]
        for job in as_completed(jobs):
            audit,frame=job.result();audits.append(audit)
            if len(frame):frames.append(frame)
            if len(audits)%20==0:print(json.dumps(dict(completed_symbols=len(audits),total_symbols=len(codes))),flush=True)
    bars=pd.concat(frames,ignore_index=True).sort_values(["code","minute"]).reset_index(drop=True) if frames else \
        pd.DataFrame(columns=["code","minute","open","high","low","close","volume_shares","amount"])
    path=target/"five_minute_bars_UNRECONCILED.csv";bars.to_csv(path,index=False)
    groups={code:frame for code,frame in bars.groupby("code")}
    empty=bars.iloc[:0]
    checks=[check_signal(groups.get(row.code,empty),reference,row.code,row.date) for row in cohort.itertuples()]
    check_path=target/"signal_checks.json"
    check_path.write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    valid={r["code"] for r in audits if r.get("status")=="RAW_PARSED"}
    report=dict(version="v287-full-september-five-minute-1",recorded_at=datetime.now().astimezone().isoformat(),
        status="RETROSPECTIVE_PATH_DATA_NOT_FORWARD_EVIDENCE",candidate_rows=len(cohort),
        candidate_dates=int(cohort.date.nunique()),requested_symbols=len(codes),available_symbols=len(valid),
        unavailable_symbols=sorted(set(codes)-valid),raw_rows=len(bars),interval_minutes=5,
        price_reconciled_rows=sum(r["price_reconciled"] for r in checks),
        full_grid_rows=sum(r["grid_complete"] for r in checks),
        volume_reconciled_rows=sum(r["volume_reconciled"] for r in checks),
        no_missing_rows_imputed=True,outcome_selection_used=False,original_forecast_rewritten=False,
        accuracy_claim_allowed=False,trade_authorized=False,audit=sorted(audits,key=lambda r:r["code"]),
        source_hashes={str(Path(p).resolve()):sha(p) for p in (cohort_path,daily_path,Path(__file__),Path(minute.__file__))},
        artifact_hashes={p.name:sha(p) for p in (path,check_path,target/"request_scope.json")},
        scope="Original provider-reported September candidate rows; no exchange-wide completeness certificate",
        limitations=["Five-minute bars do not reconstruct one-minute or tick/auction paths",
            "Source window may start mid-session; missing intervals are not zero or unchanged prices",
            "OHLC and volume-unit inference are separately checked against existing daily reference",
            "Previously viewed outcomes; finite September sample cannot establish stable sixty percent"])
    (target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {k:report[k] for k in ("status","candidate_rows","candidate_dates","requested_symbols","available_symbols",
        "raw_rows","price_reconciled_rows","full_grid_rows","volume_reconciled_rows","unavailable_symbols")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("cohort","daily","output"):parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.cohort,args.daily,args.output),ensure_ascii=True))
