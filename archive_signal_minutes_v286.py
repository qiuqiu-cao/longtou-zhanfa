"""Archive the exact frozen relay pool's recent public minute bars.

Reception is after the original forecast. OHLC agreement, observed VWAP and
bar transitions cannot establish auction facts, tick paths, queues or fills.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
import public_minute_capture_v1 as minute
from next_session_expectation_contract_v271 import load_registered


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_minutes(day):
    # Explicit continuous-session and closing-auction representation expected
    # by this feed. Missing entries remain missing, including zero-volume gaps.
    return set(pd.date_range(day+" 09:31",day+" 11:30",freq="min").strftime("%H:%M")) | \
        set(pd.date_range(day+" 13:01",day+" 14:57",freq="min").strftime("%H:%M")) | {"15:00"}


def reconcile(bars, daily, day, code):
    frame = bars[bars.minute.str[:10].eq(day)].sort_values("minute").copy()
    symbol = ("sz" if code.startswith("00") else "sh") + code
    ref = daily[daily.symbol.eq(symbol) & daily.date.eq(day)]
    result = dict(code=code, signal_date=day, observed_bars=len(frame),
        status="UNAVAILABLE", missing_minutes=sorted(expected_minutes(day)),
        auction_0925_available=False, tick_absorption_verified=False,
        exact_reseal_time_verified=False, actual_fill_verified=False,
        eligible_for_original_frozen_forecast=False)
    if frame.empty or len(ref) != 1:
        return result
    ref = ref.iloc[0]
    times = set(frame.minute.str[11:16])
    missing = sorted(expected_minutes(day)-times)
    extra = sorted(times-expected_minutes(day))
    numeric = ["open","high","low","close","volume_shares","amount"]
    missing_fields = [key for key in numeric if frame[key].isna().any()]
    finite_prices = frame[["open","high","low","close"]].notna().all(axis=1)
    valid_prices = finite_prices & frame.high.ge(frame[["open","close","low"]].max(axis=1)) \
        & frame.low.le(frame[["open","close","high"]].min(axis=1)) & frame.low.gt(0)
    ohlc = dict(open=float(frame.open.iloc[0]), close=float(frame.close.iloc[-1]),
                high=float(frame.high.max()), low=float(frame.low.min()))
    agreement = {key:abs(value-float(ref[key]))<=.005001 for key,value in ohlc.items()}
    volume = float(frame.volume_shares.sum())
    # This factor is inferred from cross-provider consistency, not silently
    # asserted as official unit documentation. Volume rounding is preserved.
    scale = volume/float(ref.volume) if ref.volume>0 else None
    volume_agrees_at_inferred_scale_100 = bool(scale is not None and abs(volume-float(ref.volume)*100)<=100)
    traded = frame[frame.volume_shares.gt(0) & frame.amount.notna()]
    price_per_raw_volume = traded.amount/traded.volume_shares
    unit_consistency = bool(len(traded) and
        (price_per_raw_volume.ge(traded.low-.020001)&price_per_raw_volume.le(traded.high+.020001)).all())
    duplicates = bool(frame.minute.duplicated().any())
    valid = all(agreement.values()) and bool(valid_prices.all()) and not extra and not duplicates
    result.update(status="PRICE_RECONCILED_MINUTE_GAPS_RETAINED" if valid else "RECONCILIATION_FAILED",
        missing_minutes=missing, unexpected_minutes=extra, duplicate_minutes=duplicates,
        minute_grid_complete=not missing and not extra and not duplicates,
        missing_numeric_fields=missing_fields, all_bar_price_bounds_valid=bool(valid_prices.all()),
        daily_ohlc_agreement=agreement, observed_ohlc=ohlc,
        observed_volume_sum=volume, volume_to_daily_raw_scale_inferred=scale,
        volume_agrees_at_inferred_scale_100=volume_agrees_at_inferred_scale_100,
        amount_volume_implied_price_within_bar_bounds=unit_consistency,
        vwap_over_observed_bars=(float(frame.amount.sum()/volume)
            if valid and volume>0 and unit_consistency and "amount" not in missing_fields else None),
        vwap_scope="OBSERVED_MINUTE_BARS_ONLY_MISSING_BARS_NOT_IMPUTED")
    return result


def run(registered_path, daily_path, output, bar_limit=1023):
    frozen = load_registered(registered_path)
    codes = [r["code"] for r in frozen["plan"]["candidates"]]
    if len(codes)!=len(set(codes)) or not 240<=bar_limit<=1023:
        raise ValueError("unique complete relay pool and bounded minute request required")
    day = frozen["plan"]["signal_date"]
    daily = pd.read_csv(daily_path)
    target = Path(output)
    target.mkdir(parents=True,exist_ok=False)
    responses = {}
    def get(url, **kwargs):
        symbol = kwargs["params"]["symbol"]
        started = datetime.now().astimezone().isoformat()
        response = requests.get(url,**kwargs)
        raw = target/(symbol+".response")
        raw.write_bytes(response.content)
        responses[symbol]=dict(url=response.url,http_status=response.status_code,
            request_started_at=started,available_at=datetime.now().astimezone().isoformat(),
            raw_path=str(raw.resolve()),sha256=sha(raw))
        return response
    fetched=minute.fetch_minute_bars(codes,get=get,workers=3,timeout=8,retries=0,datalen=bar_limit)
    bars=fetched.bars
    if bars.empty:
        bars=pd.DataFrame(columns=["code","minute","open","high","low","close","volume_shares","amount"])
    bars_path=target/"minute_bars_UNRECONCILED.csv"
    bars.to_csv(bars_path,index=False)
    checks=[reconcile(bars[bars.code.eq(code)],daily,day,code) for code in codes]
    report=dict(version="v286-post-freeze-minute-source-1",recorded_at=datetime.now().astimezone().isoformat(),
        status="RETROSPECTIVE_MINUTE_EVIDENCE_NOT_A_FORECAST",registration_hash=frozen["sha256"],
        signal_date=day,target_date=frozen["plan"]["target_date"],candidate_denominator=len(codes),
        request_bar_limit=bar_limit,parsed_rows=len(bars),responses=responses,errors=fetched.errors,
        candidate_checks=checks,price_reconciled_candidates=sum(r["status"]=="PRICE_RECONCILED_MINUTE_GAPS_RETAINED" for r in checks),
        minute_grid_complete_candidates=sum(r.get("minute_grid_complete") is True for r in checks),
        original_forecast_rewritten=False,accuracy_claim_allowed=False,trade_authorized=False,
        source_hashes={str(Path(p).resolve()):sha(p) for p in [registered_path,daily_path,Path(__file__),Path(minute.__file__)]},
        artifact_hashes={bars_path.name:sha(bars_path)},
        limitations=["Post-freeze reception; no original-call revision or forward accuracy credit",
            "No 09:25 auction, tick/queue/withdrawal/actual-fill evidence",
            "Minute gaps and cross-provider volume-unit inference remain explicit",
            "Price agreement does not independently authenticate turnover or volume units",
            "Short recent window cannot establish stable sixty-percent predictive accuracy"])
    (target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {key:report[key] for key in ("status","candidate_denominator","parsed_rows","price_reconciled_candidates",
        "minute_grid_complete_candidates","errors","original_forecast_rewritten","accuracy_claim_allowed")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered",required=True)
    parser.add_argument("--daily",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.registered,args.daily,args.output),ensure_ascii=True))
