"""Auditable opening minute measurements, no tick-order or execution inference."""
import hashlib
import json
from datetime import timedelta
from leader_target_screen_v17 import timestamp, finite


def _window(rows, code, trade_date, asof, minutes):
    cutoff=timestamp(asof)
    start=timestamp(trade_date+"T09:30:00+08:00")
    expected=[start+timedelta(minutes=i) for i in range(1,minutes+1)]
    chosen={}
    for row in rows:
        if row.get("code")!=code:
            continue
        end=timestamp(row["bar_end"])
        if end not in expected:
            continue
        if end in chosen:
            raise ValueError("duplicate window bar")
        chosen[end]=row
    if set(chosen)!=set(expected):
        raise ValueError(f"complete {minutes}-minute opening window required")
    ordered=[chosen[t] for t in expected]
    volume=amount=0.
    vwaps=[]
    availability=[]
    for t,row in zip(expected,ordered):
        if (row.get("volume_unit")!="shares" or row.get("amount_unit")!="yuan"
                or row.get("interval_seconds")!=60 or row.get("timestamp_side")!="end"):
            raise ValueError("explicit per-minute shares/yuan and end timestamps required")
        if row.get("data_status")!="verified" or not row.get("source"):
            raise ValueError("verified source declaration required")
        known=timestamp(row["available_at"])
        if not t<=known<=cutoff:
            raise ValueError("bar unavailable at decision time")
        vals=[finite(row.get(k)) for k in ("open","high","low","close","volume","amount")]
        if any(v is None for v in vals):
            raise ValueError("missing numeric field")
        o,h,l,c,v,a=vals
        if min(o,h,l,c)<=0 or v<0 or a<0 or l>min(o,c) or h<max(o,c):
            raise ValueError("invalid OHLC or volume")
        if (v==0)!=(a==0) or (v>0 and not l-.011<=a/v<=h+.011):
            raise ValueError("amount/volume inconsistent with minute prices")
        volume+=v
        amount+=a
        vwaps.append(amount/volume if volume else None)
        availability.append(known)
    if vwaps[0] is None or vwaps[-1] is None:
        raise ValueError("opening and final cumulative VWAP require trades")
    digest=hashlib.sha256(json.dumps(ordered,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    base=dict(unit="ratio",window_seconds=minutes*60,observed_at=expected[-1].isoformat(),
              available_at=max(availability).isoformat(),data_status="verified",
              source="derived_from_supplied_minute_rows",raw_sha256=digest,
              source_verification="supplier_declared",execution_verified=False)
    return ordered, vwaps, digest, base


def derive(rows,code,trade_date,asof):
    ordered,vwaps,digest,base=_window(rows,code,trade_date,asof,5)
    values={"opening_vwap_change_5m":vwaps[-1]/vwaps[0]-1,
            "opening_close_vs_vwap_5m":float(ordered[-1]["close"])/vwaps[-1]-1}
    return {metric:dict(base,metric=metric,value=value,definition_id=metric+"_v1",
                        evidence_id=digest+":"+metric) for metric,value in values.items()}


def derive_absorption(rows,code,trade_date,asof):
    """09:31--09:45 bar proxies; within-bar event order remains unknown."""
    ordered,vwaps,digest,base=_window(rows,code,trade_date,asof,15)
    limits=[finite(row.get("limit_price")) for row in ordered]
    if any(v is None or v<=0 for v in limits) or max(limits)-min(limits)>.000001:
        raise ValueError("one explicit constant limit price is required")
    limit_price=limits[0]
    tolerance=.00001
    if any(float(row["high"])>limit_price+.011 or float(row["close"])>limit_price+.011 for row in ordered):
        raise ValueError("minute price exceeds supplied limit")
    previous=[float(ordered[0]["open"]),*[float(row["close"]) for row in ordered[:-1]]]
    falling=[float(row["close"])<prior for row,prior in zip(ordered,previous)]
    total=sum(float(row["volume"]) for row in ordered)
    pullback=sum(float(row["volume"]) for row,down in zip(ordered,falling) if down)
    touched=[float(row["high"])>=limit_price-tolerance for row in ordered]
    failed=[touch and float(row["close"])<limit_price-tolerance for row,touch in zip(ordered,touched)]
    episodes=sum(flag and (i==0 or not failed[i-1]) for i,flag in enumerate(failed))
    resealed=any(float(row["close"])>=limit_price-tolerance and any(failed[:i])
                 for i,row in enumerate(ordered))
    values={
        "opening_pullback_volume_share_15m": pullback/total if total else None,
        "opening_close_vs_vwap_15m": float(ordered[-1]["close"])/vwaps[-1]-1,
        "opening_failed_seal_episodes_15m": float(episodes),
        "opening_resealed_after_failure_15m": float(resealed),
        "opening_closed_at_limit_0945": float(float(ordered[-1]["close"])>=limit_price-tolerance),
    }
    if any(value is None for value in values.values()):
        raise ValueError("window has no total traded volume")
    return {metric:dict(base,metric=metric,value=value,
                        unit="count" if metric=="opening_failed_seal_episodes_15m" else "ratio",
                        definition_id=metric+"_v1_minute_proxy",evidence_id=digest+":"+metric,
                        event_order_precision="one_minute_bar_only")
            for metric,value in values.items()}
