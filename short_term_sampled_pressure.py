"""Causal pressure/recovery observations on source-qualified five-minute bars.

Discrete OHLC intervals do not establish tick sequence within a bar, continuous
seals, real absorption or fills. Outcomes and model scores never enter this API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from archive_september_5min_v287 import check_signal, expected_times

VERSION="causal-sampled-pressure-1"
PRICE_FIELDS=["sampled_cap_close_then_below_intervals","sampled_cap_close_recovery_transitions",
    "sampled_negative_prior_bar_peak_drawdown","sampled_intervals_after_last_below_close",
    "sampled_first_cap_close_progress"]
VOLUME_FIELDS=["sampled_final_six_volume_fraction","sampled_recovery_interval_volume_fraction",
    "sampled_final_six_cap_volume_fraction","sampled_close_above_full_day_vwap"]
FIELDS=PRICE_FIELDS+VOLUME_FIELDS


def describe(bars,reference,code,day,cap):
    """Only this code's exact signal day is consumed; every output is <= T."""
    rows=bars[bars.code.eq(code)&bars.minute.str[:10].eq(day)].sort_values("minute").copy()
    result=dict(version=VERSION,date=day,code=code,feature_cutoff=day,
        interval_minutes=5,observed_intervals=len(rows),price_path_qualified=False,volume_path_qualified=False,
        source_status="UNAVAILABLE_OR_UNQUALIFIED",continuous_seal_verified=False,
        real_reseal_count=None,verified_absorption=None,actual_fills=0,
        **{key:None for key in FIELDS})
    if rows.empty or not np.isfinite(cap) or cap<=0:return result
    # The original check distinguishes price/grid/volume/amount qualification.
    check=check_signal(rows,reference,code,day)
    result["source_checks"]=check
    qualified=check["price_reconciled"] and check["grid_complete"]
    if not qualified or abs(float(rows.close.iloc[-1])-cap)>.005001 or abs(float(rows.high.max())-cap)>.005001:
        return result
    close_cap=rows.close.ge(cap-.005001)
    low_below=rows.low.lt(cap-.005001)
    previous_cap=close_cap.shift(1,fill_value=False)
    # Only ordering across two separate intervals is asserted. An interval
    # starting at cap and ending below it is not a tick-certified open event.
    after_cap_low=previous_cap & low_below
    recover=close_cap & ~previous_cap
    # The first observed interval is not called a recovery from a prior state.
    recover.iloc[0]=False
    prior_peak=rows.high.cummax().shift(1)
    drawdown=rows.low/prior_peak-1
    below_close=~close_cap
    last_below=np.flatnonzero(below_close.to_numpy())
    first_cap=np.flatnonzero(close_cap.to_numpy())
    n=len(rows)
    result.update(price_path_qualified=True,source_status="RECONCILED_DISCRETE_PRICE_PATH_NOT_TICK",
        sampled_cap_close_then_below_intervals=int(after_cap_low.sum()),
        sampled_cap_close_recovery_transitions=int(recover.sum()),
        sampled_negative_prior_bar_peak_drawdown=min(0.,float(drawdown.min())),
        sampled_intervals_after_last_below_close=n-1-int(last_below[-1]) if len(last_below) else n,
        sampled_first_cap_close_progress=float(first_cap[0]/(n-1)) if len(first_cap) and n>1 else None)
    vol_ok=check["volume_reconciled"] and check.get("amount_volume_unit_consistent") and rows.volume_shares.sum()>0
    if vol_ok:
        volume=rows.volume_shares.sum();last=rows.tail(6)
        result.update(volume_path_qualified=True,source_status="RECONCILED_DISCRETE_PRICE_AND_VOLUME_PATH_NOT_TICK",
            sampled_final_six_volume_fraction=float(last.volume_shares.sum()/volume),
            sampled_recovery_interval_volume_fraction=float(rows.loc[recover,"volume_shares"].sum()/volume),
            sampled_final_six_cap_volume_fraction=float(last.loc[last.close.ge(cap-.005001),"volume_shares"].sum()/volume),
            sampled_close_above_full_day_vwap=float(cap/check["vwap_observed_bars"]-1))
    return result


def cohort_relative(frame):
    """Entire nonword same-height cohort must be observed before a path rank."""
    output=frame.copy()
    output["sampled_same_height_pressure_rank"]=np.nan
    output["sampled_same_height_comparison_complete"]=False
    for _,group in output[output.daily_one_word.eq(0)].groupby(["date","boards"]):
        field="sampled_negative_prior_bar_peak_drawdown"
        if not group.price_path_qualified.fillna(False).all() or group[field].isna().any():continue
        output.loc[group.index,"sampled_same_height_pressure_rank"]=group[field].rank(method="average",pct=True)
        output.loc[group.index,"sampled_same_height_comparison_complete"]=True
    return output
