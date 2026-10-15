"""Reclassify exact-price failed attempts without dropping their observations.

Original V215 labels and all frozen challenger inputs remain immutable. This
module reconstructs provider-scope attempts, not a verified exchange universe.
"""
from __future__ import annotations

from decimal import Decimal,ROUND_HALF_UP

import numpy as np
import pandas as pd

VERSION="exact-price-attempt-reclassification-1"


def cent(value):
    number=Decimal(str(value))
    if not number.is_finite() or number<=0 or number*100!=(number*100).to_integral_value():raise ValueError("positive exact cents required")
    return int(number*100)


def reconstruct(frame,sessions):
    """Every classification and attempted height uses observations only <= T."""
    columns=["signal_date","code","name","pool_assignment","board_height","limit_price_10","high","close","pre_close"]
    data=frame[columns].copy()
    data["code"]=data.code.astype(str).str.zfill(6)
    if data.duplicated(["signal_date","code"]).any():raise ValueError("duplicate original attempt observation")
    dates={d:i for i,d in enumerate(sessions)}
    if not set(data.signal_date)<=set(dates):raise ValueError("observation outside official calendar")
    previous={};kept=[];quarantine=[];scope_unknown=set()
    for stock in data.sort_values(["signal_date","code"]).to_dict("records"):
        day,code=stock["signal_date"],stock["code"]
        reason=None
        try:
            hi,cl,prior,upper=[cent(stock[k]) for k in ["high","close","pre_close","limit_price_10"]]
            computed=int((Decimal(prior)*Decimal("1.10")).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
            if upper!=computed:reason="DECLARED_CAP_DIFFERS_FROM_EXACT_10PCT_ROUNDING"
            elif cl>hi:reason="DAILY_CLOSE_ABOVE_HIGH"
            elif hi>upper:reason="PRICE_ABOVE_ASSUMED_10PCT_CAP_REGIME_UNRESOLVED"
            elif hi<upper:reason="NO_EXACT_UPPER_PRICE_TOUCH_NOT_AN_ATTEMPT"
        except (ValueError,TypeError,ArithmeticError):reason="INVALID_OR_NONCENT_PRICE"
        if reason:
            quarantine.append(dict(stock,qualification_reason=reason))
            if reason!="NO_EXACT_UPPER_PRICE_TOUCH_NOT_AN_ATTEMPT":scope_unknown.add(day)
            # An excluded record does not manufacture a positive closing state.
            # The next valid observation must still be officially adjacent.
            continue
        assigned="CLOSING_LIMIT_UP" if cl==upper else "FAILED_BOARD_OBSERVATION"
        old=previous.get(code)
        height=old[1]+1 if old and old[0]==dates[day]-1 else 1
        previous[code]=(dates[day],height if assigned=="CLOSING_LIMIT_UP" else 0)
        kept.append(dict(stock,original_pool_assignment=stock["pool_assignment"],original_board_height=stock["board_height"],
            pool_assignment=assigned,board_height=height,reclassified=assigned!=stock["pool_assignment"],
            qualification_reason="EXACT_UPPER_TOUCH_AND_CLOSE_CLASSIFIED",high_cents=hi,close_cents=cl,upper_cents=upper,
            source_scope="CURRENT_UNIVERSE_ASSUMED_MAINBOARD10PCT_NOT_EXCHANGE_STATUS_CERTIFIED"))
    result=pd.DataFrame(kept)
    result.attrs["failed_scope_complete_dates"]=sorted(set(data.signal_date)-scope_unknown)
    result.attrs["source_scope"]="EXACT_PRICE_RECONSTRUCTED_SOURCE_UNIVERSE_NOT_EXCHANGE_FULL_MARKET"
    return result,pd.DataFrame(quarantine)
