"""Past-only archive price highs; imported reference column is prior close, not ex-rights truth."""
from __future__ import annotations
import numpy as np
import pandas as pd
import first_board_leader_model as base

DISTANCE = "archive_nominal_log_distance_high120"
AGE = "archive_nominal_latest_high_age120"
DIAGNOSTIC250 = "archive_nominal_log_distance_high250_DIAGNOSTIC_ONLY"
FIELDS = [DISTANCE, AGE]
KEYS = ["signal_date", "code"]


def load_source():
    columns = ["date", "code", "open", "high", "low", "close", "pre_close", "volume", "amount"]
    frames = [pd.read_csv(base.INPUTS[name], usecols=columns, dtype={"code": str}, float_precision="round_trip")
              for name in ("bars", "recovered")]
    bars = pd.concat(frames, ignore_index=True)
    bars["code"] = bars.code.str.zfill(6)
    bars["date"] = bars.date.astype(str).str[:10]
    if bars.duplicated(["date", "code"]).any():
        duplicate = bars[bars.duplicated(["date", "code"], keep=False)]
        if duplicate.groupby(["date", "code"])[["open", "high", "low", "close", "pre_close"]].nunique().max(axis=1).gt(1).any():
            raise ValueError("Conflicting duplicated source prices")
        bars = bars.drop_duplicates(["date", "code"], keep="last")
    sessions = __import__("json").loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
    positions = {day: index for index, day in enumerate(sessions)}
    bars["p"] = bars.date.map(positions)
    if bars.p.isna().any():
        raise ValueError("Source has dates outside frozen official calendar")
    bars["p"] = bars.p.astype(int)
    return bars.sort_values(["code", "p"]).reset_index(drop=True)


def from_bars(bars, horizons=(120, 250)):
    if bars.duplicated(["code", "p"]).any():
        raise ValueError("Duplicate stock and official-session key")
    output = []
    for code, part in bars.sort_values(["code", "p"]).groupby("code", sort=False):
        p = part.p.to_numpy(dtype=int)
        prices = part[["open", "high", "low", "close", "pre_close"]].to_numpy(dtype=float)
        o, hi, lo, cl, ref = prices.T
        valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        valid &= hi >= np.maximum.reduce([o, cl, lo])
        valid &= lo <= np.minimum.reduce([o, cl, hi])
        # Reference prices may have sub-cent precision; traded OHLC must be cent prices.
        valid &= np.isclose(prices[:, :4] * 100, np.rint(prices[:, :4] * 100), atol=1e-6, rtol=0).all(axis=1)
        valid &= part.volume.to_numpy(dtype=float) > 0
        valid &= part.amount.to_numpy(dtype=float) > 0
        valid &= np.rint(hi * 100) <= base.limit_cents(ref)
        valid &= np.rint(lo * 100) >= base.limit_cents(ref, False)
        adjacent = np.r_[False, np.diff(p) == 1]
        previous_valid = np.r_[False, valid[:-1]]
        links = adjacent & valid & previous_valid
        log_link = np.zeros(len(part))
        index = np.flatnonzero(links)
        log_link[index] = np.log(ref[index]) - np.log(cl[index - 1])
        cumulative = np.cumsum(log_link)
        scaled_high = np.full(len(part), np.nan)
        scaled_close = np.full(len(part), np.nan)
        scaled_high[valid] = np.log(hi[valid]) - cumulative[valid]
        scaled_close[valid] = np.log(cl[valid]) - cumulative[valid]
        result = part[["date", "code", "p"]].copy()
        result["source_close"] = cl
        result["source_valid_traded_bar"] = valid
        for horizon in horizons:
            distance = np.full(len(part), np.nan)
            age = np.full(len(part), np.nan)
            changed = np.full(len(part), np.nan)
            raw_distance = np.full(len(part), np.nan)
            if len(part) > horizon:
                windows = np.lib.stride_tricks.sliding_window_view(scaled_high, horizon)[:-1]
                all_valid = np.lib.stride_tricks.sliding_window_view(valid, horizon + 1).all(axis=1)
                contiguous = p[horizon:] - p[:-horizon] == horizon
                good = all_valid & contiguous
                safe = np.where(np.isfinite(windows), windows, -np.inf)
                maxima = safe.max(axis=1)
                # Reverse argmax selects the most recent equal reported reference-basis high.
                ages = np.argmax(safe[:, ::-1], axis=1) + 1
                distance[horizon:] = np.where(good, scaled_close[horizon:] - maxima, np.nan)
                age[horizon:] = np.where(good, ages, np.nan)
                jump = (np.abs(log_link) > 1e-10).astype(int)
                count = np.r_[0, np.cumsum(jump)]
                changed[horizon:] = np.where(good, count[horizon+1:] - count[1:-horizon], np.nan)
                raw_windows = np.lib.stride_tricks.sliding_window_view(hi, horizon)[:-1]
                raw_distance[horizon:] = np.where(good, np.log(cl[horizon:] / raw_windows.max(axis=1)), np.nan)
            name = DISTANCE if horizon == 120 else DIAGNOSTIC250
            result[name] = distance
            if horizon == 120:
                result[AGE] = age
            result[f"reported_reference_changes{horizon}"] = changed
            result[f"raw_unadjusted_distance{horizon}_DIAGNOSTIC_ONLY"] = raw_distance
        output.append(result)
    return pd.concat(output, ignore_index=True).rename(columns={"date": "signal_date"})


def attach_full_pool(facts, features):
    if facts.duplicated(KEYS).any() or features.duplicated(KEYS).any():
        raise ValueError("Duplicate full-pool key")
    result = facts.merge(features.drop(columns="p"), on=KEYS, how="left", sort=False, validate="one_to_one")
    assert len(result) == len(facts)
    matched = result.source_close.notna()
    mismatch = matched & np.rint(result.close * 100).ne(np.rint(result.source_close * 100))
    if mismatch.any():
        raise ValueError("Signal-close cent mismatch")
    return result
