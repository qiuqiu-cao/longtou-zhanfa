"""Past nominal limit/probe event -> correction -> first-close restart features.

These price-path observations do not label washout, control, actor identity or
historical exchange-adjusted limit prices. No outcome columns are consumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import first_board_leader_model as base

WINDOW = 20
KINDS = ("sealed", "probe")
SUFFIXES = ("found", "age", "correction_low_log_return", "worst_correction_close_return",
            "correction_volume_to_event", "signal_volume_to_correction", "signal_close_to_event_high")
FIELDS = {kind: [f"nominal_{kind}_path20_{name}" for name in SUFFIXES] for kind in KINDS}
KEYS = ["signal_date", "code"]


def valid_bars(part):
    prices = part[["open", "high", "low", "close", "pre_close"]].to_numpy(float)
    o, hi, lo, cl, ref = prices.T
    valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid &= hi >= np.maximum.reduce([o, lo, cl])
    valid &= lo <= np.minimum.reduce([o, hi, cl])
    valid &= np.isclose(prices[:, :4] * 100, np.rint(prices[:, :4] * 100), atol=1e-6, rtol=0).all(axis=1)
    valid &= np.isfinite(part.volume) & part.volume.gt(0)
    valid &= np.isfinite(part.amount) & part.amount.gt(0)
    valid &= np.rint(hi * 100) <= base.limit_cents(ref)
    valid &= np.rint(lo * 100) >= base.limit_cents(ref, False)
    return np.asarray(valid, dtype=bool)


def from_bars(bars, candidates):
    """Only requested keys are emitted; each key uses its own source prefix."""
    if candidates.duplicated(KEYS).any() or bars.duplicated(["code", "p"]).any():
        raise ValueError("Duplicate original stock/session keys")
    if any("LABEL_ONLY" in c or c == "label_end" for c in candidates):
        raise ValueError("Feature function must not receive target columns")
    by_code = {code: part for code, part in candidates.groupby("code", sort=False)}
    output = []
    for code, part in bars.sort_values(["code", "p"]).groupby("code", sort=False):
        wanted = by_code.get(code)
        if wanted is None:
            continue
        part = part.reset_index(drop=True)
        lookup = {day: j for j, day in enumerate(part.date)}
        p = part.p.to_numpy(int)
        valid = valid_bars(part)
        hi, lo, cl, ref, volume = (part[c].to_numpy(float) for c in ("high", "low", "close", "pre_close", "volume"))
        cap = base.limit_cents(ref)
        events = {"sealed": valid & (np.rint(cl * 100) == cap),
                  "probe": valid & (np.rint(hi * 100) == cap) & (np.rint(cl * 100) < cap)}
        for row in wanted.itertuples(index=False):
            record = {"signal_date": row.signal_date, "code": code,
                      "event_path_full_window_observed": False, "event_path_source_close": np.nan}
            for kind in KINDS:
                record.update(dict.fromkeys(FIELDS[kind], np.nan))
                record[f"nominal_{kind}_event_date_DIAGNOSTIC"] = None
            j = lookup.get(row.signal_date)
            if j is None:
                output.append(record)
                continue
            record["event_path_source_close"] = cl[j]
            if j < WINDOW or p[j] - p[j-WINDOW] != WINDOW or not valid[j-WINDOW:j+1].all():
                output.append(record)
                continue
            record["event_path_full_window_observed"] = True
            for kind in KINDS:
                found = np.flatnonzero(events[kind][j-WINDOW:j])
                record[FIELDS[kind][0]] = float(len(found) > 0)
                if not len(found):
                    continue
                e = j - WINDOW + int(found[-1])
                values = [1., float(p[j]-p[e]), np.nan, np.nan, np.nan, np.nan, float(np.log(cl[j]/hi[e]))]
                if j-e > 1:
                    values[2] = float(np.log(lo[e+1:j].min()/cl[e]))
                    # Include every correction-day return, including E -> E+1;
                    # exclude the signal-day jump by construction.
                    values[3] = float((cl[e+1:j]/cl[e:j-1]-1).min())
                    median_volume = float(np.median(volume[e+1:j]))
                    values[4] = median_volume/volume[e]
                    values[5] = volume[j]/median_volume
                record.update(zip(FIELDS[kind], values))
                record[f"nominal_{kind}_event_date_DIAGNOSTIC"] = part.date.iloc[e]
            output.append(record)
    result = pd.DataFrame(output)
    if len(result) != len(candidates):
        # Preserve candidates whose code is altogether absent from the source.
        result = candidates[KEYS].merge(result, on=KEYS, how="left", validate="one_to_one")
    else:
        result = candidates[KEYS].merge(result, on=KEYS, how="left", validate="one_to_one")
    result["event_path_full_window_observed"] = result.event_path_full_window_observed.fillna(False).astype(bool)
    return result
