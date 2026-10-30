"""Historical STATIC industry timing proxy, never a dynamic-theme classifier.

Inputs are only S-day fields. All original candidates stay in the pool. Unknown
industry/timing cannot be converted into evidence that a stock is a follower or
isolated. Counts cover qualified first boards only, not all market participants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["signal_date", "code"]
GROUP = "legacy_industry_group_PROXY"
TIME = "reported_first_seal_trading_minutes"
SCOPE = "STATIC_2021_GROUP_WITHIN_QUALIFIED_FIRST_BOARDS_ONLY"


def timing_features(frame, window=30.):
    if not np.isfinite(window) or window <= 0:
        raise ValueError("Positive fixed trading-minute window required")
    x = frame[KEYS + ["s1_score", GROUP, TIME]].copy()
    if x.duplicated(KEYS).any() or x[KEYS].isna().any().any():
        raise ValueError("Unique complete candidate keys required")
    if not np.isfinite(x.s1_score).all():
        raise ValueError("Frozen S1 score missing")
    x[GROUP] = x[GROUP].astype("string").str.strip().replace({"": pd.NA, "UNKNOWN": pd.NA})
    time = pd.to_numeric(x[TIME], errors="coerce")
    x[TIME] = time.where(np.isfinite(time) & time.between(-5., 240.))
    x["timing_known"] = x[TIME].notna()
    x["local_group_size"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x["is_earliest_in_observed_group"] = pd.Series(pd.NA, index=x.index, dtype="boolean")
    x["earliest_tie_count"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x["later_first_seals_within_window"] = pd.Series(pd.NA, index=x.index, dtype="Int64")
    x["proxy_status"] = "UNKNOWN_GROUP"
    for _, g in x[x[GROUP].notna()].groupby(["signal_date", GROUP], sort=True):
        ix = g.index
        x.loc[ix, "local_group_size"] = len(g)
        if not g.timing_known.all():
            # One unobserved peer could have sealed earlier; no forced role.
            x.loc[ix, "proxy_status"] = "UNKNOWN_INCOMPLETE_GROUP_TIMES"
            continue
        first_time = float(g[TIME].min())
        first = g[TIME].eq(first_time)
        later = g[TIME].gt(first_time) & g[TIME].le(first_time + window)
        x.loc[ix, "is_earliest_in_observed_group"] = first
        x.loc[ix, "earliest_tie_count"] = int(first.sum())
        x.loc[ix, "later_first_seals_within_window"] = 0
        x.loc[g.index[first], "later_first_seals_within_window"] = int(later.sum())
        x.loc[ix, "proxy_status"] = "OBSERVED_LOCAL_STATIC_GROUP"
    x["group_scope"] = SCOPE
    x["window_trading_minutes"] = window
    x["dynamic_theme_membership_verified"] = False
    x["causal_leadership_verified"] = False
    x["production_ranking_eligible"] = False
    return x


def baseline(frame):
    d = frame[KEYS+["s1_score"]].copy()
    if d.duplicated(KEYS).any() or not np.isfinite(d.s1_score).all():
        raise ValueError("Invalid baseline pool")
    d = d.sort_values(["signal_date", "s1_score", "code"], ascending=[True, False, True])
    d["slot"] = d.groupby("signal_date").cumcount()+1
    return d[d.slot.le(3)].reset_index(drop=True)


def choose_proxy_swap(frame, challenger_cap=None):
    """One experimental swap at most; no numeric fitting or live authorization.

    The original trial checks slots 2 then 3; keep that order to avoid silently
    optimizing another policy. None scans the COMPLETE qualified challenger pool;
    cap=8 exists only to compare against Gemini's archived restricted trial.
    Unknown evidence always leaves a candidate eligible for the frozen baseline.
    """
    if challenger_cap is not None and (int(challenger_cap) != challenger_cap or challenger_cap < 4):
        raise ValueError("Challenger cap must be None or an integer >=4")
    cols = KEYS+["s1_score", "proxy_status", "is_earliest_in_observed_group",
                 "later_first_seals_within_window"]
    d = frame[cols].copy()
    if d.duplicated(KEYS).any() or not np.isfinite(d.s1_score).all():
        raise ValueError("Invalid scored pool")
    d = d.sort_values(["signal_date", "s1_score", "code"], ascending=[True, False, True])
    picks, changes = [], []
    for day, g in d.groupby("signal_date", sort=True):
        g = g.reset_index(drop=True)
        selected = list(range(min(3, len(g))))
        target = None
        for slot_index in [1, 2]:
            if slot_index >= len(g):
                continue
            row = g.iloc[slot_index]
            if row.proxy_status != "OBSERVED_LOCAL_STATIC_GROUP":
                continue
            if not row.is_earliest_in_observed_group or row.later_first_seals_within_window == 0:
                target = slot_index
                break
        if target is not None:
            stop = len(g) if challenger_cap is None else min(challenger_cap, len(g))
            for index in range(3, stop):
                row = g.iloc[index]
                if (row.proxy_status == "OBSERVED_LOCAL_STATIC_GROUP" and
                        row.is_earliest_in_observed_group and row.later_first_seals_within_window >= 1):
                    changes.append(dict(signal_date=day, displaced_code=g.iloc[target].code,
                        promoted_code=row.code, displaced_slot=target+1, promoted_original_rank=index+1))
                    selected[target] = index
                    break
        for slot, index in enumerate(selected, 1):
            row = g.iloc[index]
            picks.append(dict(signal_date=day, code=row.code, slot=slot,
                              original_s1_rank=index+1, s1_score=float(row.s1_score)))
    return pd.DataFrame(picks), pd.DataFrame(changes, columns=["signal_date", "displaced_code", "promoted_code",
                                                              "displaced_slot", "promoted_original_rank"])
