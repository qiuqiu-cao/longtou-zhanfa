"""Frozen, pre-first-board OHLCV effort/result proxy; no outcomes or fitting.

The proxy is an observed price/volume clue, not verified distribution, a
Wyckoff phase, an absence-of-limit-touch certificate, or an entry signal.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import json
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE = "wyckoff_preboard_effort_result_risk"
NUMERIC = ["open", "high", "low", "close", "volume"]
MEASURES = ["volume_ratio", "spread_ratio", "close_location",
            "background_return", "event_return"]
OUTPUT = ["date", "code", "boards", FEATURE, "wyckoff_input_missing",
          "feature_date", "window_dates", "feature_status", *MEASURES,
          "age_sessions", "source"]


def _dates(values: Iterable) -> list[str]:
    return pd.to_datetime(list(values), errors="raise").strftime("%Y-%m-%d").tolist()


def _cents(value: float) -> int:
    # Quantize to stock-price cents so float noise cannot turn exactly 1 cent
    # into an eligible endpoint distance.
    return int(Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) * 100)


def extract_features(candidates: pd.DataFrame, bars: pd.DataFrame,
                     official_sessions: Iterable) -> pd.DataFrame:
    """Return one evidence row per original candidate, retaining all unknowns.

    ``official_sessions`` is an ordered sequence of ISO dates (or a DataFrame
    with a ``date`` column). T=signal index-boards. Only the seven official
    sessions T-6..T are looked up; signal/target/future bars cannot enter the
    measurements. Duplicate bar keys invalidate only windows using that key.
    """
    missing = {"date", "code", "boards"} - set(candidates)
    bar_missing = {"date", "code", "source", *NUMERIC} - set(bars)
    if missing or bar_missing:
        raise ValueError(f"missing candidate fields {sorted(missing)}; bar fields {sorted(bar_missing)}")
    if isinstance(official_sessions, pd.DataFrame):
        official_sessions = official_sessions["date"]
    calendar = _dates(official_sessions)
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("official sessions must be nonempty, unique and increasing")
    session_index = {date: i for i, date in enumerate(calendar)}

    c = candidates[["date", "code", "boards"]].copy()
    c["date"] = _dates(c.date)
    c["code"] = c.code.astype(str).str.zfill(6)
    candidate_duplicate = c.duplicated(["date", "code"], keep=False)
    b = bars[["date", "code", "source", *NUMERIC]].copy()
    b["date"] = _dates(b.date)
    b["code"] = b.code.astype(str).str.zfill(6)
    for field in NUMERIC:
        b[field] = pd.to_numeric(b[field], errors="coerce")
    duplicate_keys = set(map(tuple, b.loc[b.duplicated(["date", "code"], keep=False), ["date", "code"]].to_numpy()))
    unique = b.drop_duplicates(["date", "code"], keep=False)
    lookup = {(row.date, row.code): row for row in unique.itertuples(index=False)}
    result = []
    for position, candidate in enumerate(c.itertuples(index=False)):
        row = dict(date=candidate.date, code=candidate.code, boards=candidate.boards,
                   wyckoff_preboard_effort_result_risk=np.nan, wyckoff_input_missing=1,
                   feature_date=None, window_dates="[]", feature_status="UNKNOWN",
                   age_sessions=np.nan, source=None, **{name: np.nan for name in MEASURES})

        def unknown(reason: str):
            row["feature_status"] = reason
            result.append(row)

        try:
            age = float(candidate.boards)
        except (ValueError, TypeError):
            age = np.nan
        if not np.isfinite(age) or age < 1 or age != int(age):
            unknown("INVALID_BOARDS")
            continue
        row["age_sessions"] = int(age)
        signal_index = session_index.get(candidate.date)
        if signal_index is None:
            unknown("SIGNAL_NOT_OFFICIAL_SESSION")
            continue
        event_index = signal_index - int(age)
        if event_index >= 0:
            row["feature_date"] = calendar[event_index]
        if event_index < 6:
            unknown("INSUFFICIENT_OFFICIAL_WINDOW")
            continue
        window_dates = calendar[event_index-6:event_index+1]
        row["window_dates"] = json.dumps(window_dates, separators=(",", ":"))
        if candidate_duplicate.iloc[position]:
            unknown("DUPLICATE_CANDIDATE_KEY")
            continue
        keys = [(date, candidate.code) for date in window_dates]
        if any(key in duplicate_keys for key in keys):
            unknown("DUPLICATE_BAR_KEY")
            continue
        if any(key not in lookup for key in keys):
            unknown("MISSING_OFFICIAL_SESSION_BAR")
            continue
        window = [lookup[key] for key in keys]
        sources = [record.source for record in window]
        if any(pd.isna(value) or not str(value).strip() for value in sources):
            unknown("MISSING_SOURCE")
            continue
        if len(set(sources)) != 1:
            row["source"] = "|".join(sorted(map(str, set(sources))))
            unknown("MIXED_SOURCE_WINDOW")
            continue
        row["source"] = str(sources[0])
        values = np.array([[getattr(record, name) for name in NUMERIC] for record in window], dtype=float)
        if not np.isfinite(values).all() or not (values > 0).all():
            unknown("NONFINITE_OR_NONPOSITIVE_OHLCV")
            continue
        opening, high, low, close, volume = values.T
        if not ((high >= low) & (opening >= low) & (opening <= high)
                & (close >= low) & (close <= high)).all():
            unknown("INVALID_OHLC_BOUNDS")
            continue
        if high[-1] <= low[-1] or _cents(high[-1])-_cents(close[-1]) <= 1 or _cents(close[-1])-_cents(low[-1]) <= 1:
            unknown("EVENT_ENDPOINT_EXCLUDED")
            continue
        spread = (high[1:]-low[1:])/close[:-1]
        volume_ratio = volume[-1]/volume[1:6].mean()
        spread_ratio = spread[-1]
        close_location = (close[-1]-low[-1])/(high[-1]-low[-1])
        background_return = close[-2]/close[0]-1
        event_return = close[-1]/close[-2]-1
        risk = (volume_ratio >= 1.5 and spread_ratio <= np.median(spread[:-1])
                and close[-1] < close[-2] and close_location <= .5
                and close[-2] > close[0])
        row.update(wyckoff_preboard_effort_result_risk=int(risk), wyckoff_input_missing=0,
                   feature_status="OBSERVED_PROXY_NOT_VERIFIED_DISTRIBUTION",
                   volume_ratio=float(volume_ratio), spread_ratio=float(spread_ratio),
                   close_location=float(close_location), background_return=float(background_return),
                   event_return=float(event_return))
        result.append(row)
    return pd.DataFrame(result, columns=OUTPUT)
