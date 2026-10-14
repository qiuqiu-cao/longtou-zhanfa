#!/usr/bin/env python3
"""Build auditable V9 open-confirmation factors from primitive exports.

The builder accepts three point-in-time auction observations and one-minute
OHLCVA bars.  It derives the factors consumed by ``open_confirmation_v4`` and
never accepts a pre-computed cancellation ratio, VWAP slope or reseal flag.
The output is still paper/research evidence: minute bars cannot prove queue
position or an actual limit-up fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import open_confirmation_v4 as opening
from leader_system_v3 import is_mainboard, zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = ROOT / "data" / "v7" / "point_in_time_rebuild" / "leader_candidates.csv"
DEFAULT_THEME_SYNC = ROOT / "data" / "v4" / "theme_open_sync.csv"
DEFAULT_OUTPUT = ROOT / "data" / "v9" / "derived_open_snapshots.csv"
VERSION = "v9-intraday-evidence-1"
AUCTION_TIMES = ("09:20", "09:24:50", "09:25")
VERIFIED_RAW_STATUSES = {
    "verified_broker_export", "verified_terminal_export", "verified_official_export",
}
AUCTION_REQUIRED = {
    "signal_date", "trade_date", "asof", "code", "name", "source",
    "source_timestamp", "data_status", "matched_amount", "indicative_price",
    "prior_close", "prior_day_amount",
}
MINUTE_REQUIRED = {
    "signal_date", "trade_date", "bar_end", "code", "name", "source",
    "source_timestamp", "data_status", "open", "high", "low", "close",
    "volume_shares", "amount_yuan", "limit_up_price",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_date(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_source_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("source_timestamp must be present and parseable")
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return stamp


def _trade_cutoff(trade_date: str, clock: str) -> pd.Timestamp:
    return pd.Timestamp(trade_date) + pd.Timedelta(seconds=opening.clock_seconds(clock))


def _validate_identity(frame: pd.DataFrame) -> None:
    if not frame.apply(lambda row: is_mainboard(row["code"], row["name"]), axis=1).all():
        raise ValueError("primitive input contains non-mainboard/ST/delisting-risk code")


def validate_auction(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(AUCTION_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"auction input missing columns: {missing}")
    checked = frame.copy()
    checked["signal_date"] = _normalize_date(checked["signal_date"], "signal_date")
    checked["trade_date"] = _normalize_date(checked["trade_date"], "trade_date")
    checked["asof"] = checked["asof"].map(opening.normalize_asof)
    if (~checked["asof"].isin(AUCTION_TIMES)).any():
        raise ValueError(f"auction asof must be one of {AUCTION_TIMES}")
    checked["code"] = checked["code"].map(zfill_code)
    _validate_identity(checked)
    if checked.duplicated(["signal_date", "trade_date", "asof", "code"]).any():
        raise ValueError("auction input contains duplicate keys")
    checked["data_status"] = checked["data_status"].astype(str).str.strip().str.lower()
    if (~checked["data_status"].isin(VERIFIED_RAW_STATUSES)).any():
        raise ValueError("auction rows require a verified raw export status")
    for field in ("matched_amount", "indicative_price", "prior_close", "prior_day_amount"):
        checked[field] = pd.to_numeric(checked[field], errors="coerce")
        if checked[field].isna().any() or (checked[field] < 0).any():
            raise ValueError(f"auction {field} must be non-negative and present")
    if (checked[["indicative_price", "prior_close", "prior_day_amount"]] <= 0).any().any():
        raise ValueError("auction prices and prior_day_amount must be positive")
    normalized_timestamps = []
    for index, row in checked.iterrows():
        stamp = _normalize_source_timestamp(row["source_timestamp"])
        cutoff = _trade_cutoff(row["trade_date"], row["asof"])
        if stamp > cutoff or stamp < cutoff - pd.Timedelta(seconds=10):
            raise ValueError(f"auction row {index} source_timestamp is not a point-in-time capture")
        normalized_timestamps.append(stamp)
    checked["_source_timestamp"] = normalized_timestamps
    return checked


def validate_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(MINUTE_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"minute input missing columns: {missing}")
    checked = frame.copy()
    checked["signal_date"] = _normalize_date(checked["signal_date"], "signal_date")
    checked["trade_date"] = _normalize_date(checked["trade_date"], "trade_date")
    checked["code"] = checked["code"].map(zfill_code)
    _validate_identity(checked)
    checked["bar_end"] = pd.to_datetime(checked["bar_end"], errors="coerce")
    if checked["bar_end"].isna().any():
        raise ValueError("minute bar_end must be a full trade-date timestamp")
    if (checked["bar_end"].dt.strftime("%Y-%m-%d") != checked["trade_date"]).any():
        raise ValueError("minute bar_end date differs from trade_date")
    seconds = checked["bar_end"].dt.hour * 3600 + checked["bar_end"].dt.minute * 60 + checked["bar_end"].dt.second
    if (~seconds.between(opening.clock_seconds("09:30") + 1, opening.clock_seconds("10:00"))).any():
        raise ValueError("minute bars must end after 09:30 and no later than 10:00")
    if checked.duplicated(["signal_date", "trade_date", "bar_end", "code"]).any():
        raise ValueError("minute input contains duplicate bars")
    checked["data_status"] = checked["data_status"].astype(str).str.strip().str.lower()
    if (~checked["data_status"].isin(VERIFIED_RAW_STATUSES)).any():
        raise ValueError("minute rows require a verified raw export status")
    price_fields = ("open", "high", "low", "close", "limit_up_price")
    for field in (*price_fields, "volume_shares", "amount_yuan"):
        checked[field] = pd.to_numeric(checked[field], errors="coerce")
        if checked[field].isna().any() or (checked[field] < 0).any():
            raise ValueError(f"minute {field} must be non-negative and present")
    if (checked[list(price_fields)] <= 0).any().any():
        raise ValueError("minute prices must be positive")
    if ((checked["high"] < checked[["open", "close", "low"]].max(axis=1))
            | (checked["low"] > checked[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("minute OHLC relationship is invalid")
    normalized_timestamps = []
    for index, row in checked.iterrows():
        stamp = _normalize_source_timestamp(row["source_timestamp"])
        if stamp.date() != pd.Timestamp(row["trade_date"]).date():
            raise ValueError(f"minute row {index} source timestamp date differs from trade_date")
        if stamp < row["bar_end"]:
            raise ValueError(f"minute row {index} source timestamp predates bar_end")
        normalized_timestamps.append(stamp)
    checked["_source_timestamp"] = normalized_timestamps
    return checked.sort_values(["code", "bar_end"]).reset_index(drop=True)


def _auction_factors(rows: pd.DataFrame, asof: str) -> dict[str, float]:
    values = {"auction_gap": math.nan, "auction_amount_ratio": math.nan, "auction_cancel_ratio": math.nan}
    if rows.empty:
        return values
    point = opening.clock_seconds(asof)
    eligible = rows[rows["asof"].map(opening.clock_seconds) <= point]
    if eligible.empty:
        return values
    lookup = eligible.sort_values("asof", key=lambda x: x.map(opening.clock_seconds)).set_index("asof")
    latest = lookup.iloc[-1]
    values["auction_amount_ratio"] = float(latest["matched_amount"] / latest["prior_day_amount"])
    if point >= opening.clock_seconds("09:25") and "09:25" in lookup.index:
        final = lookup.loc["09:25"]
        values["auction_gap"] = float(final["indicative_price"] / final["prior_close"] - 1)
    if point >= opening.clock_seconds("09:24:50") and {"09:20", "09:24:50"}.issubset(lookup.index):
        early = float(lookup.loc["09:20", "matched_amount"])
        late = float(lookup.loc["09:24:50", "matched_amount"])
        values["auction_cancel_ratio"] = max(0.0, (early - late) / early) if early > 0 else math.nan
    return values


def _vwap_series(rows: pd.DataFrame) -> pd.Series:
    cumulative_volume = rows["volume_shares"].cumsum()
    cumulative_amount = rows["amount_yuan"].cumsum()
    return cumulative_amount.div(cumulative_volume.where(cumulative_volume > 0))


def _slope(vwap: pd.Series, count: int) -> float:
    values = vwap.iloc[:count].dropna()
    if len(values) < count or values.iloc[0] <= 0:
        return math.nan
    return float(values.iloc[-1] / values.iloc[0] - 1)


def _minute_factors(rows: pd.DataFrame, asof: str) -> dict[str, float]:
    result = {
        "vwap_slope_5m": math.nan, "vwap_slope_15m": math.nan,
        "pullback_volume_ratio": math.nan, "vwap_reclaimed": math.nan,
        "reseal_minute": math.nan, "below_vwap_expanding": math.nan,
        "failed_seal_twice": math.nan,
    }
    if rows.empty or opening.clock_seconds(asof) < opening.clock_seconds("09:35"):
        return result
    ordered = rows.sort_values("bar_end").reset_index(drop=True)
    vwap = _vwap_series(ordered)
    result["vwap_slope_5m"] = _slope(vwap, 5)
    if opening.clock_seconds(asof) >= opening.clock_seconds("10:00"):
        result["vwap_slope_15m"] = _slope(vwap, 15)
    previous_close = ordered["close"].shift(1).fillna(ordered["open"])
    pullback_volume = ordered.loc[ordered["close"] < previous_close, "volume_shares"].sum()
    advance_volume = ordered.loc[ordered["close"] >= previous_close, "volume_shares"].sum()
    result["pullback_volume_ratio"] = float(pullback_volume / advance_volume) if advance_volume > 0 else math.nan
    current_vwap = vwap.iloc[-1]
    result["vwap_reclaimed"] = float(pd.notna(current_vwap) and ordered.iloc[-1]["close"] >= current_vwap)
    tolerance = 1e-6
    touched = ordered["high"] >= ordered["limit_up_price"] * (1 - tolerance)
    closed_below = ordered["close"] < ordered["limit_up_price"] * (1 - tolerance)
    failed = touched & closed_below
    episodes = int((failed & ~failed.shift(1, fill_value=False)).sum())
    result["failed_seal_twice"] = float(episodes >= 2)
    sealed = ordered["close"] >= ordered["limit_up_price"] * (1 - tolerance)
    prior_below = closed_below.cummax().shift(1, fill_value=False)
    reseals = ordered[sealed & (prior_below | (ordered["low"] < ordered["limit_up_price"] * (1 - tolerance)))]
    if not reseals.empty:
        elapsed = (reseals.iloc[0]["bar_end"] - pd.Timestamp(reseals.iloc[0]["trade_date"] + " 09:30:00")).total_seconds() / 60
        result["reseal_minute"] = float(max(0, elapsed))
    if len(ordered) >= 4 and pd.notna(current_vwap):
        recent = ordered["volume_shares"].iloc[-2:].sum()
        prior = ordered["volume_shares"].iloc[-4:-2].sum()
        below = ordered.iloc[-1]["close"] < current_vwap and ordered.iloc[-1]["close"] < ordered.iloc[-2]["close"]
        result["below_vwap_expanding"] = float(bool(below and prior > 0 and recent > prior * 1.20))
    else:
        result["below_vwap_expanding"] = 0.0
    return result


def _rivals_eliminated(code: str, board: int, factors: dict[str, dict[str, float]], boards: dict[str, int]) -> float:
    rivals = [other for other, other_board in boards.items() if other != code and other_board == board and other in factors]
    if not rivals:
        return math.nan
    eliminated = [
        other for other in rivals
        if factors[other].get("below_vwap_expanding") == 1 or factors[other].get("vwap_reclaimed") == 0
    ]
    return float(len(eliminated) / len(rivals))


def _theme_row(registry: pd.DataFrame, signal: str, trade: str, asof: str, code: str) -> dict[str, Any]:
    empty = {
        "theme_sync_open": math.nan, "theme_sync_basis": "unavailable",
        "theme_evidence_id": "", "theme_group_id": "",
        "theme_member_coverage": math.nan, "theme_observed_peers": math.nan,
    }
    if registry.empty:
        return empty
    selected = registry[
        registry["signal_date"].astype(str).eq(signal)
        & registry["trade_date"].astype(str).eq(trade)
        & registry["asof"].astype(str).eq(asof)
        & registry["code"].map(zfill_code).eq(code)
    ]
    if len(selected) != 1:
        return empty
    row = selected.iloc[0]
    return {key: row.get(key, empty[key]) for key in empty}


def build_snapshots(
    auction: pd.DataFrame,
    minutes: pd.DataFrame,
    candidates: pd.DataFrame,
    signal_date: str,
    trade_date: str,
    asof: str,
    theme_sync: pd.DataFrame | None = None,
) -> pd.DataFrame:
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    point = opening.normalize_asof(asof)
    checked_auction = validate_auction(auction)
    checked_minutes = validate_minutes(minutes)
    checked_auction = checked_auction[
        checked_auction["signal_date"].eq(signal) & checked_auction["trade_date"].eq(trade)
    ]
    cutoff = _trade_cutoff(trade, point)
    checked_auction = checked_auction[
        checked_auction["asof"].map(opening.clock_seconds) <= opening.clock_seconds(point)
    ]
    checked_minutes = checked_minutes[
        checked_minutes["signal_date"].eq(signal)
        & checked_minutes["trade_date"].eq(trade)
        & (checked_minutes["bar_end"] <= cutoff)
        & (checked_minutes["_source_timestamp"] <= cutoff)
    ]
    candidate_rows = candidates[candidates["date"].astype(str).eq(signal)].copy()
    candidate_rows["code"] = candidate_rows["code"].map(zfill_code)
    candidate_rows = candidate_rows[candidate_rows["boards"].between(2, 5)]
    if candidate_rows.empty:
        raise ValueError(f"no 2--5 board candidates for signal date {signal}")
    boards = candidate_rows.set_index("code")["boards"].astype(int).to_dict()
    minute_factors = {
        code: _minute_factors(group, point)
        for code, group in checked_minutes.groupby("code", sort=False)
    }
    registry = pd.DataFrame() if theme_sync is None else theme_sync.copy()
    rows = []
    for _, candidate in candidate_rows.iterrows():
        code = candidate["code"]
        auction_rows = checked_auction[checked_auction["code"].eq(code)]
        minute_rows = checked_minutes[checked_minutes["code"].eq(code)]
        factors = {
            **_auction_factors(auction_rows, point),
            **minute_factors.get(code, _minute_factors(pd.DataFrame(), point)),
        }
        factors["rivals_eliminated"] = _rivals_eliminated(code, int(candidate["boards"]), minute_factors, boards)
        raw = pd.concat([auction_rows[["source", "_source_timestamp"]], minute_rows[["source", "_source_timestamp"]]])
        source = " | ".join(sorted(raw["source"].dropna().astype(str).unique())) if not raw.empty else "unavailable"
        source_timestamp = raw["_source_timestamp"].max() if not raw.empty else cutoff
        status = "verified_derived_intraday_v9" if not raw.empty else "unavailable"
        rows.append({
            "signal_date": signal, "trade_date": trade, "asof": point,
            "code": code, "name": str(candidate.get("name", "")),
            "source": source, "source_timestamp": source_timestamp.isoformat(sep=" "),
            "data_status": status, "expectation_gap": math.nan,
            "auction_cancel_ratio_semantics": (
                "LEVEL1_MATCHED_AMOUNT_NET_DECAY_PROXY_NOT_GROSS_CANCELLATION"),
            **factors, **_theme_row(registry, signal, trade, point, code),
        })
    return opening.validate_snapshots(pd.DataFrame(rows), registry if not registry.empty else None)


def append_snapshots(snapshots: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Append immutable point-in-time keys to the derived archive."""
    keys = ["signal_date", "trade_date", "asof", "code"]
    incoming = opening.validate_snapshots(snapshots)
    if output.exists() and output.stat().st_size:
        existing = opening.validate_snapshots(pd.read_csv(output, dtype={"code": "string"}))
        existing_keys = set(map(tuple, existing[keys].astype(str).to_numpy()))
        duplicate = [tuple(row) for row in incoming[keys].astype(str).to_numpy() if tuple(row) in existing_keys]
        if duplicate:
            raise ValueError(f"immutable V9 snapshot keys already exist: {duplicate[:5]}")
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(keys).to_csv(output, index=False, encoding="utf-8-sig")
    return combined


def run(
    auction_path: Path,
    minute_path: Path,
    signal_date: str,
    trade_date: str,
    asof: str,
    candidates_path: Path = DEFAULT_CANDIDATES,
    theme_sync_path: Path = DEFAULT_THEME_SYNC,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    auction = pd.read_csv(auction_path, dtype={"code": "string"})
    minutes = pd.read_csv(minute_path, dtype={"code": "string"})
    candidates = pd.read_csv(candidates_path, dtype={"code": "string"}, low_memory=False)
    theme = pd.read_csv(theme_sync_path, dtype={"code": "string"}) if theme_sync_path.exists() and theme_sync_path.stat().st_size else pd.DataFrame()
    snapshots = build_snapshots(auction, minutes, candidates, signal_date, trade_date, asof, theme)
    stem = f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')}_to_{pd.Timestamp(trade_date).strftime('%Y-%m-%d')}_{opening.normalize_asof(asof).replace(':', '')}"
    manifest_path = output.parent / f"{stem}.manifest.json"
    if manifest_path.exists():
        raise ValueError("immutable V9 import manifest already exists")
    combined = append_snapshots(snapshots, output)
    manifest = {
        "version": VERSION, "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
        "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        "asof": opening.normalize_asof(asof), "imported_rows": int(len(snapshots)),
        "archive_rows": int(len(combined)),
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        "inputs": {
            "auction": {"path": str(auction_path.resolve()), "sha256": sha256_file(auction_path)},
            "minutes": {"path": str(minute_path.resolve()), "sha256": sha256_file(minute_path)},
            "candidates": {"path": str(candidates_path.resolve()), "sha256": sha256_file(candidates_path)},
        },
        "data_quality": {
            "auction_and_minute": "VERIFIED_RAW_EXPORT_REQUIRED",
            "derived_factors": "VERIFIED_DERIVED_INTRADAY_V9",
            "level2_queue": "UNAVAILABLE",
            "actual_fill": "UNAVAILABLE",
            "execution": "CONDITIONAL_PAPER_ONLY",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive V9 open factors from primitive intraday exports")
    parser.add_argument("--auction", type=Path, required=True)
    parser.add_argument("--minutes", type=Path, required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--asof", choices=opening.SNAPSHOT_TIMES, required=True)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--theme-sync", type=Path, default=DEFAULT_THEME_SYNC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(
        args.auction, args.minutes, args.signal_date, args.trade_date, args.asof,
        args.candidates, args.theme_sync, args.output,
    ), ensure_ascii=False, indent=2))
