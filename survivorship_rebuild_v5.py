#!/usr/bin/env python3
"""Rebuild a research-only emotion tape with verified delisted securities.

This writes a separate V5 challenger dataset. It imports the frozen V3 feature
construction functions but never overwrites V3 data or the locked V4 model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leader_system_v3 import (
    append_only_recent_pools,
    attach_daily_execution_outcomes,
    attach_terminal_height_labels,
    build_candidate_table,
    build_market_table,
    build_outcomes,
    derive_daily_market_context,
    pools_from_prepared_daily_bars,
    pools_from_local_cache,
    prepare_daily_bars,
)
from security_master_v4 import validate_master


ROOT = Path(__file__).resolve().parent
CURRENT_BARS = ROOT / "data" / "v3" / "imported" / "daily-bars.csv"
RECOVERED_BARS = ROOT / "data" / "v4" / "historical_universe" / "delisted_daily_bars.csv"
MASTER = ROOT / "data" / "v4" / "security_master_intervals.csv"
BASE_MARKET = ROOT / "data" / "v3" / "market_emotion.csv"
OUTPUT = ROOT / "data" / "v5" / "survivorship_rebuild"
RESULT = ROOT / "results" / "v5" / "survivorship_rebuild_manifest.json"
VERSION = "v5-survivorship-rebuild-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_recovered_raw(recovered: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    bars = recovered.copy()
    bars["code"] = bars["code"].astype("string").str.zfill(6)
    bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    bars["_verified"] = False
    bars["_eligible"] = False
    bars["_historical_name"] = ""
    evidence = master[master["code"].isin(bars["code"].unique())].copy()
    evidence["_start"] = pd.to_datetime(evidence["valid_from"])
    evidence["_end"] = pd.to_datetime(evidence["valid_to"]).fillna(pd.Timestamp("2262-04-11"))
    for code, part in evidence.groupby("code", sort=False):
        code_index = bars.index[bars["code"].eq(code)]
        for _, interval in part.iterrows():
            covered = code_index[
                (bars.loc[code_index, "date"] >= interval["_start"])
                & (bars.loc[code_index, "date"] <= interval["_end"])
            ]
            if len(covered) == 0:
                continue
            verified = bool(interval["source_verified"])
            eligible = bool(
                verified and interval["board"] == "MAINBOARD" and interval["listing_status"] == "LISTED"
                and interval["st_status"] == "NORMAL" and interval["price_limit_regime"] == "MAINBOARD_10"
            )
            bars.loc[covered, "_verified"] = verified
            bars.loc[covered, "_eligible"] = eligible
            bars.loc[covered, "_historical_name"] = str(interval["name"])
    eligible = bars[bars["_eligible"]].copy()
    raw = pd.DataFrame({
        "date": eligible["date"].dt.strftime("%Y-%m-%d"),
        "code": eligible["code"],
        "name": eligible["_historical_name"],
        "open": eligible["open"], "high": eligible["high"], "low": eligible["low"],
        "close": eligible["close"], "pre_close": eligible["pre_close"],
        "volume": eligible.get("volume", np.nan), "amount": eligible.get("amount", np.nan),
        "turnover": eligible.get("turnover", np.nan), "float_cap": np.nan,
        "st": False, "new_listing": False, "status_verified": True,
        "industry": "", "theme": "", "benchmark_return": np.nan,
        "source": eligible.get("market_source", "recovered delayed archival"),
        "universe_complete": False,
    })
    return raw.sort_values(["date", "code"]).reset_index(drop=True)


def rebuild(
    current_bars_path: Path = CURRENT_BARS,
    recovered_bars_path: Path = RECOVERED_BARS,
    master_path: Path = MASTER,
    base_market_path: Path = BASE_MARKET,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    master = validate_master(pd.read_csv(master_path, dtype={"code": "string"}))
    recovered = pd.read_csv(recovered_bars_path, dtype={"code": "string"})
    recovered_raw = verified_recovered_raw(recovered, master)
    recovered_raw_path = output / "verified_recovered_daily_bars.csv"
    recovered_raw.to_csv(recovered_raw_path, index=False, encoding="utf-8-sig")

    current_prepared = prepare_daily_bars(current_bars_path)
    recovered_prepared = prepare_daily_bars(recovered_raw_path)
    duplicate = pd.concat([
        current_prepared[["date", "code"]], recovered_prepared[["date", "code"]]
    ]).duplicated(["date", "code"]).any()
    if duplicate:
        raise ValueError("current and recovered daily bars overlap on date/code")
    prepared = pd.concat([current_prepared, recovered_prepared], ignore_index=True, sort=False)
    prepared = prepared.sort_values(["code", "date"]).reset_index(drop=True)
    context = derive_daily_market_context(prepared)
    pools = append_only_recent_pools(
        pools_from_prepared_daily_bars(prepared), pools_from_local_cache(),
    )
    market = build_market_table(pools, context)
    candidates = build_candidate_table(pools, market)
    candidates = attach_daily_execution_outcomes(candidates, prepared)
    candidates = attach_terminal_height_labels(candidates)
    outcomes = build_outcomes(candidates)

    market_path = output / "market_emotion.csv"
    candidate_path = output / "leader_candidates.csv"
    outcome_path = output / "trade_outcomes.csv"
    context_path = output / "daily_market_context.csv"
    market.to_csv(market_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    outcomes.to_csv(outcome_path, index=False, encoding="utf-8-sig")
    context.to_csv(context_path, index=False, encoding="utf-8-sig")

    base = pd.read_csv(base_market_path)
    compare = base[["date", "emotion_state", "limit_count", "failed_count", "down_limit_count", "max_board"]].merge(
        market[["date", "emotion_state", "limit_count", "failed_count", "down_limit_count", "max_board"]],
        on="date", suffixes=("_v3", "_v5"), how="inner",
    )
    state_changed = compare["emotion_state_v3"].ne(compare["emotion_state_v5"])
    current_codes = set(current_prepared["code"].unique())
    recovered_candidate_rows = int((~candidates["code"].isin(current_codes)).sum())
    metadata = {
        "version": VERSION, "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_v3_modified": False, "locked_v4_modified": False,
        "market_days": int(market["date"].nunique()), "candidate_rows": int(len(candidates)),
        "recovered_verified_bar_rows": int(len(recovered_raw)),
        "recovered_verified_codes": int(recovered_raw["code"].nunique()),
        "recovered_candidate_rows": recovered_candidate_rows,
        "emotion_state_changed_days_vs_v3": int(state_changed.sum()),
        "emotion_state_changed_rate_vs_v3": float(state_changed.mean()),
        "max_limit_count_increase": float((compare["limit_count_v5"] - compare["limit_count_v3"]).max()),
        "max_failed_count_increase": float((compare["failed_count_v5"] - compare["failed_count_v3"]).max()),
        "max_down_limit_count_increase": float((compare["down_limit_count_v5"] - compare["down_limit_count_v3"]).max()),
        "max_board_changed_days": int(compare["max_board_v3"].ne(compare["max_board_v5"]).sum()),
        "universe_complete": False,
        "point_in_time_status_complete": False,
        "deployment_effect": "diagnostic challenger only; cannot replace frozen V3 or locked V4",
        "files": {},
        "input_hashes": {
            "current_bars": sha256_file(current_bars_path),
            "recovered_bars": sha256_file(recovered_bars_path),
            "security_master": sha256_file(master_path),
            "base_market": sha256_file(base_market_path),
        },
    }
    for key, path in {
        "recovered_raw": recovered_raw_path, "market": market_path, "candidates": candidate_path,
        "outcomes": outcome_path, "daily_context": context_path,
    }.items():
        metadata["files"][key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(RESULT.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build separate V5 survivorship-corrected challenger data")
    parser.add_argument("--current-bars", type=Path, default=CURRENT_BARS)
    parser.add_argument("--recovered-bars", type=Path, default=RECOVERED_BARS)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--base-market", type=Path, default=BASE_MARKET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = rebuild(args.current_bars, args.recovered_bars, args.master, args.base_market, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
