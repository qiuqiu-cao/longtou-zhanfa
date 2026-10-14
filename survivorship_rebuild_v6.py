#!/usr/bin/env python3
"""Rebuild the research emotion tape with point-in-time candidate eligibility.

V6 keeps unmatched ordinary mainboard bars in market-breadth calculations, but
removes every bar proven ineligible by the official security master.  It then
audits the resulting 2--5 board candidates and admits only verified eligible
observations.  Frozen V3 and locked V4 artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
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
    pools_from_local_cache,
    pools_from_prepared_daily_bars,
    prepare_daily_bars,
)
from security_master_v4 import audit_candidates, validate_master
from survivorship_rebuild_v5 import sha256_file, verified_recovered_raw


ROOT = Path(__file__).resolve().parent
CURRENT_BARS = ROOT / "data" / "v3" / "imported" / "daily-bars.csv"
RECOVERED_BARS = ROOT / "data" / "v4" / "historical_universe" / "delisted_daily_bars.csv"
MASTER = ROOT / "data" / "v4" / "security_master_intervals.csv"
BASE_MARKET = ROOT / "data" / "v5" / "survivorship_rebuild" / "market_emotion.csv"
OUTPUT = ROOT / "data" / "v6" / "point_in_time_rebuild"
RESULT = ROOT / "results" / "v6" / "point_in_time_rebuild_manifest.json"
VERSION = "v6-point-in-time-rebuild-1"


def correct_current_raw(raw: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply known official intervals without pretending unmatched breadth is verified."""
    bars = raw.copy()
    bars["code"] = bars["code"].astype("string").str.zfill(6)
    bars["_date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    bars["_master_verified"] = False
    bars["_master_eligible"] = False
    bars["_historical_name"] = ""

    evidence = master[master["code"].isin(bars["code"].unique())].copy()
    evidence["_start"] = pd.to_datetime(evidence["valid_from"])
    evidence["_end"] = pd.to_datetime(evidence["valid_to"]).fillna(pd.Timestamp("2262-04-11"))
    indices = {code: part.index for code, part in bars.groupby("code", sort=False)}
    for code, part in evidence.groupby("code", sort=False):
        code_index = indices.get(code)
        if code_index is None:
            continue
        for _, interval in part.iterrows():
            covered = code_index[
                (bars.loc[code_index, "_date"] >= interval["_start"])
                & (bars.loc[code_index, "_date"] <= interval["_end"])
            ]
            if len(covered) == 0:
                continue
            verified = bool(interval["source_verified"])
            eligible = bool(
                verified
                and interval["board"] == "MAINBOARD"
                and interval["listing_status"] == "LISTED"
                and interval["st_status"] == "NORMAL"
                and interval["price_limit_regime"] == "MAINBOARD_10"
            )
            bars.loc[covered, "_master_verified"] = verified
            bars.loc[covered, "_master_eligible"] = eligible
            bars.loc[covered, "_historical_name"] = str(interval["name"])

    proven_ineligible = bars["_master_verified"] & ~bars["_master_eligible"]
    corrected = bars.loc[~proven_ineligible].copy()
    eligible = corrected["_master_eligible"]
    corrected.loc[eligible, "name"] = corrected.loc[eligible, "_historical_name"]
    corrected.loc[eligible, "st"] = False
    corrected.loc[eligible, "new_listing"] = False
    corrected.loc[eligible, "status_verified"] = True
    corrected["universe_complete"] = False
    corrected = corrected.drop(
        columns=["_date", "_master_verified", "_master_eligible", "_historical_name"]
    )
    metadata = {
        "raw_rows": int(len(bars)),
        "officially_covered_rows": int(bars["_master_verified"].sum()),
        "officially_eligible_rows": int(bars["_master_eligible"].sum()),
        "officially_ineligible_rows_removed": int(proven_ineligible.sum()),
        "unverified_breadth_rows_retained": int((~bars["_master_verified"]).sum()),
        "unverified_breadth_codes_retained": int(bars.loc[~bars["_master_verified"], "code"].nunique()),
    }
    return corrected, metadata


def _apply_end_cutoff(frame: pd.DataFrame, end_date: str | None, label: str) -> pd.DataFrame:
    """Apply an explicit historical cutoff before any feature construction."""
    if end_date is None:
        return frame
    if "date" not in frame:
        raise ValueError(f"{label} is missing date for end-date cutoff")
    cutoff = pd.Timestamp(end_date).normalize()
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    return frame.loc[dates.le(cutoff)].copy()


def rebuild(
    current_bars_path: Path = CURRENT_BARS,
    recovered_bars_path: Path = RECOVERED_BARS,
    master_path: Path = MASTER,
    base_market_path: Path = BASE_MARKET,
    output: Path = OUTPUT,
    result_path: Path = RESULT,
    end_date: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    master = validate_master(pd.read_csv(master_path, dtype={"code": "string"}))

    raw_current = pd.read_csv(current_bars_path, dtype={"code": "string"})
    raw_current = _apply_end_cutoff(raw_current, end_date, "current daily bars")
    corrected_current, correction = correct_current_raw(raw_current, master)
    corrected_current_path = output / "point_in_time_current_daily_bars.csv"
    corrected_current.to_csv(corrected_current_path, index=False, encoding="utf-8-sig")

    recovered = pd.read_csv(recovered_bars_path, dtype={"code": "string"})
    recovered = _apply_end_cutoff(recovered, end_date, "recovered daily bars")
    recovered_raw = verified_recovered_raw(recovered, master)
    recovered_raw_path = output / "verified_recovered_daily_bars.csv"
    recovered_raw.to_csv(recovered_raw_path, index=False, encoding="utf-8-sig")

    current_prepared = prepare_daily_bars(corrected_current_path)
    recovered_prepared = prepare_daily_bars(recovered_raw_path)
    if pd.concat([
        current_prepared[["date", "code"]], recovered_prepared[["date", "code"]]
    ]).duplicated(["date", "code"]).any():
        raise ValueError("current and recovered daily bars overlap on date/code")
    prepared = pd.concat([current_prepared, recovered_prepared], ignore_index=True, sort=False)
    prepared = prepared.sort_values(["code", "date"]).reset_index(drop=True)

    context = derive_daily_market_context(prepared)
    prepared_pools = pools_from_prepared_daily_bars(prepared)
    # A fixed historical cutoff must also exclude append-only recent API pools;
    # otherwise post-cutoff candidates leak back into the reconstruction.
    pools = (prepared_pools if end_date is not None else append_only_recent_pools(
        prepared_pools, pools_from_local_cache()
    ))
    market = build_market_table(pools, context)
    raw_candidates = build_candidate_table(pools, market)
    candidate_audit = audit_candidates(raw_candidates, master)
    admitted = candidate_audit["security_status"].eq("VERIFIED_ELIGIBLE").to_numpy()
    candidates = raw_candidates.loc[admitted].reset_index(drop=True)
    candidates = attach_daily_execution_outcomes(candidates, prepared)
    candidates = attach_terminal_height_labels(candidates)
    outcomes = build_outcomes(candidates)

    paths = {
        "market": output / "market_emotion.csv",
        "candidates": output / "leader_candidates.csv",
        "outcomes": output / "trade_outcomes.csv",
        "daily_context": output / "daily_market_context.csv",
        "candidate_audit": output / "candidate_security_audit.csv",
    }
    market.to_csv(paths["market"], index=False, encoding="utf-8-sig")
    candidates.to_csv(paths["candidates"], index=False, encoding="utf-8-sig")
    outcomes.to_csv(paths["outcomes"], index=False, encoding="utf-8-sig")
    context.to_csv(paths["daily_context"], index=False, encoding="utf-8-sig")
    candidate_audit.to_csv(paths["candidate_audit"], index=False, encoding="utf-8-sig")

    base = pd.read_csv(base_market_path)
    fields = ["date", "emotion_state", "limit_count", "failed_count", "down_limit_count", "max_board"]
    compare = base[fields].merge(market[fields], on="date", suffixes=("_v5", "_v6"), how="inner")
    status_counts = candidate_audit["security_status"].value_counts().to_dict()
    metadata: dict[str, Any] = {
        "version": VERSION,
        "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_v3_modified": False,
        "locked_v4_modified": False,
        "market_days": int(market["date"].nunique()),
        "raw_candidate_rows": int(len(raw_candidates)),
        "admitted_candidate_rows": int(len(candidates)),
        "candidate_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "candidate_point_in_time_status_complete": bool(candidate_audit["status_verified"].all()),
        "candidate_eligibility_rate": float(admitted.mean()) if len(admitted) else np.nan,
        "current_bar_correction": correction,
        "recovered_verified_bar_rows": int(len(recovered_raw)),
        "recovered_verified_codes": int(recovered_raw["code"].nunique()),
        "emotion_state_changed_days_vs_v5": int(compare["emotion_state_v5"].ne(compare["emotion_state_v6"]).sum()),
        "max_board_changed_days_vs_v5": int(compare["max_board_v5"].ne(compare["max_board_v6"]).sum()),
        "universe_complete": False,
        "market_breadth_point_in_time_status_complete": correction["unverified_breadth_rows_retained"] == 0,
        "deployment_effect": "research challenger only; candidate status is verified but historical breadth universe remains incomplete",
        "historical_end_cutoff": (pd.Timestamp(end_date).strftime("%Y-%m-%d") if end_date else None),
        "input_hashes": {
            "current_bars": sha256_file(current_bars_path),
            "recovered_bars": sha256_file(recovered_bars_path),
            "security_master": sha256_file(master_path),
            "base_market": sha256_file(base_market_path),
        },
        "files": {},
    }
    artifact_paths = {
        "corrected_current": corrected_current_path,
        "recovered_raw": recovered_raw_path,
        **paths,
    }
    for key, path in artifact_paths.items():
        metadata["files"][key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(result_path.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V6 point-in-time emotion challenger data")
    parser.add_argument("--current-bars", type=Path, default=CURRENT_BARS)
    parser.add_argument("--recovered-bars", type=Path, default=RECOVERED_BARS)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--base-market", type=Path, default=BASE_MARKET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--end-date", help="optional inclusive historical cutoff applied before rebuilding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(rebuild(
        args.current_bars, args.recovered_bars, args.master,
        args.base_market, args.output, args.result,
        args.end_date,
    ), ensure_ascii=False, indent=2))
