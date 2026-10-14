#!/usr/bin/env python3
"""Audit a historical main-board window from official current and delisted enumerations."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VERSION = "v50-official-window-universe-1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def build(szse_master: pd.DataFrame, sse_current: pd.DataFrame, delisted: pd.DataFrame,
          daily_codes: set[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start cannot exceed end")

    sz = szse_master.copy()
    required_sz = {"code", "name", "listing_date", "exchange", "board", "source_type", "data_status"}
    if not required_sz.issubset(sz.columns):
        raise ValueError("SZSE master schema incomplete")
    sz["code"] = normalize_code(sz["code"])
    if (not sz["exchange"].eq("SZSE").all() or not sz["board"].eq("MAINBOARD").all()
            or not sz["source_type"].eq("szse_official").all()
            or not sz["data_status"].eq("verified_import").all()):
        raise ValueError("SZSE current enumeration is not fully official and verified")
    sz["valid_to"] = pd.to_datetime(sz.get("valid_to"), errors="coerce")
    sz = sz.sort_values(["code", "valid_to"]).groupby("code", as_index=False).tail(1)
    sz = sz[["code", "name", "listing_date"]].assign(
        exchange="SZSE", enumeration_source="OFFICIAL_CURRENT")

    sh = sse_current.copy()
    required_sh = {"A_STOCK_CODE", "SEC_NAME_CN", "LIST_DATE", "LIST_BOARD", "STOCK_TYPE"}
    if not required_sh.issubset(sh.columns):
        raise ValueError("SSE current schema incomplete")
    sh = sh[sh["LIST_BOARD"].astype(str).eq("1") & sh["STOCK_TYPE"].astype(str).eq("1")].copy()
    sh["code"] = normalize_code(sh["A_STOCK_CODE"])
    sh["listing_date"] = pd.to_datetime(sh["LIST_DATE"].astype(str), format="%Y%m%d", errors="raise").dt.strftime("%Y-%m-%d")
    sh = sh.rename(columns={"SEC_NAME_CN": "name"})[["code", "name", "listing_date"]].assign(
        exchange="SSE", enumeration_source="OFFICIAL_CURRENT")

    old = delisted.copy()
    required_old = {"code", "final_name", "listing_date", "delisting_date", "exchange", "source_ref"}
    if not required_old.issubset(old.columns):
        raise ValueError("delisted enumeration schema incomplete")
    old["code"] = normalize_code(old["code"])
    official_ref = old["source_ref"].astype(str).str.contains(r"(?:sse\.com\.cn|szse\.cn)", regex=True)
    if not official_ref.all():
        raise ValueError("delisted enumeration contains a non-exchange source")
    old["listing_date"] = pd.to_datetime(old["listing_date"], errors="raise")
    old["delisting_date"] = pd.to_datetime(old["delisting_date"], errors="raise")
    old = old[(old["listing_date"] <= end_ts) & (old["delisting_date"] >= start_ts)].copy()
    old = old.rename(columns={"final_name": "name"})
    old["listing_date"] = old["listing_date"].dt.strftime("%Y-%m-%d")
    old["delisting_date"] = old["delisting_date"].dt.strftime("%Y-%m-%d")
    old = old[["code", "name", "listing_date", "delisting_date", "exchange"]].assign(
        enumeration_source="OFFICIAL_DELISTED")

    current = pd.concat([sz, sh], ignore_index=True)
    current["listing_date"] = pd.to_datetime(current["listing_date"], errors="raise")
    current = current[current["listing_date"] <= end_ts].copy()
    current["listing_date"] = current["listing_date"].dt.strftime("%Y-%m-%d")
    current["delisting_date"] = ""
    overlap = set(current["code"]).intersection(old["code"])
    if overlap:
        raise ValueError(f"current and delisted enumerations overlap: {sorted(overlap)[:5]}")
    universe = pd.concat([current, old], ignore_index=True)
    if universe["code"].duplicated().any():
        raise ValueError("duplicate code in official window enumeration")
    universe = universe.sort_values(["exchange", "code"]).reset_index(drop=True)
    daily_codes = {str(code).zfill(6) for code in daily_codes}
    universe["daily_bars_present"] = universe["code"].isin(daily_codes)
    missing = universe[~universe["daily_bars_present"]].copy()
    missing["universe_complete"] = False
    missing = missing[["code", "name", "exchange", "listing_date", "delisting_date",
                       "enumeration_source", "universe_complete"]]
    report = {
        "version": VERSION,
        "status": ("OFFICIAL_ENUMERATION_AND_DAILY_COVERAGE_COMPLETE" if missing.empty
                   else "OFFICIAL_ENUMERATION_COMPLETE_DAILY_COVERAGE_INCOMPLETE"),
        "created_at": datetime.now().astimezone().isoformat(),
        "window_start": start,
        "window_end": end,
        "official_enumeration_complete": True,
        "official_window_codes": int(len(universe)),
        "official_current_codes": int(current["code"].nunique()),
        "official_window_delisted_codes": int(old["code"].nunique()),
        "daily_covered_codes": int(universe["daily_bars_present"].sum()),
        "daily_missing_codes": int(len(missing)),
        "daily_code_coverage_rate": float(universe["daily_bars_present"].mean()),
        "missing_by_exchange": {str(k): int(v) for k, v in missing["exchange"].value_counts().items()},
        "survivorship_bias_resolved": bool(missing.empty),
        "claim": "official code enumeration only; historical status and daily bars require separate verification",
    }
    return universe, missing, report


def run(szse_path: Path, sse_path: Path, delisted_path: Path, daily_path: Path,
        start: str, end: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    daily = pd.read_csv(daily_path, dtype={"code": str}, usecols=["code"])
    universe, missing, report = build(
        pd.read_csv(szse_path, dtype={"code": str}),
        pd.read_csv(sse_path, dtype={"A_STOCK_CODE": str}),
        pd.read_csv(delisted_path, dtype={"code": str}),
        set(normalize_code(daily["code"])), start, end)
    output.mkdir(parents=True)
    universe.to_csv(output / "official_window_universe.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output / "missing_daily_codes.csv", index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): sha256_file(path) for path in
                               (szse_path, sse_path, delisted_path, daily_path, Path(__file__))}
    for name in ("official_window_universe.csv", "missing_daily_codes.csv"):
        report["source_hashes"][name] = sha256_file(output / name)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--szse-master", type=Path, required=True)
    parser.add_argument("--sse-current", type=Path, required=True)
    parser.add_argument("--delisted", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.szse_master, args.sse_current, args.delisted, args.daily,
                         args.start, args.end, args.output), ensure_ascii=False, indent=2))
