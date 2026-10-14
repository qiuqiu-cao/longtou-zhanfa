#!/usr/bin/env python3
"""Build verified SZSE intervals for delisted securities in the V4 window."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from collect_security_master_v4 import SZSE_NAME_URL, name_status
from security_master_v4 import MASTER_COLUMNS, validate_master


ROOT = Path(__file__).resolve().parent
DELISTS = ROOT / "data" / "v4" / "historical_universe" / "official_delisted_mainboard.csv"
NAMES = ROOT / "data" / "v4" / "security_evidence" / "szse_short_name_changes_20260831.csv"
OUTPUT = ROOT / "data" / "v4" / "security_master_szse_delisted_verified.csv"
MANIFEST = ROOT / "results" / "v4" / "security_master" / "szse_delisted_interval_manifest.json"
SZSE_DELIST_URL = "https://www.szse.cn/market/stock/suspend/index.html"
VERSION = "v4-szse-delisted-security-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_name(events: pd.DataFrame, final_name: str, moment: pd.Timestamp) -> str:
    before = events[events["change_date"] <= moment]
    if not before.empty:
        return str(before.iloc[-1]["after_name"])
    after = events[events["change_date"] > moment]
    if not after.empty:
        return str(after.iloc[0]["before_name"])
    return final_name


def regime_for_status(status: str) -> str:
    if status in {"ST", "*ST"}:
        return "ST_5"
    if status == "DELISTING_RISK":
        return "OTHER"
    return "MAINBOARD_10"


def build_delisted_intervals(
    delists: pd.DataFrame,
    names: pd.DataFrame,
    coverage_start: pd.Timestamp,
    asof: pd.Timestamp,
    source_timestamp: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_code = {code: part.sort_values("change_date") for code, part in names.groupby("code")}
    for stock in delists.itertuples(index=False):
        listing = pd.Timestamp(stock.listing_date).normalize()
        delisting = pd.Timestamp(stock.delisting_date).normalize()
        listed_start = max(listing, coverage_start)
        listed_end = min(delisting - timedelta(days=1), asof)
        events = by_code.get(stock.code, names.iloc[0:0])
        if listed_start <= listed_end:
            boundaries = {listed_start}
            boundaries.update(
                events.loc[events["change_date"].between(listed_start, listed_end), "change_date"].tolist()
            )
            ordered = sorted(boundaries)
            for index, boundary in enumerate(ordered):
                valid_to = ordered[index + 1] - timedelta(days=1) if index + 1 < len(ordered) else listed_end
                current_name = effective_name(events, stock.final_name, boundary)
                status = name_status(current_name)
                rows.append({
                    "code": stock.code, "name": current_name,
                    "valid_from": boundary.strftime("%Y-%m-%d"), "valid_to": valid_to.strftime("%Y-%m-%d"),
                    "exchange": "SZSE", "board": "MAINBOARD", "listing_status": "LISTED",
                    "st_status": status, "price_limit_regime": regime_for_status(status),
                    "listing_date": listing.strftime("%Y-%m-%d"), "delisting_date": delisting.strftime("%Y-%m-%d"),
                    "source_type": "szse_official", "source_ref": f"{SZSE_NAME_URL}|{SZSE_DELIST_URL}",
                    "source_timestamp": source_timestamp, "data_status": "verified_import",
                })
        if delisting <= asof:
            rows.append({
                "code": stock.code, "name": stock.final_name,
                "valid_from": delisting.strftime("%Y-%m-%d"), "valid_to": asof.strftime("%Y-%m-%d"),
                "exchange": "SZSE", "board": "MAINBOARD", "listing_status": "DELISTED",
                "st_status": "DELISTING_RISK", "price_limit_regime": "NONE",
                "listing_date": listing.strftime("%Y-%m-%d"), "delisting_date": delisting.strftime("%Y-%m-%d"),
                "source_type": "szse_official", "source_ref": f"{SZSE_NAME_URL}|{SZSE_DELIST_URL}",
                "source_timestamp": source_timestamp, "data_status": "verified_import",
            })
    return validate_master(pd.DataFrame(rows, columns=MASTER_COLUMNS)).loc[:, MASTER_COLUMNS]


def collect(
    coverage_start: str = "2023-11-28",
    asof: str = "2026-08-31",
    delists_path: Path = DELISTS,
    names_path: Path = NAMES,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    source_timestamp = datetime.now(timezone.utc).isoformat()
    delists = pd.read_csv(delists_path, dtype={"code": "string"})
    delists["listing_date"] = pd.to_datetime(delists["listing_date"], errors="raise")
    delists["delisting_date"] = pd.to_datetime(delists["delisting_date"], errors="raise")
    delists = delists[
        delists["exchange"].eq("SZSE")
        & delists["delisting_date"].ge(pd.Timestamp(coverage_start))
        & delists["listing_date"].le(pd.Timestamp(asof))
    ].copy()
    names = pd.read_csv(names_path, dtype={"code": "string"})
    names["change_date"] = pd.to_datetime(names["change_date"], errors="raise")
    missing_history = sorted(set(delists["code"]).difference(names["code"]))
    if missing_history:
        raise ValueError(f"delisted SZSE codes lack dated name history: {missing_history}")
    intervals = build_delisted_intervals(
        delists, names, pd.Timestamp(coverage_start), pd.Timestamp(asof), source_timestamp,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output, index=False, encoding="utf-8-sig")
    metadata = {
        "version": VERSION, "status": "VERIFIED_SZSE_DELISTED_INTERVALS",
        "coverage_start": coverage_start, "asof": asof,
        "delisted_codes": int(delists["code"].nunique()), "interval_rows": int(len(intervals)),
        "normal_interval_rows": int(intervals["st_status"].eq("NORMAL").sum()),
        "risk_interval_rows": int(intervals["st_status"].ne("NORMAL").sum()),
        "all_source_verified": True,
        "source_timestamp": source_timestamp,
        "sources": [SZSE_NAME_URL, SZSE_DELIST_URL],
        "files": {
            "delists": {"path": str(delists_path.resolve()), "sha256": sha256_file(delists_path)},
            "names": {"path": str(names_path.resolve()), "sha256": sha256_file(names_path)},
            "intervals": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(MANIFEST.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified SZSE delisted security intervals")
    parser.add_argument("--coverage-start", default="2023-11-28")
    parser.add_argument("--asof", default="2026-08-31")
    parser.add_argument("--delists", type=Path, default=DELISTS)
    parser.add_argument("--names", type=Path, default=NAMES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args.coverage_start, args.asof, args.delists, args.names, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
