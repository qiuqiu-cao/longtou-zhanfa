#!/usr/bin/env python3
"""Collect verified SZSE point-in-time security-status intervals.

The collector deliberately limits its eligibility claim to the Shenzhen
mainboard.  SZSE publishes both the current stock list (including listing
dates) and a dated, internally consistent short-name change history.  That is
enough to reconstruct ST-name intervals without using today's name as a proxy
for the past.  SSE is not promoted to verified coverage until an equivalent
dated name history is available.

AKShare is used only as a transport adapter for the official SZSE XLSX
endpoints.  Raw response tables, source URLs, retrieval timestamps and hashes
are archived before derived intervals are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from security_master_v4 import MASTER_COLUMNS, validate_master


ROOT = Path(__file__).resolve().parent
DAILY_BARS = ROOT / "data" / "v3" / "imported" / "daily-bars.csv"
EVIDENCE_DIR = ROOT / "data" / "v4" / "security_evidence"
DERIVED_INTERVALS = ROOT / "data" / "v4" / "security_master_szse_verified.csv"
RESULT_DIR = ROOT / "results" / "v4" / "security_master"
VERSION = "v4-security-collector-1"

SZSE_LIST_URL = "https://www.szse.cn/market/product/stock/list/index.html"
SZSE_NAME_URL = "https://www.szse.cn/www/market/stock/changename/index.html"
SZSE_RULE_URL = "https://www.szse.cn/www/investor/knowledge/stock/deal/t20230525_600599.html"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def name_status(name: str) -> str:
    """Map an exchange short name to a conservative risk status."""
    compact = normalize_name(name)
    if re.match(r"^S?\*ST", compact):
        return "*ST"
    if re.match(r"^S?ST", compact):
        return "ST"
    if compact.startswith("PT") or "退" in compact:
        return "DELISTING_RISK"
    return "NORMAL"


def load_trading_dates(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for chunk in pd.read_csv(path, usecols=["date"], chunksize=500_000):
        parsed = pd.to_datetime(chunk["date"], errors="coerce").dropna().dt.normalize()
        dates.update(value for value in parsed if start <= value <= end)
    return sorted(dates)


def first_limited_date(listing_date: pd.Timestamp, trading_dates: Iterable[pd.Timestamp]) -> pd.Timestamp | None:
    """Return the sixth exchange trading date for post-registration IPOs.

    Before the first registration-based mainboard listings in April 2023, only
    listing day is excluded here; the model's historical observation window
    begins later, so this branch is retained mainly for completeness.
    """
    on_or_after = [day for day in trading_dates if day >= listing_date]
    required_index = 5 if listing_date >= pd.Timestamp("2023-04-10") else 1
    if len(on_or_after) <= required_index:
        return None
    return on_or_after[required_index]


def _effective_name(
    current_name: str,
    events: pd.DataFrame,
    moment: pd.Timestamp,
) -> str:
    before = events[events["change_date"] <= moment]
    if not before.empty:
        return str(before.iloc[-1]["after_name"])
    after = events[events["change_date"] > moment]
    if not after.empty:
        return str(after.iloc[0]["before_name"])
    return current_name


def verify_name_history(current: pd.DataFrame, changes: pd.DataFrame) -> dict[str, Any]:
    duplicates = changes.duplicated(["code", "change_date"], keep=False)
    if duplicates.any():
        raise ValueError("SZSE name history contains duplicate code/date events")
    latest = changes.sort_values("change_date").groupby("code", as_index=False).tail(1)
    merged = current.merge(latest[["code", "after_name"]], on="code", how="left")
    with_event = merged["after_name"].notna()
    mismatch = merged.loc[
        with_event & merged.apply(
            lambda row: normalize_name(row["current_name"]) != normalize_name(row["after_name"]), axis=1
        )
    ]
    if not mismatch.empty:
        raise ValueError(f"latest SZSE name event conflicts with current list: {mismatch['code'].head().tolist()}")
    return {
        "current_mainboard_codes": int(current["code"].nunique()),
        "codes_with_name_events": int(merged.loc[with_event, "code"].nunique()),
        "latest_name_mismatches": int(len(mismatch)),
        "duplicate_code_date_events": int(duplicates.sum()),
    }


def build_intervals(
    current: pd.DataFrame,
    changes: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    coverage_start: pd.Timestamp,
    asof: pd.Timestamp,
    source_timestamp: str,
    calendar_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = {code: part.sort_values("change_date") for code, part in changes.groupby("code")}
    for stock in current.itertuples(index=False):
        listing_date = pd.Timestamp(stock.listing_date).normalize()
        start = max(listing_date, coverage_start)
        if start > asof:
            continue
        events = grouped.get(stock.code, changes.iloc[0:0])
        events = events[(events["change_date"] >= start) & (events["change_date"] <= asof)]
        limited_from = first_limited_date(listing_date, trading_dates) if listing_date >= coverage_start else start
        boundaries = {start}
        boundaries.update(events["change_date"].tolist())
        if limited_from is not None and start < limited_from <= asof:
            boundaries.add(limited_from)
        ordered = sorted(boundaries)
        for index, boundary in enumerate(ordered):
            valid_to = (ordered[index + 1] - timedelta(days=1)) if index + 1 < len(ordered) else asof
            status = name_status(_effective_name(stock.current_name, grouped.get(stock.code, changes.iloc[0:0]), boundary))
            no_limit = limited_from is None or boundary < limited_from
            if no_limit:
                regime = "NO_LIMIT"
            elif status in {"ST", "*ST"}:
                regime = "ST_5"
            elif status == "DELISTING_RISK":
                regime = "OTHER"
            else:
                regime = "MAINBOARD_10"
            rows.append({
                "code": stock.code,
                "name": _effective_name(stock.current_name, grouped.get(stock.code, changes.iloc[0:0]), boundary),
                "valid_from": boundary.strftime("%Y-%m-%d"),
                "valid_to": valid_to.strftime("%Y-%m-%d"),
                "exchange": "SZSE",
                "board": "MAINBOARD",
                "listing_status": "LISTED",
                "st_status": status,
                "price_limit_regime": regime,
                "listing_date": listing_date.strftime("%Y-%m-%d"),
                "delisting_date": "",
                "source_type": "szse_official",
                "source_ref": f"{SZSE_LIST_URL}|{SZSE_NAME_URL}|{SZSE_RULE_URL}|calendar_sha256={calendar_hash}",
                "source_timestamp": source_timestamp,
                "data_status": "verified_import",
            })
    return validate_master(pd.DataFrame(rows, columns=MASTER_COLUMNS)).loc[:, MASTER_COLUMNS]


def collect(asof: str, daily_bars: Path = DAILY_BARS) -> dict[str, Any]:
    import akshare as ak

    asof_ts = pd.Timestamp(asof).normalize()
    retrieved = datetime.now(timezone.utc).isoformat()
    raw_current = ak.stock_info_sz_name_code("A股列表")
    raw_changes = ak.stock_info_sz_change_name("简称变更")
    current = raw_current.loc[raw_current["板块"].eq("主板")].copy()
    current = current.rename(columns={"A股代码": "code", "A股简称": "current_name", "A股上市日期": "listing_date"})
    current["code"] = current["code"].astype(str).str.zfill(6)
    current["listing_date"] = pd.to_datetime(current["listing_date"], errors="raise").dt.normalize()
    current = current[["code", "current_name", "listing_date"]].drop_duplicates("code")
    changes = raw_changes.rename(columns={
        "证券代码": "code", "变更日期": "change_date", "变更前简称": "before_name", "变更后简称": "after_name"
    })
    changes["code"] = changes["code"].astype(str).str.zfill(6)
    changes["change_date"] = pd.to_datetime(changes["change_date"], errors="raise").dt.normalize()
    changes = changes[["code", "change_date", "before_name", "after_name"]].sort_values(["code", "change_date"])
    checks = verify_name_history(current, changes)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    current_path = EVIDENCE_DIR / f"szse_mainboard_list_{asof_ts:%Y%m%d}.csv"
    changes_path = EVIDENCE_DIR / f"szse_short_name_changes_{asof_ts:%Y%m%d}.csv"
    current.to_csv(current_path, index=False, encoding="utf-8-sig")
    changes.to_csv(changes_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    calendar_hash = sha256_file(daily_bars)
    coverage_start = pd.Timestamp(pd.read_csv(daily_bars, usecols=["date"], nrows=1).iloc[0, 0]).normalize()
    trading_dates = load_trading_dates(daily_bars, coverage_start, asof_ts)
    intervals = build_intervals(
        current=current,
        changes=changes,
        trading_dates=trading_dates,
        coverage_start=coverage_start,
        asof=asof_ts,
        source_timestamp=retrieved,
        calendar_hash=calendar_hash,
    )
    DERIVED_INTERVALS.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(DERIVED_INTERVALS, index=False, encoding="utf-8-sig")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "status": "SZSE_OBSERVED_CODE_STATUS_RECONSTRUCTED",
        "asof": asof_ts.strftime("%Y-%m-%d"),
        "coverage_start": coverage_start.strftime("%Y-%m-%d"),
        "retrieved_at_utc": retrieved,
        **checks,
        "change_events": int(len(changes)),
        "interval_rows": int(len(intervals)),
        "st_interval_rows": int(intervals["st_status"].ne("NORMAL").sum()),
        "no_limit_interval_rows": int(intervals["price_limit_regime"].eq("NO_LIMIT").sum()),
        "scope_limit": "SZSE MAINBOARD only; SSE remains unverified",
        "survivorship_limit": "current SZSE list does not recover securities absent from the original historical candidate universe",
        "transport": "AKShare adapters over official SZSE XLSX endpoints",
        "sources": [SZSE_LIST_URL, SZSE_NAME_URL, SZSE_RULE_URL],
        "files": {
            "current_list": {"path": str(current_path.resolve()), "sha256": sha256_file(current_path)},
            "name_changes": {"path": str(changes_path.resolve()), "sha256": sha256_file(changes_path)},
            "daily_calendar_input": {"path": str(daily_bars.resolve()), "sha256": calendar_hash},
            "intervals": {"path": str(DERIVED_INTERVALS.resolve()), "sha256": sha256_file(DERIVED_INTERVALS)},
        },
    }
    manifest_path = RESULT_DIR / "security_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect verified SZSE security status history")
    parser.add_argument("--asof", required=True, help="Evidence cutoff date, YYYY-MM-DD")
    parser.add_argument("--daily-bars", type=Path, default=DAILY_BARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args.asof, args.daily_bars), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
