#!/usr/bin/env python3
"""Independent public close-quote scan for the official current mainboard roster."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from public_live_capture_v1 import fetch_sina_quotes


SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v68-public-close-crosscheck-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limit_price(value: float) -> float:
    return float((Decimal(str(value)) * Decimal("1.10")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def official_codes(sse: pd.DataFrame, szse: pd.DataFrame, trade_date: str) -> set[str]:
    sse_codes = set(sse["A_STOCK_CODE"].astype(str).str.zfill(6))
    day = pd.Timestamp(trade_date)
    start = pd.to_datetime(szse["valid_from"], errors="coerce")
    end = pd.to_datetime(szse["valid_to"], errors="coerce")
    active = szse[
        start.le(day) & end.ge(day)
        & szse["listing_status"].astype(str).eq("LISTED")
        & szse["st_status"].astype(str).eq("NORMAL")
        & szse["price_limit_regime"].astype(str).eq("MAINBOARD_10")
    ]
    szse_codes = set(active["code"].astype(str).str.zfill(6))
    return {code for code in sse_codes | szse_codes
            if code.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))}


def classify(quotes: pd.DataFrame, trade_date: str):
    frame = quotes.copy()
    for column in ("open", "previous_close", "last", "high", "low", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    timestamps = pd.to_datetime(frame["quote_timestamp"], errors="coerce", utc=True)
    local_dates = timestamps.dt.tz_convert(SHANGHAI).dt.strftime("%Y-%m-%d")
    frame["fresh_close_date"] = local_dates.eq(pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
    frame = frame[frame.fresh_close_date & frame.previous_close.gt(0)].copy()
    frame["expected_10pct"] = frame.previous_close.map(limit_price)
    tolerance = 0.011
    frame["touched_limit"] = frame.high.ge(frame.expected_10pct - tolerance)
    frame["closed_limit"] = frame.last.ge(frame.expected_10pct - tolerance)
    frame["pct"] = (frame.last / frame.previous_close - 1) * 100
    frame["turnover"] = pd.NA
    frame["raw"] = ""
    out = pd.DataFrame({
        "code": frame.code.astype(str).str.zfill(6), "name": frame.provider_name,
        "close": frame.last, "prev": frame.previous_close, "open": frame.open,
        "timestamp": frame.quote_timestamp, "pct": frame.pct, "high": frame.high,
        "low": frame.low, "amount_wan": frame.amount / 10000,
        "turnover": frame.turnover, "limit_up": frame.expected_10pct,
        "raw": frame.raw, "expected_10pct": frame.expected_10pct,
    })
    closing = out[frame.closed_limit.to_numpy()].copy()
    failed = out[(frame.touched_limit & ~frame.closed_limit).to_numpy()].copy()
    return out, closing, failed, int(len(frame))


def run(sse_path: Path, szse_path: Path, trade_date: str, output: Path):
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable quote crosscheck")
    sse = pd.read_csv(sse_path, dtype=str)
    szse = pd.read_csv(szse_path, dtype=str)
    codes = official_codes(sse, szse, trade_date)
    fetched = fetch_sina_quotes(codes)
    quotes, closing, failed, fresh = classify(fetched.quotes, trade_date)
    output.mkdir(parents=True)
    paths = {"quotes": output / "quotes.csv", "closing": output / "closing.csv",
             "failed": output / "failed.csv"}
    quotes.to_csv(paths["quotes"], index=False, encoding="utf-8-sig")
    closing.to_csv(paths["closing"], index=False, encoding="utf-8-sig")
    failed.to_csv(paths["failed"], index=False, encoding="utf-8-sig")
    manifest = {
        "version": VERSION, "trade_date": trade_date,
        "collected_at": datetime.now(SHANGHAI).isoformat(),
        "source": "Sina public Level-1 close quotes",
        "official_roster_codes": len(codes), "returned_quotes": len(fetched.quotes),
        "fresh_trade_date_quotes": fresh, "provider_errors": fetched.errors,
        "closing_count": len(closing), "failed_count": len(failed),
        "classification": "10% limit rounded half-up to cent; 0.011 yuan tolerance",
        "limitations": [
            "independent close-price crosscheck, not tick or Level-2 evidence",
            "turnover is unavailable from this source and remains missing",
            "official rosters are current-day snapshots, not historical survivorship-free universes",
        ],
        "source_hashes": {str(sse_path.resolve()): sha(sse_path),
                          str(szse_path.resolve()): sha(szse_path)},
        "files": {key: {"path": str(path.resolve()), "sha256": sha(path)}
                  for key, path in paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sse", required=True)
    parser.add_argument("--szse", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.sse), Path(args.szse), args.date, Path(args.output)),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
