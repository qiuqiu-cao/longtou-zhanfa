#!/usr/bin/env python3
"""Recover delisted mainboard securities omitted by a current-code universe.

This is a V4 sidecar.  It does not mutate the frozen V3 data or model.  Official
exchange lists define which delisted securities belong to the historical
universe; Eastmoney is used only to recover unadjusted OHLCV.  SZSE dated name
history can verify point-in-time ST status.  SSE status remains unavailable
until equivalent dated evidence is collected.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from collect_security_master_v4 import name_status, normalize_name


ROOT = Path(__file__).resolve().parent
CURRENT_UNIVERSE = ROOT / "data" / "v3" / "imported" / "current-universe.csv"
SZSE_NAMES = ROOT / "data" / "v4" / "security_evidence" / "szse_short_name_changes_20260831.csv"
DATA_DIR = ROOT / "data" / "v4" / "historical_universe"
BARS_DIR = DATA_DIR / "bars"
RESULT_DIR = ROOT / "results" / "v4" / "historical_universe"
VERSION = "v4-historical-universe-1"

SSE_DELIST_URL = "https://www.sse.com.cn/assortment/stock/list/delisting/"
SSE_QUERY_URL = "https://query.sse.com.cn/commonQuery.do"
SZSE_DELIST_URL = "https://www.szse.cn/market/stock/suspend/index.html"
SZSE_REPORT_URL = "https://www.szse.cn/api/report/ShowReport"
HISTORY_URL = "https://quote.eastmoney.com/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_limit_price(pre_close: Any, ratio: float = 0.10) -> float:
    if pd.isna(pre_close):
        return math.nan
    value = Decimal(str(float(pre_close))) * (Decimal("1") + Decimal(str(ratio)))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def zfill_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    return digits[-6:].zfill(6)


def fetch_official_delists(timeout: int = 25) -> pd.DataFrame:
    headers = {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"}
    params = {
        "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "isPagination": "true",
        "STOCK_CODE": "", "CSRC_CODE": "", "REG_PROVINCE": "", "STOCK_TYPE": "1,2",
        "COMPANY_STATUS": "3", "type": "inParams", "pageHelp.beginPage": "1",
        "pageHelp.pageSize": "500", "pageHelp.pageNo": "1", "pageHelp.endPage": "1",
    }
    response = requests.get(SSE_QUERY_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    raw_sse = pd.DataFrame(response.json()["result"])
    sse = raw_sse.rename(columns={
        "COMPANY_CODE": "code", "COMPANY_ABBR": "final_name",
        "LIST_DATE": "listing_date", "DELIST_DATE": "delisting_date",
    })[["code", "final_name", "listing_date", "delisting_date"]]
    sse["exchange"] = "SSE"
    sse["source_ref"] = SSE_DELIST_URL

    sz_response = requests.get(SZSE_REPORT_URL, params={
        "SHOWTYPE": "xlsx", "CATALOGID": "1793_ssgs", "TABKEY": "tab2",
        "random": "0.6935816432433362",
    }, timeout=timeout)
    sz_response.raise_for_status()
    raw_szse = pd.read_excel(BytesIO(sz_response.content))
    szse = raw_szse.rename(columns={
        "证券代码": "code", "证券简称": "final_name",
        "上市日期": "listing_date", "终止上市日期": "delisting_date",
    })[["code", "final_name", "listing_date", "delisting_date"]]
    szse["exchange"] = "SZSE"
    szse["source_ref"] = SZSE_DELIST_URL

    frame = pd.concat([sse, szse], ignore_index=True)
    frame["code"] = frame["code"].map(zfill_code)
    frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="coerce", format="mixed").dt.normalize()
    frame["delisting_date"] = pd.to_datetime(frame["delisting_date"], errors="coerce", format="mixed").dt.normalize()
    frame = frame.dropna(subset=["listing_date", "delisting_date"])
    frame = frame[frame["code"].str.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))]
    # A duplicated SSE code can reflect legacy share-class records.  For the
    # A-share universe, retain the earliest listing and the common delist date.
    frame = frame.sort_values(["code", "listing_date"]).drop_duplicates("code", keep="first")
    exchanges = set(frame["exchange"].astype(str))
    if exchanges != {"SSE", "SZSE"}:
        raise ValueError(f"official delist collection lost an exchange: {sorted(exchanges)}")
    return frame.sort_values(["delisting_date", "code"]).reset_index(drop=True)


def current_codes(path: Path = CURRENT_UNIVERSE) -> set[str]:
    frame = pd.read_csv(path, dtype={"code": "string"})
    return set(frame["code"].astype(str).str.zfill(6))


def missing_window_universe(delists: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    observed = current_codes()
    relevant = delists[(delists["delisting_date"] >= start) & (delists["listing_date"] <= end)].copy()
    relevant["missing_from_current_universe"] = ~relevant["code"].isin(observed)
    return relevant.sort_values(["exchange", "code"]).reset_index(drop=True)


def load_szse_name_history(path: Path = SZSE_NAMES) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": "string"})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["change_date"] = pd.to_datetime(frame["change_date"], errors="raise").dt.normalize()
    return frame.sort_values(["code", "change_date"])


def effective_szse_name(events: pd.DataFrame, final_name: str, date: pd.Timestamp) -> str:
    before = events[events["change_date"] <= date]
    if not before.empty:
        return str(before.iloc[-1]["after_name"])
    after = events[events["change_date"] > date]
    if not after.empty:
        return str(after.iloc[0]["before_name"])
    return final_name


def normalize_history(raw: pd.DataFrame, security: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                      names: pd.DataFrame, provider: str = "Eastmoney unadjusted daily via AKShare") -> pd.DataFrame:
    columns = {
        "日期": "date", "股票代码": "provider_code", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover", "涨跌额": "change", "涨跌幅": "pct_change", "振幅": "amplitude",
    }
    if "日期" in raw.columns:
        frame = raw.rename(columns=columns).copy()
    else:
        frame = raw.copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError(f"history response missing columns: {sorted(required.difference(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].sort_values("date")
    for field in ["open", "high", "low", "close", "volume", "amount", "turnover", "change", "pct_change", "amplitude"]:
        if field not in frame:
            frame[field] = math.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    derived_pre_close = frame["close"] - frame["change"]
    frame["pre_close"] = derived_pre_close.where(derived_pre_close > 0, frame["close"].shift(1))
    frame["limit_price_10"] = frame["pre_close"].map(round_limit_price)
    frame["down_limit_price_10"] = frame["pre_close"].map(lambda value: round_limit_price(value, -0.10))
    frame["is_limit_10"] = (frame["close"] - frame["limit_price_10"]).abs() <= 0.011
    frame["touched_limit_10"] = frame["high"] >= frame["limit_price_10"] - 0.011
    frame["is_down_limit_10"] = (frame["close"] - frame["down_limit_price_10"]).abs() <= 0.011
    frame["code"] = str(security["code"])
    frame["exchange"] = str(security["exchange"])
    frame["final_name"] = str(security["final_name"])
    frame["delisting_date"] = pd.Timestamp(security["delisting_date"]).strftime("%Y-%m-%d")
    if security["exchange"] == "SZSE":
        events = names[names["code"].eq(security["code"])]
        frame["name"] = frame["date"].map(lambda date: effective_szse_name(events, security["final_name"], date))
        frame["st_status"] = frame["name"].map(name_status)
        frame["status_basis"] = "VERIFIED_SZSE_DATED_NAME_HISTORY"
    else:
        frame["name"] = ""
        frame["st_status"] = ""
        frame["status_basis"] = "UNAVAILABLE_SSE_DATED_NAME_HISTORY"
    frame["market_source"] = provider
    frame["market_data_status"] = "delayed_archival"
    keep = [
        "date", "code", "exchange", "name", "final_name", "open", "high", "low", "close", "pre_close",
        "volume", "amount", "turnover", "pct_change", "amplitude", "limit_price_10", "down_limit_price_10",
        "is_limit_10", "touched_limit_10", "is_down_limit_10", "st_status", "status_basis",
        "delisting_date", "market_source", "market_data_status",
    ]
    return frame[keep].dropna(subset=["pre_close"]).reset_index(drop=True)


def fetch_history_one(security: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                      names: pd.DataFrame, retries: int = 3) -> tuple[pd.DataFrame | None, str]:
    import akshare as ak

    fetch_start = (start - pd.Timedelta(days=35)).strftime("%Y%m%d")
    fetch_end = min(end, pd.Timestamp(security["delisting_date"])).strftime("%Y%m%d")
    errors: list[str] = []
    for attempt in range(retries):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=str(security["code"]), period="daily", start_date=fetch_start,
                end_date=fetch_end, adjust="", timeout=20,
            )
            if raw is not None and not raw.empty:
                return normalize_history(raw, security, start, end, names), ""
            errors.append("empty response")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {str(exc)[:180]}")
        time.sleep(0.5 * (attempt + 1))
    symbol = ("sh" if str(security["code"]).startswith("6") else "sz") + str(security["code"])
    for attempt in range(retries):
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=symbol, start_date=fetch_start, end_date=fetch_end, adjust="", timeout=20,
            )
            if raw is not None and not raw.empty:
                # AKShare labels Tencent's sixth field as amount, but the
                # provider payload is share volume (confirmed against the
                # overlapping Eastmoney rows).  Do not invent traded value.
                raw = raw.rename(columns={"amount": "volume"})
                raw["amount"] = math.nan
                return normalize_history(
                    raw, security, start, end, names,
                    provider="Tencent unadjusted daily via AKShare fallback",
                ), ""
            errors.append("Tencent: empty response")
        except Exception as exc:
            errors.append(f"Tencent {type(exc).__name__}: {str(exc)[:180]}")
        time.sleep(0.5 * (attempt + 1))
    return None, " | ".join(errors)[:700]


def longest_consecutive(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def collect(start: str, end: str, workers: int = 4, refresh: bool = False) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    retrieved = datetime.now(timezone.utc).isoformat()
    delists = fetch_official_delists()
    universe = missing_window_universe(delists, start_ts, end_ts)
    missing = universe[universe["missing_from_current_universe"]].copy()
    names = load_szse_name_history()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    delist_path = DATA_DIR / "official_delisted_mainboard.csv"
    universe_path = DATA_DIR / f"window_universe_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}.csv"
    delists.to_csv(delist_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    progress_rows: list[dict[str, Any]] = []
    todo: list[pd.Series] = []
    for _, security in missing.iterrows():
        path = BARS_DIR / f"{security['code']}.csv"
        if path.exists() and path.stat().st_size and not refresh:
            continue
        todo.append(security)
    with futures.ThreadPoolExecutor(max_workers=max(1, min(5, workers))) as executor:
        jobs = {executor.submit(fetch_history_one, security, start_ts, end_ts, names): security for security in todo}
        for future in futures.as_completed(jobs):
            security = jobs[future]
            try:
                frame, error = future.result()
            except Exception as exc:
                frame, error = None, f"{type(exc).__name__}: {str(exc)[:500]}"
            path = BARS_DIR / f"{security['code']}.csv"
            if frame is not None and not frame.empty:
                frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
                status, rows = "success", len(frame)
            else:
                status, rows = "failed", 0
            progress_rows.append({
                "code": security["code"], "exchange": security["exchange"], "status": status,
                "rows": rows, "error": error, "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            })

    audit_rows: list[dict[str, Any]] = []
    combined_parts: list[pd.DataFrame] = []
    for _, security in missing.iterrows():
        path = BARS_DIR / f"{security['code']}.csv"
        if path.exists() and path.stat().st_size:
            frame = pd.read_csv(path, dtype={"code": "string"}, parse_dates=["date"])
            combined_parts.append(frame)
            verified = frame[frame["status_basis"].eq("VERIFIED_SZSE_DATED_NAME_HISTORY")]
            normal = verified[verified["st_status"].eq("NORMAL")]
            potential = normal if security["exchange"] == "SZSE" else frame
            audit_rows.append({
                "code": security["code"], "exchange": security["exchange"], "bars_status": "RECOVERED",
                "rows": len(frame), "first_bar": frame["date"].min(), "last_bar": frame["date"].max(),
                "status_verified_rows": len(verified), "normal_10pct_limit_days": int((potential["is_limit_10"] == True).sum()),
                "max_consecutive_10pct_limits": longest_consecutive(potential["is_limit_10"]),
                "possible_2plus_board": longest_consecutive(potential["is_limit_10"]) >= 2,
            })
        else:
            audit_rows.append({
                "code": security["code"], "exchange": security["exchange"], "bars_status": "MISSING",
                "rows": 0, "first_bar": "", "last_bar": "", "status_verified_rows": 0,
                "normal_10pct_limit_days": 0, "max_consecutive_10pct_limits": 0, "possible_2plus_board": False,
            })
    audit = pd.DataFrame(audit_rows)
    combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    audit_path = RESULT_DIR / "delisted_bar_recovery_audit.csv"
    combined_path = DATA_DIR / "delisted_daily_bars.csv"
    progress_path = RESULT_DIR / "collection_progress.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    pd.DataFrame(progress_rows).to_csv(progress_path, index=False, encoding="utf-8-sig")

    recovered = int(audit["bars_status"].eq("RECOVERED").sum())
    verified_codes = int(audit.loc[audit["status_verified_rows"] > 0, "code"].nunique())
    possible = int(audit["possible_2plus_board"].sum())
    manifest = {
        "version": VERSION, "status": "PARTIAL_RECOVERY" if recovered < len(missing) or verified_codes < len(missing) else "RECOVERED",
        "start": start_ts.strftime("%Y-%m-%d"), "end": end_ts.strftime("%Y-%m-%d"),
        "retrieved_at_utc": retrieved, "official_recent_delisted_codes": int(len(universe)),
        "missing_from_current_universe_codes": int(len(missing)), "bars_recovered_codes": recovered,
        "bars_recovery_coverage": recovered / len(missing) if len(missing) else 1.0,
        "point_in_time_status_verified_codes": verified_codes,
        "point_in_time_status_coverage": verified_codes / len(missing) if len(missing) else 1.0,
        "possible_2plus_board_codes": possible,
        "delisted_omission_bar_coverage_complete": bool(recovered == len(missing)),
        "candidate_universe_complete": False,
        "point_in_time_status_complete": bool(verified_codes == len(missing)),
        "survivorship_bias_resolved": False,
        "universe_limit": "recovered delisted omissions do not prove the original current-stock universe and all point-in-time statuses are complete",
        "sse_status_limit": "dated SSE short-name/ST history unavailable",
        "sources": {"sse_delist": SSE_DELIST_URL, "szse_delist": SZSE_DELIST_URL, "ohlcv": HISTORY_URL},
        "files": {},
    }
    for key, path in {
        "official_delists": delist_path, "window_universe": universe_path, "combined_bars": combined_path,
        "audit": audit_path, "progress": progress_path,
    }.items():
        manifest["files"][key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    manifest_path = RESULT_DIR / "historical_universe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover delisted A-share mainboard history omitted by current universe")
    parser.add_argument("--start", default="2023-11-28")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(collect(args.start, args.end, args.workers, args.refresh), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
