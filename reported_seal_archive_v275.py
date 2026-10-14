"""Recover reported seal timestamps from original pools, never zero placeholders.

Pool completeness is provider-reported, not exchange-certified. Historic fetch
times remain retrospective; no auction queue, reseal duration or true fill is
inferred from the first/last-seal clocks. Missing next-session pools stay unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from next_session_forecast_lab_v271 import ROOT, sha

VERSION = "v275-reported-seal-archive-1"
CALENDAR_SOURCES = [
    "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml",
    "https://www.szse.cn/www/disclosure/notice/general/t20260917_622911.html",
]


def sessions():
    # This bounded calendar only uses the read 2026 Mid-Autumn/National Day notices.
    holidays = {"2026-09-25", *pd.date_range("2026-10-01", "2026-10-07").strftime("%Y-%m-%d")}
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-28", "2026-10-09")
            if d.strftime("%Y-%m-%d") not in holidays]


def clock(value):
    if value is None or pd.isna(value):
        return np.nan
    value = str(value).strip().split(".")[0].replace(":", "").zfill(6)
    if not value.isdigit() or len(value) != 6:
        return np.nan
    h, m, s = int(value[:2]), int(value[2:4]), int(value[4:])
    if h > 23 or m > 59 or s > 59:
        return np.nan
    seconds = h*3600+m*60+s
    valid = (seconds == 9*3600+25*60 or 9*3600+30*60 <= seconds <= 11*3600+30*60
             or 13*3600 <= seconds <= 15*3600)
    return seconds if valid else np.nan


def provider_json(path, calendar):
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data") or {}
    requested = path.stem.removeprefix("zt_pool_")
    date = pd.Timestamp(requested).strftime("%Y-%m-%d")
    if date not in calendar:
        return None, {"path": str(path), "date": date, "status": "NON_SESSION_EXCLUDED"}
    if str(data.get("qdate")) != requested:
        return None, {"path": str(path), "date": date, "status": "SOURCE_DATE_MISMATCH"}
    pool = data.get("pool") or []
    if not pool or data.get("tc") != len(pool):
        return None, {"path": str(path), "date": date, "status": "EMPTY_OR_INCOMPLETE_POOL"}
    raw = pd.DataFrame(pool).rename(columns={"c": "code", "n": "name", "lbc": "boards",
        "p": "price", "hs": "turnover", "fbt": "first_seal", "lbt": "last_seal", "zbc": "bomb_count",
        "hybk": "industry", "ltsz": "float_cap", "fund": "sealed_fund"})
    raw["price"] = pd.to_numeric(raw.price, errors="coerce")/1000
    raw["date"] = date
    raw["captured_at"] = None  # File modification time is not an original capture timestamp.
    return raw, {"path": str(path), "date": date, "status": "PROVIDER_REPORTED_COMPLETE", "rows_all_market": len(raw)}


def csv_pool(path, calendar):
    meta_path = path.parent / "manifest.json"
    if not meta_path.exists():
        return None, {"path": str(path), "status": "NO_CAPTURE_MANIFEST"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    date = pd.Timestamp(str(meta["date"])).strftime("%Y-%m-%d")
    if date not in calendar:
        return None, {"path": str(path), "date": date, "status": "NON_SESSION_EXCLUDED"}
    block = meta.get("datasets", {}).get("limit_up", {})
    if block.get("sha256") and block["sha256"] != sha(path):
        return None, {"path": str(path), "date": date, "status": "CAPTURE_HASH_MISMATCH"}
    raw = pd.read_csv(path, dtype={"代码": str})
    if block.get("rows") != len(raw) or not len(raw):
        return None, {"path": str(path), "date": date, "status": "ROW_COUNT_MISMATCH"}
    raw = raw.rename(columns={"代码": "code", "名称": "name", "连板数": "boards", "最新价": "price",
        "换手率": "turnover", "首次封板时间": "first_seal", "最后封板时间": "last_seal",
        "炸板次数": "bomb_count", "所属行业": "industry", "成交额": "amount",
        "流通市值": "float_cap", "封板资金": "sealed_fund"})
    raw["date"], raw["captured_at"] = date, meta.get("collected_at")
    return raw, {"path": str(path), "date": date, "status": "MANIFEST_BOUND_PROVIDER_POOL", "rows_all_market": len(raw)}


def normalize(raw, path):
    raw = raw.copy()
    raw["code"] = raw.code.astype(str).str.zfill(6)
    if raw.code.duplicated().any():
        raise ValueError("duplicate provider pool stock code")
    mask = raw.code.str.match(r"^(60|00)\d{4}$") & ~raw.name.astype(str).str.contains(r"ST|退", regex=True)
    raw = raw[mask].copy()
    for field in ["boards", "turnover", "bomb_count", "amount", "float_cap", "sealed_fund", "price"]:
        raw[field] = pd.to_numeric(raw[field], errors="coerce") if field in raw else np.nan
    raw["reported_first_seconds"] = raw.first_seal.map(clock)
    raw["reported_last_seconds"] = raw.last_seal.map(clock)
    valid = (raw.reported_first_seconds.notna() & raw.reported_last_seconds.notna()
             & (raw.reported_last_seconds >= raw.reported_first_seconds) & raw.bomb_count.ge(0))
    raw["reported_seal_status"] = np.where(valid, "PROVIDER_REPORTED_VALID_CLOCKS", "UNAVAILABLE_OR_INVALID")
    raw["first_last_clock_gap_minutes"] = (raw.reported_last_seconds-raw.reported_first_seconds)/60
    raw.loc[~valid, "first_last_clock_gap_minutes"] = np.nan
    raw["auction_sealed_clock_proxy"] = np.where(valid,
        ((raw.reported_first_seconds == 9*3600+25*60) & (raw.reported_last_seconds == 9*3600+25*60)
         & raw.bomb_count.eq(0)).astype(float), np.nan)
    raw["actual_one_word_ohlc_status"] = "NOT_CHECKED"
    raw["queue_and_reseal_duration_status"] = "UNAVAILABLE"
    raw["source_path"], raw["source_sha256"] = str(path.relative_to(ROOT)), sha(path)
    return raw


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable reported-seal archive exists")
    calendar = sessions()
    options, audit = {}, []
    paths = list((ROOT / "data/2026-09_eastmoney").glob("zt_pool_*.json"))
    paths += list((ROOT / "data/v3/raw").glob("*/limit_up.csv"))
    paths += list((ROOT / "data").glob("review_202609*_*/limit_up.csv"))
    for path in sorted(paths):
        raw, evidence = provider_json(path, calendar) if path.suffix == ".json" else csv_pool(path, calendar)
        audit.append(evidence)
        if raw is None:
            continue
        rows = normalize(raw, path)
        day = rows.date.iloc[0] if len(rows) else None
        if day:
            # Choose dated manifests ahead of untimed retrospective JSON. Within
            # manifest pools use the latest receive time, keeping every source in audit.
            priority = (int(path.suffix == ".csv"), str(rows.captured_at.iloc[0] or ""))
            if day not in options or priority > options[day][0]:
                options[day] = (priority, rows)
    rows = pd.concat([v[1] for _, v in sorted(options.items())], ignore_index=True)
    successor = dict(zip(calendar[:-1], calendar[1:]))
    by_date = {d: g.set_index("code") for d, g in rows.groupby("date")}
    rows["target_date"] = rows.date.map(successor)
    labels, statuses = [], []
    for _, row in rows.iterrows():
        next_pool = by_date.get(row.target_date)
        if next_pool is None:
            labels.append(np.nan); statuses.append("NEXT_SESSION_POOL_UNAVAILABLE")
        else:
            promoted = row.code in next_pool.index and next_pool.loc[row.code, "boards"] == row.boards+1
            labels.append(float(promoted)); statuses.append("PROVIDER_POOL_NEXT_SESSION_CLOSING_PROMOTION")
    rows["promoted_next_day"], rows["promotion_label_status"] = labels, statuses
    output.mkdir(parents=True)
    rows.to_csv(output / "reported_seal_rows.csv", index=False, encoding="utf-8-sig")
    report = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "rows": len(rows), "dates": int(rows.date.nunique()),
        "relay_rows": int(rows.boards.ge(2).sum()),
        "valid_seal_rows": int(rows.reported_seal_status.eq("PROVIDER_REPORTED_VALID_CLOCKS").sum()),
        "next_session_labels": int(rows.promoted_next_day.notna().sum()),
        "relay_complete_label_dates": int(rows[rows.boards.ge(2)].groupby("date").promoted_next_day.apply(lambda x:x.notna().all()).sum()),
        "calendar": {"sources": CALENDAR_SOURCES, "sessions": calendar, "scope": "2026-08-28 through 2026-10-09 only"},
        "source_audit": audit, "source_hashes": {str(path.relative_to(ROOT)):sha(path) for path in paths},
        "limitations": ["Provider-reported pool completeness, not exchange-certified universe",
            "Zero original reported bomb count is valid; zero reconstructed timestamp placeholder is invalid",
            "First-last clock gap includes non-trading time and cannot prove actual absorption/reseal duration",
            "Auction-seal proxy is not true OHLC one-word status or queue fill proof",
            "File times are not manufactured into original capture times; retrospective rows remain retrospective"],
        "forecast_accuracy_60_proved": False, "trade_authorized": False}
    (output / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    result=run(args.output)
    print(json.dumps({k:result[k] for k in ["status", "dates", "rows", "relay_rows", "valid_seal_rows", "relay_complete_label_dates"]}))
