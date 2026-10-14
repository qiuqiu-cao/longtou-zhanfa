"""Collect bounded, immutable daily inputs for current severe-abnormal research.

The collector covers every main-board closing-limit candidate in the bound pool.
Provider OHLCV is archived with receipt time and hashes.  It is deliberately
not called official exchange reconciliation, corporate-action reconciliation,
or verified regulatory-distance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


VERSION = "v191-current-abnormal-input-collector-1"


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _parse_eastmoney(payload: dict) -> pd.DataFrame:
    data = payload.get("data") or {}
    rows = [str(line).split(",") for line in data.get("klines", [])]
    if not rows:
        raise ValueError("provider returned no rows")
    frame = pd.DataFrame(
        [row[:7] for row in rows],
        columns=["date", "open", "close", "high", "low", "volume", "amount"],
    )
    return frame


def _parse_tencent(payload: dict, symbol: str) -> pd.DataFrame:
    data = payload.get("data", {}).get(symbol, {})
    rows = data.get("day") or []
    if not rows:
        raise ValueError("provider returned no raw day rows")
    frame = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    frame["amount"] = pd.NA
    return frame


def _validate_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.strftime("%Y-%m-%d")
    result = result[result["date"].between(start, end)].sort_values("date")
    if result.empty or result["date"].duplicated().any():
        raise ValueError("empty or duplicate dated rows")
    for column in ("open", "close", "high", "low", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["open", "close", "high", "low"]].isna().any().any():
        raise ValueError("invalid price values")
    if not (result[["open", "close", "high", "low"]] > 0).all().all():
        raise ValueError("non-positive price values")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("invalid high price")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid low price")
    return result


def _fetch_one(code: str, start: str, end: str,
               raw_dir: Path, received_at: str) -> tuple[pd.DataFrame | None, dict]:
    symbol = ("sh" if code.startswith("6") else "sz") + code
    errors = []
    providers = (
        ("eastmoney", "https://push2his.eastmoney.com/api/qt/stock/kline/get", {
            "secid": ("1." if code.startswith("6") else "0.") + code,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "101", "fqt": "0",
            "beg": start.replace("-", ""), "end": end.replace("-", ""),
        }),
        ("tencent", "https://web.ifzq.gtimg.cn/appstock/app/kline/kline", {
            "param": f"{symbol},day,{start},{end},1000",
        }),
    )
    for provider, url, params in providers:
        try:
            response = requests.get(url, params=params, timeout=(8, 20))
            response.raise_for_status()
            payload = response.json()
            frame = (_parse_eastmoney(payload) if provider == "eastmoney"
                     else _parse_tencent(payload, symbol))
            frame = _validate_frame(frame, start, end)
            raw_path = raw_dir / f"{code}_{provider}.json"
            raw_path.write_bytes(response.content)
            frame["code"] = code
            frame["provider"] = provider
            frame["price_basis"] = "provider_unadjusted"
            frame["received_at"] = received_at
            frame["data_status"] = "provider_observed_not_official_reconciled"
            record = {
                "code": code, "status": "provider_observed", "provider": provider,
                "rows": int(len(frame)), "first_date": str(frame["date"].min()),
                "last_date": str(frame["date"].max()), "request_url": response.url,
                "raw_sha256": _sha256(raw_path), "errors": errors,
                "price_basis": "provider_unadjusted",
                "official_exchange_reconciled": False,
                "corporate_action_reconciled": False,
            }
            return frame, record
        except Exception as exc:  # archive provider failure and try bounded fallback
            errors.append({"provider": provider, "error": str(exc)[:500]})
    return None, {
        "code": code, "status": "unavailable", "errors": errors,
        "official_exchange_reconciled": False,
        "corporate_action_reconciled": False,
    }


def collect(pool_path: str | Path, pool_manifest_path: str | Path,
            output: str | Path, start: str, end: str,
            workers: int = 5) -> dict:
    pool_path, pool_manifest_path, target = (
        Path(pool_path), Path(pool_manifest_path), Path(output))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V191 archive")
    if not 1 <= workers <= 5:
        raise ValueError("workers must be between one and five")
    manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    signal_date = str(manifest.get("signal_date"))
    if end != signal_date:
        raise ValueError("collection end must equal bound signal_date")
    pool = pd.read_csv(pool_path, dtype={"code": str})
    required = {"signal_date", "code", "name", "pool_assignment", "is_mainboard"}
    if required - set(pool):
        raise ValueError("pool fields missing")
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    closing = pool[
        pool["signal_date"].astype(str).eq(signal_date)
        & pool["pool_assignment"].eq("CLOSING_LIMIT_UP")
        & pool["is_mainboard"].map(_truthy)
    ].copy()
    expected = int(manifest.get("mainboard_closing_limit_rows", -1))
    if closing.empty or closing["code"].duplicated().any() or len(closing) != expected:
        raise ValueError("pool does not match manifest-bound mainboard closing count")
    received_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    target.mkdir(parents=True, exist_ok=False)
    raw_dir = target / "raw"
    raw_dir.mkdir()
    frames, records = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_one, code, start, end, raw_dir, received_at): code
            for code in sorted(closing["code"])
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                frame, record = future.result()
            except Exception as exc:
                frame, record = None, {
                    "code": code, "status": "unavailable",
                    "errors": [{"provider": "collector", "error": str(exc)[:500]}],
                    "official_exchange_reconciled": False,
                    "corporate_action_reconciled": False,
                }
            records.append(record)
            if frame is not None:
                frames.append(frame)
    history = (pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
               if frames else pd.DataFrame())
    history_path = target / "candidate_daily_history.csv"
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    records = sorted(records, key=lambda row: row["code"])
    returned = [record for record in records if record["status"] == "provider_observed"]
    stale = [record for record in returned if record.get("last_date") != signal_date]
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_INPUT_ONLY",
        "collected_at": received_at, "signal_date": signal_date,
        "requested_start": start, "requested_end": end,
        "candidate_count": int(len(closing)),
        "provider_observed_count": int(len(returned)),
        "unavailable_count": int(len(closing) - len(returned)),
        "signal_date_complete_count": int(len(returned) - len(stale)),
        "stale_count": int(len(stale)), "rows": int(len(history)),
        "official_exchange_reconciled": False,
        "corporate_action_reconciled": False,
        "reset_chronology_reconciled": False,
        "point_in_time_security_eligibility_complete": bool(
            manifest.get("point_in_time_security_scope_complete") is True),
        "records": records,
        "source_hashes": {
            str(pool_path.resolve()): _sha256(pool_path),
            str(pool_manifest_path.resolve()): _sha256(pool_manifest_path),
        },
        "history_sha256": _sha256(history_path),
        "limitations": [
            "provider daily bars are not official exchange reconciliation",
            "corporate actions and historical no-price-limit sessions are not reconciled",
            "exchange-publication reset chronology is not collected",
            "input collection does not authorize ranking, probability, or entry",
        ],
    }
    (target / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(collect(args.pool, args.pool_manifest, args.output,
                             args.start, args.end, args.workers),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
