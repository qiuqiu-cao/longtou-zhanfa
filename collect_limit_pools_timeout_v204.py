"""Collect Eastmoney limit pools with explicit network timeouts and no overwrite.

AKShare's pool parsers are retained so field semantics stay identical to the
existing archive.  Only the HTTP transport is wrapped with finite timeouts and
bounded retries.  Partial collections are written to a new immutable folder
and the manifest never labels a failed pool as an empty verified pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(date: str, output: str, connect_timeout: float = 8,
            read_timeout: float = 15, retries: int = 3) -> dict:
    import akshare as ak
    import akshare.stock_feature.stock_ztb_em as ztb

    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable pool archive")
    target.mkdir(parents=True)

    original_get = requests.get

    def timed_get(*args, **kwargs):
        kwargs.setdefault("timeout", (connect_timeout, read_timeout))
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("User-Agent", "Mozilla/5.0")
        return original_get(*args, headers=headers, **kwargs)

    ztb.requests.get = timed_get
    calls = {
        "limit_up": ak.stock_zt_pool_em,
        "failed_board": ak.stock_zt_pool_zbgc_em,
        "limit_down": ak.stock_zt_pool_dtgc_em,
        "previous_limit": ak.stock_zt_pool_previous_em,
        "strong": ak.stock_zt_pool_strong_em,
    }
    manifest = {
        "version": "v204-timeout-limit-pool-collector-1",
        "date": pd.Timestamp(date).strftime("%Y%m%d"),
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Eastmoney via AKShare parser with bounded HTTP timeout",
        "transport": {
            "connect_timeout_seconds": connect_timeout,
            "read_timeout_seconds": read_timeout,
            "maximum_attempts": retries,
        },
        "datasets": {},
    }
    for kind, function in calls.items():
        errors = []
        frame = None
        for attempt in range(1, retries + 1):
            try:
                frame = function(date=manifest["date"])
                break
            except Exception as exc:  # explicit evidence, never silent fallback
                errors.append({"attempt": attempt, "error": str(exc)[:500]})
                if attempt < retries:
                    time.sleep(min(attempt, 2))
        if frame is None:
            manifest["datasets"][kind] = {
                "status": "unavailable", "attempts": errors}
            continue
        path = target / f"{kind}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest["datasets"][kind] = {
            "status": "delayed_provider_observation",
            "rows": int(len(frame)), "path": path.name,
            "sha256": _sha256(path), "attempt_errors": errors,
        }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--connect-timeout", type=float, default=8)
    parser.add_argument("--read-timeout", type=float, default=15)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(collect(
        args.date, args.output, args.connect_timeout,
        args.read_timeout, args.retries), ensure_ascii=False, indent=2))
