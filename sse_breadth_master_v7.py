#!/usr/bin/env python3
"""Archive point-in-time SSE status for the remaining market-breadth codes.

The immutable scope is the set of SSE mainboard codes present in the imported
daily bars but absent from the security master at first collection.  The actual
official-list, announcement-search, PDF, parsing and interval logic is shared
with ``sse_current_master_v5`` so candidate and breadth evidence use one rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import sse_current_master_v5 as core


ROOT = Path(__file__).resolve().parent
DAILY_BARS = ROOT / "data" / "v3" / "imported" / "daily-bars.csv"
MASTER = ROOT / "data" / "v4" / "security_master_intervals.csv"
BASE = ROOT / "data" / "v7" / "security_evidence" / "sse_breadth_gap"
SCOPE = BASE / "scope_audit.csv"
INTERVALS = ROOT / "data" / "v7" / "security_master_sse_breadth_evidenced.csv"
MANIFEST = ROOT / "results" / "v7" / "sse_breadth_security_master_manifest.json"
VERSION = "v7-sse-breadth-security-1"
SSE_PREFIXES = ("600", "601", "603", "605")
MAINBOARD_PREFIXES = SSE_PREFIXES + ("000", "001", "002", "003")


def derive_scope(daily_bars_path: Path = DAILY_BARS, master_path: Path = MASTER) -> list[str]:
    bars = pd.read_csv(daily_bars_path, usecols=["code"], dtype={"code": "string"})
    master = pd.read_csv(master_path, usecols=["code"], dtype={"code": "string"})
    bar_codes = bars["code"].astype("string").str.zfill(6)
    master_codes = set(master["code"].astype("string").str.zfill(6))
    mainboard = set(bar_codes[bar_codes.str.startswith(MAINBOARD_PREFIXES)])
    missing = sorted(mainboard.difference(master_codes))
    non_sse = [code for code in missing if not code.startswith(SSE_PREFIXES)]
    if non_sse:
        raise ValueError(f"uncovered daily mainboard includes non-SSE codes: {non_sse[:20]}")
    return missing


def freeze_scope(
    daily_bars_path: Path = DAILY_BARS,
    master_path: Path = MASTER,
    scope_path: Path = SCOPE,
) -> dict[str, Any]:
    if scope_path.exists():
        archived = pd.read_csv(scope_path, dtype={"code": "string"})
        codes = sorted(archived["code"].astype("string").str.zfill(6).unique())
        source = "IMMUTABLE_ARCHIVED_SCOPE"
    else:
        codes = derive_scope(daily_bars_path, master_path)
        if not codes:
            raise ValueError("no uncovered SSE breadth codes found before scope freeze")
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "code": codes,
            "security_status": "UNVERIFIED_POINT_IN_TIME_STATUS",
            "scope_reason": "daily mainboard code absent from security master at V7 scope freeze",
        }).to_csv(scope_path, index=False, encoding="utf-8-sig")
        source = "FROZEN_FROM_DAILY_BARS_MINUS_MASTER"
    if not codes or any(not code.startswith(SSE_PREFIXES) for code in codes):
        raise ValueError("archived breadth scope must contain only SSE mainboard codes")
    return {"scope_codes": len(codes), "scope_source": source, "scope": str(scope_path.resolve())}


def configure_core() -> None:
    core.GAP_AUDIT = SCOPE
    core.BASE = BASE
    core.RAW = BASE / "raw"
    core.PDF = BASE / "pdf"
    core.TEXT = BASE / "text"
    core.CURRENT_CSV = BASE / "sse_current_mainboard.csv"
    core.INDEX_CSV = BASE / "sse_action_index.csv"
    core.EVENTS_CSV = BASE / "sse_action_events.csv"
    core.INTERVALS_CSV = INTERVALS
    core.MANIFEST = MANIFEST
    core.VERSION = VERSION


def run(command: str, start: str, end: str, workers: int) -> dict[str, Any]:
    scope = freeze_scope()
    configure_core()
    if command == "scope":
        return scope
    if command == "collect":
        return {"scope": scope, "collect": core.collect_actions(start, end, workers)}
    if command == "download":
        return {"scope": scope, "download": core.download_candidates(workers)}
    if command == "extract":
        return {"scope": scope, "extract": core.extract_events()}
    if command == "build":
        return {"scope": scope, "build": core.build_intervals(start, end)}
    result = {"scope": scope}
    result["collect"] = core.collect_actions(start, end, workers)
    result["download"] = core.download_candidates(workers)
    result["extract"] = core.extract_events()
    result["build"] = core.build_intervals(start, end)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V7 SSE market-breadth point-in-time master")
    parser.add_argument("command", choices=["scope", "collect", "download", "extract", "build", "all"])
    parser.add_argument("--start", default="2023-11-28")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.command, args.start, args.end, args.workers), ensure_ascii=False, indent=2))
