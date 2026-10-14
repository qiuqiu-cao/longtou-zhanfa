#!/usr/bin/env python3
"""Freeze all verified first-board and consecutive-board candidates in an interval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_fixed_cohort_v40 import mapping


VERSION = "v48-full-pool-cohort-1"
ALLOWED_GROUPS = {"FIRST_BOARD", "CONSECUTIVE_BOARD"}


def build(frame: pd.DataFrame, sessions: pd.Series, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    dates = mapping(sessions.astype(str).tolist(), start, end)
    required = {"code", "signal_date", "boards", "start_known", "board_group", "board_status",
                "candidate_scope"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing candidate fields: " + ",".join(missing))
    data = frame.loc[frame["signal_date"].isin(dates)].copy()
    excluded = data.loc[~data["board_group"].isin(ALLOWED_GROUPS)].copy()
    data = data.loc[data["board_group"].isin(ALLOWED_GROUPS)].copy()
    if not data["start_known"].map(lambda value: value is True or str(value).lower() == "true").all():
        raise ValueError("eligible first/consecutive rows must have a known board-chain start")
    if data.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate candidate identity")
    data["entry_date"] = data["signal_date"].map(dates)
    data["provider_code"] = data["code"].map(lambda code: ("sh." if code.startswith("6") else "sz.") + code)
    columns = ["code", "signal_date", "boards", "start_known", "board_group", "board_status",
               "candidate_scope", "entry_date", "provider_code"]
    data = data[columns].sort_values(["entry_date", "code"]).reset_index(drop=True)
    counts = {str(key): int(value) for key, value in data.groupby("board_group").size().items()}
    audit = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "start": start,
        "end": end,
        "trading_sessions": len(dates),
        "eligible_rows": len(data),
        "board_group_counts": counts,
        "excluded_left_censored_or_unknown": len(excluded),
        "selection": "all supplied verified first and consecutive closing-limit candidates; no outcomes read",
        "not_blind": True,
    }
    return data, audit


def run(candidates: Path, calendar: Path, start: str, end: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("choose a new immutable cohort directory")
    frame = pd.read_csv(candidates, dtype={"code": str})
    frame["code"] = frame["code"].str.zfill(6)
    sessions = pd.read_csv(calendar)["date"]
    data, report = build(frame, sessions, start, end)
    output.mkdir(parents=True)
    cohort_path = output / "cohort.csv"
    data.to_csv(cohort_path, index=False)
    for day, group in data.groupby("entry_date"):
        (output / f"{day}_requests.json").write_text(
            json.dumps(group[["provider_code", "entry_date"]].values.tolist(), indent=2), encoding="utf-8")
    report["session_counts"] = {str(key): int(value) for key, value in data.groupby("signal_date").size().items()}
    report["source_hashes"] = {
        str(candidates.resolve()): hashlib.sha256(candidates.read_bytes()).hexdigest(),
        str(calendar.resolve()): hashlib.sha256(calendar.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str(cohort_path.resolve()): hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates, args.calendar, args.start, args.end, args.output),
                     ensure_ascii=False, indent=2))
