"""Reconstruct same-height competitor elimination from adjacent archived pools.

A failed board's attempted height is previous-session board height plus one when
the name closed limit-up in the immediately preceding verified calendar
session; otherwise it is a first-board attempt.  Sparse archive dates are never
skipped.  Archived-roster-only sessions remain descriptive and are not marked
formal point-in-time evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v213-competitor-elimination-history-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> int:
    values = re.findall(r"_v(\d+)", path.as_posix())
    return int(values[-1]) if values else 0


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def discover(root: Path) -> dict[str, tuple[pd.DataFrame, dict, Path]]:
    selected = {}
    for path in root.rglob("all_closing_and_failed_candidates.csv"):
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except Exception:
            continue
        if frame.empty or "signal_date" not in frame:
            continue
        dates = frame.signal_date.astype(str).str[:10].unique()
        if len(dates) != 1:
            continue
        date = dates[0]
        item = (_revision(path), frame, _load(path.parent / "manifest.json"), path)
        if date not in selected or item[0] > selected[date][0]:
            selected[date] = item
    return {date: (item[1], item[2], item[3]) for date, item in selected.items()}


def build(root: Path, candidates: pd.DataFrame, calendar: list[str]):
    if not {"signal_date", "code", "name"} <= set(candidates):
        raise ValueError("candidate input requires signal_date/code/name")
    sessions = sorted(set(str(value)[:10] for value in calendar))
    previous = {sessions[index]: sessions[index - 1] for index in range(1, len(sessions))}
    pools = discover(root)
    output, sources = [], []
    for date, group in candidates.groupby(candidates.signal_date.astype(str).str[:10]):
        current = pools.get(date)
        prior = pools.get(previous.get(date, ""))
        if current:
            sources.append(current[2])
        if prior:
            sources.append(prior[2])
        if current and prior:
            frame, manifest, _ = current
            prev_frame, _, _ = prior
            closing = frame[frame.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
            failed = frame[frame.pool_assignment.eq("FAILED_BOARD_OBSERVATION")].copy()
            previous_closing = prev_frame[prev_frame.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
            previous_closing["code"] = previous_closing.code.astype(str).str.zfill(6)
            previous_boards = pd.to_numeric(previous_closing.boards, errors="coerce")
            prior_map = dict(zip(previous_closing.code, previous_boards))
            failed["code"] = failed.code.astype(str).str.zfill(6)
            failed["attempted_height"] = failed.code.map(
                lambda code: int(prior_map[code]) + 1
                if code in prior_map and pd.notna(prior_map[code]) else 1)
            closing["boards"] = pd.to_numeric(closing.boards, errors="coerce")
            closing_counts = closing.groupby("boards").code.size().to_dict()
            failed_counts = failed.groupby("attempted_height").code.size().to_dict()
            full_market = manifest.get("full_market_completeness") == "verified_full_market"
            roster_complete = manifest.get("complete_within_archived_roster") is True
            current_available_at = frame.get("available_at", pd.Series(dtype=object)).dropna().astype(str).max()
        else:
            closing_counts, failed_counts, full_market, roster_complete = {}, {}, False, False
            current_available_at = None
        for row in group.to_dict("records"):
            height = int(row.get("boards", 1))
            failed_count = int(failed_counts.get(height, 0))
            survivor_count = int(closing_counts.get(height, 0))
            available = bool(current and prior and roster_complete and survivor_count > 0)
            value = failed_count / (failed_count + survivor_count) if available else np.nan
            status = ("POINT_IN_TIME_VERIFIED" if available and full_market
                      else "ARCHIVED_ROSTER_ONLY" if available else "UNAVAILABLE")
            output.append({
                "signal_date": date, "code": str(row["code"]).zfill(6), "name": row.get("name"),
                "boards": height, "competitor_elimination": value,
                "competitor_elimination_evidence_status": status,
                "same_height_failed_attempt_count": failed_count if available else np.nan,
                "same_height_survivor_count": survivor_count if available else np.nan,
                "competitor_available_at": current_available_at if available else None,
                "immediate_previous_session": previous.get(date),
                "sparse_archive_date_skipped": False,
            })
    rows = pd.DataFrame(output)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)), "sessions": int(rows.signal_date.nunique()) if len(rows) else 0,
        "descriptive_roster_rows": int(rows.competitor_elimination_evidence_status.eq(
            "ARCHIVED_ROSTER_ONLY").sum()) if len(rows) else 0,
        "formal_point_in_time_rows": int(rows.competitor_elimination_evidence_status.eq(
            "POINT_IN_TIME_VERIFIED").sum()) if len(rows) else 0,
        "missing_is_zero": False,
        "formal_factor_allowed": bool(len(rows) and rows.competitor_elimination_evidence_status.eq(
            "POINT_IN_TIME_VERIFIED").all()),
    }
    return rows, report, sources


def run(root_path, candidates_path, calendar_path, output_dir):
    root, source, calendar_source, target = map(
        Path, (root_path, candidates_path, calendar_path, output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable elimination history")
    candidates = pd.read_csv(source, dtype={"code": str})
    calendar_frame = pd.read_csv(calendar_source)
    date_column = "date" if "date" in calendar_frame else "signal_date"
    rows, report, evidence = build(root, candidates, calendar_frame[date_column].astype(str).tolist())
    target.mkdir(parents=True)
    output = target / "competitor_elimination_evidence.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    paths = {source, calendar_source, Path(__file__), *evidence}
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths if path.exists()}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.candidates, args.calendar, args.output), ensure_ascii=False, indent=2))
