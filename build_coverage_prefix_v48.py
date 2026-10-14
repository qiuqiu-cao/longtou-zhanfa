#!/usr/bin/env python3
"""Freeze the maximal contiguous complete prefix from a V48 coverage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v48-complete-prefix-1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(cohort_path: Path, coverage_report_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    cohort = pd.read_csv(cohort_path, dtype={"code": str, "entry_date": str})
    if cohort.empty or cohort.duplicated(["code", "entry_date"]).any():
        raise ValueError("non-empty unique security-date cohort required")
    report = json.loads(coverage_report_path.read_text(encoding="utf-8"))
    if report.get("cohort_rows") != len(cohort):
        raise ValueError("coverage report does not match cohort row count")
    coverage_rows = report.get("coverage_by_entry_date", [])
    expected_days = sorted(cohort["entry_date"].unique())
    if [str(row.get("entry_date")) for row in coverage_rows] != expected_days:
        raise ValueError("coverage report dates do not match cohort dates")
    prefix_days = []
    first_incomplete = None
    for row in coverage_rows:
        if row.get("complete") is True and row.get("missing_rows") == 0:
            prefix_days.append(str(row["entry_date"]))
            continue
        first_incomplete = str(row["entry_date"])
        break
    if not prefix_days:
        raise ValueError("no complete prefix sessions")
    selected = cohort[cohort["entry_date"].isin(prefix_days)].copy()
    expected_rows = sum(int(row["cohort_rows"]) for row in coverage_rows[:len(prefix_days)])
    if len(selected) != expected_rows:
        raise ValueError("complete prefix row count mismatch")
    output.mkdir(parents=True)
    cohort_output = output / "cohort.csv"
    selected.to_csv(cohort_output, index=False)
    manifest = {
        "version": VERSION,
        "status": "DEVELOPMENT_ONLY_NOT_BLIND",
        "selection_rule": "maximal contiguous complete entry-date prefix ending before first coverage gap",
        "entry_date_start": prefix_days[0],
        "entry_date_end": prefix_days[-1],
        "complete_sessions": len(prefix_days),
        "cohort_rows": len(selected),
        "board_group_counts": selected["board_group"].value_counts().to_dict(),
        "first_incomplete_entry_date": first_incomplete,
        "later_complete_dates_ignored": [
            str(row["entry_date"]) for row in coverage_rows[len(prefix_days):]
            if row.get("complete") is True
        ],
        "outcomes_read": False,
        "source_hashes": {
            str(cohort_path.resolve()): sha256_file(cohort_path),
            str(coverage_report_path.resolve()): sha256_file(coverage_report_path),
            str(Path(__file__).resolve()): sha256_file(Path(__file__)),
            str(cohort_output.resolve()): sha256_file(cohort_output),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cohort, args.coverage_report, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
