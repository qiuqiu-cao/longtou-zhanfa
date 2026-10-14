"""Join point-in-time theme membership to theme-level active-driver evidence.

The output is a historical coverage ledger and a prioritized backfill queue,
not a theme-strength score.  Membership and a review-time narrative are never
treated as proof of an active driver.  Same-height races receive first backfill
priority because they are required by the group comparison model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v207-theme-driver-history-1"
KEYS = ["signal_date", "code"]
DRIVER_FIELDS = [
    "theme_granularity", "driver_type", "driver_summary",
    "driver_first_available_at", "driver_claim_status", "expectation_status",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(archive: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_archive = set(KEYS + [
        "name", "boards", "theme_group_id", "theme_status",
        "point_in_time_available_for_next_session", "next_session_preopen_cutoff",
        "full_market_completeness",
    ])
    required_evidence = set(["signal_date", "theme_group_id", "available_at", "evidence_status"] + DRIVER_FIELDS)
    if required_archive - set(archive):
        raise ValueError(f"archive fields missing: {sorted(required_archive - set(archive))}")
    if required_evidence - set(evidence):
        raise ValueError(f"driver evidence fields missing: {sorted(required_evidence - set(evidence))}")
    if archive.duplicated(KEYS).any():
        raise ValueError("candidate archive keys must be unique")
    evidence_keys = ["signal_date", "theme_group_id"]
    if evidence.duplicated(evidence_keys).any():
        raise ValueError("driver evidence must have one selected revision per date/theme")

    left, right = archive.copy(), evidence.copy()
    left["code"] = left.code.astype(str).str.zfill(6)
    left["signal_date"] = left.signal_date.astype(str).str[:10]
    right["signal_date"] = right.signal_date.astype(str).str[:10]
    rename = {field: "driver_" + field if field in left else field for field in DRIVER_FIELDS}
    right = right.rename(columns=rename)
    joined_fields = evidence_keys + [rename[field] for field in DRIVER_FIELDS] + ["available_at", "evidence_status"]
    rows = left.merge(right[joined_fields], on=evidence_keys, how="left", validate="many_to_one")

    available = pd.to_datetime(rows.available_at, errors="coerce", utc=True)
    cutoff = pd.to_datetime(rows.next_session_preopen_cutoff, errors="coerce", utc=True)
    evidence_status = rows.evidence_status.fillna("UNAVAILABLE").astype(str).str.lower()
    factor_columns = [rename[field] for field in DRIVER_FIELDS]
    complete = rows[factor_columns].notna().all(axis=1)
    rows["driver_evidence_point_in_time"] = available.notna() & cutoff.notna() & available.le(cutoff)
    rows["driver_schema_complete"] = complete
    rows["active_driver_verified"] = (
        rows.driver_evidence_point_in_time & complete
        & evidence_status.isin({"verified", "inferred"})
        & rows.point_in_time_available_for_next_session.fillna(False).astype(bool)
    )
    rows["theme_driver_factor_model_eligible"] = (
        rows.active_driver_verified
        & rows.full_market_completeness.astype(str).eq("verified_full_market")
    )
    rows["missing_driver_is_zero"] = False

    rows["height_group_size"] = rows.groupby(["signal_date", "boards"]).code.transform("size")
    rows["same_height_race"] = rows.height_group_size.gt(1)
    rows["backfill_priority"] = (
        rows.same_height_race.astype(int) * 100
        + pd.to_numeric(rows.boards, errors="coerce").fillna(0).clip(upper=9) * 10
        + rows.theme_group_id.notna().astype(int) * 2
        + rows.point_in_time_available_for_next_session.fillna(False).astype(bool).astype(int)
    )
    rows["backfill_reason"] = np.select(
        [rows.theme_group_id.isna(), ~rows.driver_evidence_point_in_time,
         ~rows.driver_schema_complete, ~rows.active_driver_verified],
        ["MISSING_SPECIFIC_THEME", "MISSING_POINT_IN_TIME_DRIVER_EVIDENCE",
         "INCOMPLETE_DRIVER_OR_EXPECTATION_SCHEMA", "DRIVER_CLAIM_NOT_VERIFIED"],
        default="COMPLETE",
    )
    queue = rows[~rows.active_driver_verified].sort_values(
        ["backfill_priority", "signal_date", "boards", "code"],
        ascending=[False, True, False, True]).copy()
    queue = queue[[
        "signal_date", "code", "name", "boards", "theme_group_id", "theme_status",
        "same_height_race", "height_group_size", "backfill_priority", "backfill_reason",
        "next_session_preopen_cutoff",
    ]]
    daily = rows.groupby("signal_date", as_index=False).agg(
        candidate_count=("code", "size"),
        specific_theme_count=("theme_group_id", lambda x: x.notna().sum()),
        active_driver_verified_count=("active_driver_verified", "sum"),
        model_eligible_count=("theme_driver_factor_model_eligible", "sum"),
        same_height_race_candidates=("same_height_race", "sum"),
    )
    daily["active_driver_coverage"] = daily.active_driver_verified_count / daily.candidate_count
    ready_sessions = daily.model_eligible_count.eq(daily.candidate_count)
    report = {
        "version": VERSION,
        "status": "READY_FOR_FORMAL_FACTOR" if len(daily) >= 250 and ready_sessions.all() else "DIAGNOSTIC_ONLY",
        "sessions": int(len(daily)), "candidate_rows": int(len(rows)),
        "active_driver_verified_rows": int(rows.active_driver_verified.sum()),
        "model_eligible_rows": int(rows.theme_driver_factor_model_eligible.sum()),
        "fully_model_eligible_sessions": int(ready_sessions.sum()),
        "same_height_backfill_rows": int(queue.same_height_race.sum()),
        "minimum_sessions": 250,
        "formal_theme_factor_allowed": bool(len(daily) >= 250 and ready_sessions.all()),
        "missing_values_treated_as_zero": False,
        "daily_coverage": daily.to_dict("records"),
    }
    return rows, queue, report


def run(archive_path, evidence_path, output_dir):
    sources = [Path(archive_path), Path(evidence_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable theme-driver history")
    rows, queue, report = build(
        pd.read_csv(sources[0], dtype={"code": str}), pd.read_csv(sources[1]))
    target.mkdir(parents=True)
    ledger, backlog = target / "theme_driver_coverage_ledger.csv", target / "theme_driver_backfill_queue.csv"
    rows.to_csv(ledger, index=False, encoding="utf-8-sig")
    queue.to_csv(backlog, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in sources}
    report["artifact_hashes"] = {ledger.name: _sha256(ledger), backlog.name: _sha256(backlog)}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.evidence, args.output), ensure_ascii=False, indent=2))
