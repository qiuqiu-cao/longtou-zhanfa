"""Build and validate the catalyst/expectation evidence backlog for themes.

The archive in V164 proves some point-in-time theme memberships but leaves the
leading-information fields empty.  This module turns those gaps into one task
per date/theme and validates completed evidence without treating rumours,
secondary-review publication times or unknown expectation states as facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v165-theme-catalyst-evidence-1"
REQUIRED = {
    "signal_date", "theme_group_id", "theme", "theme_granularity",
    "driver_type", "driver_summary", "source_url", "source_published_at",
    "driver_first_available_at", "driver_first_source_url", "first_time_basis", "driver_claim_status",
    "novelty_status", "expectation_status", "expectation_evidence",
    "expectation_observed_at", "expectation_source_url", "available_at", "evidence_status",
}
ENUMS = {
    "theme_granularity": {"specific", "broad", "mixed", "unavailable"},
    "driver_type": {"POLICY", "INDUSTRY", "EVENT", "ANNOUNCEMENT", "PURE_EMOTION", "MIXED", "UNAVAILABLE"},
    "first_time_basis": {"official_timestamp", "primary_source_timestamp", "verified_earliest_search",
                         "inferred_from_secondary", "unavailable"},
    "driver_claim_status": {"verified_official", "verified_primary", "verified_secondary",
                            "inferred", "rumor", "unavailable"},
    "novelty_status": {"new_information", "continuing_catalyst", "no_new_information", "unknown"},
    "expectation_status": {"underrecognized", "partially_priced", "fully_priced", "crowded", "unknown"},
    "evidence_status": {"verified", "delayed", "inferred", "unavailable"},
}
VERIFIED_FIRST_TIME = {"official_timestamp", "primary_source_timestamp", "verified_earliest_search"}
VERIFIED_CLAIMS = {"verified_official", "verified_primary", "verified_secondary"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    result = pd.Timestamp(value)
    return result.tz_localize("Asia/Shanghai") if result.tzinfo is None else result.tz_convert("Asia/Shanghai")


def _calendar_map(calendar: list[str]):
    dates = sorted(set(str(value)[:10] for value in calendar))
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def build_backlog(archive: pd.DataFrame) -> pd.DataFrame:
    missing = {"signal_date", "code", "theme_group_id", "theme", "boards",
               "point_in_time_available_for_next_session"} - set(archive)
    if missing:
        raise ValueError(f"archive fields missing: {sorted(missing)}")
    data = archive.copy()
    point_in_time = data.point_in_time_available_for_next_session.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"})
    data = data.loc[point_in_time & data.theme_group_id.notna()].copy()
    rows = []
    for (date, group_id), group in data.groupby(["signal_date", "theme_group_id"], sort=True):
        boards = pd.to_numeric(group.boards, errors="coerce")
        relay_count = int((boards >= 2).sum())
        max_board = int(boards.max()) if boards.notna().any() else None
        candidate_count = int(len(group))
        priority = "HIGH" if (max_board or 0) >= 3 or relay_count >= 2 else (
            "MEDIUM" if candidate_count >= 3 or relay_count >= 1 else "LOW")
        themes = group.theme.dropna().astype(str).unique().tolist()
        evidence_refs = group.get("primary_evidence_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        roster_times = group.get("theme_evidence_available_at", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        granularities = group.get("theme_granularity", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        summaries = group.get("driver_summary", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        rows.append({
            "signal_date": str(date), "theme_group_id": str(group_id),
            "theme": themes[0] if len(themes) == 1 else " | ".join(themes),
            "candidate_count": candidate_count,
            "first_board_count": int((boards == 1).sum()), "relay_count": relay_count,
            "max_board": max_board, "candidate_codes": ";".join(sorted(group.code.astype(str).str.zfill(6))),
            "collection_priority": priority,
            "theme_roster_source_urls": ";".join(evidence_refs),
            "theme_roster_available_at": ";".join(roster_times),
            "theme_granularity": granularities[0] if len(granularities) == 1 else "unavailable",
            "driver_type": "UNAVAILABLE",
            "driver_summary": summaries[0] if len(summaries) == 1 else "",
            "source_url": "", "source_published_at": "",
            "driver_first_available_at": "", "driver_first_source_url": "",
            "first_time_basis": "unavailable",
            "driver_claim_status": "unavailable", "novelty_status": "unknown",
            "expectation_status": "unknown", "expectation_evidence": "",
            "expectation_observed_at": "", "expectation_source_url": "",
            "available_at": "",
            "evidence_status": "unavailable", "backlog_status": "NEEDS_RESEARCH",
            "note": "theme roster time is not substituted for catalyst first-publication time",
        })
    if not rows:
        return pd.DataFrame(columns=[*sorted(REQUIRED), "backlog_status"])
    priority_order = pd.CategoricalDtype(["HIGH", "MEDIUM", "LOW"], ordered=True)
    result = pd.DataFrame(rows)
    result["collection_priority"] = result.collection_priority.astype(priority_order)
    return result.sort_values(["collection_priority", "signal_date", "theme_group_id"]).reset_index(drop=True)


def validate(frame: pd.DataFrame, calendar: list[str]) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"evidence fields missing: {sorted(missing)}")
    next_sessions = _calendar_map(calendar)
    rows = []
    for row in frame.to_dict("records"):
        issues = []
        for column, allowed in ENUMS.items():
            if str(row.get(column, "")) not in allowed:
                issues.append(f"INVALID_ENUM:{column}")
        date = str(row.get("signal_date", ""))[:10]
        next_session = next_sessions.get(date)
        cutoff = _stamp(f"{next_session}T09:15:00+08:00") if next_session else pd.NaT
        source_time = _stamp(row.get("source_published_at"))
        first_time = _stamp(row.get("driver_first_available_at"))
        expectation_time = _stamp(row.get("expectation_observed_at"))
        available_time = _stamp(row.get("available_at"))
        for url_field in ("source_url", "driver_first_source_url", "expectation_source_url"):
            source_url = str(row.get(url_field, "")).strip()
            if source_url and not source_url.startswith(("https://", "http://")):
                issues.append(f"INVALID_SOURCE_URL:{url_field}")
        timestamp_complete = all(pd.notna(value) for value in
                                 (source_time, first_time, expectation_time, available_time, cutoff))
        if timestamp_complete:
            if first_time > available_time or source_time > available_time or expectation_time > available_time:
                issues.append("EVIDENCE_AVAILABLE_BEFORE_OBSERVATION")
            if available_time > cutoff:
                issues.append("LATE_FOR_NEXT_SESSION_PREOPEN")
        else:
            issues.append("TIMESTAMP_INCOMPLETE")
        point_in_time = timestamp_complete and not any(
            issue in issues for issue in ("EVIDENCE_AVAILABLE_BEFORE_OBSERVATION", "LATE_FOR_NEXT_SESSION_PREOPEN"))
        verified_claim = str(row.get("driver_claim_status")) in VERIFIED_CLAIMS
        verified_first = str(row.get("first_time_basis")) in VERIFIED_FIRST_TIME
        known_expectation = str(row.get("expectation_status")) not in {"", "unknown"}
        known_novelty = str(row.get("novelty_status")) not in {"", "unknown"}
        specific = str(row.get("theme_granularity")) == "specific"
        complete_text = all(str(row.get(column, "")).strip() for column in
                            ("theme_group_id", "theme", "driver_summary", "source_url",
                             "driver_first_source_url", "expectation_evidence",
                             "expectation_source_url"))
        factor_ready = bool(
            not issues and point_in_time and verified_claim and verified_first
            and known_expectation and known_novelty and specific and complete_text
            and str(row.get("driver_type")) != "UNAVAILABLE"
            and str(row.get("evidence_status")) == "verified"
        )
        rumor = str(row.get("driver_claim_status")) == "rumor"
        rows.append({
            **row, "next_session": next_session,
            "next_session_preopen_cutoff": cutoff.isoformat() if pd.notna(cutoff) else None,
            "point_in_time_available": bool(point_in_time),
            "factor_schema_ready": factor_ready,
            "hard_fact_use_allowed": bool(verified_claim and not rumor and factor_ready),
            "positive_buy_reason_allowed": False,
            "model_use_allowed": False,
            "validation_issues": ";".join(sorted(set(issues))),
        })
    result = pd.DataFrame(rows)
    ready = int(result.factor_schema_ready.sum()) if len(result) else 0
    report = {
        "version": VERSION,
        "status": "EVIDENCE_SCHEMA_READY_FOR_FACTOR_TESTING" if ready == len(result) and ready else
                  "DIAGNOSTIC_ONLY",
        "rows": int(len(result)), "factor_schema_ready_rows": ready,
        "rumor_rows": int(result.driver_claim_status.astype(str).eq("rumor").sum()) if len(result) else 0,
        "late_or_incomplete_timestamp_rows": int(
            (~result.point_in_time_available).sum()) if len(result) else 0,
        "positive_buy_reason_allowed_rows": 0,
        "model_use_allowed_rows": 0,
        "policy": {
            "schema_ready_does_not_authorize_model_use": True,
            "rumor_cannot_be_hard_fact_or_buy_reason": True,
            "roster_publication_is_not_catalyst_first_time": True,
            "expectation_status_requires_contemporaneous_evidence": True,
        },
    }
    return result, report


def run_backlog(archive_path, output_dir):
    source, target = Path(archive_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable backlog")
    backlog = build_backlog(pd.read_csv(source, dtype={"code": str}))
    target.mkdir(parents=True)
    backlog_path = target / "theme_catalyst_evidence_backlog.csv"
    backlog.to_csv(backlog_path, index=False, encoding="utf-8-sig")
    report = {
        "version": VERSION, "status": "BACKLOG_READY",
        "theme_tasks": int(len(backlog)),
        "candidate_rows_covered": int(backlog.candidate_count.sum()) if len(backlog) else 0,
        "priority_counts": backlog.collection_priority.astype(str).value_counts().to_dict() if len(backlog) else {},
        "completed_factor_rows": 0,
        "source": {"path": str(source.resolve()), "sha256": _sha256(source)},
        "output": {"path": str(backlog_path.resolve()), "sha256": _sha256(backlog_path)},
        "next_action": "research driver first-publication, novelty and expectation evidence without overwriting roster timestamps",
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_validate(input_path, calendar_path, output_dir):
    source, calendar_source, target = Path(input_path), Path(calendar_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable validation")
    normalized, report = validate(
        pd.read_csv(source), pd.read_csv(calendar_source).date.astype(str).tolist())
    target.mkdir(parents=True)
    output = target / "validated_theme_catalyst_evidence.csv"
    normalized.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(source.resolve()): _sha256(source), str(calendar_source.resolve()): _sha256(calendar_source),
    }
    report["output"] = {"path": str(output.resolve()), "sha256": _sha256(output)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    backlog_parser = sub.add_parser("backlog")
    backlog_parser.add_argument("--archive", required=True)
    backlog_parser.add_argument("--output", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--calendar", default="data/v3/market_emotion.csv")
    validate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (run_backlog(args.archive, args.output) if args.command == "backlog"
              else run_validate(args.input, args.calendar, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
