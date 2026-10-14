"""Audit whether archived theme/competition evidence can train leader models.

Existing point-in-time theme snapshots are useful for daily review, but a
snapshot is not automatically fit for historical factor training.  This audit
requires explicit granularity, catalyst timing/type and expectation-state
fields, while carrying full-market completeness separately from theme coverage.
Missing fields remain unavailable; they are never converted to zero or to a
negative theme label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v149-theme-factor-readiness-1"
REQUIRED_FACTOR_FIELDS = {
    "theme_granularity",
    "driver_type",
    "driver_summary",
    "driver_first_available_at",
    "driver_claim_status",
    "expectation_status",
}
FULL_MARKET_VALUES = {
    "verified_full_market",
    "VERIFIED_FULL_MARKET",
    "proven_full_market",
    "PROVEN_FULL_MARKET",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _source_path(manifest: dict, suffix: str) -> Path | None:
    matches = [Path(path) for path in manifest.get("source_hashes", {}) if path.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def discover(root: Path):
    records = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        if (
            "closing_candidate_count" not in manifest
            or "verified_theme_candidate_count" not in manifest
            or "coverage" not in manifest
            or "signal_date" not in manifest
        ):
            continue
        competition_path = manifest_path.parent / "candidate_competition_snapshot.csv"
        theme_path = _source_path(manifest, "candidate_theme_roles.csv")
        theme_manifest_path = _source_path(manifest, "manifest.json")
        records.append({
            "competition_manifest_path": manifest_path,
            "competition_path": competition_path,
            "theme_path": theme_path,
            "theme_manifest_path": theme_manifest_path,
            "manifest": manifest,
        })
    return records


def _truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def audit(records, minimum_sessions=250, minimum_coverage=.80):
    if minimum_sessions < 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("invalid readiness policy")
    rows = []
    for record in records:
        manifest = record["manifest"]
        issues = []
        date = str(manifest.get("signal_date", ""))
        closing = manifest.get("closing_candidate_count")
        verified = manifest.get("verified_theme_candidate_count")
        coverage = manifest.get("coverage")
        try:
            closing, verified, coverage = int(closing), int(verified), float(coverage)
            if closing <= 0 or verified < 0 or verified > closing:
                raise ValueError
            if abs(coverage - verified / closing) > 1e-9:
                issues.append("COVERAGE_ARITHMETIC_MISMATCH")
        except (TypeError, ValueError, ZeroDivisionError):
            closing = verified = None
            coverage = None
            issues.append("INVALID_COVERAGE_COUNTS")

        competition_path, theme_path = record.get("competition_path"), record.get("theme_path")
        competition = theme = None
        if not competition_path or not Path(competition_path).exists():
            issues.append("COMPETITION_ROWS_MISSING")
        else:
            competition = pd.read_csv(competition_path, dtype={"code": str})
        if not theme_path or not Path(theme_path).exists():
            issues.append("THEME_ROWS_MISSING_OR_AMBIGUOUS_SOURCE")
        else:
            theme = pd.read_csv(theme_path, dtype={"code": str})

        factor_fields = set(theme.columns) if theme is not None else set()
        missing_factor_fields = sorted(REQUIRED_FACTOR_FIELDS - factor_fields)
        if missing_factor_fields:
            issues.append("THEME_FACTOR_SCHEMA_INCOMPLETE")
        point_in_time = False
        if competition is not None and not competition.empty:
            needed = {"signal_date", "code", "available_at", "observed_at", "data_status"}
            if needed <= set(competition):
                available = pd.to_datetime(competition.available_at, errors="coerce", utc=True)
                observed = pd.to_datetime(competition.observed_at, errors="coerce", utc=True)
                point_in_time = bool(
                    available.notna().all() and observed.notna().all()
                    and (observed <= available).all()
                    and competition.data_status.eq("verified").all()
                    and competition.signal_date.astype(str).eq(date).all()
                )
        if not point_in_time:
            issues.append("POINT_IN_TIME_COMPETITION_NOT_VERIFIED")

        full_market = str(manifest.get("full_market_completeness", "")) in FULL_MARKET_VALUES
        if not full_market:
            issues.append("FULL_MARKET_COMPLETENESS_NOT_PROVEN")
        if coverage is None or coverage < minimum_coverage:
            issues.append("THEME_COVERAGE_BELOW_POLICY")
        if not _truthy(manifest.get("complete_within_declared_roster")):
            issues.append("DECLARED_ROSTER_NOT_COMPLETE")

        descriptive_ready = bool(
            competition is not None and theme is not None and point_in_time
            and coverage is not None and coverage >= minimum_coverage
            and _truthy(manifest.get("complete_within_declared_roster"))
        )
        model_ready = descriptive_ready and full_market and not missing_factor_fields
        rows.append({
            "signal_date": date,
            "competition_manifest": str(Path(record["competition_manifest_path"]).resolve()),
            "closing_candidate_count": closing,
            "verified_theme_candidate_count": verified,
            "theme_coverage": coverage,
            "full_market_complete": full_market,
            "point_in_time_competition": point_in_time,
            "missing_factor_fields": ";".join(missing_factor_fields),
            "descriptive_review_ready": descriptive_ready,
            "leader_factor_model_ready": model_ready,
            "issues": ";".join(sorted(set(issues))),
        })

    frame = pd.DataFrame(rows).sort_values(
        ["signal_date", "competition_manifest"] if rows else []
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=[
        "signal_date", "competition_manifest", "closing_candidate_count",
        "verified_theme_candidate_count", "theme_coverage", "full_market_complete",
        "point_in_time_competition", "missing_factor_fields",
        "descriptive_review_ready", "leader_factor_model_ready", "issues",
    ])
    # Multiple immutable revisions of one date are visible, but only the best
    # currently evidenced revision counts once toward date-level readiness.
    by_date = []
    for date, group in frame.groupby("signal_date", sort=True):
        selected = group.sort_values(
            ["leader_factor_model_ready", "descriptive_review_ready", "theme_coverage"],
            ascending=[False, False, False],
        ).iloc[0]
        by_date.append(selected.to_dict())
    daily = pd.DataFrame(by_date, columns=frame.columns)
    model_sessions = int(daily.leader_factor_model_ready.sum()) if len(daily) else 0
    descriptive_sessions = int(daily.descriptive_review_ready.sum()) if len(daily) else 0
    summary = {
        "version": VERSION,
        "status": "READY_FOR_TIME_ISOLATED_THEME_FACTOR_MODEL" if model_sessions >= minimum_sessions
                  else "DIAGNOSTIC_ONLY",
        "minimum_sessions": minimum_sessions,
        "minimum_theme_coverage": minimum_coverage,
        "discovered_revisions": int(len(frame)),
        "distinct_sessions": int(len(daily)),
        "descriptive_review_ready_sessions": descriptive_sessions,
        "leader_factor_model_ready_sessions": model_sessions,
        "remaining_sessions_to_minimum": max(0, minimum_sessions - model_sessions),
        "full_market_complete_sessions": int(daily.full_market_complete.sum()) if len(daily) else 0,
        "complete_factor_schema_sessions": int(daily.missing_factor_fields.eq("").sum()) if len(daily) else 0,
        "missing_values_treated_as_zero": False,
        "allowed_use": (
            "time_isolated_theme_factor_training" if model_sessions >= minimum_sessions
            else "coverage_audit_and_daily_descriptive_review_only"
        ),
        "blocking_requirements": [
            "prove full-market closing-limit coverage for each training date",
            "archive explicit specific-theme granularity and catalyst timing/type",
            "archive expectation status without hindsight",
            f"reach at least {minimum_sessions} distinct model-ready sessions",
        ] if model_sessions < minimum_sessions else [],
    }
    return frame, daily, summary


def run(root_path, output_dir, minimum_sessions=250, minimum_coverage=.80):
    root, target = Path(root_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable readiness audit")
    records = discover(root)
    revisions, daily, summary = audit(records, minimum_sessions, minimum_coverage)
    target.mkdir(parents=True)
    revisions.to_csv(target / "all_revisions.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(target / "daily_best_evidence.csv", index=False, encoding="utf-8-sig")
    summary["root"] = str(root.resolve())
    summary["source_manifest_hashes"] = {
        str(Path(record["competition_manifest_path"]).resolve()): _sha(Path(record["competition_manifest_path"]))
        for record in records
    }
    (target / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-sessions", type=int, default=250)
    parser.add_argument("--minimum-coverage", type=float, default=.80)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.minimum_sessions, args.minimum_coverage),
                     ensure_ascii=False, indent=2))

