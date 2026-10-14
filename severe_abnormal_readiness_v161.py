"""Fail-closed readiness audit for the 100%/200% severe-abnormal path model.

The official rule calculation is not the same thing as a supervised label.  A
usable historical label additionally needs the contemporaneous benchmark,
corporate-action treatment, security eligibility, the exchange publication and
reset chronology, a complete candidate universe, and a right-censoring-safe
outcome.  This module keeps a diagnostic distance/path calculation available
while forbidding a calibrated hit probability whenever any of those items is
missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from severe_abnormal_rule_registry_v145 import verified_rule_for_code


VERSION = "v161-severe-abnormal-readiness-1"
OFFICIAL_BENCHMARK_STATUS = {"verified", "official_reconciled", "verified_official"}
VERIFIED_ROW_STATUS = {"verified", "official_reconciled", "verified_official"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _event_rule(event: dict) -> dict | None:
    try:
        return verified_rule_for_code(str(event["code"]).zfill(6), event["event_date"])
    except (KeyError, TypeError, ValueError):
        return None


def _ready_labels(library: dict) -> list[dict]:
    rows = list(library.get("events", [])) + list(library.get("negative_labels", []))
    return [row for row in rows if _truthy(row.get("training_label_ready"))]


def _label_value(row: dict):
    value = row.get("label")
    if value in (0, 1, False, True):
        return int(value)
    return 1 if _truthy(row.get("official_confirmed")) else None


def _benchmark_audit(events: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    required = {}
    for event in events:
        rule = _event_rule(event)
        if rule:
            required.setdefault(rule["benchmark_code"], []).append(event)
    issues = []
    per_symbol = {}
    for symbol, symbol_events in sorted(required.items()):
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            issues.append(f"MISSING_BENCHMARK:{symbol}")
            per_symbol[symbol] = {"present": False}
            continue
        frame = frame.copy()
        frame["date"] = frame["date"].astype(str).str[:10]
        frame = frame.sort_values("date").drop_duplicates("date", keep="last")
        statuses = set(frame.get("data_status", pd.Series(dtype=str)).dropna().astype(str))
        official = bool(statuses) and statuses <= OFFICIAL_BENCHMARK_STATUS
        if not official:
            issues.append(f"BENCHMARK_NOT_OFFICIAL_RECONCILED:{symbol}")
        missing_dates, insufficient_history = [], []
        dates = frame["date"].tolist()
        positions = {value: index for index, value in enumerate(dates)}
        for event in symbol_events:
            event_date = str(event.get("event_date", ""))[:10]
            target = int(event.get("target_pct", 100) or 100)
            needed = 31 if target >= 200 else 11
            if event_date not in positions:
                missing_dates.append(event_date)
            elif positions[event_date] + 1 < needed:
                insufficient_history.append(event_date)
        if missing_dates:
            issues.append(f"BENCHMARK_EVENT_DATES_MISSING:{symbol}")
        if insufficient_history:
            issues.append(f"BENCHMARK_WINDOW_HISTORY_INSUFFICIENT:{symbol}")
        per_symbol[symbol] = {
            "present": True,
            "rows": int(len(frame)),
            "date_min": dates[0] if dates else None,
            "date_max": dates[-1] if dates else None,
            "data_status_values": sorted(statuses),
            "official_reconciled": official,
            "missing_event_date_count": len(set(missing_dates)),
            "insufficient_window_history_count": len(set(insufficient_history)),
        }
    calendar_complete = not any(
        token in issue for issue in issues
        for token in ("MISSING_BENCHMARK", "EVENT_DATES_MISSING", "WINDOW_HISTORY_INSUFFICIENT")
    )
    official_complete = calendar_complete and not any("NOT_OFFICIAL_RECONCILED" in x for x in issues)
    return {
        "required_symbols": sorted(required),
        "per_symbol": per_symbol,
        "calendar_complete_for_scanned_events": calendar_complete,
        "official_reconciliation_complete": official_complete,
        "issues": issues,
    }


def _stock_data_audit(events: list[dict], daily: pd.DataFrame) -> dict:
    required_columns = {
        "date", "code", "status_verified", "st", "new_listing",
        "universe_complete", "adjustment_status", "price_limit_status",
    }
    missing = sorted(required_columns - set(daily.columns))
    if not {"date", "code"} <= set(daily.columns):
        return {
            "required_columns_present": False,
            "missing_columns": missing,
            "matched_event_rows": 0,
            "point_in_time_eligibility_complete": False,
            "corporate_action_treatment_complete": False,
            "price_limit_eligibility_complete": False,
            "full_market_universe_complete": False,
        }
    work = daily.copy()
    work["code"] = work["code"].astype(str).str.zfill(6)
    work["date"] = work["date"].astype(str).str[:10]
    keys = {(str(row.get("code", "")).zfill(6), str(row.get("event_date", ""))[:10])
            for row in events}
    matched = work.loc[[((code, date) in keys) for code, date in zip(work.code, work.date)]]
    matched_keys = set(zip(matched.code, matched.date))
    all_matched = keys <= matched_keys
    status_ok = (
        "status_verified" in matched and bool(len(matched)) and all_matched
        and matched.status_verified.map(_truthy).all()
    )
    st_ok = "st" in matched and not matched.st.map(_truthy).any()
    new_listing_ok = "new_listing" in matched and not matched.new_listing.map(_truthy).any()
    eligibility_ok = status_ok and st_ok and new_listing_ok
    adjustment_ok = (
        "adjustment_status" in matched and bool(len(matched)) and all_matched
        and matched.adjustment_status.astype(str).isin(VERIFIED_ROW_STATUS).all()
    )
    price_limit_ok = (
        "price_limit_status" in matched and bool(len(matched)) and all_matched
        and matched.price_limit_status.astype(str).isin(
            {"verified_limit_applicable", "verified"}).all()
    )
    universe_ok = (
        "universe_complete" in work and bool(len(work))
        and work.universe_complete.map(_truthy).all()
    )
    return {
        "required_columns_present": not missing,
        "missing_columns": missing,
        "matched_event_rows": int(len(matched)),
        "all_event_rows_matched": all_matched,
        "status_verified_rows": int(matched.status_verified.map(_truthy).sum())
        if "status_verified" in matched else 0,
        "universe_complete_rows": int(work.universe_complete.map(_truthy).sum())
        if "universe_complete" in work else 0,
        "point_in_time_eligibility_complete": bool(eligibility_ok),
        "corporate_action_treatment_complete": bool(adjustment_ok),
        "price_limit_eligibility_complete": bool(price_limit_ok),
        "full_market_universe_complete": bool(universe_ok),
    }


def audit(library: dict, benchmark_frames: dict[str, pd.DataFrame], daily: pd.DataFrame,
          minimum_positive_labels: int = 100, minimum_negative_labels: int = 100,
          minimum_sessions: int = 250, minimum_group_labels: int = 30) -> dict:
    if min(minimum_positive_labels, minimum_negative_labels, minimum_sessions,
           minimum_group_labels) < 1:
        raise ValueError("readiness thresholds must be positive")
    events = list(library.get("events", []))
    rule_rows = [_event_rule(event) for event in events]
    rule_coverage = bool(events) and all(rule_rows)
    rule_ids = sorted({row["rule_id"] for row in rule_rows if row})
    benchmarks = _benchmark_audit(events, benchmark_frames)
    stock = _stock_data_audit(events, daily)

    ready = _ready_labels(library)
    labelled = [(row, _label_value(row)) for row in ready]
    positives = [row for row, value in labelled if value == 1]
    negatives = [row for row, value in labelled if value == 0]
    sessions = {str(row.get("signal_date") or row.get("event_date"))[:10]
                for row, value in labelled if value in (0, 1)}
    group_counts = {
        group: sum(str(row.get("candidate_group", "")).upper() == group for row, value in labelled
                   if value in (0, 1))
        for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD")
    }
    timestamp_complete = bool(labelled) and all(
        row.get("feature_available_at") and row.get("outcome_observed_at")
        and row.get("signal_date") for row, value in labelled if value in (0, 1)
    )
    horizon_complete = bool(labelled) and all(
        int(row.get("target_horizon_sessions", 0) or 0) > 0
        and not _truthy(row.get("right_censored"))
        for row, value in labelled if value in (0, 1)
    )
    reset_events = library.get("reset_events")
    reset_complete = (
        library.get("reset_chronology_status") == "verified_complete"
        and isinstance(reset_events, list)
        and bool(reset_events or not events)
        and all(row.get("publication_date") and row.get("reset_effective_date")
                and str(row.get("status", "")) in VERIFIED_ROW_STATUS
                for row in (reset_events or []))
    )
    universe_complete = (
        library.get("candidate_universe_status") == "verified_full_market"
        and stock["full_market_universe_complete"]
    )
    counts_ready = (
        len(positives) >= minimum_positive_labels
        and len(negatives) >= minimum_negative_labels
        and len(sessions) >= minimum_sessions
        and all(value >= minimum_group_labels for value in group_counts.values())
    )
    gates = [
        {"name": "official_rule_version_coverage", "passed": rule_coverage,
         "observed": len(rule_ids), "requirement": "one verified exchange rule for every event date"},
        {"name": "benchmark_calendar_coverage", "passed": benchmarks["calendar_complete_for_scanned_events"],
         "observed": benchmarks["per_symbol"], "requirement": "benchmark event dates and complete rule windows"},
        {"name": "benchmark_official_reconciliation", "passed": benchmarks["official_reconciliation_complete"],
         "observed": benchmarks["per_symbol"], "requirement": "officially reconciled benchmark closes"},
        {"name": "official_publication_reset_chronology", "passed": reset_complete,
         "observed": library.get("reset_chronology_status"), "requirement": "verified complete publications and resets"},
        {"name": "stock_corporate_action_treatment", "passed": stock["corporate_action_treatment_complete"],
         "observed": stock, "requirement": "verified treatment for every event row"},
        {"name": "point_in_time_security_and_limit_status",
         "passed": stock["point_in_time_eligibility_complete"] and stock["price_limit_eligibility_complete"],
         "observed": stock, "requirement": "verified non-ST/new-listing and limit applicability"},
        {"name": "complete_candidate_universe", "passed": universe_complete,
         "observed": {"library": library.get("candidate_universe_status"),
                      "daily": stock["full_market_universe_complete"]},
         "requirement": "verified full-market eligible candidate set"},
        {"name": "point_in_time_feature_timestamps", "passed": timestamp_complete,
         "observed": sum(bool(row.get("feature_available_at")) for row, _ in labelled),
         "requirement": "feature and outcome availability timestamps on every label"},
        {"name": "right_censoring_safe_horizon", "passed": horizon_complete,
         "observed": sum(not _truthy(row.get("right_censored")) for row, _ in labelled),
         "requirement": "declared positive horizon and fully observed outcome"},
        {"name": "minimum_verified_label_support", "passed": counts_ready,
         "observed": {"positive": len(positives), "negative": len(negatives),
                      "sessions": len(sessions), "groups": group_counts},
         "requirement": {"positive": minimum_positive_labels, "negative": minimum_negative_labels,
                         "sessions": minimum_sessions, "each_group": minimum_group_labels}},
    ]
    training_ready = all(gate["passed"] for gate in gates)
    diagnostic_path_allowed = bool(
        rule_coverage and benchmarks["calendar_complete_for_scanned_events"]
    )
    return {
        "version": VERSION,
        "status": "READY_FOR_TIME_ISOLATED_CALIBRATION" if training_ready else "DIAGNOSTIC_ONLY",
        "scanned_events": len(events),
        "official_confirmed_events": sum(_truthy(row.get("official_confirmed")) for row in events),
        "training_ready_labels": len(labelled),
        "rule_ids": rule_ids,
        "benchmark_audit": benchmarks,
        "stock_data_audit": stock,
        "label_support": {"positive": len(positives), "negative": len(negatives),
                          "distinct_sessions": len(sessions), "candidate_groups": group_counts},
        "gates": gates,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "calibrated_hit_probability_allowed": training_ready,
        "candidate_reranking_allowed": training_ready,
        "diagnostic_counterfactual_path_rating_allowed": diagnostic_path_allowed,
        "missing_values_treated_as_zero": False,
        "interpretation": (
            "A formal 100%/200% rule-distance path may be shown as a diagnostic only; it is not a "
            "hit probability, buy signal, overflow guarantee, or proof that an exchange publication will occur."
            if not training_ready else
            "Inputs pass schema and support gates; calibration still requires a purged walk-forward and sealed blind test."
        ),
    }


def run(event_library_path: str | Path, benchmark_dir: str | Path,
        daily_bars_path: str | Path, output_path: str | Path, **thresholds) -> dict:
    event_path, bench_dir, daily_path, target = (
        Path(event_library_path), Path(benchmark_dir), Path(daily_bars_path), Path(output_path)
    )
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable readiness audit")
    library = json.loads(event_path.read_text(encoding="utf-8"))
    frames = {}
    for symbol in ("sh000002", "sz399107"):
        path = bench_dir / f"{symbol}.csv"
        if path.exists():
            frames[symbol] = pd.read_csv(path)
    daily = pd.read_csv(daily_path, dtype={"code": str}, low_memory=False)
    report = audit(library, frames, daily, **thresholds)
    report["source_hashes"] = {
        str(event_path.resolve()): _sha256(event_path),
        str(daily_path.resolve()): _sha256(daily_path),
        **{str((bench_dir / f"{symbol}.csv").resolve()): _sha256(bench_dir / f"{symbol}.csv")
           for symbol in frames},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-library", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--daily-bars", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-positive-labels", type=int, default=100)
    parser.add_argument("--minimum-negative-labels", type=int, default=100)
    parser.add_argument("--minimum-sessions", type=int, default=250)
    parser.add_argument("--minimum-group-labels", type=int, default=30)
    args = parser.parse_args()
    result = run(
        event_library_path=args.event_library,
        benchmark_dir=args.benchmark_dir,
        daily_bars_path=args.daily_bars,
        output_path=args.output,
        minimum_positive_labels=args.minimum_positive_labels,
        minimum_negative_labels=args.minimum_negative_labels,
        minimum_sessions=args.minimum_sessions,
        minimum_group_labels=args.minimum_group_labels,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
