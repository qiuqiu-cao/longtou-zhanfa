"""Audit whether point-in-time theme coverage can support profit ranking.

Theme evidence availability is not itself theme strength.  This audit joins the
calendar-corrected, time-isolated first-board profit sample to the archived
theme roster, keeps missing membership unknown, and measures coverage selection
without promoting it into a stock factor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _metrics(part: pd.DataFrame) -> dict:
    observed = part[part.profit_target_available.eq(1)]
    filled = part[part.fill_status.eq("FILLED_OPEN_PROXY")]
    b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    big = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return {
        "candidate_n": int(len(part)),
        "session_n": int(part.signal_date.nunique()),
        "promotion_rate": _rate(part.promotion_label),
        "profit_target_observed_n": int(len(observed)),
        "executable_B_plus_2_positive_rate": _rate(
            observed.executable_positive_B_plus_2),
        "filled_n": int(len(filled)),
        "B_plus_2_known_n": int(len(b2)),
        "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
        "B_plus_2_positive_rate_filled": float(b2.gt(0).mean()) if len(b2) else None,
        "B_plus_1_big_loss_known_n": int(len(big)),
        "B_plus_1_big_loss_rate": float(big.mean()) if len(big) else None,
    }


def build_rows(predictions: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    required_predictions = set(KEYS + [
        "promotion_label", "fill_status", "profit_target_available",
        "executable_positive_B_plus_2", "net_return_B_plus_2",
        "big_loss_B_plus_1",
    ])
    required_archive = set(KEYS + [
        "theme_status", "point_in_time_available_for_next_session",
        "theme_factor_model_eligible", "full_market_completeness",
        "driver_claim_status", "expectation_status",
    ])
    if required_predictions - set(predictions):
        raise ValueError("profit prediction evidence fields missing")
    if required_archive - set(archive):
        raise ValueError("theme archive evidence fields missing")
    if predictions.duplicated(KEYS).any() or archive.duplicated(KEYS).any():
        raise ValueError("candidate keys must be unique")
    archive_view = archive[list(required_archive)].copy()
    rows = predictions.merge(archive_view, on=KEYS, how="left", validate="one_to_one")
    point_in_time = rows.point_in_time_available_for_next_session.fillna(False).astype(bool)
    verified_roster = rows.theme_status.fillna("UNKNOWN").astype(str).str.startswith("verified_")
    rows["theme_roster_available_preopen"] = (point_in_time & verified_roster).astype(int)
    rows["theme_factor_eligible"] = (
        rows.theme_factor_model_eligible.fillna(False).astype(bool)).astype(int)
    rows["theme_missing_or_unavailable"] = rows.theme_roster_available_preopen.eq(0).astype(int)
    rows["theme_missing_is_weak"] = 0
    return rows


def _difference(rows: pd.DataFrame, field: str) -> float | None:
    known = _metrics(rows[rows.theme_roster_available_preopen.eq(1)]).get(field)
    unknown = _metrics(rows[rows.theme_roster_available_preopen.eq(0)]).get(field)
    return float(known - unknown) if known is not None and unknown is not None else None


def _cluster_bootstrap(rows: pd.DataFrame, repetitions: int = 2000,
                       seed: int = 197) -> dict:
    mixed_dates = [
        date for date, group in rows.groupby("signal_date")
        if group.theme_roster_available_preopen.nunique() == 2
    ]
    fields = [
        "promotion_rate", "executable_B_plus_2_positive_rate",
        "B_plus_2_mean_net_return", "B_plus_1_big_loss_rate",
    ]
    if not mixed_dates:
        return {"mixed_sessions": 0, "intervals": {field: None for field in fields}}
    groups = {date: rows[rows.signal_date.eq(date)] for date in mixed_dates}
    rng = np.random.default_rng(seed)
    draws = {field: [] for field in fields}
    for _ in range(repetitions):
        sampled = rng.choice(mixed_dates, len(mixed_dates), replace=True)
        frame = pd.concat([groups[date] for date in sampled], ignore_index=True)
        for field in fields:
            value = _difference(frame, field)
            if value is not None and np.isfinite(value):
                draws[field].append(value)
    return {
        "mixed_sessions": int(len(mixed_dates)),
        "intervals": {
            field: np.quantile(values, [.025, .975]).tolist() if values else None
            for field, values in draws.items()
        },
    }


def audit(rows: pd.DataFrame, bootstrap_repetitions: int = 2000) -> dict:
    known = rows[rows.theme_roster_available_preopen.eq(1)]
    unknown = rows[rows.theme_roster_available_preopen.eq(0)]
    date_coverage = (rows.groupby("signal_date").agg(
        candidates=("code", "size"),
        theme_roster_available=("theme_roster_available_preopen", "sum"),
        model_eligible=("theme_factor_eligible", "sum"),
    ).reset_index())
    date_coverage["coverage"] = (
        date_coverage.theme_roster_available / date_coverage.candidates)
    bootstrap = _cluster_bootstrap(rows, repetitions=bootstrap_repetitions)
    differences = {
        field: _difference(rows, field) for field in (
            "promotion_rate", "executable_B_plus_2_positive_rate",
            "B_plus_2_mean_net_return", "B_plus_1_big_loss_rate",
        )
    }
    full_market_complete = rows.full_market_completeness.fillna("").astype(str).eq("verified")
    eligible = int(rows.theme_factor_eligible.sum())
    factor_allowed = bool(
        eligible > 0
        and full_market_complete.groupby(rows.signal_date).all().sum() >= 20
        and bootstrap["mixed_sessions"] >= 20
    )
    return {
        "version": "v197-theme-profit-coverage-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "sessions": int(rows.signal_date.nunique()),
        "candidate_rows": int(len(rows)),
        "point_in_time_theme_roster_rows": int(len(known)),
        "theme_unknown_or_unavailable_rows": int(len(unknown)),
        "theme_factor_model_eligible_rows": eligible,
        "full_market_complete_sessions": int(
            full_market_complete.groupby(rows.signal_date).all().sum()),
        "daily_coverage": date_coverage.to_dict("records"),
        "groups": {
            "point_in_time_theme_roster": _metrics(known),
            "unknown_or_unavailable_theme": _metrics(unknown),
        },
        "roster_minus_unknown": differences,
        "date_cluster_bootstrap95": bootstrap,
        "governance": {
            "theme_evidence_availability_is_theme_strength": False,
            "unknown_theme_is_weak_theme": False,
            "theme_roster_direct_profit_factor_allowed": factor_allowed,
            "theme_driver_factor_allowed": False,
            "expectation_saturation_factor_allowed": False,
            "direct_stock_reranking_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "A name being covered by an archived roster may reflect attention and evidence "
            "availability rather than a verified active driver. Membership, catalyst timing, "
            "expectation saturation and payoff must remain separate."),
        "limitations": [
            "theme coverage exists on only part of the later profit evaluation window",
            "no row has a complete model-eligible driver and expectation schema",
            "no session is independently certified as full-market theme complete",
            "available rosters are selected evidence and cannot define an unbiased control group",
            "daily-open execution remains a proxy rather than a verified intraday fill",
        ],
    }


def run(predictions_path, archive_path, output_dir) -> dict:
    paths = [Path(predictions_path), Path(archive_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable theme coverage audit")
    predictions = pd.read_csv(paths[0], dtype={"code": str})
    archive = pd.read_csv(paths[1], dtype={"code": str})
    # Exclude right-censored sessions, but retain missing execution evidence as unavailable.
    right_censored = (
        predictions.fill_status.eq("FILLED_OPEN_PROXY")
        & pd.to_numeric(predictions.net_return_B_plus_2, errors="coerce").isna())
    complete_dates = set(
        predictions.assign(_right_censored=right_censored)
        .groupby("signal_date")._right_censored.any().loc[lambda x: ~x].index.astype(str))
    predictions = predictions[
        predictions.signal_date.astype(str).isin(complete_dates)].copy()
    rows = build_rows(predictions, archive)
    report = audit(rows)
    target.mkdir(parents=True)
    rows_path = target / "theme_profit_coverage_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["rows_sha256"] = _sha256(rows_path)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--theme-archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions, args.theme_archive, args.output),
                     ensure_ascii=False, indent=2))
