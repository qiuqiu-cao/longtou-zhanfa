"""Audit trading-session chronology before building promotion or B+1/B+2/B+3 labels.

Daily review dates are observations, not a trading calendar.  This module
cross-checks successful exchange-date queries with a complete provider
cross-section and identifies labels that skipped an unobserved trading day.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(official_queries: pd.DataFrame, provider_history: pd.DataFrame,
          observations: pd.DataFrame, start: str, end: str,
          pool_dates: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not {"date", "status"} <= set(official_queries):
        raise ValueError("official query audit requires date and status")
    if not {"date", "code"} <= set(provider_history):
        raise ValueError("provider history requires date and code")
    if not {"signal_date", "target_date"} <= set(observations):
        raise ValueError("observations require signal_date and target_date")

    official = official_queries.copy()
    official["date"] = pd.to_datetime(official.date, errors="raise").dt.strftime("%Y-%m-%d")
    official = official[
        official.status.eq("SUCCESS") & official.date.between(start, end)
    ].drop_duplicates("date").sort_values("date")
    if official.empty:
        raise ValueError("no successful official-query sessions in range")

    history = provider_history.copy()
    history["date"] = pd.to_datetime(history.date, errors="raise").dt.strftime("%Y-%m-%d")
    history["code"] = history.code.astype(str).str.zfill(6)
    history = history[history.date.between(start, end)]
    expected_codes = int(history.code.nunique())
    if expected_codes < 2:
        raise ValueError("provider calendar needs a cross-section, not one stock")
    coverage = (history.groupby("date").code.nunique().rename("observed_codes")
                .reset_index())
    coverage["expected_codes"] = expected_codes
    coverage["coverage"] = coverage.observed_codes / expected_codes

    calendar = official[["date"]].merge(coverage, on="date", how="left", validate="one_to_one")
    calendar["observed_codes"] = calendar.observed_codes.fillna(0).astype(int)
    calendar["expected_codes"] = expected_codes
    calendar["coverage"] = calendar.observed_codes / expected_codes
    calendar["official_query_success"] = True
    calendar["provider_cross_section_complete"] = calendar.coverage.eq(1)
    calendar["data_status"] = calendar.provider_cross_section_complete.map(
        {True: "official_query_success_provider_cross_section_complete",
         False: "provider_cross_section_incomplete"})

    sessions = calendar.loc[
        calendar.provider_cross_section_complete, "date"].tolist()
    successor = dict(zip(sessions[:-1], sessions[1:]))
    if pool_dates is not None:
        if "date" not in pool_dates:
            raise ValueError("pool dates require date")
        pool_series = pd.to_datetime(
            pool_dates.date, errors="raise").dt.strftime("%Y-%m-%d")
        available_pool_dates = set(pool_series[pool_series.between(start, end)])
    else:
        available_pool_dates = set(observations.signal_date.astype(str))
    obs = observations[["signal_date", "target_date"]].drop_duplicates().copy()
    obs["signal_date"] = pd.to_datetime(obs.signal_date, errors="raise").dt.strftime("%Y-%m-%d")
    obs["target_date"] = pd.to_datetime(obs.target_date, errors="coerce").dt.strftime("%Y-%m-%d")
    obs["correct_next_session"] = obs.signal_date.map(successor)
    obs["old_target_matches_calendar"] = obs.target_date.eq(obs.correct_next_session)
    obs["correct_next_pool_available"] = obs.correct_next_session.isin(available_pool_dates)
    obs["issue"] = ""
    mismatch = obs.correct_next_session.notna() & ~obs.old_target_matches_calendar
    obs.loc[mismatch, "issue"] = "SPARSE_OBSERVATION_CALENDAR_SKIPPED_TRADING_SESSION"
    missing_pool = obs.correct_next_session.notna() & ~obs.correct_next_pool_available
    obs.loc[missing_pool, "issue"] = obs.loc[missing_pool, "issue"].map(
        lambda value: (value + ";" if value else "") + "IMMEDIATE_NEXT_POOL_MISSING")

    report = {
        "version": "v195-trading-calendar-integrity-1",
        "status": ("CALENDAR_LABEL_REBUILD_REQUIRED" if mismatch.any()
                   else "CALENDAR_CHRONOLOGY_MATCHED"),
        "range": {"start": start, "end": end},
        "official_query_success_sessions": int(len(official)),
        "provider_complete_sessions": int(calendar.provider_cross_section_complete.sum()),
        "provider_expected_codes": expected_codes,
        "observation_signal_sessions": int(obs.signal_date.nunique()),
        "available_pool_sessions": int(len(available_pool_dates)),
        "sparse_successor_mismatch_sessions": int(mismatch.sum()),
        "immediate_next_pool_missing_sessions": int(missing_pool.sum()),
        "affected_signal_dates": obs.loc[obs.issue.ne(""), [
            "signal_date", "target_date", "correct_next_session", "issue"
        ]].to_dict("records"),
        "governance": {
            "daily_review_dates_may_define_trading_calendar": False,
            "missing_pool_session_may_be_skipped_for_promotion_label": False,
            "missing_session_may_be_skipped_for_holding_horizon": False,
            "calendar_fix_authorizes_entry": False,
        },
        "interpretation": (
            "A sequence of captured review dates is not an exchange calendar.  Promotion and "
            "holding horizons must use the immediate verified trading session; if that session's "
            "pool is missing, the promotion label is unavailable rather than shifted forward."
        ),
    }
    return calendar, obs, report


def run(official_path, provider_path, observations_path, output_dir, start, end,
        pool_dates_path=None) -> dict:
    paths = list(map(Path, (official_path, provider_path, observations_path)))
    pool_path = Path(pool_dates_path) if pool_dates_path else None
    if pool_path:
        paths.append(pool_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable calendar audit")
    calendar, rows, report = audit(
        pd.read_csv(paths[0]),
        pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2], dtype={"code": str}),
        start, end,
        pd.read_csv(pool_path) if pool_path else None,
    )
    target.mkdir(parents=True)
    calendar_path = target / "verified_trading_calendar.csv"
    rows_path = target / "observation_successor_audit.csv"
    calendar.to_csv(calendar_path, index=False, encoding="utf-8-sig")
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["calendar_sha256"] = _sha256(calendar_path)
    report["successor_audit_sha256"] = _sha256(rows_path)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-dates")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.official, args.provider, args.observations,
                         args.output, args.start, args.end, args.pool_dates),
                     ensure_ascii=False, indent=2))
