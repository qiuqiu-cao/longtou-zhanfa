"""Build point-in-time inputs for the five-factor first-board challenger.

Only facts supported by the supplied archives are populated.  In particular,
closing limit-up rows cannot prove which failed-board competitors were
eliminated, and a missing announcement archive cannot prove zero soft risk.
Those fields therefore remain unavailable rather than becoming zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v209-first-board-five-factor-inputs-1"
KEYS = ["signal_date", "code"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(features: pd.DataFrame, theme: pd.DataFrame,
          announcements: pd.DataFrame | None = None,
          competition: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    required_features = set(KEYS + [
        "target_date",
        "turnover", "amount_to_float_cap", "last_minus_first_minute",
        "failed_seal_count", "one_price_proxy", "thin_control_risk",
        "feature_available_at",
    ])
    required_theme = set(KEYS + [
        "theme_group_id", "theme_support_ex_self", "point_in_time_available_for_next_session",
        "active_driver_verified", "driver_theme_granularity", "available_at",
    ])
    if required_features - set(features):
        raise ValueError(f"feature fields missing: {sorted(required_features - set(features))}")
    if required_theme - set(theme):
        raise ValueError(f"theme fields missing: {sorted(required_theme - set(theme))}")
    if features.duplicated(KEYS).any() or theme.duplicated(KEYS).any():
        raise ValueError("feature and theme keys must be unique")
    rows = features.copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    theme_view = theme[list(required_theme)].copy()
    theme_view["code"] = theme_view.code.astype(str).str.zfill(6)
    theme_view = theme_view.rename(columns={"available_at": "theme_factor_available_at"})
    rows = rows.merge(theme_view, on=KEYS, how="left", validate="one_to_one")

    rows["same_height_peer_count"] = rows.groupby("signal_date").code.transform("size") - 1
    turnover = pd.to_numeric(rows.turnover, errors="coerce")
    amount = pd.to_numeric(rows.amount_to_float_cap, errors="coerce")
    delay = pd.to_numeric(rows.last_minus_first_minute, errors="coerce")
    failed = pd.to_numeric(rows.failed_seal_count, errors="coerce")
    turn_rank = turnover.groupby(rows.signal_date).rank(pct=True)
    amount_rank = amount.groupby(rows.signal_date).rank(pct=True)
    repair = 1 / (1 + delay.clip(lower=0) / 30)
    failure_penalty = failed.clip(lower=0, upper=3) / 6
    thin_penalty = rows.one_price_proxy.fillna(0).astype(float).mul(.15) + rows.thin_control_risk.fillna(0).astype(float).mul(.15)
    observed = turnover.notna() & amount.notna() & delay.notna() & failed.notna()
    rows["turnover_absorption_quality"] = (
        .35 * turn_rank + .35 * amount_rank + .30 * repair - failure_penalty - thin_penalty
    ).clip(0, 1).where(observed)
    rows["turnover_absorption_quality_evidence_status"] = np.where(
        observed, "POINT_IN_TIME_VERIFIED", "UNAVAILABLE")

    pit_theme = rows.point_in_time_available_for_next_session.fillna(False).astype(bool)
    active = rows.active_driver_verified.fillna(False).astype(bool)
    specific = rows.driver_theme_granularity.fillna("").astype(str).str.lower().eq("specific")
    rows["theme_purity"] = pd.Series(np.nan, index=rows.index)
    rows.loc[pit_theme & active & specific, "theme_purity"] = 1.0
    rows["theme_purity_evidence_status"] = np.where(
        rows.theme_purity.notna(), "POINT_IN_TIME_VERIFIED", "UNAVAILABLE")

    support = pd.to_numeric(rows.theme_support_ex_self, errors="coerce")
    denominator = rows.groupby("signal_date").code.transform("size").clip(lower=1)
    rows["sector_leadership"] = ((support + 1) / denominator).clip(0, 1).where(pit_theme & support.notna())
    rows["sector_leadership_evidence_status"] = np.where(
        rows.sector_leadership.notna(), "POINT_IN_TIME_VERIFIED", "UNAVAILABLE")

    # These require a complete failed-board peer archive and a complete
    # point-in-time announcement window respectively.  Neither can be inferred
    # from the closing-limit and theme tables supplied here.
    rows["competitor_elimination"] = np.nan
    rows["competitor_elimination_evidence_status"] = "UNAVAILABLE"
    rows["competitor_factor_available_at"] = pd.NA
    if competition is not None:
        required_competition = set(KEYS + [
            "competitor_elimination", "competitor_elimination_evidence_status",
            "competitor_available_at",
        ])
        if required_competition - set(competition):
            raise ValueError("competition evidence fields missing")
        if competition.duplicated(KEYS).any():
            raise ValueError("competition evidence keys must be unique")
        view = competition[list(required_competition)].copy()
        view["code"] = view.code.astype(str).str.zfill(6)
        view = view.rename(columns={
            "competitor_elimination": "_competitor_elimination",
            "competitor_elimination_evidence_status": "_competitor_elimination_evidence_status",
            "competitor_available_at": "_competitor_factor_available_at",
        })
        rows = rows.merge(view, on=KEYS, how="left", validate="one_to_one")
        present = rows._competitor_elimination_evidence_status.notna() & rows._competitor_elimination_evidence_status.ne("UNAVAILABLE")
        rows.loc[present, "competitor_elimination"] = pd.to_numeric(
            rows.loc[present, "_competitor_elimination"], errors="coerce")
        rows.loc[present, "competitor_elimination_evidence_status"] = (
            rows.loc[present, "_competitor_elimination_evidence_status"])
        rows.loc[present, "competitor_factor_available_at"] = rows.loc[
            present, "_competitor_factor_available_at"]
        rows = rows.drop(columns=[
            "_competitor_elimination", "_competitor_elimination_evidence_status",
            "_competitor_factor_available_at"])
    rows["announcement_soft_risk"] = np.nan
    rows["announcement_soft_risk_evidence_status"] = "UNAVAILABLE"
    rows["announcement_factor_available_at"] = pd.NA
    if announcements is not None:
        required_announcements = set(KEYS + [
            "announcement_soft_risk", "announcement_soft_risk_evidence_status",
            "announcement_available_at",
        ])
        if required_announcements - set(announcements):
            raise ValueError("announcement evidence fields missing")
        if announcements.duplicated(KEYS).any():
            raise ValueError("announcement evidence keys must be unique")
        view = announcements[list(required_announcements)].copy()
        view["code"] = view.code.astype(str).str.zfill(6)
        view = view.rename(columns={
            "announcement_soft_risk": "_announcement_soft_risk",
            "announcement_soft_risk_evidence_status": "_announcement_soft_risk_evidence_status",
            "announcement_available_at": "_announcement_factor_available_at",
        })
        rows = rows.merge(view, on=KEYS, how="left", validate="one_to_one")
        present = rows._announcement_soft_risk_evidence_status.eq("POINT_IN_TIME_VERIFIED")
        rows.loc[present, "announcement_soft_risk"] = pd.to_numeric(
            rows.loc[present, "_announcement_soft_risk"], errors="coerce")
        rows.loc[present, "announcement_soft_risk_evidence_status"] = (
            rows.loc[present, "_announcement_soft_risk_evidence_status"])
        rows.loc[present, "announcement_factor_available_at"] = (
            rows.loc[present, "_announcement_factor_available_at"])
        rows = rows.drop(columns=[
            "_announcement_soft_risk", "_announcement_soft_risk_evidence_status",
            "_announcement_factor_available_at"])
    availability = pd.DataFrame({
        "market": pd.to_datetime(rows.feature_available_at, errors="coerce", utc=True),
        "theme": pd.to_datetime(rows.theme_factor_available_at, errors="coerce", utc=True).where(
            rows.theme_purity.notna() | rows.sector_leadership.notna()),
        "competition": pd.to_datetime(rows.competitor_factor_available_at,
                                       errors="coerce", utc=True).where(
            rows.competitor_elimination.notna()),
        "announcement": pd.to_datetime(rows.announcement_factor_available_at,
                                        errors="coerce", utc=True).where(
            rows.announcement_soft_risk.notna()),
    })
    rows["evidence_available_at"] = availability.max(axis=1).dt.tz_convert(
        "Asia/Shanghai").map(lambda value: value.isoformat() if pd.notna(value) else None)
    rows["feature_cutoff_at"] = pd.to_datetime(
        rows.target_date.astype(str) + "T09:15:00+08:00", errors="coerce", utc=True
    ).dt.tz_convert("Asia/Shanghai").map(lambda value: value.isoformat() if pd.notna(value) else None)
    statuses = [
        "turnover_absorption_quality_evidence_status", "theme_purity_evidence_status",
        "sector_leadership_evidence_status", "competitor_elimination_evidence_status",
        "announcement_soft_risk_evidence_status",
    ]
    rows["five_factor_point_in_time_complete"] = rows[statuses].isin(
        {"VERIFIED", "POINT_IN_TIME_VERIFIED"}).all(axis=1)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)), "sessions": int(rows.signal_date.nunique()),
        "coverage": {field.replace("_evidence_status", ""): int(
            rows[field].isin({"VERIFIED", "POINT_IN_TIME_VERIFIED"}).sum()) for field in statuses},
        "descriptive_nonformal_coverage": {
            "competitor_elimination": int(rows.competitor_elimination.notna().sum()),
        },
        "five_factor_complete_rows": int(rows.five_factor_point_in_time_complete.sum()),
        "missing_values_treated_as_zero": False,
        "limitations": [
            "competitor elimination is formal only with attempted-height labels from a point-in-time complete failed-board pool; current-universe reconstruction remains descriptive",
            "announcement soft risk requires a complete timestamped announcement window and full-text classification",
            "theme purity and sector leadership remain unavailable where specific active-driver evidence is absent",
        ],
    }
    return rows, report


def run(features_path, theme_path, output_dir, announcements_path=None,
        competition_path=None):
    sources = [Path(features_path), Path(theme_path)]
    if announcements_path:
        sources.append(Path(announcements_path))
    if competition_path:
        sources.append(Path(competition_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable five-factor inputs")
    rows, report = build(
        pd.read_csv(sources[0], dtype={"code": str}),
        pd.read_csv(sources[1], dtype={"code": str}),
        pd.read_csv(Path(announcements_path), dtype={"code": str}) if announcements_path else None,
        pd.read_csv(Path(competition_path), dtype={"code": str}) if competition_path else None)
    target.mkdir(parents=True)
    output = target / "first_board_features_with_five_factor_evidence.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in sources}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--announcements")
    parser.add_argument("--competition")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.theme, args.output, args.announcements,
                         args.competition), ensure_ascii=False, indent=2))
