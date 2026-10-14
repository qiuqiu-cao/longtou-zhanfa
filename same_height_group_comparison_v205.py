"""Fail-closed same-height comparison for closing limit-up candidates.

Every candidate on the same signal date and board height is compared in one
group.  Seal speed and zero failed seals are retained as observations, but may
not declare a winner.  A research leader exists only when all five requested
evidence families are point-in-time available and both independently trained
first-board targets are present.  This module never authorizes entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v205-same-height-group-comparison-1"
KEYS = ["signal_date", "code"]
FIVE_FACTORS = {
    "turnover_absorption_quality": False,
    "theme_purity": False,
    "sector_leadership": False,
    "competitor_elimination": False,
    "announcement_soft_risk": True,
}
STATUS_SUFFIX = "_evidence_status"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stamp(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")


def build(frame: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, dict]:
    required = set(KEYS + [
        "name", "boards", "first_seal_minute", "failed_seal_count",
        "full_market_complete", "b2_positive_probability",
        "b1_big_loss_probability", "dual_target_status",
        "evidence_available_at",
    ])
    for factor in FIVE_FACTORS:
        required.update({factor, factor + STATUS_SUFFIX})
    missing = required - set(frame)
    if missing:
        raise ValueError(f"same-height fields missing: {sorted(missing)}")
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("candidate set must be non-empty and unique by signal_date/code")

    rows = frame.copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    rows["signal_date"] = pd.to_datetime(rows.signal_date, errors="raise").dt.strftime("%Y-%m-%d")
    rows["boards"] = pd.to_numeric(rows.boards, errors="raise")
    if not rows.boards.mod(1).eq(0).all() or rows.boards.lt(1).any():
        raise ValueError("boards must be positive integers")
    rows["boards"] = rows.boards.astype(int)
    cutoff_stamp = _stamp(cutoff)
    if pd.isna(cutoff_stamp):
        raise ValueError("comparison cutoff is required")
    evidence_at = rows.evidence_available_at.map(_stamp)
    rows["point_in_time_available"] = evidence_at.notna() & evidence_at.le(cutoff_stamp)
    rows["height_group_id"] = rows.signal_date + ":H" + rows.boards.astype(str)
    rows["same_height_group_size"] = rows.groupby("height_group_id").code.transform("size")
    rows["same_height_peer_count"] = rows.same_height_group_size - 1
    rows["forced_group_comparison"] = True

    factor_complete = pd.Series(True, index=rows.index)
    for factor, is_risk in FIVE_FACTORS.items():
        numeric = pd.to_numeric(rows[factor], errors="coerce")
        status = rows[factor + STATUS_SUFFIX].fillna("UNAVAILABLE").astype(str).str.upper()
        verified = status.isin({"VERIFIED", "POINT_IN_TIME_VERIFIED"}) & numeric.notna()
        factor_complete &= verified
        rows[factor] = numeric
        # Relative ranks are calculated strictly within height group.  Risk is
        # reversed so that a larger percentile always means more supportive.
        supportive = -numeric if is_risk else numeric
        rows[factor + "_group_percentile"] = supportive.groupby(rows.height_group_id).rank(
            method="average", pct=True)

    rows["five_factor_evidence_complete"] = factor_complete & rows.point_in_time_available
    rows["full_group_scope"] = rows.groupby("height_group_id").full_market_complete.transform(
        lambda values: values.map(_truthy).all())
    rows["five_factor_group_complete"] = rows.groupby("height_group_id").five_factor_evidence_complete.transform("all")
    dual = rows.dual_target_status.astype(str).eq("TIME_ISOLATED_DIAGNOSTIC")
    probabilities = rows[["b2_positive_probability", "b1_big_loss_probability"]].apply(
        pd.to_numeric, errors="coerce")
    rows[["b2_positive_probability", "b1_big_loss_probability"]] = probabilities
    rows["dual_target_available"] = dual & probabilities.notna().all(axis=1)
    rows["group_model_eligible"] = (
        rows.full_group_scope & rows.five_factor_group_complete
        & rows.groupby("height_group_id").dual_target_available.transform("all")
    )

    percentiles = [factor + "_group_percentile" for factor in FIVE_FACTORS]
    rows["five_factor_relative_score"] = rows[percentiles].mean(axis=1, skipna=False)
    rows["dual_target_utility_RESEARCH_ONLY"] = (
        rows.b2_positive_probability - rows.b1_big_loss_probability
    )
    rows["group_research_score_NOT_PROBABILITY"] = (
        .7 * rows.dual_target_utility_RESEARCH_ONLY
        + .3 * (rows.five_factor_relative_score - .5)
    )
    rows.loc[~rows.group_model_eligible, "group_research_score_NOT_PROBABILITY"] = np.nan
    rows["research_rank_within_height"] = rows.groupby("height_group_id")[
        "group_research_score_NOT_PROBABILITY"].rank(method="min", ascending=False)
    rows["research_group_leader"] = rows.group_model_eligible & rows.research_rank_within_height.eq(1)
    rows["official_champion"] = False
    rows["entry_authorized"] = False
    rows["seal_speed_zero_open_can_select_champion"] = False
    rows["comparison_status"] = np.where(
        rows.group_model_eligible, "RESEARCH_RANK_AVAILABLE_NOT_VALIDATED",
        "GROUP_EVIDENCE_INCOMPLETE_NO_RANK")

    groups = rows.groupby("height_group_id", as_index=False).agg(
        signal_date=("signal_date", "first"), boards=("boards", "first"),
        candidate_count=("code", "size"), full_group_scope=("full_group_scope", "all"),
        five_factor_group_complete=("five_factor_group_complete", "all"),
        group_model_eligible=("group_model_eligible", "all"),
        research_leader_count=("research_group_leader", "sum"),
    )
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)),
        "height_groups": int(len(groups)),
        "multi_candidate_height_groups": int(groups.candidate_count.gt(1).sum()),
        "model_eligible_groups": int(groups.group_model_eligible.sum()),
        "policy": {
            "same_height_candidates_forced_into_one_group": True,
            "seal_speed_or_zero_failed_seals_can_select_champion": False,
            "five_factor_group_completeness_required": True,
            "missing_factor_filled_as_zero": False,
            "dual_targets_required": ["B+2_positive", "B+1_big_loss"],
            "official_champion_claimed": False,
            "entry_authorized": False,
        },
        "group_summary": groups.to_dict("records"),
    }
    return rows, report


def run(input_path, cutoff, output_dir):
    source, target = Path(input_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable same-height comparison")
    rows, report = build(pd.read_csv(source, dtype={"code": str}), cutoff)
    target.mkdir(parents=True)
    output = target / "same_height_group_comparison.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_sha256"] = _sha256(source)
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.cutoff, args.output), ensure_ascii=False, indent=2))
