"""Build fail-closed candidate roles for the 100% severe-abnormal path.

The closest regulatory-distance observation and the strongest leader-path
candidate are deliberately separate.  Neither is a probability, ranking
factor, entry signal, or reason to relax execution-risk controls.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from severe_abnormal_rule_registry_v145 import verified_rule_for_code


VERSION = "v171-abnormal-path-role-1"
VERIFIED = {"verified", "verified_official", "official_reconciled"}
IDENTITY_PRIORITY = {
    "height_anchor": 0,
    "unique_survivor": 1,
    "pioneer": 2,
    "turnover_core": 3,
    "capacity_core": 4,
    "supplement": 5,
}


def _truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("timestamp is missing")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "code", "name", "boards", "pool_assignment"}
    if required - set(candidates):
        raise ValueError("candidate pool fields missing")
    rows = candidates[candidates.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    if "is_mainboard" in rows:
        rows = rows[rows.is_mainboard.map(_truthy)].copy()
    if rows.empty:
        raise ValueError("closing-limit candidate pool is empty")
    rows["code"] = rows.code.astype(str).str.zfill(6)
    rows["signal_date"] = pd.to_datetime(rows.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    rows["boards"] = pd.to_numeric(rows.boards, errors="coerce")
    if (rows.signal_date.isna().any() or rows.code.duplicated().any()
            or rows.boards.isna().any() or (rows.boards < 1).any()
            or not rows.boards.map(lambda value: float(value).is_integer()).all()):
        raise ValueError("candidate pool contains invalid date/code/board records")
    if rows.signal_date.nunique() != 1:
        raise ValueError("one signal_date is required")
    return rows


def _evidence_map(evidence: pd.DataFrame | None, signal_date: str, cutoff) -> dict:
    if evidence is None:
        return {}
    required = {
        "signal_date", "code", "available_at", "rule_id", "rule_status",
        "benchmark_code", "benchmark_status", "reset_status",
        "corporate_action_status", "security_status", "coverage_complete",
        "indicator_status", "distance_100_pp", "distance_200_pp",
        "scenario_limit_boards_to_100", "scenario_status",
        "active_theme_path_status", "competition_identity_status",
        "competition_identity", "theme_support_ex_self", "same_height_rivals",
    }
    if required - set(evidence):
        raise ValueError("abnormal-path evidence fields missing")
    frame = evidence.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["signal_date"] = pd.to_datetime(
        frame.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame[frame.signal_date.eq(signal_date)].copy()
    if frame.empty or frame.code.duplicated().any():
        raise ValueError("abnormal-path evidence must be unique and nonempty for signal_date")
    available = frame.available_at.map(_timestamp)
    if available.gt(cutoff).any():
        raise ValueError("abnormal-path evidence is after asof")
    frame["available_at"] = available.map(lambda value: value.isoformat())
    return {row.code: row.to_dict() for _, row in frame.iterrows()}


def _diagnostic(code: str, signal_date: str, source: dict | None) -> tuple[dict, list[str]]:
    blockers = []
    official_rule = verified_rule_for_code(code, signal_date)
    if source is None:
        blockers.append("CANDIDATE_RULE_DISTANCE_EVIDENCE_UNAVAILABLE")
        return {
            "status": "UNAVAILABLE", "rule_id": official_rule["rule_id"],
            "benchmark_code": official_rule["benchmark_code"],
            "distance_100_pp": None, "distance_200_pp": None,
            "scenario_limit_boards_to_100": None,
            "scenario_status": "UNAVAILABLE", "available_at": None,
        }, blockers

    if source.get("rule_id") != official_rule["rule_id"]:
        blockers.append("RULE_VERSION_MISMATCH")
    if source.get("benchmark_code") != official_rule["benchmark_code"]:
        blockers.append("BENCHMARK_MISMATCH")
    for field in ("rule_status", "benchmark_status", "reset_status",
                  "corporate_action_status", "security_status", "indicator_status"):
        if str(source.get(field, "")).lower() not in VERIFIED:
            blockers.append(field.upper() + "_NOT_VERIFIED")
    if not _truthy(source.get("coverage_complete")):
        blockers.append("RULE_WINDOW_COVERAGE_INCOMPLETE")
    distance_100 = _finite(source.get("distance_100_pp"))
    distance_200 = _finite(source.get("distance_200_pp"))
    steps = _finite(source.get("scenario_limit_boards_to_100"))
    if distance_100 is None or distance_200 is None:
        blockers.append("RULE_DISTANCE_MISSING")
    if (steps is not None and not steps.is_integer()):
        blockers.append("SCENARIO_STEP_NOT_INTEGER")
    if source.get("scenario_status") != "verified_inputs_counterfactual":
        blockers.append("SCENARIO_INPUTS_NOT_VERIFIED")
    return {
        "status": "VERIFIED_DIAGNOSTIC_ONLY" if not blockers else "INCOMPLETE",
        "rule_id": official_rule["rule_id"],
        "benchmark_code": official_rule["benchmark_code"],
        "distance_100_pp": distance_100,
        "distance_200_pp": distance_200,
        "scenario_limit_boards_to_100": int(steps) if steps is not None and steps.is_integer() else None,
        "scenario_status": source.get("scenario_status"),
        "available_at": source.get("available_at"),
    }, blockers


def build(candidates: pd.DataFrame, evidence: pd.DataFrame | None, asof: str) -> dict:
    rows = _candidate_rows(candidates)
    signal_date = rows.signal_date.iloc[0]
    cutoff = _timestamp(asof)
    evidence_by_code = _evidence_map(evidence, signal_date, cutoff)
    candidate_codes = set(rows.code)
    if not set(evidence_by_code) <= candidate_codes:
        raise ValueError("abnormal-path evidence contains non-candidate codes")

    records = []
    eligible = []
    for _, row in rows.sort_values(["boards", "code"], ascending=[False, True]).iterrows():
        source = evidence_by_code.get(row.code)
        diagnostic, blockers = _diagnostic(row.code, signal_date, source)
        identity = str((source or {}).get("competition_identity", "")).lower()
        support = _finite((source or {}).get("theme_support_ex_self"))
        rivals = _finite((source or {}).get("same_height_rivals"))
        role_blockers = list(blockers)
        if int(row.boards) < 2:
            role_blockers.append("FIRST_BOARD_NOT_YET_RELAY_IDENTITY")
        if (source or {}).get("active_theme_path_status") != "verified":
            role_blockers.append("ACTIVE_THEME_PATH_NOT_VERIFIED")
        if (source or {}).get("competition_identity_status") != "verified" or identity not in IDENTITY_PRIORITY:
            role_blockers.append("COMPETITION_IDENTITY_NOT_VERIFIED")
        if support is None or support < 1:
            role_blockers.append("THEME_SUPPORT_NOT_VERIFIED")
        if rivals is None or rivals < 0:
            role_blockers.append("RIVAL_COUNT_NOT_VERIFIED")
        steps = diagnostic["scenario_limit_boards_to_100"]
        distance = diagnostic["distance_100_pp"]
        if steps is None or not 1 <= steps <= 3:
            role_blockers.append("THRESHOLD_NOT_REACHED_WITHIN_THREE_LIMIT_SCENARIO_STEPS")
        if distance is None or distance <= 0:
            role_blockers.append("NO_POSITIVE_PRE_THRESHOLD_DISTANCE")
        record = {
            "signal_date": signal_date, "code": row.code, "name": row.get("name"),
            "boards": int(row.boards), "diagnostic": diagnostic,
            "active_theme_path_status": (source or {}).get("active_theme_path_status", "unavailable"),
            "competition_identity_status": (source or {}).get("competition_identity_status", "unavailable"),
            "competition_identity": identity or None,
            "theme_support_ex_self": support, "same_height_rivals": rivals,
            "path_role_status": "ELIGIBLE_DIAGNOSTIC_PATH" if not role_blockers else "NOT_ELIGIBLE",
            "blocking_reasons": sorted(set(role_blockers)),
            "calibrated_probability": None, "buy_signal": False,
            "candidate_reranking_allowed": False,
        }
        records.append(record)
        if not role_blockers:
            eligible.append((
                IDENTITY_PRIORITY[identity], -support, steps, rivals,
                -int(row.boards), row.code, record,
            ))

    verified_distances = [
        item for item in records
        if item["diagnostic"]["status"] == "VERIFIED_DIAGNOSTIC_ONLY"
        and item["diagnostic"]["distance_100_pp"] is not None
        and item["diagnostic"]["distance_100_pp"] > 0
    ]
    verified_distances.sort(key=lambda item: (
        item["diagnostic"]["distance_100_pp"], -item["boards"], item["code"]))
    closest = verified_distances[0] if verified_distances else None
    eligible.sort(key=lambda item: item[:-1])
    potential = eligible[0][-1] if eligible else None

    def view(item, role):
        if item is None:
            return None
        return {
            "role": role, "code": item["code"], "name": item["name"],
            "boards": item["boards"],
            "distance_100_pp": item["diagnostic"]["distance_100_pp"],
            "scenario_limit_boards_to_100": item["diagnostic"]["scenario_limit_boards_to_100"],
            "interpretation": "diagnostic_counterfactual_not_probability_or_buy_reason",
            "calibrated_probability": None, "buy_signal": False,
            "candidate_reranking_allowed": False,
        }

    return {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY", "asof": cutoff.isoformat(),
        "signal_date": signal_date, "closing_candidate_count": int(len(rows)),
        "evidence_attached_count": len(set(evidence_by_code) & candidate_codes),
        "verified_distance_count": len(verified_distances),
        "path_eligible_count": len(eligible),
        "rule_distance_closest": view(closest, "RULE_DISTANCE_CLOSEST_DIAGNOSTIC"),
        "hundred_percent_abnormal_path_potential": view(
            potential, "HUNDRED_PERCENT_ABNORMAL_PATH_POTENTIAL"),
        "closest_and_path_potential_may_differ": True,
        "plain_price_doubling_is_sufficient": False,
        "distance_is_positive_ranking_factor": False,
        "calibrated_probability": None, "buy_signal": False,
        "candidate_reranking_allowed": False,
        "candidate_records": records,
        "interpretation": (
            "The closest verified rule distance is arithmetic only. A path-potential role also "
            "requires verified reset/benchmark/security data, an active specific-theme path, "
            "leader identity and support. Neither role authorizes entry."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--evidence")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable abnormal-path role artifact")
    report = build(
        pd.read_csv(args.candidates, dtype={"code": str}),
        pd.read_csv(args.evidence, dtype={"code": str}) if args.evidence else None,
        args.asof,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
