"""Evaluate the frozen V239 3/4-board cross-theme expectation tree.

The evaluator requires one timestamped observation for every member of the
frozen 3/4-board denominator.  It distinguishes the first passing member of
the predeclared tradable hierarchy from the actual identity outcome: an
excluded one-price or unknown-theme name can still expose a model miss by
winning the board race.  A fallback result is observation only and never an
intraday replacement-trade instruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v240-relay-cross-theme-confirmation-1"
TZ = ZoneInfo("Asia/Shanghai")
VERIFIED_SOURCE = {"VERIFIED", "PRIMARY_VERIFIED", "OFFICIAL_VERIFIED"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def _number(value: Any, field: str) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"missing numeric field: {field}")
    return float(parsed)


def _dt(value: str) -> datetime:
    stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        raise ValueError("available_at must include timezone")
    return stamp.astimezone(TZ)


def _min_sync(gap: float) -> float:
    return .8 if gap > 7 else (.7 if gap > 5 or gap <= 0 else .6)


def _row_evaluation(row: pd.Series, decision_date: str) -> dict:
    if str(row.trade_date) != decision_date:
        raise ValueError("observation trade_date differs from decision date")
    if _dt(row.available_at) > _dt(f"{decision_date}T09:35:59+08:00"):
        raise ValueError("09:35 decision uses evidence available after 09:35:59")
    gap = _number(row.auction_gap_pct, "auction_gap_pct")
    amount_ratio = _number(row.auction_amount_ratio_pct, "auction_amount_ratio_pct")
    decay = _number(row.match_amount_decay_proxy, "match_amount_decay_proxy")
    failed = int(_number(row.failed_seal_episodes, "failed_seal_episodes"))
    sync = _number(row.theme_sync_ratio, "theme_sync_ratio")
    coverage = _number(row.theme_coverage_ratio, "theme_coverage_ratio")
    peers = int(_number(row.theme_peer_count, "theme_peer_count"))
    if not 0 <= decay <= 1:
        raise ValueError("match_amount_decay_proxy must be within [0,1]")
    if not 0 <= sync <= 1 or not 0 <= coverage <= 1:
        raise ValueError("theme sync and coverage must be within [0,1]")
    reasons: list[str] = []
    source_verified = str(row.source_status).strip().upper() in VERIFIED_SOURCE
    if not source_verified:
        reasons.append("SOURCE_NOT_VERIFIED")
    if gap < -3:
        reasons.append("AUCTION_BELOW_MINUS_3")
    if gap > 5 and (amount_ratio < 2 or decay > .45):
        reasons.append("HOT_OPEN_WITH_WEAK_AMOUNT_OR_DECAY")
    if not _truthy(row.above_vwap_0935):
        reasons.append("BELOW_VWAP_AT_0935")
    if not _truthy(row.vwap_slope_positive):
        reasons.append("VWAP_SLOPE_NOT_POSITIVE")
    if failed >= 2:
        reasons.append("TWO_OR_MORE_FAILED_SEALS")
    sealed = _truthy(row.sealed_next_board_by_0935)
    if not sealed:
        reasons.append("NO_NEXT_BOARD_SEAL_BY_0935")
    if coverage < .7 or peers < 2 or sync < _min_sync(gap):
        reasons.append("LADDER_SYNC_OR_COVERAGE_FAILED")
    if gap > 7:
        reseal_time = str(row.get("reseal_time", "")).strip()
        if decay > .2 or not reseal_time or reseal_time > "09:35":
            reasons.append("EXTREME_GAP_WITHOUT_ABSORPTION_RESEAL")
    return {
        "source_verified": source_verified,
        "gap_pct": gap,
        "amount_ratio_pct": amount_ratio,
        "match_amount_decay_proxy": decay,
        "above_vwap_0935": _truthy(row.above_vwap_0935),
        "vwap_slope_positive": _truthy(row.vwap_slope_positive),
        "failed_seal_episodes": failed,
        "sealed_next_board_by_0935": sealed,
        "theme_sync_ratio": sync,
        "theme_coverage_ratio": coverage,
        "theme_peer_count": peers,
        "evidence_lane": str(row.evidence_lane),
        "meets_expectation": not reasons,
        "failure_reasons": ";".join(reasons),
    }


def evaluate(pool: pd.DataFrame, tree: pd.DataFrame, v239_report: dict,
             observations: pd.DataFrame | None, decision_date: str) -> tuple[pd.DataFrame, dict]:
    pool = pool.copy()
    pool["code"] = pool.code.astype(str).str.zfill(6)
    tree = tree.copy()
    tree["code"] = tree.code.astype(str).str.zfill(6)
    frozen_codes = set(pool.code)
    if observations is None:
        observations = pd.DataFrame()
    observations = observations.copy()
    if not observations.empty:
        observations["code"] = observations.code.astype(str).str.zfill(6)
    base = {
        "version": VERSION,
        "signal_date": v239_report.get("signal_date"),
        "decision_date": decision_date,
        "full_3_4_board_denominator": len(pool),
        "replacement_trade_allowed": False,
        "entry_authorized": False,
        "level2_used": False,
        "decision": "DATA_UNAVAILABLE_INCOMPLETE_3_4_BOARD_DENOMINATOR",
        "conditional_winner": None,
        "confirmed_hierarchy_candidates": [],
        "competition_status": "UNAVAILABLE",
        "actual_identity_winners": [],
        "unexpected_winners": [],
    }
    if observations.empty:
        base["missing_codes"] = sorted(frozen_codes)
        return pd.DataFrame(), base
    if observations.code.duplicated().any():
        raise ValueError("duplicate candidate observations")
    observed_codes = set(observations.code)
    if observed_codes != frozen_codes:
        base["missing_codes"] = sorted(frozen_codes - observed_codes)
        base["extra_codes"] = sorted(observed_codes - frozen_codes)
        return pd.DataFrame(), base
    expected_signal_date = str(v239_report.get("signal_date"))
    if set(observations.signal_date.astype(str)) != {expected_signal_date}:
        raise ValueError("observation signal_date differs from V239")

    evaluations = []
    for _, row in observations.iterrows():
        evaluations.append({"code": row.code, **_row_evaluation(row, decision_date)})
    evaluated = pool.merge(pd.DataFrame(evaluations), on="code", how="left", validate="one_to_one")
    identity = evaluated[
        evaluated.source_verified & evaluated.sealed_next_board_by_0935
    ].copy()
    base["actual_identity_winners"] = [
        {"code": row.code, "name": row["name"], "boards_at_close": int(row.boards),
         "predeclared_role": row.predeclared_role,
         "tradable_expectation_eligible": bool(row.tradable_expectation_eligible)}
        for _, row in identity.sort_values(["boards", "code"], ascending=[False, True]).iterrows()
    ]

    hierarchy = evaluated[evaluated.winner_hierarchy_rank.notna()].sort_values(
        "winner_hierarchy_rank")
    passing = hierarchy[hierarchy.meets_expectation]
    if passing.empty:
        base["decision"] = "NO_WINNER_RETREAT_CASH"
        base["competition_status"] = "NO_CONFIRMED_WINNER"
    else:
        base["confirmed_hierarchy_candidates"] = [
            {"code": row.code, "name": row["name"],
             "starting_boards": int(row.boards),
             "expected_next_boards": int(row.boards) + 1,
             "hierarchy_rank": int(row.winner_hierarchy_rank)}
            for _, row in passing.iterrows()
        ]
        maximum_next_board = int((passing.boards + 1).max())
        top_identity = passing[(passing.boards + 1).eq(maximum_next_board)]
        unique_highest = len(top_identity) == 1
        base["competition_status"] = (
            "UNIQUE_HIGHEST_CONFIRMED" if unique_highest
            else "SAME_HEIGHT_CO_WINNERS_NO_UNIQUE_LEADER"
        )
        base["highest_confirmed_identity_codes"] = top_identity.code.tolist()
        winner = passing.iloc[0]
        lane = str(winner.evidence_lane).strip().upper()
        if lane == "VERIFIED_REAL_THEME":
            strength = "VERIFIED_REAL_THEME"
        elif lane == "VERIFIED_SYMBOLIC_ROSTER":
            strength = "SYMBOLIC_ONLY_NOT_REAL_THEME"
        else:
            strength = "UNVERIFIED_THEME"
        primary_code = str(v239_report.get("primary_expected_winner", {}).get("code", "")).zfill(6)
        is_primary = winner.code == primary_code
        if not unique_highest:
            base["decision"] = "SAME_HEIGHT_CO_WINNERS_NO_UNIQUE_LEADER"
        else:
            base["decision"] = (
                "PRIMARY_EXPECTATION_CONFIRMED" if is_primary
                else "CONDITIONAL_FALLBACK_WINNER_OBSERVED_NO_REPLACEMENT_TRADE"
            )
        base["conditional_winner"] = {
            "code": winner.code, "name": winner["name"],
            "hierarchy_rank": int(winner.winner_hierarchy_rank),
            "predeclared_role": winner.predeclared_role,
            "evidence_strength": strength,
            "unique_highest_confirmed": unique_highest,
            "trade_replacement_allowed": False,
        }

    hierarchy_codes = set(tree.code)
    unexpected = identity[~identity.code.isin(hierarchy_codes)]
    base["unexpected_winners"] = [
        {"code": row.code, "name": row["name"], "reason": "UNEXPECTED_WINNER_MODEL_MISS"}
        for _, row in unexpected.iterrows()
    ]
    if base["unexpected_winners"]:
        base["model_miss_flag"] = True
    else:
        base["model_miss_flag"] = False
    base["all_source_rows_verified"] = bool(evaluated.source_verified.all())
    return evaluated, base


def run(pool_path: str, tree_path: str, report_path: str,
        decision_date: str, output_dir: str, observations_path: str | None = None) -> dict:
    sources = [Path(pool_path), Path(tree_path), Path(report_path)]
    pool_file, tree_file, report_file = sources
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v239-relay-cross-theme-expectation-1":
        raise ValueError("V239 report version mismatch")
    hashes = report.get("artifact_hashes", {})
    if hashes.get(pool_file.name) != _sha(pool_file) or hashes.get(tree_file.name) != _sha(tree_file):
        raise ValueError("V239 artifact hash mismatch")
    pool = pd.read_csv(pool_file, dtype={"code": str})
    tree = pd.read_csv(tree_file, dtype={"code": str})
    observations = None
    if observations_path:
        observation_file = Path(observations_path)
        sources.append(observation_file)
        observations = pd.read_csv(observation_file, dtype={"code": str})
    evaluated, result = evaluate(pool, tree, report, observations, decision_date)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V240 output")
    target.mkdir(parents=True)
    evaluated_path = target / "candidate_confirmations.csv"
    evaluated.to_csv(evaluated_path, index=False, encoding="utf-8-sig")
    result["source_hashes"] = {str(path.resolve()): _sha(path) for path in sources}
    result["artifact_hashes"] = {evaluated_path.name: _sha(evaluated_path)}
    (target / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--decision-date", required=True)
    ap.add_argument("--observations")
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.pool, ns.tree, ns.report, ns.decision_date,
                         ns.output, ns.observations), ensure_ascii=False, indent=2))
