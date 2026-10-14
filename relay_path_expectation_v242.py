"""Build path-conditioned expectations for every frozen 3/4-board candidate.

V239 supplied a close-time hierarchy.  V242 adds a community-derived but
falsifiable expectation reference for each prior board path.  It does not use
one universal auction band: locked boards need strong-to-strong continuation,
quick-repair boards need divergence absorption, and one-price boards require
first-open price discovery before any tradability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v242-relay-path-expectation-1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def _profile(row: pd.Series) -> dict:
    path = str(row.board_path_status)
    turnover = pd.to_numeric(pd.Series([row.turnover]), errors="coerce").iloc[0]
    turnover = None if pd.isna(turnover) else float(turnover)
    if path == "ONE_PRICE_BUYABILITY_UNPROVEN":
        return {
            "expectation_type": "FIRST_OPEN_PRICE_DISCOVERY",
            "normal_gap_low": None, "normal_gap_high": None,
            "expected_auction_relation": "NO_VALID_GAP_BAND_WHILE_STILL_ONE_PRICE",
            "required_repair": "OPEN;REAL_TURNOVER;ABOVE_VWAP;RESEAL;THEME_SUPPORT",
            "overheat_rule": "ONE_PRICE_CONTINUATION_IS_IDENTITY_ONLY_NOT_BUYABLE",
            "trade_expectation_allowed": False,
        }
    if path == "LOCKED_NO_REOPEN":
        return {
            "expectation_type": "STRONG_TO_STRONG_CONTINUATION",
            "normal_gap_low": 3.0, "normal_gap_high": 7.0,
            "expected_auction_relation": "BELOW_3_IS_UNDER_EXPECTATION_UNLESS_OPEN_RECOVERY_CONFIRMS",
            "required_repair": "IF_BELOW_BAND_RECLAIM_PREVCLOSE_AND_VWAP_THEN_SEAL_BY_1000",
            "overheat_rule": "ABOVE_7_NEEDS_AMOUNT_ABSORPTION_LOW_DECAY_AND_RESEAL",
            "trade_expectation_allowed": _truthy(row.tradable_expectation_eligible),
        }
    if path == "QUICK_REPAIR" and turnover is not None and turnover > 25:
        return {
            "expectation_type": "HIGH_TURNOVER_DIVERGENCE_CONTINUATION",
            "normal_gap_low": 0.0, "normal_gap_high": 4.0,
            "expected_auction_relation": "MODEST_PREMIUM_WITHOUT_EXPECTATION_OVERDRAFT",
            "required_repair": "HOLD_VWAP;NO_SECOND_FAILED_SEAL;SEAL_BY_1000;THEME_LEAD",
            "overheat_rule": "ABOVE_5_IS_OVERDRAFT_UNLESS_STRICT_ABSORPTION_CONFIRMS",
            "trade_expectation_allowed": _truthy(row.tradable_expectation_eligible),
        }
    if path == "QUICK_REPAIR":
        return {
            "expectation_type": "DIVERGENCE_ABSORPTION_CONTINUATION",
            "normal_gap_low": 0.0, "normal_gap_high": 5.0,
            "expected_auction_relation": "NORMAL_PREMIUM_THEN_ACTIVE_ATTACK",
            "required_repair": "IF_LOW_OPEN_RECLAIM_PREVCLOSE_AND_VWAP;SEAL_BY_1000;THEME_LEAD",
            "overheat_rule": "ABOVE_5_NEEDS_STRONGER_AMOUNT_AND_LOW_DECAY",
            "trade_expectation_allowed": _truthy(row.tradable_expectation_eligible),
        }
    return {
        "expectation_type": "SUPPLY_RELEASE_WATCH_ONLY",
        "normal_gap_low": None, "normal_gap_high": None,
        "expected_auction_relation": "NO_TRADABLE_AUCTION_EXPECTATION",
        "required_repair": "FULL_PRICE_DISCOVERY_AND_RESEAL_REQUIRED_FOR_IDENTITY_ONLY",
        "overheat_rule": "DO_NOT_TREAT_GAP_AS_STRENGTH",
        "trade_expectation_allowed": False,
    }


def build(pool: pd.DataFrame, tree: pd.DataFrame, v239: dict,
          review: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pool = pool.copy(); tree = tree.copy()
    pool["code"] = pool.code.astype(str).str.zfill(6)
    tree["code"] = tree.code.astype(str).str.zfill(6)
    if len(pool) != int(v239["full_3_4_board_denominator"]):
        raise ValueError("V239 full denominator count mismatch")
    if set(tree.code) - set(pool.code):
        raise ValueError("V239 hierarchy contains code outside full denominator")
    profiles = pd.DataFrame([{"code": row.code, **_profile(row)}
                             for _, row in pool.iterrows()])
    matrix = pool.merge(profiles, on="code", how="left", validate="one_to_one")
    matrix["auction_stage_is_final_winner"] = False
    matrix["required_stage_chain"] = (
        "0925_AUCTION_RELATIVE_TO_OWN_EXPECTATION>0935_OPEN_AND_ATTACK>"
        "1000_SEAL_AND_THEME_LEAD>CLOSE_PROMOTION_SETTLEMENT"
    )
    matrix["cross_height_takeover_rule"] = (
        "3_BOARD_CAN_TAKE_OVER_ONLY_IF_NO_4_BOARD_CONFIRMS_BY_1000"
    )
    matrix["same_height_co_winner_rule"] = (
        "MULTIPLE_CONFIRMED_AT_HIGHEST_RESULTING_BOARD_MEANS_NO_UNIQUE_LEADER"
    )
    matrix["entry_authorized"] = False

    scenarios = pd.DataFrame([
        {"scenario_rank": 1, "scenario": "ONE_OR_MORE_4_BOARD_CONFIRM",
         "winner_rule": "CONFIRMED_4_TO_5_NAMES_RETAIN_HEIGHT;TWO_OR_MORE_ARE_CO_WINNERS",
         "three_board_status": "FOLLOWER_NOT_CROSS_HEIGHT_WINNER"},
        {"scenario_rank": 2, "scenario": "NO_4_BOARD_CONFIRM_AND_ONE_3_BOARD_CONFIRMS",
         "winner_rule": "THAT_3_TO_4_NAME_IS_UNIQUE_CROSS_HEIGHT_TAKEOVER",
         "three_board_status": "CONDITIONAL_IDENTITY_WINNER"},
        {"scenario_rank": 3, "scenario": "NO_4_BOARD_CONFIRM_AND_MULTIPLE_3_BOARD_CONFIRM",
         "winner_rule": "CO_WINNERS_NO_UNIQUE_LEADER_UNTIL_LATER_SEPARATION",
         "three_board_status": "CONTESTED_TAKEOVER"},
        {"scenario_rank": 4, "scenario": "MARKET_WIDE_HIGH_AND_MIDDLE_NEGATIVE",
         "winner_rule": "ANY_SURVIVOR_IS_RETREAT_SURVIVOR_NOT_PROVEN_CORE_SWITCH",
         "three_board_status": "NO_TRADE_WINNER"},
        {"scenario_rank": 5, "scenario": "NO_CANDIDATE_CONFIRMS_BY_1000",
         "winner_rule": "NO_WINNER_RETREAT_CASH",
         "three_board_status": "NO_WINNER"},
    ])
    dimensions = review.get("emotion_dimensions") or {}
    env = {key: (dimensions.get(key) or {}).get("state") for key in
           ("money_effect", "broken_loss_effect", "height_feedback",
            "theme_divergence", "repair_quality")}
    report = {
        "version": VERSION,
        "status": "COMMUNITY_DERIVED_PATH_EXPECTATIONS_DIAGNOSTIC_ONLY",
        "signal_date": str(v239["signal_date"]),
        "full_denominator": len(matrix),
        "market_environment_at_close": env,
        "primary_close_expectation": v239.get("primary_expected_winner"),
        "improvements_over_v239": [
            "NO_UNIVERSAL_AUCTION_BAND",
            "PRIOR_BOARD_PATH_DETERMINES_NORMAL_EXPECTATION",
            "AUCTION_OPEN_ATTACK_SEAL_ARE_SEPARATE_STAGES",
            "THREE_BOARD_CANNOT_BEAT_A_CONFIRMED_FOUR_BOARD",
            "SYSTEMIC_RETREAT_SEPARATED_FROM_IDIOSYNCRATIC_FAILURE",
            "CO_WINNERS_NOT_FORCED_INTO_ONE_WINNER",
        ],
        "community_hypothesis_sources": [
            "https://m.tgb.cn/a/2rrtaeVU3FC",
            "https://m.tgb.cn/a/2uhDe9qyrhX?type=hot",
            "https://m.tgb.cn/a/24ojnS4l404",
        ],
        "governance": {
            "community_posts_are_hypothesis_sources_only": True,
            "validated_probability_claimed": False,
            "minimum_forward_date_clusters": 60,
            "level2_claimed": False,
            "entry_authorized": False,
        },
        "entry_authorized": False,
    }
    return matrix, scenarios, report


def run(pool_path: str, tree_path: str, v239_report_path: str,
        review_path: str, output_dir: str) -> dict:
    pool_file, tree_file, report_file, review_file = map(
        Path, (pool_path, tree_path, v239_report_path, review_path))
    v239 = json.loads(report_file.read_text(encoding="utf-8-sig"))
    review = json.loads(review_file.read_text(encoding="utf-8-sig"))
    if v239.get("version") != "v239-relay-cross-theme-expectation-1":
        raise ValueError("V239 report version mismatch")
    hashes = v239.get("artifact_hashes", {})
    if hashes.get(pool_file.name) != _sha(pool_file) or hashes.get(tree_file.name) != _sha(tree_file):
        raise ValueError("V239 artifact hash mismatch")
    pool = pd.read_csv(pool_file, dtype={"code": str})
    tree = pd.read_csv(tree_file, dtype={"code": str})
    matrix, scenarios, report = build(pool, tree, v239, review)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V242 output")
    target.mkdir(parents=True)
    matrix_path, scenarios_path = target / "path_expectation_matrix.csv", target / "scenario_tree.csv"
    matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    scenarios.to_csv(scenarios_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha(path)
                               for path in (pool_file, tree_file, report_file, review_file)}
    report["artifact_hashes"] = {
        matrix_path.name: _sha(matrix_path), scenarios_path.name: _sha(scenarios_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True); ap.add_argument("--tree", required=True)
    ap.add_argument("--v239-report", required=True); ap.add_argument("--review", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.pool, ns.tree, ns.v239_report, ns.review, ns.output),
                     ensure_ascii=False, indent=2))
