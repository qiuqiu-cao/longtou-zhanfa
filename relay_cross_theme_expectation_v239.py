"""Freeze a falsifiable cross-theme expectation tree for 3/4-board relays.

Every 3-board and 4-board closing candidate remains in the denominator,
including one-price, unknown-theme and untradable names.  A separate tradable
winner hierarchy is frozen before the next session: primary expectation,
same-height fallback, lower-height fallback, then no winner/retreat.  The tree
is descriptive and cannot be used as a same-morning replacement trade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "v239-relay-cross-theme-expectation-1"
TARGET_BOARDS = {3, 4}
TRADEABLE_PATHS = {"QUICK_REPAIR", "LOCKED_NO_REOPEN"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def build(scores: pd.DataFrame, role_candidates: pd.DataFrame,
          role_report: dict, review: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores = scores.copy()
    scores["code"] = scores.code.astype(str).str.zfill(6)
    scores["boards"] = pd.to_numeric(scores.boards, errors="raise").astype(int)
    pool = scores[scores.boards.isin(TARGET_BOARDS)].copy()
    if pool.empty or set(pool.boards) != TARGET_BOARDS:
        raise ValueError("V239 requires both 3-board and 4-board candidates")
    review_map = {str(item["code"]).zfill(6): item for item in review.get("relay_candidates", [])}
    if not set(pool.code).issubset(review_map):
        raise ValueError("3/4-board candidates are missing from the daily review")
    role_candidates = role_candidates.copy()
    role_candidates["code"] = role_candidates.code.astype(str).str.zfill(6)
    if set(role_candidates.code) != set(scores.code):
        raise ValueError("V235 role candidates do not cover the V225 relay pool")

    primary_obj = role_report.get("close_relay_challenger")
    if not primary_obj:
        primary_code = None
    else:
        primary_code = str(primary_obj["code"]).zfill(6)
        if primary_code not in set(pool.code):
            raise ValueError("V235 close challenger is outside the 3/4-board race")

    theme_fields = []
    for code in pool.code:
        evidence = review_map[code].get("theme_evidence") or {}
        theme_id = str(evidence.get("theme_group_id") or "").strip()
        theme_name = str(evidence.get("theme") or "").strip()
        theme_fields.append({
            "code": code,
            "theme_group_id": theme_id if theme_id else f"UNKNOWN_THEME:{code}",
            "theme_name": theme_name if theme_name else "题材未知",
            "theme_evidence_status": str(evidence.get("status") or "UNAVAILABLE"),
            "theme_support_ex_self": _number(evidence.get("support_ex_self")),
            "theme_ladder_observed": _truthy(evidence.get("ladder_verified", False)),
            "turnover": _number(review_map[code].get("turnover")),
            "amount": _number(review_map[code].get("amount")),
            "security_eligibility_status": str(
                review_map[code].get("eligibility_status") or "UNAVAILABLE"),
        })
    theme_frame = pd.DataFrame(theme_fields)
    pool = pool.merge(theme_frame, on="code", how="left", validate="one_to_one")
    pool["theme_known"] = ~pool.theme_group_id.str.startswith("UNKNOWN_THEME:")
    pool["closing_path_tradeable"] = pool.board_path_status.isin(TRADEABLE_PATHS)
    pool["announcement_hard_veto"] = pool.announcement_hard_veto.map(_truthy)
    pool["turnover_not_blowoff"] = pd.to_numeric(
        pool.factor_turnover_quality, errors="coerce").fillna(-99).ge(0)
    pool["tradable_expectation_eligible"] = (
        pool.theme_known & pool.closing_path_tradeable
        & pool.turnover.notna() & pool.turnover_not_blowoff
        & ~pool.announcement_hard_veto
    )
    pool["in_full_identity_denominator"] = True
    pool["predeclared_role"] = "FULL_DENOMINATOR_WATCH_ONLY"
    pool.loc[pool.code.eq(primary_code), "predeclared_role"] = "PRIMARY_EXPECTED_WINNER"

    same_height = pool[
        pool.tradable_expectation_eligible
        & pool.boards.eq(int(pool.loc[pool.code.eq(primary_code), "boards"].iloc[0]) if primary_code else 4)
        & ~pool.code.eq(primary_code)
    ].sort_values(["research_score_NOT_PROBABILITY", "code"], ascending=[False, True])
    lower_height = pool[
        pool.tradable_expectation_eligible
        & pool.boards.eq(3)
        & ~pool.code.eq(primary_code)
    ].sort_values(["research_score_NOT_PROBABILITY", "code"], ascending=[False, True])
    hierarchy_codes = ([primary_code] if primary_code else []) + same_height.code.tolist() + lower_height.code.tolist()
    hierarchy_codes = list(dict.fromkeys(code for code in hierarchy_codes if code))
    for index, code in enumerate(hierarchy_codes, start=1):
        role = ("PRIMARY_EXPECTED_WINNER" if index == 1
                else "SAME_HEIGHT_FALLBACK_WINNER" if int(pool.loc[pool.code.eq(code), "boards"].iloc[0]) == 4
                else "LOWER_HEIGHT_FALLBACK_WINNER")
        pool.loc[pool.code.eq(code), "predeclared_role"] = role
        pool.loc[pool.code.eq(code), "winner_hierarchy_rank"] = index
    pool["winner_hierarchy_rank"] = pd.to_numeric(pool.get("winner_hierarchy_rank"), errors="coerce")

    expectation_rows = []
    for _, row in pool[pool.winner_hierarchy_rank.notna()].sort_values("winner_hierarchy_rank").iterrows():
        rank = int(row.winner_hierarchy_rank)
        if rank == 1:
            activation = "PRIMARY_MEETS_EXPECTATION"
        elif row.boards == 4:
            activation = "ALL_EARLIER_CHOICES_FAIL_AND_SAME_HEIGHT_RIVAL_CONFIRMS"
        else:
            activation = "ALL_4_BOARD_CHOICES_FAIL_AND_THIS_3_BOARD_CHOICE_CONFIRMS"
        expectation_rows.append({
            "signal_date": row.signal_date, "hierarchy_rank": rank,
            "code": row.code, "name": row["name"], "boards": int(row.boards),
            "theme_group_id": row.theme_group_id, "theme_name": row.theme_name,
            "predeclared_role": row.predeclared_role,
            "activation_condition": activation,
            "expected_auction": "0_TO_5_IDEAL;5_TO_7_STRICT;ABOVE_7_NEEDS_ABSORPTION_RESEAL",
            "required_by_0935": (
                "NEXT_BOARD_SEALED;ABOVE_VWAP;VWAP_SLOPE_POSITIVE;"
                "FAILED_SEAL_EPISODES_LT_2;LADDER_SYNC_AND_COVERAGE_PASS"
            ),
            "underperformance": (
                "AUCTION_BELOW_MINUS_3_OR_HOT_WITH_WEAK_AMOUNT;BELOW_VWAP_EXPANDING;"
                "TWO_FAILED_SEALS;NO_NEXT_BOARD_SEAL_BY_0935;LADDER_COLLAPSE"
            ),
            "trade_replacement_allowed": False,
            "identity_winner_claimed_at_close": False,
            "entry_authorized": False,
        })
    tree = pd.DataFrame(expectation_rows)

    primary = None
    if primary_code:
        row = pool[pool.code.eq(primary_code)].iloc[0]
        primary = {"code": primary_code, "name": row["name"], "boards": int(row.boards),
                   "theme_group_id": row.theme_group_id, "theme_name": row.theme_name}
    fallbacks = [
        {"rank": int(row.winner_hierarchy_rank), "code": row.code, "name": row["name"],
         "boards": int(row.boards), "theme_group_id": row.theme_group_id,
         "theme_name": row.theme_name, "role": row.predeclared_role}
        for _, row in pool[pool.winner_hierarchy_rank.gt(1)].sort_values(
            "winner_hierarchy_rank").iterrows()
    ]
    report = {
        "version": VERSION,
        "status": "PREDECLARED_3_4_BOARD_CROSS_THEME_EXPECTATION_DIAGNOSTIC_ONLY",
        "signal_date": str(review["signal_date"]),
        "full_3_4_board_denominator": int(len(pool)),
        "board_counts": {str(key): int(value) for key, value in pool.boards.value_counts().sort_index().items()},
        "known_theme_rows": int(pool.theme_known.sum()),
        "unknown_theme_rows_retained": int((~pool.theme_known).sum()),
        "primary_expected_winner": primary,
        "conditional_fallback_winners": fallbacks,
        "no_winner_scenario": {
            "condition": "NO_PREDECLARED_3_OR_4_BOARD_CHOICE_CONFIRMS_BY_0935",
            "interpretation": "HEIGHT_COMPETITION_FAILED_OR_RETREAT;DO_NOT_FORCE_A_WINNER",
            "action": "CASH_NO_REPLACEMENT_TRADE",
        },
        "expectation_definition": {
            "primary_meets_expectation": (
                "auction_not_cancelled;next_board_sealed_by_0935;above_VWAP;"
                "positive_VWAP_slope;fewer_than_two_failed_seals;ladder_passes"
            ),
            "fallback_can_win_only_after_earlier_choice_fails": True,
            "same_height_fallback_precedes_lower_height_fallback": True,
            "one_price_or_unknown_theme_can_still_be_identity_outcome": True,
            "one_price_or_unknown_theme_is_tradable_fallback": False,
            "fallback_is_observation_not_replacement_trade": True,
        },
        "governance": {
            "higher_board_win_rate_claimed": False,
            "minimum_forward_date_clusters": 60,
            "full_denominator_required": True,
            "unexpected_winner_is_model_miss": True,
            "next_session_high_or_close_not_used_in_close_ranking": True,
            "entry_authorized": False,
        },
        "entry_authorized": False,
    }
    ordered = [
        "signal_date", "code", "name", "boards", "theme_group_id", "theme_name",
        "theme_evidence_status", "theme_known", "theme_support_ex_self",
        "theme_ladder_observed", "turnover", "amount", "board_path_status",
        "announcement_risk_tier", "announcement_hard_veto", "symbolic_tags",
        "research_score_NOT_PROBABILITY", "closing_path_tradeable",
        "turnover_not_blowoff", "tradable_expectation_eligible",
        "in_full_identity_denominator", "predeclared_role", "winner_hierarchy_rank",
        "security_eligibility_status", "entry_authorized",
    ]
    pool["entry_authorized"] = False
    return pool[ordered].sort_values(
        ["boards", "research_score_NOT_PROBABILITY", "code"], ascending=[False, False, True]), tree, report


def _markdown(pool: pd.DataFrame, tree: pd.DataFrame, report: dict) -> str:
    primary = report["primary_expected_winner"] or {}
    lines = [
        "# 3/4板跨题材卡位预期树 V239", "",
        f"状态：`{report['status']}`；信号日：{report['signal_date']}；买入授权：否。", "",
        f"- 主预期：{primary.get('name', '无')}({primary.get('code', '-')})，"
        f"{primary.get('boards', '-')}板，题材 `{primary.get('theme_name', '不可得')}`。",
        "- 主预期不及后：只按盘前冻结的同高度→低一板顺序观察条件赢家，不允许交易替换。", "",
        "## 预先冻结的赢家顺序", "",
        "|顺位|股票|板位|题材|角色|启动条件|", "|---:|---|---:|---|---|---|",
    ]
    for _, row in tree.iterrows():
        lines.append(f"|{int(row.hierarchy_rank)}|{row['name']}({row.code})|{row.boards}|"
                     f"{row.theme_name}|{row.predeclared_role}|{row.activation_condition}|")
    lines += ["", "## 完整3/4板分母", "",
              "|股票|板位|题材/状态|收盘路径|可交易预期资格|预定角色|",
              "|---|---:|---|---|---|---|"]
    for _, row in pool.iterrows():
        lines.append(f"|{row['name']}({row.code})|{row.boards}|{row.theme_name}/"
                     f"{row.theme_evidence_status}|{row.board_path_status}|"
                     f"{'YES' if row.tradable_expectation_eligible else 'NO'}|{row.predeclared_role}|")
    lines += ["", "若所有预定候选均未在09:35前完成晋级封板、VWAP与梯队确认，"
              "结论固定为“无赢家/退潮”，不从后排强行选股。", ""]
    return "\n".join(lines)


def run(scores_path: str, same_height_report_path: str, role_candidates_path: str,
        role_report_path: str, review_path: str, output_dir: str) -> dict:
    paths = [Path(value) for value in (scores_path, same_height_report_path,
                                        role_candidates_path, role_report_path, review_path)]
    scores_file, v225_file, roles_file, v235_file, review_file = paths
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V239 output")
    v225 = json.loads(v225_file.read_text(encoding="utf-8-sig"))
    v235 = json.loads(v235_file.read_text(encoding="utf-8-sig"))
    review = json.loads(review_file.read_text(encoding="utf-8-sig"))
    if v225.get("version") != "v225-same-height-leader-pk-1":
        raise ValueError("V225 report version mismatch")
    if v235.get("version") != "v235-relay-leader-role-gate-1":
        raise ValueError("V235 report version mismatch")
    if v225.get("artifact_hashes", {}).get("same_height_scores.csv") != _sha(scores_file):
        raise ValueError("V225 score CSV hash mismatch")
    if v235.get("artifact_hashes", {}).get("relay_role_candidates.csv") != _sha(roles_file):
        raise ValueError("V235 role candidate CSV hash mismatch")
    if _sha(review_file) not in set(v225.get("source_hashes", {}).values()):
        raise ValueError("daily review is not a hashed V225 source")
    if _sha(v225_file) not in set(v235.get("source_hashes", {}).values()):
        raise ValueError("V235 report was not built from this V225 report")
    scores = pd.read_csv(scores_file, dtype={"code": str})
    roles = pd.read_csv(roles_file, dtype={"code": str})
    pool, tree, report = build(scores, roles, v235, review)
    target.mkdir(parents=True)
    pool_path, tree_path = target / "three_four_board_pool.csv", target / "expectation_tree.csv"
    pool.to_csv(pool_path, index=False, encoding="utf-8-sig")
    tree.to_csv(tree_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha(path) for path in paths}
    report["artifact_hashes"] = {
        pool_path.name: _sha(pool_path), tree_path.name: _sha(tree_path),
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "report.md").write_text(_markdown(pool, tree, report), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--same-height-report", required=True)
    ap.add_argument("--role-candidates", required=True)
    ap.add_argument("--role-report", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.scores, ns.same_height_report, ns.role_candidates,
                         ns.role_report, ns.review, ns.output),
                     ensure_ascii=False, indent=2))
