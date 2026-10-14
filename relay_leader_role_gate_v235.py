"""Separate the relay height anchor from the close-time tradable challenger.

V225 compares stocks only inside the same board height.  V235 performs the
missing cross-height role step without replacing the original leader model:
the highest board remains the emotion/space anchor, while a separate close-time
challenger may be selected from the top two height levels only when its closing
path was actually tradable.  The result is a frozen research hypothesis, not a
probability or a buy signal; security identity, verified theme evidence and the
next-session auction/open confirmation remain hard gates for an official relay
primary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "v235-relay-leader-role-gate-1"
TRADEABLE_CLOSE_PATHS = {"QUICK_REPAIR", "LOCKED_NO_REOPEN"}
PATH_PRIORITY = {
    "QUICK_REPAIR": 2,
    "LOCKED_NO_REOPEN": 1,
    "REPAIR_DURATION_UNAVAILABLE": 0,
    "PROLONGED_CONTEST": -1,
    "ONE_PRICE_BUYABILITY_UNPROVEN": -2,
    "UNAVAILABLE": -3,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _candidate_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "code": str(row.code).zfill(6),
        "name": row["name"],
        "boards": int(row.boards),
        "height_gap_to_anchor": int(row.height_gap_to_anchor),
        "relay_role": row.relay_role,
        "board_path_status": row.board_path_status,
        "repair_span_minutes": None if pd.isna(row.repair_span_minutes)
        else float(row.repair_span_minutes),
        "turnover": None if pd.isna(row.turnover) else float(row.turnover),
        "amount": None if pd.isna(row.amount) else float(row.amount),
        "symbolic_tags": row.symbolic_tags if pd.notna(row.symbolic_tags) else "",
        "announcement_risk_tier": row.announcement_risk_tier,
        "security_eligibility_status": row.security_eligibility_status,
        "theme_evidence_status": row.theme_evidence_status,
        "close_relay_challenger_eligible": bool(row.close_relay_challenger_eligible),
        "official_primary_gate_passed": bool(row.official_primary_gate_passed),
        "entry_authorized": False,
    }


def build(scores: pd.DataFrame, review: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {
        "signal_date", "code", "name", "boards", "research_score_NOT_PROBABILITY",
        "board_path_status", "repair_span_minutes", "announcement_risk_tier",
        "announcement_hard_veto", "symbolic_tags", "missing_evidence",
        "factor_verified_real_theme", "factor_turnover_quality",
    }
    missing = required - set(scores)
    if missing:
        raise ValueError(f"V225 score fields missing: {sorted(missing)}")
    frame = scores.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    if frame.code.duplicated().any():
        raise ValueError("V225 scores contain duplicate codes")
    if frame.empty:
        raise ValueError("V225 scores contain no relay candidates")
    signal_dates = set(frame.signal_date.astype(str))
    if signal_dates != {str(review.get("signal_date"))}:
        raise ValueError("V225 scores and daily review signal dates differ")

    review_candidates = {
        str(item["code"]).zfill(6): item for item in review.get("relay_candidates", [])
    }
    if set(frame.code) != set(review_candidates):
        raise ValueError("V225 scores do not exactly cover the daily relay candidate pool")

    maximum = int(pd.to_numeric(frame.boards, errors="raise").max())
    frame["boards"] = pd.to_numeric(frame.boards, errors="raise").astype(int)
    frame["height_gap_to_anchor"] = maximum - frame.boards
    frame["path_priority_NOT_SCORE"] = frame.board_path_status.map(PATH_PRIORITY).fillna(-3).astype(int)
    frame["turnover"] = frame.code.map(
        lambda code: _number(review_candidates[code].get("turnover")))
    frame["amount"] = frame.code.map(
        lambda code: _number(review_candidates[code].get("amount")))
    frame["security_eligibility_status"] = frame.code.map(
        lambda code: str(review_candidates[code].get("eligibility_status") or "UNAVAILABLE"))
    frame["theme_evidence_status"] = frame.code.map(
        lambda code: str((review_candidates[code].get("theme_evidence") or {}).get("status")
                         or "UNAVAILABLE"))
    frame["is_height_anchor"] = frame.boards.eq(maximum)
    frame["within_anchor_or_one_board"] = frame.height_gap_to_anchor.le(1)
    frame["closing_path_tradeable"] = frame.board_path_status.isin(TRADEABLE_CLOSE_PATHS)
    frame["turnover_evidence_available"] = frame.turnover.notna()
    frame["turnover_not_blowoff"] = pd.to_numeric(
        frame.factor_turnover_quality, errors="coerce").fillna(-99).ge(0)
    frame["announcement_hard_veto"] = frame.announcement_hard_veto.map(_truthy)
    frame["close_relay_challenger_eligible"] = (
        frame.within_anchor_or_one_board
        & frame.closing_path_tradeable
        & frame.turnover_evidence_available
        & frame.turnover_not_blowoff
        & ~frame.announcement_hard_veto
    )
    frame["verified_real_theme"] = pd.to_numeric(
        frame.factor_verified_real_theme, errors="coerce").fillna(0).gt(0)
    frame["security_scope_verified"] = frame.security_eligibility_status.eq("VERIFIED_ELIGIBLE")
    frame["next_session_confirmation_status"] = "UNAVAILABLE_AT_CLOSE"
    frame["official_primary_gate_passed"] = (
        frame.close_relay_challenger_eligible
        & frame.verified_real_theme
        & frame.security_scope_verified
        & frame.next_session_confirmation_status.ne("UNAVAILABLE_AT_CLOSE")
    )

    roles = []
    for _, row in frame.iterrows():
        if row.announcement_hard_veto:
            role = "HARD_RISK_REJECT"
        elif row.is_height_anchor and row.board_path_status == "ONE_PRICE_BUYABILITY_UNPROVEN":
            role = "HEIGHT_EMOTION_ANCHOR_NOT_CLOSE_ENTRY"
        elif row.close_relay_challenger_eligible and row.board_path_status == "QUICK_REPAIR":
            role = "DIVERGENCE_ABSORBED_RELAY_CHALLENGER"
        elif row.close_relay_challenger_eligible:
            role = "LOCKED_BOARD_RELAY_CHALLENGER"
        elif row.board_path_status == "PROLONGED_CONTEST":
            role = "PROLONGED_CONTEST_WATCH_ONLY"
        elif row.height_gap_to_anchor > 1:
            role = "LOWER_LADDER_SUPPORT_OR_ALTERNATE"
        else:
            role = "EVIDENCE_INCOMPLETE_WATCH_ONLY"
        roles.append(role)
    frame["relay_role"] = roles
    frame["entry_authorized"] = False

    anchors = frame[frame.is_height_anchor].sort_values(
        ["research_score_NOT_PROBABILITY", "code"], ascending=[False, True])
    anchor = _candidate_dict(anchors.iloc[0])
    eligible = frame[frame.close_relay_challenger_eligible].copy()
    eligible = eligible.sort_values(
        ["boards", "path_priority_NOT_SCORE", "research_score_NOT_PROBABILITY", "code"],
        ascending=[False, False, False, True],
    )
    challenger = _candidate_dict(eligible.iloc[0]) if not eligible.empty else None

    duel_rows = []
    if challenger:
        anchor_row = frame[frame.code.eq(anchor["code"])].iloc[0]
        challenger_row = frame[frame.code.eq(challenger["code"])].iloc[0]
        anchor_advantages, challenger_advantages = [], []
        if int(anchor_row.boards) > int(challenger_row.boards):
            anchor_advantages.append("board_height")
        if challenger_row.board_path_status == "QUICK_REPAIR":
            challenger_advantages.append("real_divergence_absorption")
        if (anchor_row.board_path_status == "ONE_PRICE_BUYABILITY_UNPROVEN"
                and challenger_row.closing_path_tradeable):
            challenger_advantages.append("close_buyability_path")
        if float(challenger_row.research_score_NOT_PROBABILITY) > float(
                anchor_row.research_score_NOT_PROBABILITY):
            challenger_advantages.append("V225_cross_evidence_strength")
        if str(anchor_row.symbolic_tags):
            anchor_advantages.append("symbolic_height_identity")
        if str(challenger_row.symbolic_tags):
            challenger_advantages.append("symbolic_market_identity")
        duel_rows.append({
            "signal_date": str(review["signal_date"]),
            "anchor_code": anchor["code"], "anchor_name": anchor["name"],
            "challenger_code": challenger["code"], "challenger_name": challenger["name"],
            "anchor_advantages": "|".join(anchor_advantages),
            "challenger_advantages": "|".join(challenger_advantages),
            "role_conclusion": "ANCHOR_AND_TRADABLE_CHALLENGER_ARE_DIFFERENT_ROLES",
            "official_winner_claimed": False, "entry_authorized": False,
        })
    duel = pd.DataFrame(duel_rows)

    report = {
        "version": VERSION,
        "status": "CASE_DERIVED_CROSS_HEIGHT_ROLE_CHALLENGER_DIAGNOSTIC_ONLY",
        "signal_date": str(review["signal_date"]),
        "relay_candidate_rows": int(len(frame)),
        "maximum_board": maximum,
        "height_emotion_anchor": anchor,
        "close_relay_challenger": challenger,
        "current_market_core": None,
        "official_relay_primary": None,
        "role_separation": {
            "highest_board_equals_automatic_buy_candidate": False,
            "height_anchor_and_tradable_core_may_differ": True,
            "candidate_search_depth_below_maximum_boards": 1,
            "quick_repair_preferred_to_locked_board_at_same_height": True,
            "prolonged_contest_is_close_candidate": False,
            "one_price_board_is_close_entry": False,
            "replacement_after_next_session_failure": False,
        },
        "next_session_contract": {
            "scope": "FROZEN_CLOSE_RELAY_CHALLENGER_ONLY",
            "ideal_auction_gap_pct": [0, 5],
            "strict_confirmation_gap_pct": [5, 7],
            "above_7_pct_default": "CANCEL_UNLESS_REAL_ABSORPTION_AND_RESEAL_WITH_THEME_CONFIRMATION",
            "required": [
                "security_scope_verified", "no_new_hard_announcement_risk",
                "auction_not_overheated", "first_pullback_absorbed",
                "VWAP_reclaimed_or_held", "same_theme_or_symbolic_ladder_not_collapsing",
            ],
            "cancel": [
                "opens_hot_then_loses_VWAP_on_expanding_sell_volume",
                "two_limit_attempts_without_reseal", "height_anchor_severe_negative_feedback",
                "support_ladder_batch_failure", "new_hard_risk_or_regulatory_notice",
            ],
            "level2_queue_or_cancel_data_claimed": False,
        },
        "governance": {
            "case_derived_rule": True,
            "research_score_is_probability": False,
            "verified_real_theme_required_for_official_core": True,
            "next_session_confirmation_required": True,
            "minimum_forward_date_clusters_before_promotion": 60,
            "targets": ["next_session_promotion", "B_plus_2_executable_payoff",
                        "B_plus_1_big_loss"],
            "entry_authorized": False,
        },
        "limitations": [
            "V225 weights and this cross-height gate are frozen case-derived hypotheses",
            "provider seal timestamps are not Level-2 queue, cancel or tick-order evidence",
            "security eligibility and verified real-theme evidence can block an official core",
            "auction/open confirmation is unavailable at the close",
        ],
        "entry_authorized": False,
    }
    ordered = [
        "signal_date", "code", "name", "boards", "height_gap_to_anchor",
        "relay_role", "is_height_anchor", "within_anchor_or_one_board",
        "board_path_status", "repair_span_minutes", "turnover", "amount",
        "closing_path_tradeable", "turnover_evidence_available", "turnover_not_blowoff",
        "announcement_risk_tier", "announcement_hard_veto", "symbolic_tags",
        "theme_evidence_status", "verified_real_theme", "security_eligibility_status",
        "security_scope_verified", "close_relay_challenger_eligible",
        "next_session_confirmation_status", "official_primary_gate_passed",
        "research_score_NOT_PROBABILITY", "path_priority_NOT_SCORE",
        "missing_evidence", "entry_authorized",
    ]
    return frame[ordered].sort_values(
        ["boards", "research_score_NOT_PROBABILITY", "code"],
        ascending=[False, False, True]), duel, report


def _markdown(rows: pd.DataFrame, duel: pd.DataFrame, report: dict) -> str:
    anchor, challenger = report["height_emotion_anchor"], report["close_relay_challenger"]
    lines = [
        "# 连板龙头跨高度角色分层 V235", "",
        f"状态：`{report['status']}`；信号日：{report['signal_date']}；买入授权：否。", "",
        f"- 空间/情绪锚：{anchor['name']}({anchor['code']})，{anchor['boards']}板，"
        f"收盘路径 `{anchor['board_path_status']}`。",
        (f"- 收盘可接力挑战者：{challenger['name']}({challenger['code']})，"
         f"{challenger['boards']}板，收盘路径 `{challenger['board_path_status']}`。"
         if challenger else "- 收盘可接力挑战者：无。"),
        "- 正式市场核心：无；正式连板主候选：无。需真实题材、证券身份与次日开盘共同确认。", "",
        "## 角色候选", "",
        "|代码|名称|板位|距高标|角色|封板路径|换手|证券身份|收盘挑战资格|", 
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for _, row in rows.iterrows():
        turnover = "不可得" if pd.isna(row.turnover) else f"{row.turnover:.2f}%"
        lines.append(
            f"|{row.code}|{row['name']}|{row.boards}|{row.height_gap_to_anchor}|"
            f"{row.relay_role}|{row.board_path_status}|{turnover}|"
            f"{row.security_eligibility_status}|"
            f"{'YES' if row.close_relay_challenger_eligible else 'NO'}|"
        )
    if not duel.empty:
        item = duel.iloc[0]
        lines += ["", "## 高标与挑战者对决", "",
                  f"- 高标优势：{item.anchor_advantages or '无'}",
                  f"- 挑战者优势：{item.challenger_advantages or '无'}",
                  "- 结论：空间锚与可接力核心是两个角色，不得因为板位最高就自动追高。"]
    lines += ["", "## 验收边界", "",
              "至少60个前向交易日簇，比较次日晋级、可成交B+2收益和B+1大亏。"
              "未同时改善前，不替换原正式连板规则。", ""]
    return "\n".join(lines)


def run(scores_path: str, same_height_report_path: str, review_path: str,
        output_dir: str) -> dict:
    scores_file, v225_file, review_file = (Path(scores_path), Path(same_height_report_path),
                                            Path(review_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V235 output")
    v225 = json.loads(v225_file.read_text(encoding="utf-8-sig"))
    if v225.get("version") != "v225-same-height-leader-pk-1":
        raise ValueError("V225 report version mismatch")
    if v225.get("artifact_hashes", {}).get("same_height_scores.csv") != _sha(scores_file):
        raise ValueError("V225 score CSV hash mismatch")
    if _sha(review_file) not in set(v225.get("source_hashes", {}).values()):
        raise ValueError("daily review is not a hashed V225 source")
    review = json.loads(review_file.read_text(encoding="utf-8-sig"))
    scores = pd.read_csv(scores_file, dtype={"code": str})
    rows, duel, report = build(scores, review)
    rows_path, duel_path = target / "relay_role_candidates.csv", target / "anchor_core_duel.csv"
    target.mkdir(parents=True)
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    duel.to_csv(duel_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(scores_file.resolve()): _sha(scores_file),
        str(v225_file.resolve()): _sha(v225_file),
        str(review_file.resolve()): _sha(review_file),
    }
    report["artifact_hashes"] = {
        rows_path.name: _sha(rows_path), duel_path.name: _sha(duel_path),
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "report.md").write_text(_markdown(rows, duel, report), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--same-height-report", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.scores, ns.same_height_report, ns.review, ns.output),
                     ensure_ascii=False, indent=2))
