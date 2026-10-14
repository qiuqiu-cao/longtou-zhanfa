"""Render the validated scope of V144 into an actionable market forecast card.

The card answers the user's practical question -- tomorrow leans toward
continuation or divergence -- while preserving the model's real boundary.  It
does not promote the rejected seven-state or three-session forecasts, and it
does not turn aggregate emotion risk into a stock-ranking bonus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_MARKET = {
    "date", "emotion_state", "limit_count", "consecutive_count", "max_board",
    "failed_count", "down_limit_count", "failed_rate", "seal_rate",
    "yesterday_premium", "previous_limit_tail_loss", "promotion_mid",
    "ladder_gap_count", "middle_density", "data_status", "source_timestamp",
}


def _path(state, band):
    if band == "HIGH":
        if state in {"加速", "一致"}:
            return "CONSENSUS_TO_DIVERGENCE_RISK", "偏分歧"
        if state in {"启动", "升温"}:
            return "WARMING_INTERRUPTED_BY_DIVERGENCE_RISK", "偏分歧"
        if state == "修复":
            return "REPAIR_FAILURE_OR_ROTATION_DIVERGENCE_RISK", "偏分歧"
        return "RISK_OFF_CONTINUATION", "偏分歧"
    if band == "LOW":
        if state in {"启动", "升温"}:
            return "WARMING_OR_ACCELERATION_CAN_CONTINUE", "偏一致但未确认"
        if state in {"加速", "一致"}:
            return "CONSENSUS_CAN_CONTINUE_BUT_CROWDING_STILL_MATTERS", "偏一致但未确认"
        if state == "修复":
            return "REPAIR_CONTINUATION_CANDIDATE", "偏修复延续"
        return "STOP_FALL_OR_REPAIR_CANDIDATE", "偏修复但未确认"
    return "MIXED_ROTATION_OR_DIVERGENCE", "一致/分歧无法明确"


def build(market: pd.DataFrame, emotion_report: dict, overlay_report: dict | None = None):
    missing = REQUIRED_MARKET - set(market)
    if missing:
        raise ValueError(f"market fields missing: {sorted(missing)}")
    row = market.sort_values("date").iloc[-1]
    latest = emotion_report.get("latest_research_prediction", {})
    signal_date = pd.to_datetime(row.date, errors="raise").strftime("%Y-%m-%d")
    probability = float(latest["next_session_risk_off_probability"])
    band = latest["risk_band"]
    if (
        emotion_report.get("status") != "DIAGNOSTIC_ONLY"
        or not emotion_report.get("outer_evaluation", {}).get("risk_warning_gate_passed")
        or latest.get("signal_date") != signal_date
        or band not in {"LOW", "MIXED", "HIGH"}
        or not 0 <= probability <= 1
        or row.data_status != "verified"
    ):
        raise ValueError("V144 forecast and verified latest market row required")
    policy = emotion_report.get("policy", {})
    if policy.get("individual_stock_ranking_use_allowed") is not False:
        raise ValueError("emotion model must not authorize stock reranking")
    primary_path, consistency_judgement = _path(str(row.emotion_state), band)

    overlay = {
        "available": False,
        "high_risk_top1_veto_supported": False,
        "low_risk_positive_rank_bonus_supported": False,
    }
    if overlay_report is not None:
        evidence = overlay_report.get("policy_evidence", {})
        if overlay_report.get("status") != "DIAGNOSTIC_ONLY":
            raise ValueError("diagnostic overlay audit required")
        overlay = {
            "available": True,
            "high_risk_top1_veto_supported": bool(
                evidence.get("high_risk_top1_veto_supported")
            ),
            "low_risk_positive_rank_bonus_supported": bool(
                evidence.get("low_risk_positive_rank_bonus_supported")
            ),
        }

    if band == "HIGH":
        selection_policy = [
            "do not expand from the emotion anchor to random followers",
            "require verified specific-theme synchronization and stronger opening absorption",
            "cash is preferred when the frozen primary lacks those confirmations",
        ]
    elif band == "MIXED":
        selection_policy = [
            "treat rotation and divergence as coexisting paths",
            "prefer a unique specific-theme pioneer/core over a crowded height follower",
            "freeze one primary and do not promote an intraday backup",
        ]
    else:
        selection_policy = [
            "low aggregate risk does not add points to any stock",
            "theme, competition, price and buyability gates remain unchanged",
            "a crowded acceleration or isolated height board can still be rejected",
        ]

    return {
        "version": "v148-emotion-market-forecast-card-1",
        "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_date,
        "current_state": str(row.emotion_state),
        "next_session": {
            "risk_off_probability": probability,
            "risk_band": band,
            "primary_path": primary_path,
            "consistency_vs_divergence": consistency_judgement,
            "alternative_path": "must_be_confirmed_by_next_session_market_and_theme_breadth",
        },
        "market_evidence": {
            "limit_count": int(float(row.limit_count)),
            "consecutive_count": int(float(row.consecutive_count)),
            "max_board": int(float(row.max_board)),
            "failed_count": int(float(row.failed_count)),
            "down_limit_count": int(float(row.down_limit_count)),
            "failed_rate": float(row.failed_rate),
            "seal_rate": float(row.seal_rate),
            "yesterday_premium": float(row.yesterday_premium),
            "previous_limit_tail_loss": float(row.previous_limit_tail_loss),
            "promotion_mid": float(row.promotion_mid),
            "ladder_gap_count": int(float(row.ladder_gap_count)),
            "middle_density": float(row.middle_density),
            "data_status": str(row.data_status),
            "source": str(row.get("source", "")),
            "observed_at": str(row.source_timestamp),
        },
        "leader_selection_effect": {
            "policy": selection_policy,
            "stock_reranking_allowed": False,
            "positive_score_bonus_allowed": False,
            "mechanical_top1_veto_allowed": overlay["high_risk_top1_veto_supported"],
            "overlay_audit": overlay,
        },
        "two_to_three_session_outlook": {
            "status": "UNAVAILABLE_FAILED_HISTORICAL_GATE",
            "probability": None,
            "required_separate_evidence": [
                "specific-theme catalyst persistence",
                "theme ladder and same-height competition",
                "candidate B+2/B+3 value and large-loss model",
            ],
        },
        "guarantee": False,
        "entry_authorized": False,
    }


def run(market_path, emotion_path, output_path, overlay_path=None):
    market_source, emotion_source, target = map(Path, (market_path, emotion_path, output_path))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable forecast card")
    overlay_source = Path(overlay_path) if overlay_path else None
    sources = [market_source, emotion_source] + ([overlay_source] if overlay_source else [])
    card = build(
        pd.read_csv(market_source),
        json.loads(emotion_source.read_text(encoding="utf-8")),
        json.loads(overlay_source.read_text(encoding="utf-8")) if overlay_source else None,
    )
    card["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.market, args.emotion, args.output, args.overlay),
                     ensure_ascii=False, indent=2))

