"""Build an emotion-first next-session scenario card from governed evidence."""
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_path(state: str, risk_band: str) -> tuple[str, str, str]:
    if risk_band == "HIGH":
        return (
            "DIVERGENCE_OR_RISK_RELEASE",
            "stronger opening breadth, positive prior-limit premium and intact middle boards",
            "failed repair, expanding down-limits or synchronized middle-board breaks",
        )
    if risk_band == "LOW":
        if state in {"大分歧", "退潮"}:
            primary = "STOP_FALL_OR_REPAIR_CANDIDATE"
        elif state == "修复":
            primary = "REPAIR_CONTINUATION_CANDIDATE"
        else:
            primary = "CONSISTENCY_OR_WARMING_CAN_CONTINUE"
        return (
            primary,
            "breadth and theme ladders strengthen without crowded acceleration",
            "high-board acceleration with weakening followers or sudden loss expansion",
        )
    return (
        "ROTATION_AND_DIVERGENCE_COEXIST",
        "specific-theme breadth, middle-board promotion and prior-limit premium improve together",
        "height survives alone while middle boards and prior winners deteriorate",
    )


def build(market: pd.DataFrame, risk: dict, damage: dict) -> dict:
    missing = REQUIRED_MARKET - set(market)
    if missing:
        raise ValueError(f"market fields missing: {sorted(missing)}")
    row = market.sort_values("date").iloc[-1]
    signal_date = pd.to_datetime(row.date, errors="raise").strftime("%Y-%m-%d")
    risk_latest = risk.get("latest_research_prediction", {})
    damage_latest = damage.get("latest_research_prediction", {})
    if (
        risk.get("status") != "DIAGNOSTIC_ONLY"
        or not risk.get("outer_evaluation", {}).get("risk_warning_gate_passed")
        or risk_latest.get("signal_date") != signal_date
        or damage.get("status") != "DIAGNOSTIC_ONLY"
        or damage_latest.get("signal_date") != signal_date
        or str(row.data_status) != "verified"
    ):
        raise ValueError("aligned governed emotion evidence and a verified market row are required")
    probability = float(risk_latest["next_session_risk_off_probability"])
    risk_band = str(risk_latest["risk_band"])
    if risk_band not in {"LOW", "MIXED", "HIGH"} or not 0 <= probability <= 1:
        raise ValueError("invalid V144 probability or risk band")

    components = damage_latest.get("components", {})
    validated = {
        name: {
            "probability": float(details["probability"]),
            "outcome": details["outcome"],
            "damage_direction": details["damage_direction"],
        }
        for name, details in components.items()
        if details.get("validated_for_aggregate") is True
    }
    aggregate_passed = damage.get("aggregate_gate", {}).get("passed") is True
    damage_band = damage_latest.get("damage_band") if aggregate_passed else "UNAVAILABLE"
    if aggregate_passed and damage_band not in {
        "LOW_DAMAGE_RISK", "MIXED_DAMAGE_RISK", "HIGH_DAMAGE_RISK"
    }:
        raise ValueError("invalid validated damage band")

    primary, support, invalidate = _base_path(str(row.emotion_state), risk_band)
    if risk_band == "HIGH" and damage_band == "HIGH_DAMAGE_RISK":
        confidence = "CORROBORATED_HIGH"
    elif risk_band == "LOW" and damage_band == "LOW_DAMAGE_RISK":
        confidence = "CORROBORATED_HIGH"
    elif aggregate_passed:
        confidence = "CONFLICTING_OR_MIXED"
    else:
        confidence = "V144_ONLY_DAMAGE_VECTOR_NOT_READY"

    if risk_band == "HIGH":
        allowed_candidate = "verified unique core after strong absorption; otherwise cash"
        forbidden = "isolated followers, crowded acceleration and unbuyable one-price boards"
    elif risk_band == "LOW":
        allowed_candidate = "specific-theme pioneer or turnover core after buyability confirmation"
        forbidden = "using low aggregate risk as an automatic stock score bonus"
    else:
        allowed_candidate = "one frozen specific-theme core; first-board and relay compared separately"
        forbidden = "promoting an intraday backup when the frozen primary fails"

    return {
        "version": "v156-emotion-first-scenario-card-1",
        "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_date,
        "current_market": {
            "emotion_state": str(row.emotion_state),
            "limit_count": int(float(row.limit_count)),
            "consecutive_count": int(float(row.consecutive_count)),
            "max_board": int(float(row.max_board)),
            "failed_count": int(float(row.failed_count)),
            "failed_rate": float(row.failed_rate),
            "down_limit_count": int(float(row.down_limit_count)),
            "yesterday_premium": float(row.yesterday_premium),
            "previous_limit_tail_loss": float(row.previous_limit_tail_loss),
            "promotion_mid": float(row.promotion_mid),
            "ladder_gap_count": int(float(row.ladder_gap_count)),
            "middle_density": float(row.middle_density),
            "observed_at": str(row.source_timestamp),
            "data_status": str(row.data_status),
        },
        "next_session_forecast": {
            "risk_off_probability": probability,
            "risk_band": risk_band,
            "primary_scenario": primary,
            "confidence": confidence,
            "supporting_market_change": support,
            "invalidation_market_change": invalidate,
            "damage_vector_aggregate_ready": aggregate_passed,
            "damage_band": damage_band,
            "validated_damage_components": validated,
            "unvalidated_component_count": len(components) - len(validated),
        },
        "transmission_logic": [
            "prior winners lose premium or show deeper tail loss",
            "middle-board promotion weakens and ladder gaps widen",
            "failed boards and down-limits expand",
            "height board becomes isolated and theme followers stop responding",
            "leader selection threshold rises and cash becomes a valid outcome",
        ],
        "leader_selection_policy": {
            "allowed_candidate_type": allowed_candidate,
            "forbidden_inference": forbidden,
            "emotion_direct_stock_reranking_allowed": False,
            "first_board_and_relay_must_be_ranked_separately": True,
            "one_comprehensive_primary_or_cash": True,
        },
        "horizon": {
            "next_session": "AVAILABLE_DIAGNOSTIC",
            "next_two_to_three_sessions": "UNAVAILABLE_FAILED_HISTORICAL_GATE",
        },
        "entry_authorized": False,
        "guarantee": False,
    }


def run(market_path, risk_path, damage_path, output_path):
    market_source, risk_source, damage_source, target = map(
        Path, (market_path, risk_path, damage_path, output_path))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable scenario card")
    card = build(
        pd.read_csv(market_source),
        json.loads(risk_source.read_text(encoding="utf-8")),
        json.loads(damage_source.read_text(encoding="utf-8")),
    )
    card["source_hashes"] = {
        str(path.resolve()): _sha256(path)
        for path in (market_source, risk_source, damage_source)
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--damage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.market, args.risk, args.damage, args.output),
                     ensure_ascii=False, indent=2))
