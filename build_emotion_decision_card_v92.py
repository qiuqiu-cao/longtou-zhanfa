"""Build a single evidence-bound daily emotion decision card."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(market: pd.DataFrame, baseline: dict, ensemble: dict, policy: dict,
          review: dict, recovery: dict | None = None,
          recovery_governance: dict | None = None,
          repair_quality: dict | None = None,
          policy_evidence: dict | None = None) -> dict:
    row = market.sort_values("date").iloc[-1]
    signal_date = pd.to_datetime(row.date).strftime("%Y-%m-%d")
    baseline_latest = baseline["latest_prediction"]
    ensemble_latest = ensemble["latest_research_prediction"]
    policy_latest = policy["current_shadow_application"]
    dates = {signal_date, baseline_latest["signal_date"], ensemble_latest["signal_date"],
             review["signal_date"]}
    if len(dates) != 1:
        raise ValueError(f"signal dates do not match: {sorted(dates)}")
    threshold = float(policy["risk_off_threshold"])
    ensemble_risk = float(ensemble_latest["probabilities"]["RISK_OFF"])
    triggered = ensemble_risk >= threshold
    policy_shadow_same_date = policy_latest.get("signal_date") == signal_date
    if policy_shadow_same_date and bool(policy_latest["triggered"]) != triggered:
        raise ValueError("policy shadow trigger disagrees with current ensemble probability")
    top = ensemble_latest["top_class"]
    if triggered:
        stance = "PAPER_RISK_VETO_TRIGGERED"
    elif top == "RISK_OFF":
        stance = "RISK_OFF_LEAN_BELOW_VETO"
    elif top == "REPAIR":
        stance = "REPAIR_LEAN_UNCONFIRMED"
    else:
        stance = "RISK_ON_LEAN_UNCONFIRMED"
    evidence = {
        "limit_count": int(float(row.limit_count)),
        "relay_count": int(float(row.consecutive_count)),
        "failed_count": int(float(row.failed_count)),
        "down_limit_count": int(float(row.down_limit_count)),
        "seal_rate": float(row.seal_rate),
        "failed_rate": float(row.failed_rate),
        "promotion_mid": float(row.promotion_mid),
        "yesterday_premium": float(row.yesterday_premium),
        "previous_limit_tail_loss": float(row.previous_limit_tail_loss),
        "max_board": int(float(row.max_board)),
        "ladder_gap_count": int(float(row.ladder_gap_count)),
    }
    card = {
        "version": "v92-emotion-decision-card-2",
        "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_date,
        "current_emotion_state": row.emotion_state,
        "market_evidence": evidence,
        "forecasts": {
            "seven_state_markov": baseline_latest["probabilities"],
            "three_regime_markov": baseline_latest["group_probabilities"],
            "three_regime_nested_ensemble": ensemble_latest["probabilities"],
            "ensemble_model_gate_passed": bool(ensemble["promotion_gate_passed"]),
        },
        "paper_risk_veto": {
            "threshold": threshold, "risk_off_probability": ensemble_risk,
            "distance_to_threshold": ensemble_risk - threshold,
            "triggered": triggered, "policy_status": policy["status"],
            "forward_gate_status": policy["forward_gate"]["status"],
            "threshold_is_frozen": True,
            "policy_reference_signal_date": policy_latest.get("signal_date"),
            "current_trigger_recomputed_from_frozen_threshold": not policy_shadow_same_date,
        },
        "stance": stance,
        "trade_effect": {
            "review_decision": review["status"],
            "high_board_relay": "VETO" if triggered else "NOT_VETOED_BY_PAPER_MODEL",
            "middle_board_relay": "VETO" if triggered else "NOT_VETOED_BY_PAPER_MODEL",
            "first_board": "WATCH_ONLY" if triggered else "SEPARATE_STOCK_EVIDENCE_REQUIRED",
            "direct_buy_authorized": False,
            "reason": "emotion model is a risk overlay; the daily review and executable stock plan remain separate gates",
        },
        "interpretation": {
            "money_effect": "MIXED" if evidence["yesterday_premium"] >= 0 else "ADVERSE",
            "broken_board_loss": "ADVERSE" if evidence["failed_rate"] >= .35 or evidence["previous_limit_tail_loss"] <= -.05 else "NORMAL",
            "height_structure": "FRAGILE" if evidence["ladder_gap_count"] > 0 else "CONTINUOUS",
            "repair_confirmation": "UNAVAILABLE_BEFORE_NEXT_SESSION_CLOSE",
        },
        "guarantee": False,
    }
    if recovery is not None:
        recovery_latest = recovery["latest_research_prediction"]
        if recovery_latest["signal_date"] != signal_date:
            raise ValueError("recovery forecast signal date does not match")
        card["forecasts"]["riskoff_conditional_recovery"] = {
            "recovery_probability": recovery_latest["recovery_probability"],
            "continued_risk_off_probability": recovery_latest["continued_risk_off_probability"],
            "top_class": recovery_latest["top_class"],
            "point_estimate_gate_passed": bool(recovery["promotion_gate_passed"]),
            "uncertainty_stability_gate_passed": bool(
                recovery_governance and recovery_governance.get("uncertainty_stability_gate_passed")),
            "effective_historical_gate_passed": bool(
                recovery_governance and recovery_governance.get("effective_historical_gate_passed")),
            "status": recovery_latest["status"],
        }
        if recovery_latest["top_class"] == "RECOVERY":
            card["interpretation"]["conditional_transition"] = (
                "RECOVERY_EDGE_UNSTABLE_RESEARCH_ONLY"
                if not card["forecasts"]["riskoff_conditional_recovery"]["effective_historical_gate_passed"]
                else "RECOVERY_EDGE_BUT_NOT_TRADE_SIGNAL")
        else:
            card["interpretation"]["conditional_transition"] = "CONTINUED_RISK_OFF_EDGE"
    if repair_quality is not None:
        quality_latest = repair_quality["latest_research_prediction"]
        if quality_latest["signal_date"] != signal_date:
            raise ValueError("repair-quality forecast signal date does not match")
        gate = bool(repair_quality["promotion_gate_passed"])
        card["forecasts"]["repair_quality_split_challenger"] = {
            "probabilities": quality_latest.get("probabilities"),
            "top_class": quality_latest.get("top_class"),
            "historical_gate_passed": gate,
            "status": quality_latest["status"],
            "decision_authorized": False,
        }
        card["interpretation"]["repair_quality_split"] = (
            "HISTORICAL_GATE_PASSED_RESEARCH_ONLY_FORWARD_VALIDATION_REQUIRED"
            if gate else "REJECTED_HISTORICAL_GATE_DO_NOT_USE_FOR_POSITION_OR_ENTRY"
        )
    if policy_evidence is not None:
        summary = policy_evidence["evidence_summary"]
        governance = policy_evidence["policy_governance"]
        card["trade_effect"]["economic_evidence_scope"] = {
            "aggregate_market_damage_gate_passed": bool(
                summary["aggregate_market_damage_gate_passed"]),
            "first_board_return_proxy_gate_passed": bool(
                summary["first_board_return_proxy_gate_passed"]),
            "relay_return_proxy_gate_passed": bool(
                summary["relay_return_proxy_gate_passed"]),
            "complete_trade_count": int(summary["complete_trade_count"]),
            "blanket_relay_veto_promoted_to_live": bool(
                governance["blanket_relay_veto_promoted_to_live"]),
        }
        card["trade_effect"]["paper_veto_evidence_status"] = (
            "AGGREGATE_WARNING_SUPPORTED_CANDIDATE_RETURN_BENEFIT_NOT_VALIDATED"
        )
        card["interpretation"]["policy_evidence"] = (
            "use emotion to tighten confirmation and sizing; do not claim a guaranteed candidate loss or premium"
        )
    return card


def markdown(card: dict) -> str:
    e, f, veto = card["market_evidence"], card["forecasts"], card["paper_risk_veto"]
    seven = f["seven_state_markov"]
    group = f["three_regime_nested_ensemble"]
    lines = [
        f"# {card['signal_date']} 盘后情绪决策卡",
        "",
        f"当前状态：**{card['current_emotion_state']}**；决策姿态：`{card['stance']}`；完整复盘结论：`{card['trade_effect']['review_decision']}`。",
        "",
        "## 盘面证据",
        "",
        f"涨停{e['limit_count']}只、连板{e['relay_count']}只、炸板{e['failed_count']}只、跌停{e['down_limit_count']}只；封板率{e['seal_rate']:.1%}、炸板率{e['failed_rate']:.1%}。",
        f"中位晋级率{e['promotion_mid']:.1%}，昨日涨停中位溢价{e['yesterday_premium']:.2%}，尾部反馈{e['previous_limit_tail_loss']:.2%}；最高{e['max_board']}板，梯队断层{e['ladder_gap_count']}处。",
        "",
        "## 下一交易日预测",
        "",
        f"七状态基线：大分歧{seven['大分歧']:.1%}、退潮{seven['退潮']:.1%}、修复{seven['修复']:.1%}、加速{seven['加速']:.1%}、一致{seven['一致']:.1%}。",
        f"训练内选权重后的组合：风险关闭{group['RISK_OFF']:.1%}、修复{group['REPAIR']:.1%}、风险开启{group['RISK_ON']:.1%}。",
        f"纸面风险否决线{veto['threshold']:.2%}，当前距阈值{veto['distance_to_threshold']:+.2%}，触发：{'是' if veto['triggered'] else '否'}。",
        "",
        "## 使用边界",
        "",
        "该模型只做风险减法，不能产生买点。没有触发否决也不代表可以买；仍须通过个股持续性、公告、题材竞争、价格和可成交性检查。修复只能由下一交易日实际承接与收盘结构确认。",
    ]
    recovery = f.get("riskoff_conditional_recovery")
    if recovery:
        lines.insert(-4, f"大分歧/退潮专项模型：修复或转强{recovery['recovery_probability']:.1%}、风险延续{recovery['continued_risk_off_probability']:.1%}；仅为研究预测。")
    quality = f.get("repair_quality_split_challenger")
    if quality:
        quality_text = ("历史闸门通过但仍须前向验证" if quality["historical_gate_passed"]
                        else "历史闸门未通过，不参与仓位与买点")
        lines.insert(-4, f"修复质量三分类挑战模型：{quality_text}。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--recovery")
    parser.add_argument("--recovery-governance")
    parser.add_argument("--repair-quality")
    parser.add_argument("--policy-evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)) for name in ("market", "baseline", "ensemble", "policy", "review")}
    if args.recovery:
        paths["recovery"] = Path(args.recovery)
    if args.recovery_governance:
        paths["recovery_governance"] = Path(args.recovery_governance)
    if args.repair_quality:
        paths["repair_quality"] = Path(args.repair_quality)
    if args.policy_evidence:
        paths["policy_evidence"] = Path(args.policy_evidence)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable decision card directory")
    card = build(
        pd.read_csv(paths["market"]),
        json.loads(paths["baseline"].read_text(encoding="utf-8")),
        json.loads(paths["ensemble"].read_text(encoding="utf-8")),
        json.loads(paths["policy"].read_text(encoding="utf-8")),
        json.loads(paths["review"].read_text(encoding="utf-8")),
        json.loads(paths["recovery"].read_text(encoding="utf-8")) if "recovery" in paths else None,
        json.loads(paths["recovery_governance"].read_text(encoding="utf-8")) if "recovery_governance" in paths else None,
        json.loads(paths["repair_quality"].read_text(encoding="utf-8")) if "repair_quality" in paths else None,
        json.loads(paths["policy_evidence"].read_text(encoding="utf-8")) if "policy_evidence" in paths else None,
    )
    card["source_hashes"] = {str(path.resolve()): sha256(path) for path in paths.values()}
    output.mkdir(parents=True)
    (output / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "card.md").write_text(markdown(card), encoding="utf-8")
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
