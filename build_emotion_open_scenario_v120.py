"""Freeze 09:25/09:35 emotion confirmation rules before the target session.

This protocol turns vague phrases such as "承接不错" into auditable fields.  A
positive market classification only relaxes an emotion gate; it never creates a
stock entry by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS_925 = [
    "previous_limit_auction_median_return",
    "previous_limit_negative_auction_rate",
    "down_limit_indication_count",
    "relay_positive_auction_rate",
    "verified_theme_core_positive_rate",
]
FIELDS_935 = [
    "previous_limit_return_median",
    "early_failed_board_rate",
    "down_limit_count",
    "relay_positive_rate",
    "verified_core_above_vwap_rate",
    "verified_theme_sync_rate",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(record: dict, key: str):
    value = record.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def classify_925(record: dict) -> dict:
    values = {key: _number(record, key) for key in FIELDS_925}
    adverse = []
    if values["previous_limit_auction_median_return"] is not None and values["previous_limit_auction_median_return"] <= -.02:
        adverse.append("PREVIOUS_LIMIT_AUCTION_MEDIAN_LE_-2PCT")
    if values["previous_limit_negative_auction_rate"] is not None and values["previous_limit_negative_auction_rate"] >= .60:
        adverse.append("PREVIOUS_LIMIT_NEGATIVE_RATE_GE_60PCT")
    if values["down_limit_indication_count"] is not None and values["down_limit_indication_count"] >= 10:
        adverse.append("DOWN_LIMIT_INDICATIONS_GE_10")
    if adverse:
        return {"state": "CONTINUED_RISK_OFF", "evidence": adverse,
                "emotion_gate": "CLOSED"}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        return {"state": "DATA_LIMITED", "missing": missing, "emotion_gate": "CLOSED"}
    effective = (
        values["previous_limit_auction_median_return"] >= 0
        and values["previous_limit_negative_auction_rate"] <= .45
        and values["down_limit_indication_count"] <= 5
        and values["relay_positive_auction_rate"] >= .50
        and values["verified_theme_core_positive_rate"] >= .50
    )
    if effective:
        return {"state": "EFFECTIVE_REPAIR_CANDIDATE", "evidence": [
            "ALL_FIVE_0925_REPAIR_CHECKS_PASS",
        ], "emotion_gate": "WAIT_0935"}
    return {"state": "WEAK_OR_MIXED_REPAIR", "evidence": [
        "NO_HARD_RISK_TRIGGER_BUT_NOT_ALL_REPAIR_CHECKS_PASS",
    ], "emotion_gate": "CLOSED"}


def classify_935(record: dict) -> dict:
    values = {key: _number(record, key) for key in FIELDS_935}
    adverse = []
    if values["previous_limit_return_median"] is not None and values["previous_limit_return_median"] <= -.02:
        adverse.append("PREVIOUS_LIMIT_MEDIAN_LE_-2PCT")
    if values["early_failed_board_rate"] is not None and values["early_failed_board_rate"] >= .35:
        adverse.append("EARLY_FAILED_BOARD_RATE_GE_35PCT")
    if values["down_limit_count"] is not None and values["down_limit_count"] >= 10:
        adverse.append("DOWN_LIMIT_COUNT_GE_10")
    if values["relay_positive_rate"] is not None and values["relay_positive_rate"] <= .25:
        adverse.append("RELAY_POSITIVE_RATE_LE_25PCT")
    if adverse:
        return {"state": "CONTINUED_RISK_OFF", "evidence": adverse,
                "emotion_gate": "CLOSED"}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        return {"state": "DATA_LIMITED", "missing": missing, "emotion_gate": "CLOSED"}
    effective = (
        values["previous_limit_return_median"] > 0
        and values["early_failed_board_rate"] <= .30
        and values["down_limit_count"] <= 5
        and values["relay_positive_rate"] >= .50
        and values["verified_core_above_vwap_rate"] >= .50
        and values["verified_theme_sync_rate"] >= .50
    )
    if effective:
        return {"state": "EFFECTIVE_REPAIR_CONFIRMED_AT_0935", "evidence": [
            "ALL_SIX_0935_REPAIR_CHECKS_PASS",
        ], "emotion_gate": "CONDITIONALLY_OPEN_STOCK_GATES_STILL_REQUIRED"}
    return {"state": "WEAK_OR_NARROW_REPAIR", "evidence": [
        "NO_HARD_RISK_TRIGGER_BUT_NOT_ALL_EFFECTIVE_REPAIR_CHECKS_PASS",
    ], "emotion_gate": "CLOSED"}


def build(card: dict, frozen: dict) -> dict:
    signal_date = card["signal_date"]
    if frozen["signal_date"] != signal_date:
        raise ValueError("decision card and frozen forecast signal dates differ")
    if frozen.get("outcome") is not None:
        raise ValueError("scenario plan must be frozen before target outcome is attached")
    risk = float(card["forecasts"]["three_regime_nested_ensemble"]["RISK_OFF"])
    return {
        "version": "v120-emotion-open-scenario-protocol-1",
        "status": "FROZEN_DIAGNOSTIC_PROTOCOL",
        "signal_date": signal_date,
        "target_date": frozen["target_date"],
        "preopen_default": "RISK_OFF" if card["paper_risk_veto"]["triggered"] else "MIXED",
        "frozen_close_forecast": {
            "risk_off_probability": risk,
            "repair_probability": float(card["forecasts"]["three_regime_nested_ensemble"]["REPAIR"]),
            "risk_on_probability": float(card["forecasts"]["three_regime_nested_ensemble"]["RISK_ON"]),
            "paper_veto_triggered": bool(card["paper_risk_veto"]["triggered"]),
        },
        "snapshot_0925": {
            "required_fields": FIELDS_925,
            "values": {key: None for key in FIELDS_925},
            "classification": classify_925({}),
            "rules": {
                "risk_off_if_any": ["auction median<=-2%", "negative auction rate>=60%",
                                    "down-limit indications>=10"],
                "repair_candidate_if_all": ["auction median>=0", "negative auction rate<=45%",
                                            "down-limit indications<=5", "relay positive rate>=50%",
                                            "verified theme-core positive rate>=50%"],
            },
        },
        "snapshot_0935": {
            "required_fields": FIELDS_935,
            "values": {key: None for key in FIELDS_935},
            "classification": classify_935({}),
            "rules": {
                "risk_off_if_any": ["previous-limit median<=-2%", "early failed-board rate>=35%",
                                    "down-limit count>=10", "relay positive rate<=25%"],
                "effective_repair_if_all": ["previous-limit median>0", "early failed-board rate<=30%",
                                            "down-limit count<=5", "relay positive rate>=50%",
                                            "verified cores above VWAP>=50%", "verified theme sync>=50%"],
            },
        },
        "decision_effect": {
            "CONTINUED_RISK_OFF": "CASH",
            "WEAK_OR_MIXED_REPAIR": "CASH",
            "WEAK_OR_NARROW_REPAIR": "CASH",
            "DATA_LIMITED": "CASH",
            "EFFECTIVE_REPAIR_CONFIRMED_AT_0935": (
                "emotion gate only conditionally opens; candidate, price, execution, and risk gates remain mandatory"
            ),
            "unattended_user_default": "CASH_NO_PRESET_BLIND_ORDER",
        },
        "field_definitions": {
            "early_failed_board_rate": "09:35 touched-but-unsealed count / all securities touched limit by 09:35",
            "relay_positive_rate": "share of prior-close consecutive-board candidates above prior close",
            "verified_core_above_vwap_rate": "share of predeclared verified theme cores trading above 09:30-09:35 VWAP",
            "verified_theme_sync_rate": "share of members in the selected verified theme above prior close",
            "missing_policy": "missing cannot pass a repair condition; a known hard-risk condition may still close the gate",
        },
        "historical_calibration_status": "NOT_YET_VALIDATED_AS_ENTRY_OVERLAY",
        "direct_buy_authorized": False,
        "order_submitted": False,
        "guarantee": False,
    }


def markdown(plan: dict) -> str:
    forecast = plan["frozen_close_forecast"]
    return "\n".join([
        f"# {plan['signal_date']}盘后 → {plan['target_date']}情绪路径预案",
        "",
        f"盘前默认：**{plan['preopen_default']}**。风险关闭{forecast['risk_off_probability']:.1%}、修复{forecast['repair_probability']:.1%}、风险开启{forecast['risk_on_probability']:.1%}。",
        "",
        "## 09:25",
        "",
        "任一风险条件成立即继续风险关闭；五项修复条件必须全部成立，才进入等待09:35确认。字段缺失不能按通过处理。",
        "",
        "## 09:35",
        "",
        "任一风险条件成立继续空仓；六项有效修复条件必须全部成立，才仅解除情绪层否决。个股、价格、公告、可成交性和退出规则仍须分别通过。",
        "",
        "## 不盯盘规则",
        "",
        "没有09:25与09:35快照时默认空仓，不预挂盲目追板单。该协议尚未完成历史执行校准，不能产生直接买点。",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--frozen-forecast", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    card_path, frozen_path, output = Path(args.card), Path(args.frozen_forecast), Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable scenario protocol")
    plan = build(
        json.loads(card_path.read_text(encoding="utf-8")),
        json.loads(frozen_path.read_text(encoding="utf-8")),
    )
    plan["source_hashes"] = {
        str(card_path.resolve()): sha256(card_path),
        str(frozen_path.resolve()): sha256(frozen_path),
    }
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "plan.md").write_text(markdown(plan), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
