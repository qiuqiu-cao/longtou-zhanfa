"""Settle an immutable emotion forecast without modifying the original."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


STATES = ["启动", "升温", "加速", "一致", "大分歧", "退潮", "修复"]
GROUP = {
    **{state: "RISK_ON" for state in ("启动", "升温", "加速", "一致")},
    **{state: "RISK_OFF" for state in ("大分歧", "退潮")},
    "修复": "REPAIR",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settle(forecast_path: Path, market_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite settlement")
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    if forecast.get("outcome_status") != "UNAVAILABLE_BEFORE_TARGET_CLOSE":
        raise ValueError("forecast is not an unsettled forward record")
    probabilities = forecast.get("probabilities", {})
    if set(probabilities) != set(STATES):
        raise ValueError("forecast must contain all seven state probabilities")
    market = pd.read_csv(market_path)
    market["date"] = pd.to_datetime(market.date, errors="coerce").dt.strftime("%Y-%m-%d")
    rows = market.loc[market.date.eq(forecast["target_date"])]
    if len(rows) != 1:
        raise ValueError("target-date outcome is missing or duplicated")
    actual = rows.iloc[0].emotion_state
    if actual not in STATES:
        raise ValueError("target-date emotion state is invalid")
    p = np.array([float(probabilities[state]) for state in STATES])
    if not np.isclose(p.sum(), 1.0):
        raise ValueError("forecast probabilities do not sum to one")
    predicted_state = STATES[int(p.argmax())]
    group_probabilities = {
        name: float(sum(probabilities[state] for state, mapped in GROUP.items() if mapped == name))
        for name in ("RISK_ON", "RISK_OFF", "REPAIR")
    }
    predicted_group = max(group_probabilities, key=group_probabilities.get)
    y = np.array([1.0 if state == actual else 0.0 for state in STATES])
    adverse_probability = group_probabilities["RISK_OFF"]
    adverse_actual = GROUP[actual] == "RISK_OFF"
    result = {
        "version": "v84-emotion-forecast-settlement-1",
        "status": "SETTLED_FORWARD_DIAGNOSTIC",
        "signal_date": forecast["signal_date"],
        "target_date": forecast["target_date"],
        "frozen_at": forecast["frozen_at"],
        "settled_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "predicted_state": predicted_state,
        "actual_state": actual,
        "seven_state_correct": predicted_state == actual,
        "multiclass_brier": float(np.sum((p - y) ** 2)),
        "log_loss": float(-np.log(max(float(probabilities[actual]), 1e-12))),
        "group_probabilities": group_probabilities,
        "predicted_group": predicted_group,
        "actual_group": GROUP[actual],
        "three_regime_correct": predicted_group == GROUP[actual],
        "adverse_probability": adverse_probability,
        "adverse_actual": adverse_actual,
        "adverse_binary_brier": float((adverse_probability - float(adverse_actual)) ** 2),
        "source_forecast": str(forecast_path.resolve()),
        "source_forecast_sha256": sha256(forecast_path),
        "source_market": str(market_path.resolve()),
        "source_market_sha256": sha256(market_path),
        "trade_result": None,
        "trade_result_status": "NOT_A_TRADE_SIGNAL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--market", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = settle(Path(args.forecast), Path(args.market), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
