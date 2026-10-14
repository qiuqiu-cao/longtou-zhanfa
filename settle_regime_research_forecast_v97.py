from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


GROUPS = {
    "启动": "RISK_ON", "升温": "RISK_ON", "加速": "RISK_ON", "一致": "RISK_ON",
    "大分歧": "RISK_OFF", "退潮": "RISK_OFF", "修复": "REPAIR",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Settle immutable three-regime or recovery research forecasts.")
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    forecast_path, market_path, output = map(Path, (args.forecast, args.market, args.output))
    if output.exists():
        raise FileExistsError(output)
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    market = pd.read_csv(market_path, dtype={"date": str})
    rows = market.loc[market.date.eq(forecast["target_date"])]
    if len(rows) != 1:
        raise ValueError("target date must occur exactly once")
    actual_state = rows.iloc[0]["emotion_state"]
    actual_group = GROUPS[actual_state]
    result = {
        "version": "v97-regime-research-settlement-1",
        "status": "SETTLED_FORWARD_RESEARCH_ONLY",
        "signal_date": forecast["signal_date"], "target_date": forecast["target_date"],
        "frozen_at": forecast["frozen_at"], "settled_at": datetime.now().astimezone().isoformat(),
        "actual_state": actual_state, "actual_group": actual_group,
        "decision_authorized": False, "trade_result": None,
        "source_forecast": str(forecast_path.resolve()), "source_forecast_sha256": sha256(forecast_path),
        "source_market": str(market_path.resolve()), "source_market_sha256": sha256(market_path),
    }
    if "probabilities" in forecast:
        probs = forecast["probabilities"]
        result.update({
            "forecast_type": "THREE_REGIME", "probabilities": probs,
            "predicted_group": forecast["top_class"],
            "correct": forecast["top_class"] == actual_group,
            "multiclass_brier": sum((float(probs[g]) - float(g == actual_group)) ** 2 for g in ("RISK_ON", "RISK_OFF", "REPAIR")),
            "log_loss": -math.log(max(float(probs[actual_group]), 1e-15)),
        })
    else:
        actual_recovery = actual_group != "RISK_OFF"
        p = float(forecast["recovery_probability"])
        predicted = forecast["top_class"] == "RECOVERY"
        result.update({
            "forecast_type": "RISK_OFF_RECOVERY", "recovery_probability": p,
            "continued_risk_off_probability": 1.0 - p, "actual_recovery": actual_recovery,
            "predicted_recovery": predicted, "correct": predicted == actual_recovery,
            "binary_brier": (p - float(actual_recovery)) ** 2,
            "log_loss": -math.log(max(p if actual_recovery else 1.0 - p, 1e-15)),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
