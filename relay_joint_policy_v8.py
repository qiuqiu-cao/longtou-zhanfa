#!/usr/bin/env python3
"""Select a causal joint ENTER threshold from V8 relay predictions.

The rejected large-loss model is never used as a live probability.  The policy
uses stable promotion, fill and two-more-board predictions, then validates the
actual inferred T+1 large-loss rate.  A final chronological holdout remains
sealed unless calibration and validation both pass the frozen red lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emotion_challenger_v4 import ACTIVE_STATES


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "v8" / "relay_model_lab" / "prequential_predictions.csv"
MODEL_CARD = ROOT / "results" / "v8" / "relay_model_lab" / "relay_model_lab_card.json"
OUTPUT = ROOT / "results" / "v8" / "relay_joint_policy"
VERSION = "v8-relay-joint-policy-1"
PROMOTION = "p__promotion__plus_stock_memory__hist_gb"
FILL = "p__fill__plus_stock_memory__hist_gb"
TWO_MORE = "p__two_more__plus_stock_memory__hist_gb"
FILL_STATUSES = {"inferred_daily_limit_touch_fill", "sell_blocked_one_word_down"}
TARGET_PRECISION = 0.70
MAX_LARGE_LOSS = 0.10
MIN_EXECUTED = 31
TARGET_FREQUENCY = (3.0, 5.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, n: int, z: float = 1.96) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def execution_metrics(selected: pd.DataFrame, active_dates: int) -> dict[str, Any]:
    executed = selected[selected["execution_status"].isin(FILL_STATUSES)].copy()
    labeled = executed[executed["large_loss"].notna()]
    promotion = float(executed["promoted_next_day"].mean()) if len(executed) else math.nan
    loss = float(labeled["large_loss"].mean()) if len(labeled) else math.nan
    returns = pd.to_numeric(executed["net_return"], errors="coerce").dropna().to_numpy(float)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else math.nan
    if len(returns):
        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(np.r_[1.0, equity])
        curve = np.r_[1.0, equity]
        maximum_drawdown = float(np.min(curve / peak - 1))
    else:
        maximum_drawdown = math.nan
    weeks = max(1e-12, active_dates / 5.0)
    return {
        "signals": int(len(selected)), "executed": int(len(executed)),
        "fill_rate": float(len(executed) / len(selected)) if len(selected) else math.nan,
        "executed_promotion_precision": promotion,
        "promotion_wilson_95": wilson(int(executed["promoted_next_day"].sum()), len(executed)),
        "large_loss_labeled": int(len(labeled)),
        "large_loss_label_coverage": float(len(labeled) / len(executed)) if len(executed) else math.nan,
        "large_loss_rate": loss,
        "signals_per_active_week": float(len(selected) / weeks),
        "mean_net_return": float(returns.mean()) if len(returns) else math.nan,
        "median_net_return": float(np.median(returns)) if len(returns) else math.nan,
        "average_win": float(wins.mean()) if len(wins) else math.nan,
        "average_loss": float(losses.mean()) if len(losses) else math.nan,
        "profit_factor": profit_factor, "maximum_drawdown": maximum_drawdown,
        "unresolved_sell_blocked": int(executed["net_return"].isna().sum()),
    }


def passes(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["executed"] >= MIN_EXECUTED
        and metrics["executed_promotion_precision"] >= TARGET_PRECISION
        and metrics["large_loss_label_coverage"] >= 0.90
        and metrics["large_loss_rate"] <= MAX_LARGE_LOSS
        and TARGET_FREQUENCY[0] <= metrics["signals_per_active_week"] <= TARGET_FREQUENCY[1]
    )


def prepare(path: Path = INPUT) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": "string"}, low_memory=False)
    required = {
        "date", "code", "boards", "emotion_state", "valid_identity", "buyable_eod",
        "execution_status", "promoted_next_day", "large_loss", "net_return", PROMOTION, FILL, TWO_MORE,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"V8 policy input missing fields: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for name in (PROMOTION, FILL, TWO_MORE):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["joint_score"] = 0.65 * frame[TWO_MORE] + 0.25 * frame[PROMOTION] + 0.10 * frame[FILL]
    frame["executed_proxy"] = frame["execution_status"].isin(FILL_STATUSES)
    frame["eligible_common"] = (
        frame["emotion_state"].isin(ACTIVE_STATES)
        & frame["buyable_eod"].eq(1)
        & frame["joint_score"].notna()
    )
    return frame.sort_values(["date", "joint_score"], ascending=[True, False]).reset_index(drop=True)


def policy_population(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    common = frame[frame["eligible_common"]].copy()
    if policy == "strict_identity":
        common = common[common["valid_identity"].eq(1)]
    elif policy == "height_confirmed":
        # Historical true-theme evidence is unavailable, so second boards are
        # excluded instead of being treated as verified pioneers.
        common = common[common["boards"].between(3, 5)]
    else:
        raise ValueError(policy)
    common["daily_rank"] = common.groupby("date")["joint_score"].rank(method="first", ascending=False)
    return common[common["daily_rank"].eq(1)].copy()


def window_split(dates: np.ndarray) -> dict[str, np.ndarray]:
    if len(dates) < 120:
        raise ValueError("insufficient prediction dates for joint policy split")
    first = int(len(dates) * 0.50)
    second = int(len(dates) * 0.75)
    return {
        "calibration": dates[:first],
        "embargo_1": dates[first:first + 1],
        "validation": dates[first + 1:second],
        "embargo_2": dates[second:second + 1],
        "sealed_holdout": dates[second + 1:],
    }


def scan(population: pd.DataFrame, dates: np.ndarray, all_active_dates: int) -> list[dict[str, Any]]:
    part = population[population["date"].isin(dates)].copy()
    if part.empty:
        return []
    thresholds = sorted(set(np.quantile(part["joint_score"], np.linspace(0, 0.98, 100)).round(8)))
    rows = []
    for threshold in thresholds:
        selected = part[part["joint_score"] >= threshold]
        metrics = execution_metrics(selected, all_active_dates)
        rows.append({"threshold": float(threshold), **metrics, "passes": passes(metrics)})
    return rows


def diagnostic_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["executed"] >= 10]
    if not usable:
        return {"status": "INSUFFICIENT_EXECUTED_TRADES"}
    best_precision = max(usable, key=lambda row: (row["executed_promotion_precision"], row["executed"]))
    best_loss = min(usable, key=lambda row: (row["large_loss_rate"], -row["executed"]))
    best_return = max(usable, key=lambda row: (row["mean_net_return"], row["executed"]))
    return {"best_precision": best_precision, "best_large_loss": best_loss, "best_mean_return": best_return}


def evaluate_policy(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    population = policy_population(frame, policy)
    dates = np.array(sorted(frame.loc[frame["eligible_common"], "date"].unique()))
    split = window_split(dates)
    active_by_window = {
        key: int(frame.loc[frame["date"].isin(value) & frame["emotion_state"].isin(ACTIVE_STATES), "date"].nunique())
        for key, value in split.items()
    }
    calibration_rows = scan(population, split["calibration"], active_by_window["calibration"])
    feasible = [row for row in calibration_rows if row["passes"]]
    result: dict[str, Any] = {
        "policy": policy,
        "population_rows": int(len(population)),
        "split": {key: [str(pd.Timestamp(value[0]).date()), str(pd.Timestamp(value[-1]).date()), int(len(value))]
                  for key, value in split.items() if len(value)},
        "calibration_frontier": diagnostic_frontier(calibration_rows),
        "calibration_feasible_thresholds": int(len(feasible)),
        "sealed_holdout_evaluated": False,
    }
    if not feasible:
        result.update({
            "status": "NO_FEASIBLE_CALIBRATION_THRESHOLD",
            "operational_threshold": None,
            "validation": None,
            "sealed_holdout": "NOT_EVALUATED",
        })
        return result
    chosen = max(feasible, key=lambda row: (row["executed"], row["executed_promotion_precision"], -row["large_loss_rate"]))
    threshold = chosen["threshold"]
    validation_population = population[population["date"].isin(split["validation"])]
    validation = execution_metrics(
        validation_population[validation_population["joint_score"] >= threshold],
        active_by_window["validation"],
    )
    validation["passes"] = passes(validation)
    result.update({"calibration_selected": chosen, "validation": validation})
    if not validation["passes"]:
        result.update({
            "status": "VALIDATION_GATE_FAILED", "operational_threshold": None,
            "sealed_holdout": "NOT_EVALUATED",
        })
        return result
    holdout_population = population[population["date"].isin(split["sealed_holdout"])]
    holdout = execution_metrics(
        holdout_population[holdout_population["joint_score"] >= threshold],
        active_by_window["sealed_holdout"],
    )
    holdout["passes"] = passes(holdout)
    result.update({
        "status": "HISTORICAL_HOLDOUT_PASS" if holdout["passes"] else "HISTORICAL_HOLDOUT_FAIL",
        "operational_threshold": threshold if holdout["passes"] else None,
        "sealed_holdout_evaluated": True, "sealed_holdout": holdout,
    })
    return result


def run(input_path: Path = INPUT, output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = prepare(input_path)
    policies = [evaluate_policy(frame, policy) for policy in ("strict_identity", "height_confirmed")]
    any_pass = any(item["status"] == "HISTORICAL_HOLDOUT_PASS" for item in policies)
    result = {
        "version": VERSION, "status": "HISTORICAL_GATE_ONLY" if any_pass else "DIAGNOSTIC_ONLY",
        "gate_decision": "HISTORICAL_GATE_PASS" if any_pass else "AUTO_REJECTED_NO_FEASIBLE_CALIBRATION_THRESHOLD",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_rules": {
            "executed_promotion_precision": TARGET_PRECISION,
            "maximum_large_loss_rate": MAX_LARGE_LOSS,
            "minimum_executed_trades": MIN_EXECUTED,
            "signals_per_active_week": list(TARGET_FREQUENCY),
        },
        "large_loss_model_used": False,
        "large_loss_model_reason": "V8 large-loss models failed chronological stability; actual selected-trade labels remain the hard gate",
        "daily_signal_cap": 1, "policies": policies,
        "deployment_allowed": False,
        "deployment_reason": "historical dates are development evidence; execution labels are inferred daily OHLC and no tick/VWAP confirmation exists",
        "integrity": {
            "predictions_path": str(input_path.resolve()),
            "predictions_sha256": sha256_file(input_path),
            "model_card_path": str(MODEL_CARD.resolve()),
            "model_card_sha256": sha256_file(MODEL_CARD),
            "policy_script_path": str(Path(__file__).resolve()),
            "policy_script_sha256": sha256_file(Path(__file__)),
        },
    }
    path = output / "joint_policy_card.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["card"] = str(path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V8 joint relay threshold")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
