#!/usr/bin/env python3
"""Chronological selective-risk audit for the rejected V8 large-loss model.

This layer never turns a candidate into ENTER.  It tests whether the rejected
large-loss model can be salvaged as a veto-only ranking signal on one frozen
daily top candidate.  Thresholds are selected with development and validation
dates only; the final blind window is opened once for an honest deployment gate.
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


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "v8" / "relay_model_lab" / "prequential_predictions.csv"
DEFAULT_OUTPUT = ROOT / "results" / "v12" / "selective_risk_gate"
VERSION = "v12-selective-risk-gate-1"
ACTIVE_STATES = {"启动", "升温", "加速", "修复"}

P_PROMOTION = "p__promotion__plus_stock_memory__hist_gb"
P_FILL = "p__fill__plus_stock_memory__hist_gb"
P_TWO_MORE = "p__two_more__plus_stock_memory__hist_gb"
P_LOSS = "p__large_loss__plus_stock_memory__hist_gb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(losses: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = losses / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def prepare_daily_top(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "code", "boards", "emotion_state", "buyable_eod", "one_word_proxy",
        "promotion_target", "large_loss_target", "net_return", P_PROMOTION, P_FILL, P_TWO_MORE, P_LOSS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prequential predictions missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["code"] = result["code"].astype("string").str.zfill(6)
    numeric = [
        "boards", "buyable_eod", "one_word_proxy", "promotion_target", "large_loss_target",
        "net_return", P_PROMOTION, P_FILL, P_TWO_MORE, P_LOSS,
    ]
    for field in numeric:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    result = result[
        result["date"].notna()
        & result["boards"].between(2, 5)
        & result["emotion_state"].isin(ACTIVE_STATES)
        & result["buyable_eod"].eq(1)
        & result["one_word_proxy"].eq(0)
        & result[["promotion_target", "large_loss_target", P_PROMOTION, P_FILL, P_TWO_MORE, P_LOSS]].notna().all(axis=1)
    ].copy()
    result["joint_score"] = 0.65 * result[P_TWO_MORE] + 0.25 * result[P_PROMOTION] + 0.10 * result[P_FILL]
    result = result.sort_values(["date", "joint_score", "code"], ascending=[True, False, True])
    result = result.groupby("date", sort=True, as_index=False).head(1).reset_index(drop=True)
    if result.duplicated("date").any():
        raise AssertionError("daily top selection must have at most one row per date")
    return result


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = np.array(sorted(frame["date"].dropna().unique()))
    if len(dates) < 100:
        raise ValueError("need at least 100 daily top observations for three-way risk audit")
    first = max(1, int(math.floor(len(dates) * 0.60)))
    second = max(first + 2, int(math.floor(len(dates) * 0.80)))
    embargo_1 = dates[first]
    embargo_2 = dates[second]
    development_dates = set(dates[:first])
    validation_dates = set(dates[first + 1:second])
    blind_dates = set(dates[second + 1:])
    development = frame[frame["date"].isin(development_dates)].copy()
    validation = frame[frame["date"].isin(validation_dates)].copy()
    blind = frame[frame["date"].isin(blind_dates)].copy()
    metadata = {
        "total_dates": len(dates),
        "development": {"start": development["date"].min(), "end": development["date"].max(), "n": len(development)},
        "embargo_1": pd.Timestamp(embargo_1),
        "validation": {"start": validation["date"].min(), "end": validation["date"].max(), "n": len(validation)},
        "embargo_2": pd.Timestamp(embargo_2),
        "blind": {"start": blind["date"].min(), "end": blind["date"].max(), "n": len(blind)},
    }
    return development, validation, blind, metadata


def gate_metrics(frame: pd.DataFrame, joint_min: float, loss_max: float) -> dict[str, Any]:
    accepted = frame[frame["joint_score"].ge(joint_min) & frame[P_LOSS].le(loss_max)]
    rejected = frame.drop(index=accepted.index)
    losses = int(accepted["large_loss_target"].sum())
    low, high = wilson_interval(losses, len(accepted))
    accepted_rate = float(accepted["large_loss_target"].mean()) if len(accepted) else math.nan
    rejected_rate = float(rejected["large_loss_target"].mean()) if len(rejected) else math.nan
    risk_difference = rejected_rate - accepted_rate if len(accepted) and len(rejected) else math.nan
    risk_ratio = rejected_rate / accepted_rate if len(accepted) and accepted_rate > 0 and len(rejected) else math.nan
    return {
        "n": int(len(frame)), "accepted_n": int(len(accepted)), "rejected_n": int(len(rejected)),
        "accepted_dates": int(accepted["date"].nunique()),
        "coverage": float(len(accepted) / len(frame)) if len(frame) else math.nan,
        "accepted_large_losses": losses,
        "accepted_large_loss_rate": accepted_rate,
        "accepted_loss_wilson_low": low, "accepted_loss_wilson_high": high,
        "rejected_large_loss_rate": rejected_rate,
        "risk_difference_rejected_minus_accepted": risk_difference,
        "risk_ratio_rejected_over_accepted": risk_ratio,
        "accepted_promotion_precision": float(accepted["promotion_target"].mean()) if len(accepted) else math.nan,
        "accepted_mean_net_return": float(accepted["net_return"].mean()) if accepted["net_return"].notna().any() else math.nan,
    }


def finite(value: Any, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def choose_threshold(development: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    rows = []
    for joint_min in np.round(np.arange(0.15, 0.651, 0.025), 3):
        for loss_max in np.round(np.arange(0.15, 0.551, 0.025), 3):
            dev = gate_metrics(development, float(joint_min), float(loss_max))
            val = gate_metrics(validation, float(joint_min), float(loss_max))
            dev_ok = (
                dev["accepted_n"] >= 60 and dev["accepted_dates"] >= 40
                and finite(dev["risk_difference_rejected_minus_accepted"], -1) >= 0.03
                and finite(dev["risk_ratio_rejected_over_accepted"], -1) >= 1.10
            )
            val_ok = (
                val["accepted_n"] >= 25 and val["accepted_dates"] >= 20
                and finite(val["risk_difference_rejected_minus_accepted"], -1) >= 0.03
                and finite(val["risk_ratio_rejected_over_accepted"], -1) >= 1.10
            )
            rows.append({
                "joint_min": float(joint_min), "loss_max": float(loss_max),
                "development_eligible": bool(dev_ok), "validation_eligible": bool(val_ok),
                **{f"development_{key}": value for key, value in dev.items()},
                **{f"validation_{key}": value for key, value in val.items()},
            })
    audit = pd.DataFrame(rows)
    feasible = audit[audit["development_eligible"] & audit["validation_eligible"]].copy()
    if feasible.empty:
        return None, audit
    feasible["worst_preblind_ucb"] = feasible[[
        "development_accepted_loss_wilson_high", "validation_accepted_loss_wilson_high"
    ]].max(axis=1)
    feasible = feasible.sort_values(
        ["worst_preblind_ucb", "validation_accepted_large_loss_rate", "validation_accepted_n", "joint_min", "loss_max"],
        ascending=[True, True, False, False, True],
    )
    chosen = feasible.iloc[0]
    return {"joint_min": float(chosen["joint_min"]), "loss_max": float(chosen["loss_max"])}, audit


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def render_card(card: dict[str, Any]) -> str:
    selected = card["selected_threshold"]
    lines = [
        "# V12 选择性大面风险闸门",
        "",
        f"- 状态：`{card['status']}`",
        f"- 日级候选：{card['population']['daily_top_rows']} 条，{card['population']['start']} 至 {card['population']['end']}",
        "- 用途：只能否决或降级，永远不能把WATCH升级为ENTER。",
        "- 标签：日线触板成交与T+1结果代理，不等于真实委托成交。",
        "",
    ]
    if selected is None:
        lines.extend([
            "开发期与验证期没有同时稳定的阈值，风险闸门继续禁用。",
            "最终盲测未被用于反向选择阈值。",
        ])
    else:
        lines.extend([
            f"- 冻结阈值：联合分 ≥ {selected['joint_min']:.3f}，V8拒绝风险分 ≤ {selected['loss_max']:.3f}",
            "",
            "| 区间 | 接受数 | 大面率 | Wilson上界 | 晋级率 | 拒绝/接受风险比 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name in ("development", "validation", "blind"):
            metric = card["evaluation"][name]
            def pct(value: Any) -> str:
                return "NA" if value is None else f"{100 * value:.2f}%"
            ratio = metric["risk_ratio_rejected_over_accepted"]
            lines.append(
                f"| {name} | {metric['accepted_n']} | {pct(metric['accepted_large_loss_rate'])} | "
                f"{pct(metric['accepted_loss_wilson_high'])} | {pct(metric['accepted_promotion_precision'])} | "
                f"{'NA' if ratio is None else f'{ratio:.2f}'} |"
            )
        lines.extend([
            "",
            f"- 盲测结论：{card['blind_gate_reason']}",
            f"- 否决器结论：{card['veto_gate_reason']}",
        ])
    lines.extend([
        "",
        "无论结果如何，V8大面输出仍不得展示为可靠概率；只有完全盲测和冻结后前向结果均通过，才能进入风险决策接口。",
        "",
    ])
    return "\n".join(lines)


def run(input_path: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    frame = pd.read_csv(input_path, dtype={"code": "string"}, low_memory=False)
    daily = prepare_daily_top(frame)
    development, validation, blind, split = chronological_split(daily)
    selected, audit = choose_threshold(development, validation)
    evaluation = None
    blind_gate = False
    veto_gate = False
    blind_reason = "no preblind-stable threshold; blind window not used for threshold selection"
    veto_reason = "no preblind-stable veto threshold"
    if selected is not None:
        evaluation = {
            "development": gate_metrics(development, **selected),
            "validation": gate_metrics(validation, **selected),
            "blind": gate_metrics(blind, **selected),
        }
        blind_metric = evaluation["blind"]
        blind_gate = bool(
            blind_metric["accepted_n"] >= 60
            and finite(blind_metric["accepted_large_loss_rate"], 1) <= 0.10
            and finite(blind_metric["accepted_loss_wilson_high"], 1) <= 0.15
            and finite(blind_metric["accepted_promotion_precision"], -1) >= 0.70
        )
        blind_reason = (
            "PASS: blind accepted loss and promotion gates passed"
            if blind_gate else "FAIL: blind set did not simultaneously reach <=10% large-loss, <=15% Wilson upper bound, >=70% promotion and n>=60"
        )
        veto_gate = bool(
            blind_metric["accepted_n"] >= 40
            and finite(blind_metric["risk_difference_rejected_minus_accepted"], -1) >= 0.03
            and finite(blind_metric["risk_ratio_rejected_over_accepted"], -1) >= 1.10
        )
        veto_reason = (
            "PASS_RESEARCH_ONLY: risk ranking separated blind accepted/rejected groups; forward freeze is still required"
            if veto_gate else "FAIL: blind risk separation was too small or sample was insufficient"
        )
    status = "DEPLOYMENT_CANDIDATE_REQUIRES_FORWARD" if blind_gate else "VETO_ONLY_RESEARCH_CANDIDATE" if veto_gate else "REJECTED_DIAGNOSTIC_ONLY"
    card = {
        "version": VERSION,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "blind_opened": True,
        "blind_reuse_prohibited": True,
        "population": {
            "daily_top_rows": len(daily), "start": daily["date"].min(), "end": daily["date"].max(),
            "large_loss_rate": float(daily["large_loss_target"].mean()),
            "selection": "one highest V10-style joint-score, buyable, non-one-word, 2-5 board candidate per active-state date",
        },
        "split": split,
        "threshold_grid": {"joint_min": [0.15, 0.65, 0.025], "loss_max": [0.15, 0.55, 0.025]},
        "threshold_selection": "development+validation only; minimize worst preblind Wilson upper bound after fixed sample and risk-separation gates",
        "selected_threshold": selected,
        "evaluation": evaluation,
        "deployment_gate_passed": blind_gate,
        "blind_gate_reason": blind_reason,
        "veto_only_gate_passed": veto_gate,
        "veto_gate_reason": veto_reason,
        "live_use": "NEVER_UPGRADE; no live probability; risk output may only downgrade after a new frozen forward protocol",
        "label_limit": "inferred daily limit-touch fill and T+1 proxy; no tick/queue proof",
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "threshold_audit.csv", index=False, encoding="utf-8-sig")
    card_path = output / "model_card.json"
    card_path.write_text(json.dumps(json_safe(card), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(render_card(json_safe(card)), encoding="utf-8")
    return json_safe(card)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a veto-only large-loss gate without blind-set tuning")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
