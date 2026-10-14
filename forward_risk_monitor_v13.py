#!/usr/bin/env python3
"""Forward-only large-loss calibration monitor for the V10 shadow ledger."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import intraday_forward_v10 as forward


ROOT = Path(__file__).resolve().parent
DEFAULT_LEDGER = forward.OUTPUT
DEFAULT_OUTPUT = ROOT / "results" / "v13" / "forward_risk"
VERSION = "v13-forward-risk-monitor-1"
MIN_RESOLVED = 60
MAX_LARGE_LOSS_RATE = 0.10
MAX_WILSON_UPPER = 0.15
MIN_PROMOTION_PRECISION = 0.70


def wilson_interval(losses: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = losses / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def segment_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [event for event in events if bool(event.get("executed_proxy"))]
    losses = [event for event in executed if event.get("large_loss") is not None]
    promotions = [event for event in executed if event.get("promotion_target") is not None]
    loss_count = sum(bool(event["large_loss"]) for event in losses)
    low, high = wilson_interval(loss_count, len(losses))
    return {
        "settled": len(events),
        "executed_proxy": len(executed),
        "resolved_large_loss": len(losses),
        "large_loss_label_coverage": len(losses) / len(executed) if executed else None,
        "large_losses": loss_count,
        "large_loss_rate": loss_count / len(losses) if losses else None,
        "large_loss_wilson_low": low,
        "large_loss_wilson_high": high,
        "resolved_promotions": len(promotions),
        "promotion_precision": (
            sum(bool(event["promotion_target"]) for event in promotions) / len(promotions)
            if promotions else None
        ),
        "sell_blocked": sum(bool(event.get("sell_blocked")) for event in executed),
    }


def summarize(events: list[dict[str, Any]], verification: dict[str, Any]) -> dict[str, Any]:
    plans = {str(event.get("session_id")): event for event in events if event.get("event_type") == "PLAN"}
    settlements = [event for event in events if event.get("event_type") == "SETTLEMENT"]
    enriched = []
    for event in settlements:
        row = dict(event)
        plan = plans.get(str(event.get("session_id")), {})
        row["candidate_boards"] = plan.get("candidate_boards")
        enriched.append(row)
    overall = segment_metrics(enriched)
    by_board = {}
    for board in (2, 3, 4, 5):
        part = [event for event in enriched if event.get("candidate_boards") == board]
        by_board[str(board)] = segment_metrics(part)
    resolved = overall["resolved_large_loss"]
    point_gate = bool(
        resolved >= MIN_RESOLVED
        and overall["large_loss_label_coverage"] is not None
        and overall["large_loss_label_coverage"] >= 0.90
        and overall["large_loss_rate"] is not None
        and overall["large_loss_rate"] <= MAX_LARGE_LOSS_RATE
        and overall["promotion_precision"] is not None
        and overall["promotion_precision"] >= MIN_PROMOTION_PRECISION
    )
    confidence_gate = bool(
        point_gate
        and overall["large_loss_wilson_high"] is not None
        and overall["large_loss_wilson_high"] <= MAX_WILSON_UPPER
    )
    if verification.get("status") != "PASS":
        status = "LEDGER_VERIFICATION_FAILED"
        reason = "V10 ledger failed verification; no evidence is counted"
    elif resolved < MIN_RESOLVED:
        status = "DATA_COLLECTION_REQUIRED"
        reason = f"need {MIN_RESOLVED - resolved} more resolved executed-proxy large-loss labels"
    elif not point_gate:
        status = "FORWARD_RISK_GATE_FAILED"
        reason = "resolved forward outcomes failed <=10% loss, >=90% label coverage or >=70% promotion gate"
    elif not confidence_gate:
        status = "POINT_GATE_ONLY_LOW_CONFIDENCE"
        reason = "point estimates passed but Wilson large-loss upper bound remains above 15%"
    else:
        status = "RISK_COMPONENT_FORWARD_GATE_PASSED"
        reason = "risk component passed; full strategy deployment gates remain independent and mandatory"
    return {
        "version": VERSION,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "minimum_resolved": MIN_RESOLVED,
            "maximum_large_loss_rate": MAX_LARGE_LOSS_RATE,
            "maximum_wilson_upper": MAX_WILSON_UPPER,
            "minimum_label_coverage": 0.90,
            "minimum_promotion_precision": MIN_PROMOTION_PRECISION,
        },
        "overall": overall,
        "by_board": by_board,
        "point_gate_passed": point_gate,
        "confidence_gate_passed": confidence_gate,
        "reason": reason,
        "verification": verification,
        "evidence_rule": "only hash-verified V10 settlements with on-time CONFIRM_RESEAL executed proxies are counted; missing labels remain unavailable",
        "deployment_effect": "component-only monitor; never upgrades a candidate and never overrides the full historical/forward gates",
    }


def render(report: dict[str, Any]) -> str:
    overall = report["overall"]
    def pct(value: Any) -> str:
        return "NA" if value is None else f"{100 * value:.2f}%"
    lines = [
        "# V13 前向大面风险监控",
        "",
        f"- 状态：`{report['status']}`",
        f"- 已结算：{overall['settled']}，纸面成交代理：{overall['executed_proxy']}，大面标签：{overall['resolved_large_loss']}",
        f"- 大面率：{pct(overall['large_loss_rate'])}，95% Wilson上界：{pct(overall['large_loss_wilson_high'])}",
        f"- 晋级精确率：{pct(overall['promotion_precision'])}",
        f"- 结论：{report['reason']}",
        "",
        "该监控只接受V10哈希链中按时确认并完成T+1结算的记录；缺失结果不会按安全样本计入。",
        "即使风险组件通过，也不能单独把系统升级为实盘状态。",
        "",
    ]
    return "\n".join(lines)


def run(ledger: Path = DEFAULT_LEDGER, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verification = forward.verify(ledger)
    events = forward.read_events(ledger) if verification.get("status") == "PASS" else []
    report = summarize(events, verification)
    output.mkdir(parents=True, exist_ok=True)
    (output / "forward_risk_monitor.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "forward_risk_monitor.md").write_text(render(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor forward-only V10 large-loss evidence")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.ledger, args.output), ensure_ascii=False, indent=2))
