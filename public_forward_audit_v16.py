#!/usr/bin/env python3
"""Audit public live-monitor runs without cherry-picking dates or decisions.

This is a collection-readiness ledger, not a performance backtest.  Public
Level-1 decisions cannot establish fills, complete trades, or a win rate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import public_live_monitor_v1 as monitor
from live_decision_bridge_v1 import sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_MONITOR_ROOT = ROOT / "results" / "v16" / "public_live_monitor"
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "public_forward_audit"
VERSION = "v16-public-forward-audit-1"


def _events(manifest_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(manifest.get("ledger_path", "")))
    if not path.exists():
        path = manifest_path.parent / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_records(
    monitor_root: Path = DEFAULT_MONITOR_ROOT,
    verifier: Callable[[Path], dict[str, Any]] = monitor.verify_run,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in sorted(Path(monitor_root).glob("*/*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"manifest_path": str(path.resolve()), "verification": "FAIL", "error": str(exc)})
            continue
        hashes[str(path.resolve())] = sha256_file(path)
        if manifest.get("version") != monitor.VERSION:
            records.append({
                "manifest_path": str(path.resolve()), "version": manifest.get("version"),
                "trade_date": manifest.get("trade_date"), "verification": "LEGACY_UNVERIFIABLE",
                "run_status": manifest.get("status"), "decision_statuses": [],
            })
            continue
        check = verifier(path)
        events = _events(path, manifest)
        decision_statuses = [
            str(event["decision_status"]) for event in events if event.get("decision_status") is not None
        ]
        records.append({
            "manifest_path": str(path.resolve()), "version": manifest.get("version"),
            "run_id": manifest.get("run_id"), "trade_date": manifest.get("trade_date"),
            "weekday": pd.Timestamp(manifest.get("trade_date")).dayofweek < 5,
            "verification": check.get("status"), "verification_errors": check.get("errors", []),
            "run_status": manifest.get("status"),
            "completed_captures": manifest.get("completed_captures", 0),
            "completed_decisions": manifest.get("completed_decisions", 0),
            "failed_decisions": manifest.get("failed_decisions", 0),
            "decision_statuses": decision_statuses,
            "direct_entry_authorized": manifest.get("direct_entry_authorized"),
        })
    return records, hashes


def summarize(records: list[dict[str, Any]], source_hashes: dict[str, str]) -> dict[str, Any]:
    current = [row for row in records if row.get("version") == monitor.VERSION]
    dates = Counter(str(row.get("trade_date")) for row in current if row.get("trade_date"))
    duplicate_dates = sorted(date for date, count in dates.items() if count > 1)
    verified = [row for row in current if row.get("verification") == "PASS"]
    eligible = [
        row for row in verified
        if row.get("weekday") and dates[str(row.get("trade_date"))] == 1
        and row.get("run_status") == "CAPTURE_AND_DECISION_SEQUENCE_COMPLETE_PUBLIC_ONLY"
        and row.get("completed_captures") == 6 and row.get("completed_decisions") == 3
        and row.get("failed_decisions") == 0 and row.get("direct_entry_authorized") is False
    ]
    statuses = Counter(status for row in eligible for status in row.get("decision_statuses", []))
    watch = sum(count for status, count in statuses.items() if status.startswith("WATCH"))
    cancel = sum(count for status, count in statuses.items() if status == "CANCEL")
    cash = sum(count for status, count in statuses.items() if status == "CASH")
    enter = sum(count for status, count in statuses.items() if status == "ENTER")
    return {
        "version": VERSION,
        "status": "FORWARD_PUBLIC_COLLECTION_IN_PROGRESS",
        "tradability_status": "DIAGNOSTIC_ONLY",
        "monitor_version": monitor.VERSION,
        "runs_found": len(records), "current_version_runs": len(current),
        "legacy_unverifiable_runs": sum(row.get("verification") == "LEGACY_UNVERIFIABLE" for row in records),
        "verified_current_runs": len(verified), "verification_failed_runs": len(current) - len(verified),
        "unique_current_trade_dates": len(dates), "duplicate_trade_dates": duplicate_dates,
        "eligible_complete_sessions": len(eligible),
        "ineligible_or_incomplete_current_runs": len(current) - len(eligible),
        "public_decision_observations": sum(statuses.values()),
        "decision_status_counts": dict(sorted(statuses.items())),
        "cash_decisions": cash, "cancel_decisions": cancel, "watch_decisions": watch,
        "enter_decisions": enter,
        "completed_trades": 0, "complete_trade_win_rate": None,
        "mean_net_return": None, "large_loss_rate": None, "account_max_drawdown": None,
        "forward_gate": {
            "required_completed_trades": 60, "observed_completed_trades": 0,
            "challenge_win_rate": 0.70, "challenge_large_loss_rate_max": 0.10,
            "passed": False,
        },
        "limitations": [
            "public Level-1 decisions never authorise ENTER and therefore cannot establish a fill",
            "empty and failed sessions remain in the audit; duplicate dates are ineligible rather than cherry-picked",
            "complete-trade performance requires verified broker/terminal fills and exits",
            "assurance is local tamper-evidence, not a trusted third-party timestamp",
        ],
        "deployment_allowed": False,
        "source_hashes": source_hashes,
        "records": records,
    }


def make_report(result: dict[str, Any]) -> str:
    return "\n".join([
        "# V16 公开实时前向采集审计", "",
        f"- 状态：**{result['status']}**",
        f"- 当前版本运行：{result['current_version_runs']}；验签通过：{result['verified_current_runs']}",
        f"- 合格完整交易日：{result['eligible_complete_sessions']}",
        f"- 重复运行日期：{', '.join(result['duplicate_trade_dates']) or '无'}",
        f"- 公开决策观察：{result['public_decision_observations']}（空仓{result['cash_decisions']}、取消{result['cancel_decisions']}、观察{result['watch_decisions']}、ENTER {result['enter_decisions']}）",
        "- 已完成交易：0；完整交易盈利率、平均净收益、大面率和回撤均不可计算。", "",
        "## 结论", "",
        "公开实时层目前只证明采集和决策过程能被完整保存，不能证明成交或策略收益。所有空仓、失败、迟到和重复运行均保留；只有接入可核验成交与退出后，才开始累计完整交易的60笔前向闸门。", "",
    ])


def run(monitor_root: Path = DEFAULT_MONITOR_ROOT, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    records, hashes = collect_records(monitor_root)
    result = summarize(records, hashes)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.md"
    report_path.write_text(make_report(result), encoding="utf-8")
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = sha256_file(report_path)
    json_path = output / "audit.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "audit_path": str(json_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit all public live forward sessions without cherry-picking")
    parser.add_argument("--monitor-root", type=Path, default=DEFAULT_MONITOR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.monitor_root, args.output), ensure_ascii=False, indent=2))
