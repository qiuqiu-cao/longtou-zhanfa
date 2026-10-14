#!/usr/bin/env python3
"""Aggregate broker exit reports using only completed, rule-compliant trades."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import broker_exit_import_v16 as lifecycle

VERSION = "v16-trade-report-aggregate-1"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "trade_report_aggregate"


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("strategy_complete_trade_eligible") is True]
    returns = [_finite(row.get("strategy_net_return")) for row in eligible]
    returns = [value for value in returns if value is not None]
    winners = [value for value in returns if value > 0]
    losers = [value for value in returns if value < 0]
    losses = sum(value <= -.07 for value in returns)
    # The report remains trade-level; account drawdown requires capital, sizing and
    # chronological mark-to-market data and is intentionally not inferred here.
    streak = maximum = 0
    for value in returns:
        streak = streak + 1 if value < 0 else 0
        maximum = max(maximum, streak)
    return {
        "reports": len(rows),
        "completed_strategy_trades": len(eligible),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "unresolved_or_off_plan_reports": len(rows) - len(eligible),
        # Keep unresolved states separate.  A missing exit is not a loss, and
        # a partial exit must not disappear into the generic non-eligible bin.
        "unresolved_open_position_reports": sum(bool(row.get("unresolved_open_position")) for row in rows),
        "partial_exit_reports": sum(bool(row.get("partial_exit")) for row in rows),
        "fee_incomplete_reports": sum(row.get("fee_data_complete") is not True for row in rows),
        "off_plan_or_ineligible_reports": sum(
            row.get("strategy_complete_trade_eligible") is not True
            and not row.get("unresolved_open_position")
            and not row.get("partial_exit")
            for row in rows
        ),
        "complete_trade_win_rate": len(winners) / len(returns) if returns else None,
        "large_loss_rate": losses / len(returns) if returns else None,
        "mean_net_return": sum(returns) / len(returns) if returns else None,
        "average_winning_return": sum(winners) / len(winners) if winners else None,
        "average_losing_return": sum(losers) / len(losers) if losers else None,
        "payoff_ratio": ((sum(winners) / len(winners)) / abs(sum(losers) / len(losers))
                         if winners and losers else None),
        "profit_factor": sum(winners) / abs(sum(losers)) if losers else None,
        "maximum_consecutive_losses": maximum,
        "returns_used": len(returns),
        "account_max_drawdown": None,
        "drawdown_status": "REQUIRES_CHRONOLOGICAL_ACCOUNT_LEDGER",
    }


def discover_reports(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for supplied in paths:
        path = Path(supplied)
        if path.is_dir():
            found.extend(sorted(path.rglob("report.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    result = sorted({path.resolve() for path in found})
    if not result:
        raise ValueError("no broker exit reports supplied")
    return result


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("version") != lifecycle.VERSION:
        raise ValueError(f"unsupported broker exit report version: {path}")
    if not report.get("exit_asof") or not report.get("code"):
        raise ValueError(f"broker exit report lacks identity: {path}")
    return report


def aggregate(reports: list[dict[str, Any]], source_paths: list[Path], asof: str) -> dict[str, Any]:
    reports = sorted(reports, key=lambda row: (
        str(row.get("entry_trade_date") or ""),
        str(row.get("exit_asof") or ""),
        str(row.get("code") or ""),
    ))
    identities = [(str(row.get("entry_trade_date")), str(row.get("code")), str(row.get("plan_id")))
                  for row in reports]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate entry date/code/plan reports would inflate denominators")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in reports:
        groups.setdefault(str(row.get("board_group") or "UNSPECIFIED"), []).append(row)
    group_metrics = {group: _summary(rows) for group, rows in sorted(groups.items())}
    return {
        "version": VERSION,
        "status": "COMPLETE_TRADE_AGGREGATE_DIAGNOSTIC",
        "asof": asof,
        "unit_of_analysis": "one_unique_broker_exit_lifecycle_report",
        "all_reports": len(reports),
        "strategy": _summary(reports),
        "groups": group_metrics,
        "deployment_allowed": False,
        "limitation": "only rule-compliant completed reports enter strategy statistics; account drawdown needs a chronological ledger and open-position marks",
        "source_paths": [str(path.resolve()) for path in source_paths],
    }


def run(inputs: list[Path], asof: str, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_paths = discover_reports(inputs)
    reports = [_load(path) for path in source_paths]
    result = aggregate(reports, source_paths, asof)
    result["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (*source_paths, Path(__file__))
    }
    digest = hashlib.sha256(json.dumps(result["source_hashes"], sort_keys=True).encode()).hexdigest()[:20]
    target = Path(output) / digest
    if target.exists():
        raise FileExistsError("immutable trade aggregate already exists")
    target.mkdir(parents=True, exist_ok=False)
    report_path = target / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    result["report_path"] = str(report_path.resolve())
    result["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.reports, args.asof, args.output), ensure_ascii=False, indent=2))
