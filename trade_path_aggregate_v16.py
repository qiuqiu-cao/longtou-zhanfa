#!/usr/bin/env python3
"""Aggregate immutable B+1/B+2/B+3 path reports without inventing fills.

The reports consumed here are close-mark valuations.  They are deliberately
kept separate from completed-trade P&L: a positive close mark is not a sale,
and a normal close mark is not proof that the order could have filled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

VERSION = "v16-trade-path-aggregate-1"
REPORT_VERSION = "v16-trade-path-metrics-1"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "trade_path_aggregate"


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("version") != REPORT_VERSION:
        raise ValueError(f"unsupported path report version: {path}")
    if not report.get("code") or not report.get("entry_date_B"):
        raise ValueError(f"path report lacks trade identity: {path}")
    marks = report.get("marks")
    offsets = [row.get("session_offset") for row in marks] if isinstance(marks, list) else []
    if (not isinstance(marks, list) or len(marks) != 3
            or len(set(offsets)) != 3 or set(offsets) != {1, 2, 3}):
        raise ValueError(f"path report must contain exactly B+1/B+2/B+3 marks: {path}")
    return report


def discover_reports(paths: Iterable[Path]) -> list[Path]:
    """Resolve files/directories deterministically, rejecting duplicate paths."""
    found: list[Path] = []
    for supplied in paths:
        path = Path(supplied)
        if path.is_dir():
            found.extend(sorted(path.rglob("report.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(path)
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise ValueError("no trade-path reports supplied")
    return unique


def aggregate(reports: list[dict[str, Any]], source_paths: list[Path], asof: str) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one path report is required")
    identity = [(str(row["code"]), str(row["entry_date_B"])) for row in reports]
    if len(identity) != len(set(identity)):
        raise ValueError("duplicate code/entry-date path reports would inflate denominators")

    horizon_summary: dict[str, Any] = {}
    blocked_by_offset: dict[str, int] = {}
    for offset in (1, 2, 3):
        values: list[float] = []
        blocked = 0
        for report in reports:
            row = next(item for item in report["marks"] if item["session_offset"] == offset)
            value = _finite(row.get("net_close_return"))
            if value is not None:
                values.append(value)
            if row.get("sellability_state") not in (None, "NORMAL"):
                blocked += 1
        blocked_by_offset[str(offset)] = blocked
        horizon_summary[str(offset)] = {
            "observed": len(values),
            "unresolved": len(reports) - len(values),
            "mean_net_close_return": _mean(values),
            "median_net_close_return": _quantile(values, .5),
            "positive_fraction": (sum(value > 0 for value in values) / len(values)
                                   if values else None),
            "negative_fraction": (sum(value < 0 for value in values) / len(values)
                                   if values else None),
            "min_net_close_return": min(values) if values else None,
            "max_net_close_return": max(values) if values else None,
            "blocked_sellability_count": blocked,
            "blocked_sellability_fraction": blocked / len(reports),
        }

    joint = [report["B_plus_2_and_B_plus_3_both_positive"] for report in reports
             if isinstance(report.get("B_plus_2_and_B_plus_3_both_positive"), bool)]
    mfe = [_finite(report.get("maximum_favourable_excursion_net")) for report in reports]
    mae = [_finite(report.get("maximum_adverse_excursion_net")) for report in reports]
    mfe = [value for value in mfe if value is not None]
    mae = [value for value in mae if value is not None]
    eligible = sum(bool(report.get("strategy_path_metric_eligible")) for report in reports)
    group_counts: dict[str, dict[str, int]] = {}
    for report in reports:
        group = report.get("board_group") or "UNSPECIFIED"
        group_counts.setdefault(group, {"reports": 0, "complete_windows": 0, "strategy_eligible": 0})
        group_counts[group]["reports"] += 1
        group_counts[group]["complete_windows"] += int(bool(report.get("window_B1_B3_complete")))
        group_counts[group]["strategy_eligible"] += int(bool(report.get("strategy_path_metric_eligible")))

    return {
        "version": VERSION,
        "status": "PATH_VALUATION_ONLY",
        "asof": asof,
        "unit_of_analysis": "one_unique_code_entry_date_path_report",
        "reports": len(reports),
        "complete_B1_B3_windows": sum(bool(row.get("window_B1_B3_complete")) for row in reports),
        "incomplete_B1_B3_windows": sum(not bool(row.get("window_B1_B3_complete")) for row in reports),
        "strategy_path_metric_eligible": eligible,
        "horizons": horizon_summary,
        "B_plus_2_and_B_plus_3_joint": {
            "observed": len(joint),
            "unresolved": len(reports) - len(joint),
            "positive_fraction": sum(joint) / len(joint) if joint else None,
        },
        "excursion": {
            "observed_complete_windows": len(mfe),
            "mean_mfe_net": _mean(mfe),
            "median_mfe_net": _quantile(mfe, .5),
            "mean_mae_net": _mean(mae),
            "median_mae_net": _quantile(mae, .5),
            "worst_mae_net": min(mae) if mae else None,
        },
        "blocked_sellability_by_offset": blocked_by_offset,
        "groups": group_counts,
        "completed_trade_win_rate": None,
        "completed_trade_mean_net_return": None,
        "limitation": "close marks are valuation metrics; no actual exit fill or realized-trade rate is inferred",
        "source_paths": [str(path.resolve()) for path in source_paths],
    }


def run(report_inputs: list[Path], asof: str, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    source_paths = discover_reports(report_inputs)
    reports = [_load_report(path) for path in source_paths]
    result = aggregate(reports, source_paths, asof)
    result["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (*source_paths, Path(__file__))
    }
    digest = hashlib.sha256(json.dumps(result["source_hashes"], sort_keys=True).encode()).hexdigest()[:20]
    target = Path(output) / digest
    if target.exists():
        raise FileExistsError("immutable trade-path aggregate already exists")
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
