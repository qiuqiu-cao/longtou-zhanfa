"""Preflight and launch entry for the public auction/open monitor.

The command never places orders.  It binds the newest standardized close pool
and review to the immediately preceding archived session, reports which fixed
capture windows are still reachable, and only then starts the existing
tamper-evident Level-1/minute monitor.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import public_live_capture_v1 as live
import public_live_monitor_v1 as monitor
import absorption_forward_registry_v141 as absorption_registry


ROOT = Path(__file__).resolve().parent
SHANGHAI = monitor.SHANGHAI
VERSION = "v135-auction-monitor-entry-1"


def _timestamp(value: Any) -> pd.Timestamp:
    return monitor.local_timestamp(value)


def reachable_points(trade_date: str, now: Any) -> tuple[list[str], list[str]]:
    current = _timestamp(now)
    reachable, missed = [], []
    for point in live.SNAPSHOT_TIMES:
        deadline = monitor.target_timestamp(trade_date, point) + pd.Timedelta(
            seconds=monitor.late_deadline(point))
        (missed if current > deadline else reachable).append(point)
    return reachable, missed


def assess(candidate_path: Path, review_path: Path, trade_date: str,
           market_dates_path: Path, universe_path: Path | None, now: Any,
           absorption_registry_path: Path | None = None) -> dict:
    issues: list[str] = []
    candidates = live.load_candidates(candidate_path)
    if candidates.empty:
        issues.append("EMPTY_CLOSING_LIMIT_POOL")
    signal_dates = sorted(candidates.signal_date.dropna().astype(str).unique())
    candidate_signal_date = signal_dates[0] if len(signal_dates) == 1 else None
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review_signal_date = pd.Timestamp(review.get("signal_date")).strftime("%Y-%m-%d")
    except Exception:
        review, review_signal_date = {}, None
        issues.append("REVIEW_UNREADABLE")
    if candidate_signal_date != review_signal_date:
        issues.append("CANDIDATE_REVIEW_DATE_MISMATCH")
    try:
        expected_prior = monitor.validate_prior_session_binding(
            candidate_path, review_path, trade_date, market_dates_path)
    except Exception as exc:
        expected_prior = None
        issues.append("PRIOR_SESSION_BINDING_FAILED: " + str(exc))

    current = _timestamp(now)
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    reachable, missed = reachable_points(trade, current)
    if current.date() != pd.Timestamp(trade).date():
        issues.append("TRADE_DATE_IS_NOT_LOCAL_TODAY")
    if pd.Timestamp(trade).dayofweek >= 5:
        issues.append("TRADE_DATE_IS_WEEKEND")
    if not reachable:
        issues.append("ALL_REGISTERED_WINDOWS_MISSED")
    universe_status = (
        "AVAILABLE" if universe_path is not None and universe_path.exists()
        else "UNAVAILABLE_BREADTH_WILL_BE_INCOMPLETE")
    if absorption_registry_path is None:
        absorption_status = "DISABLED"
    else:
        absorption_check = absorption_registry.verify(Path(absorption_registry_path))
        absorption_status = absorption_check["status"]
        if absorption_status != "PASS":
            issues.append("ABSORPTION_FORWARD_REGISTRY_INVALID: " + "; ".join(absorption_check["errors"]))

    first_count = int(candidates.board_group.eq("FIRST_BOARD").sum()) if "board_group" in candidates else None
    relay_count = int(candidates.board_group.eq("CONSECUTIVE_BOARD").sum()) if "board_group" in candidates else None
    roles = review.get("roles", {}) if isinstance(review, dict) else {}
    anchors = [
        {key: item.get(key) for key in ("code", "name", "boards")}
        for item in roles.get("emotion_anchors", []) if isinstance(item, dict)
    ]
    potential = roles.get("highest_leader_potential")
    challenger = roles.get("next_height_challenger")
    ready = not issues
    if ready and not missed:
        readiness = "READY_FULL_SEQUENCE_PUBLIC_ONLY"
    elif ready:
        readiness = "READY_DEGRADED_REMAINING_WINDOWS_PUBLIC_ONLY"
    else:
        readiness = "BLOCKED"
    command = (
        f'python "{(ROOT / "auction_monitor_entry_v135.py").resolve()}" start '
        f'--trade-date {trade}'
    )
    return {
        "version": VERSION,
        "status": readiness,
        "checked_at": current.isoformat(),
        "trade_date": trade,
        "candidate_path": str(candidate_path.resolve()),
        "review_path": str(review_path.resolve()),
        "market_dates_path": str(market_dates_path.resolve()),
        "expected_prior_session": expected_prior,
        "candidate_signal_date": candidate_signal_date,
        "review_signal_date": review_signal_date,
        "candidate_count": int(len(candidates)),
        "first_board_count": first_count,
        "relay_count": relay_count,
        "emotion_anchors": anchors,
        "highest_leader_potential": potential,
        "next_height_challenger": challenger,
        "reachable_capture_points": reachable,
        "missed_capture_points": missed,
        "universe_status": universe_status,
        "absorption_forward_registry_status": absorption_status,
        "issues": issues,
        "launch_command": command,
        "data_scope": {
            "public_level1_price": "CONDITIONAL_WHEN_FRESH",
            "public_1min_vwap_reseal": "INFERRED_WHEN_09:35_OR_10:00_CAPTURED",
            "level2_queue": "UNAVAILABLE",
            "auction_cancel_sequence": "UNAVAILABLE",
            "actual_fill": "UNAVAILABLE_UNTIL_BROKER_IMPORT",
        },
        "direct_entry_authorized": False,
        "automatic_order_placement": False,
    }


def render(result: dict) -> str:
    anchors = "、".join(
        f"{item.get('name')}({item.get('code')}) {item.get('boards')}板"
        for item in result["emotion_anchors"]
    ) or "无"
    potential = result.get("highest_leader_potential") or {}
    challenger = result.get("next_height_challenger") or {}
    lines = [
        "# 竞价监控启动前检查", "",
        f"- 状态：`{result['status']}`",
        f"- 检查时间：{result['checked_at']}",
        f"- 交易日 / 绑定盘后日：{result['trade_date']} / {result['expected_prior_session']}",
        f"- 候选：{result['candidate_count']}只；首板{result['first_board_count']}只；连板{result['relay_count']}只",
        f"- 当前情绪锚：{anchors}",
        f"- 最可能成为/保持最高标：{potential.get('name', '无')}({potential.get('code', '-')})",
        f"- 当前锚失败后的挑战者：{challenger.get('name', '无')}({challenger.get('code', '-')})",
        f"- 尚可采集：{'、'.join(result['reachable_capture_points']) or '无'}",
        f"- 已错过：{'、'.join(result['missed_capture_points']) or '无'}",
        f"- 市场广度名册：{result['universe_status']}", "",
        f"- 10:00影子登记簿：{result['absorption_forward_registry_status']}", "",
    ]
    if result["issues"]:
        lines += ["## 阻断原因", "", *[f"- {item}" for item in result["issues"]], ""]
    lines += [
        "## 数据边界", "",
        "- 公开Level-1价格只有在提供方时间戳新鲜时才是条件证据。",
        "- 09:35/10:00的VWAP与回封来自公开1分钟K线推断。",
        "- Level-2队列、真实撤单顺序和实际成交不可获得；监控不自动下单。",
        "- 错过的固定时点永久标记为已错过，不使用后续数据回填。", "",
        "启动命令：", "", f"`{result['launch_command']}`", "",
    ]
    return "\n".join(lines)


def discover_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    return (args.candidates or live.discover_candidates(),
            args.review or monitor.discover_review())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("preflight", "start"):
        command = sub.add_parser(name)
        command.add_argument("--trade-date", default=datetime.now(SHANGHAI).strftime("%Y-%m-%d"))
        command.add_argument("--candidates", type=Path)
        command.add_argument("--review", type=Path)
        command.add_argument("--market-dates", type=Path, default=monitor.DEFAULT_MARKET_DATES)
        command.add_argument("--universe", type=Path, default=live.DEFAULT_UNIVERSE)
        command.add_argument("--no-breadth", action="store_true")
        command.add_argument("--no-absorption-forward", action="store_true")
        command.add_argument("--output", type=Path, default=ROOT / "results" / "v135" / "auction_preflight")
    args = parser.parse_args()
    candidate_path, review_path = discover_inputs(args)
    universe = None if args.no_breadth else args.universe
    forward_registry = None if args.no_absorption_forward else absorption_registry.DEFAULT_REGISTRY
    check = assess(candidate_path, review_path, args.trade_date, args.market_dates,
                   universe, datetime.now(SHANGHAI), forward_registry)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(check["checked_at"]).strftime("%Y%m%d_%H%M%S")
    prefix = args.output / f"{check['trade_date']}_{stamp}"
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render(check), encoding="utf-8")
    if args.mode == "preflight":
        print(json.dumps({**check, "report_path": str(markdown_path.resolve())},
                         ensure_ascii=False, indent=2))
        return
    if check["status"] == "BLOCKED":
        raise SystemExit("preflight blocked; inspect " + str(markdown_path.resolve()))
    result = monitor.run_monitor(
        candidate_path, args.trade_date, monitor.DEFAULT_OUTPUT, universe,
        review_path=review_path, market_dates_path=args.market_dates,
        absorption_registry_root=forward_registry,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
