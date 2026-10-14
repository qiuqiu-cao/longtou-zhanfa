#!/usr/bin/env python3
"""Run the public quote capture automatically at the six registered times.

Starting this process before the auction removes the need to send screenshots
or manually invoke six commands.  The process records evidence only; it never
submits an order and cannot turn public Level-1 data into an ENTER decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

import public_live_capture_v1 as live
import public_minute_capture_v1 as minute
import live_decision_bridge_v1 as bridge
import absorption_forward_registry_v141 as absorption_registry


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v16" / "public_live_monitor"
DEFAULT_MARKET_DATES = ROOT / "data" / "v3" / "market_emotion.csv"
SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v16-public-live-monitor-5"
BREADTH_POINTS = {"09:25", "09:35", "10:00"}
DECISION_POINTS = {"09:25", "09:35", "10:00"}


def local_timestamp(value: Any) -> pd.Timestamp:
    return live.local_timestamp(value)


def target_timestamp(trade_date: str, point: str) -> pd.Timestamp:
    suffix = ":00" if len(point) == 5 else ""
    return local_timestamp(f"{pd.Timestamp(trade_date).strftime('%Y-%m-%d')} {point}{suffix}")


def hash_event(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_event(path: Path, event: dict[str, Any], previous_hash: str) -> str:
    chained = {**event, "previous_event_hash": previous_hash}
    chained["event_hash"] = hash_event(chained)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(chained, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return chained["event_hash"]


def file_hash_if_exists(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    return bridge.sha256_file(path) if path.exists() and path.is_file() else None


def verify_run(manifest_path: Path) -> dict[str, Any]:
    """Verify one immutable monitor run and all files bound by its events."""
    manifest_path = Path(manifest_path)
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "errors": [f"manifest unreadable: {exc}"]}
    if manifest.get("version") != VERSION:
        errors.append("monitor version is not the current verifiable version")
    ledger = Path(str(manifest.get("ledger_path", "")))
    if not ledger.exists():
        ledger = manifest_path.parent / "events.jsonl"
    if not ledger.exists():
        return {"status": "FAIL", "errors": errors + ["event ledger missing"]}
    if manifest.get("ledger_sha256") != bridge.sha256_file(ledger):
        errors.append("ledger file hash mismatch")
    try:
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        return {"status": "FAIL", "errors": errors + [f"event ledger invalid JSON: {exc}"]}
    previous = "GENESIS"
    completed = failed = skipped = decisions = decision_failures = 0
    bound_fields = (
        ("manifest_path", "quote_manifest_sha256"),
        ("minute_manifest_path", "minute_manifest_sha256"),
        ("decision_path", "decision_sha256"),
        ("decision_report_path", "decision_report_sha256"),
    )
    for number, event in enumerate(events, 1):
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_event_hash") != previous:
            errors.append(f"event {number}: previous hash mismatch")
        if event.get("event_hash") != hash_event(body):
            errors.append(f"event {number}: event hash mismatch")
        previous = str(event.get("event_hash", ""))
        completed += event.get("event_type") == "CAPTURE_COMPLETED"
        failed += event.get("event_type") == "CAPTURE_FAILED"
        skipped += event.get("event_type") == "CAPTURE_SKIPPED"
        decisions += bool(event.get("decision_path"))
        decision_failures += bool(event.get("decision_error"))
        for path_field, hash_field in bound_fields:
            value = event.get(path_field)
            expected = event.get(hash_field)
            if value:
                path = Path(str(value))
                if not path.exists():
                    errors.append(f"event {number}: {path_field} missing")
                elif not expected or bridge.sha256_file(path) != expected:
                    errors.append(f"event {number}: {path_field} hash mismatch")
    if manifest.get("last_event_hash") != previous:
        errors.append("manifest head event hash mismatch")
    for field, observed in (
        ("completed_captures", completed), ("failed_captures", failed),
        ("skipped_captures", skipped), ("completed_decisions", decisions),
        ("failed_decisions", decision_failures),
    ):
        if manifest.get(field) != observed:
            errors.append(f"manifest {field} mismatch")
    for path_field, hash_field in (
        ("candidate_path", "candidate_sha256"), ("review_path", "review_sha256"),
        ("market_dates_path", "market_dates_sha256"),
        ("universe_path", "universe_sha256"),
    ):
        value = manifest.get(path_field)
        if value:
            path = Path(str(value))
            if not path.exists() or manifest.get(hash_field) != bridge.sha256_file(path):
                errors.append(f"manifest {path_field} hash mismatch")
    if manifest.get("direct_entry_authorized") is not False:
        errors.append("public monitor must not claim direct entry authority")
    return {
        "status": "PASS" if not errors else "FAIL", "version": VERSION,
        "run_id": manifest.get("run_id"), "trade_date": manifest.get("trade_date"),
        "events": len(events), "completed_captures": completed,
        "completed_decisions": decisions, "head_event_hash": previous,
        "errors": errors, "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def late_deadline(point: str) -> float:
    if point == "09:24:50":
        return 20.0
    if point in {"09:20", "09:25"}:
        return 45.0
    return 75.0


def discover_review(root: Path = ROOT) -> Path:
    """Return the newest parseable close review by signal date, then path.

    Discovery is deliberately limited to the standardized daily-review output.
    The decision bridge later verifies that its exact candidate set matches the
    quote/minute capture, so discovery cannot silently mix two different pools.
    """
    paths = list((root / "results").glob("daily_review_*/review.json"))
    dated: list[tuple[str, Path]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            signal_date = pd.Timestamp(payload.get("signal_date")).strftime("%Y-%m-%d")
            dated.append((signal_date, path))
        except Exception:
            continue
    if not dated:
        raise FileNotFoundError("no standardized daily review found; pass --review or use --capture-only")
    return max(dated, key=lambda item: (item[0], str(item[1])))[1]


def validate_prior_session_binding(
    candidate_path: Path, review_path: Path, trade_date: str, market_dates_path: Path,
) -> str:
    """Require both inputs to represent the latest archived session before trade_date."""
    candidates = live.load_candidates(candidate_path)
    candidate_date = str(candidates["signal_date"].iloc[0])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_date = pd.Timestamp(review.get("signal_date")).strftime("%Y-%m-%d")
    if candidate_date != review_date:
        raise RuntimeError("candidate pool and close review signal dates differ")
    if not market_dates_path.exists():
        raise FileNotFoundError("market-date archive is missing")
    dates = pd.to_datetime(pd.read_csv(market_dates_path, usecols=["date"])["date"], errors="coerce").dropna()
    prior = dates[dates < pd.Timestamp(trade_date)]
    if prior.empty:
        raise RuntimeError("market-date archive has no session before trade_date")
    expected = prior.max().strftime("%Y-%m-%d")
    if candidate_date != expected:
        raise RuntimeError(
            f"stale close inputs: expected prior archived session {expected}, got {candidate_date}"
        )
    return expected


def run_monitor(
    candidate_path: Path, trade_date: str, output: Path = DEFAULT_OUTPUT,
    universe_path: Path | None = live.DEFAULT_UNIVERSE, *, review_path: Path | None = None,
    market_dates_path: Path | None = None,
    clock: Callable[[], Any] = lambda: datetime.now(SHANGHAI),
    sleeper: Callable[[float], None] = time.sleep,
    capture_runner: Callable[..., dict[str, Any]] = live.capture,
    minute_runner: Callable[..., dict[str, Any]] = minute.capture,
    decision_runner: Callable[..., dict[str, Any]] = bridge.run,
    absorption_registry_root: Path | None = None,
    absorption_snapshot_runner: Callable[..., dict[str, Any]] = absorption_registry.snapshot_decision,
) -> dict[str, Any]:
    candidate_path = Path(candidate_path)
    if review_path is not None:
        review_path = Path(review_path)
        if not review_path.exists():
            raise FileNotFoundError("daily review file does not exist")
        if market_dates_path is not None:
            validate_prior_session_binding(candidate_path, review_path, trade_date, Path(market_dates_path))
    started = local_timestamp(clock())
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    run_id = hash_event({
        "version": VERSION, "trade_date": trade, "started_at": started.isoformat(),
        "candidate_path": str(candidate_path.resolve()),
        "review_path": str(review_path.resolve()) if review_path else None,
        "review_sha256": bridge.sha256_file(review_path) if review_path else None,
        "market_dates_path": str(Path(market_dates_path).resolve()) if market_dates_path else None,
        "market_dates_sha256": bridge.sha256_file(Path(market_dates_path)) if market_dates_path else None,
    })[:20]
    target = output / trade / run_id
    if target.exists():
        raise FileExistsError("immutable live monitor run already exists")
    target.mkdir(parents=True, exist_ok=False)
    # Bind every mutable input to an immutable per-run copy.  In particular,
    # market_emotion.csv is appended each day; verifying an old run against
    # its later contents would falsely invalidate authentic forward evidence.
    inputs = target / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)

    def snapshot_input(path: Path | None, filename: str) -> Path | None:
        if path is None:
            return None
        source = Path(path)
        if not source.exists():
            return source
        destination = inputs / filename
        shutil.copy2(source, destination)
        return destination

    source_inputs = {
        "candidate": {"path": str(candidate_path.resolve()), "sha256": bridge.sha256_file(candidate_path)},
        "review": ({"path": str(review_path.resolve()), "sha256": bridge.sha256_file(review_path)}
                   if review_path else None),
        "market_dates": ({"path": str(Path(market_dates_path).resolve()),
                          "sha256": bridge.sha256_file(Path(market_dates_path))}
                         if market_dates_path else None),
        "universe": ({"path": str(Path(universe_path).resolve()),
                      "sha256": bridge.sha256_file(Path(universe_path))}
                     if universe_path is not None and Path(universe_path).exists() else None),
    }
    bound_candidate = snapshot_input(candidate_path, "candidates.csv")
    bound_review = snapshot_input(review_path, "review.json")
    bound_market_dates = snapshot_input(Path(market_dates_path) if market_dates_path else None,
                                        "market_dates.csv")
    bound_universe = snapshot_input(Path(universe_path) if universe_path is not None else None,
                                    "universe.csv")
    ledger = target / "events.jsonl"
    previous_hash = "GENESIS"
    events: list[dict[str, Any]] = []

    if started.date() != pd.Timestamp(trade).date() or started.dayofweek >= 5:
        event = {
            "event_type": "MONITOR_ABORTED", "observed_at": started.isoformat(),
            "status": "NOT_CURRENT_TRADING_DATE_OR_WEEKEND",
        }
        previous_hash = append_event(ledger, event, previous_hash)
        events.append(event)
    else:
        for point in live.SNAPSHOT_TIMES:
            target_at = target_timestamp(trade, point)
            current = local_timestamp(clock())
            if current > target_at + pd.Timedelta(seconds=late_deadline(point)):
                event = {
                    "event_type": "CAPTURE_SKIPPED", "asof": point,
                    "observed_at": current.isoformat(), "status": "MISSED_REGISTERED_WINDOW",
                }
                previous_hash = append_event(ledger, event, previous_hash)
                events.append(event)
                continue
            while current < target_at:
                sleeper(min(10.0, max(0.0, (target_at - current).total_seconds())))
                current = local_timestamp(clock())
            breadth = universe_path if point in BREADTH_POINTS else None
            quote_result: dict[str, Any] | None = None
            minute_result: dict[str, Any] | None = None
            quote_error: str | None = None
            minute_error: str | None = None
            decision_error: str | None = None
            decision_result: dict[str, Any] | None = None
            absorption_forward_result: dict[str, Any] | None = None
            absorption_forward_error: str | None = None
            try:
                quote_result = capture_runner(bound_candidate, trade, point, live.DEFAULT_OUTPUT,
                                              bound_universe if breadth is not None else None)
            except Exception as exc:
                quote_error = str(exc)[:500]
            if point in minute.MINUTE_POINTS:
                try:
                    minute_result = minute_runner(bound_candidate, trade, point, minute.DEFAULT_OUTPUT)
                except Exception as exc:
                    minute_error = str(exc)[:500]
            if (bound_review is not None and point in DECISION_POINTS
                    and quote_result is not None and quote_error is None and minute_error is None):
                try:
                    quote_manifest = Path(str(quote_result["manifest_path"]))
                    minute_manifest = (
                        Path(str(minute_result["manifest_path"])) if minute_result is not None else None
                    )
                    decision_output = target / "decisions" / point.replace(":", "")
                    decision_result = decision_runner(
                        bound_review, quote_manifest, decision_output, minute_manifest,
                    )
                except Exception as exc:
                    decision_error = str(exc)[:500]
            if (point == "10:00" and absorption_registry_root is not None
                    and decision_result is not None and decision_result.get("decision_path")):
                try:
                    absorption_forward_result = absorption_snapshot_runner(
                        Path(absorption_registry_root), Path(str(decision_result["decision_path"])))
                except Exception as exc:
                    absorption_forward_error = str(exc)[:500]
            if quote_error or minute_error or decision_error:
                event = {
                    "event_type": "CAPTURE_FAILED", "asof": point,
                    "observed_at": local_timestamp(clock()).isoformat(),
                    "status": "CASH_CAPTURE_FAILED", "quote_error": quote_error,
                    "minute_error": minute_error,
                    "decision_error": decision_error,
                    "quote_status": quote_result.get("status") if quote_result else None,
                    "minute_status": minute_result.get("status") if minute_result else None,
                    "decision_status": decision_result.get("status") if decision_result else None,
                    "absorption_forward_error": absorption_forward_error,
                    "absorption_forward_event_hash": (absorption_forward_result.get("event_hash")
                                                       if absorption_forward_result else None),
                    "direct_entry_authorized": False,
                }
            else:
                event = {
                    "event_type": "CAPTURE_COMPLETED", "asof": point,
                    "observed_at": local_timestamp(clock()).isoformat(),
                    "status": quote_result.get("status") if quote_result else "CASH_QUOTE_CAPTURE_MISSING",
                    "capture_id": quote_result.get("capture_id") if quote_result else None,
                    "manifest_path": quote_result.get("manifest_path") if quote_result else None,
                    "quote_manifest_sha256": file_hash_if_exists(
                        quote_result.get("manifest_path") if quote_result else None
                    ),
                    "minute_status": minute_result.get("status") if minute_result else None,
                    "minute_capture_id": minute_result.get("capture_id") if minute_result else None,
                    "minute_manifest_path": minute_result.get("manifest_path") if minute_result else None,
                    "minute_manifest_sha256": file_hash_if_exists(
                        minute_result.get("manifest_path") if minute_result else None
                    ),
                    "decision_status": decision_result.get("status") if decision_result else None,
                    "decision_path": decision_result.get("decision_path") if decision_result else None,
                    "decision_report_path": decision_result.get("report_path") if decision_result else None,
                    "decision_sha256": file_hash_if_exists(
                        decision_result.get("decision_path") if decision_result else None
                    ),
                    "decision_report_sha256": file_hash_if_exists(
                        decision_result.get("report_path") if decision_result else None
                    ),
                    "absorption_forward_error": absorption_forward_error,
                    "absorption_forward_event_hash": (absorption_forward_result.get("event_hash")
                                                       if absorption_forward_result else None),
                    "absorption_forward_snapshot": (absorption_forward_result.get("snapshot")
                                                    if absorption_forward_result else None),
                    "direct_entry_authorized": False,
                }
            previous_hash = append_event(ledger, event, previous_hash)
            events.append(event)

    completed = sum(event["event_type"] == "CAPTURE_COMPLETED" for event in events)
    failed = sum(event["event_type"] == "CAPTURE_FAILED" for event in events)
    skipped = sum(event["event_type"] == "CAPTURE_SKIPPED" for event in events)
    completed_decisions = sum(bool(event.get("decision_path")) for event in events)
    failed_decisions = sum(bool(event.get("decision_error")) for event in events)
    absorption_forward_snapshots = sum(bool(event.get("absorption_forward_event_hash")) for event in events)
    absorption_forward_failures = sum(bool(event.get("absorption_forward_error")) for event in events)
    expected_decisions = len(DECISION_POINTS) if review_path is not None else 0
    status = (
        "CAPTURE_AND_DECISION_SEQUENCE_COMPLETE_PUBLIC_ONLY"
        if (completed == len(live.SNAPSHOT_TIMES) and failed == 0
            and (review_path is None or completed_decisions == expected_decisions)) else
        "CAPTURE_SEQUENCE_INCOMPLETE"
    )
    if status == "CAPTURE_AND_DECISION_SEQUENCE_COMPLETE_PUBLIC_ONLY" and absorption_forward_failures:
        status = "CAPTURE_SEQUENCE_COMPLETE_FORWARD_RECORD_FAILED"
    manifest = {
        "version": VERSION, "run_id": run_id, "trade_date": trade,
        "started_at": started.isoformat(), "finished_at": local_timestamp(clock()).isoformat(),
        "candidate_path": str(bound_candidate.resolve()), "status": status,
        "candidate_sha256": bridge.sha256_file(bound_candidate),
        "review_path": str(bound_review.resolve()) if bound_review else None,
        "review_sha256": bridge.sha256_file(bound_review) if bound_review else None,
        "market_dates_path": str(bound_market_dates.resolve()) if bound_market_dates else None,
        "market_dates_sha256": bridge.sha256_file(bound_market_dates) if bound_market_dates else None,
        "universe_path": (str(bound_universe.resolve()) if bound_universe and bound_universe.exists() else None),
        "universe_sha256": (bridge.sha256_file(bound_universe)
                            if bound_universe and bound_universe.exists() else None),
        "source_inputs": source_inputs,
        "completed_captures": completed, "failed_captures": failed, "skipped_captures": skipped,
        "completed_decisions": completed_decisions, "failed_decisions": failed_decisions,
        "expected_decisions": expected_decisions,
        "absorption_forward_registry": (str(Path(absorption_registry_root).resolve())
                                        if absorption_registry_root is not None else None),
        "absorption_forward_snapshots": absorption_forward_snapshots,
        "absorption_forward_failures": absorption_forward_failures,
        "registered_points": list(live.SNAPSHOT_TIMES), "breadth_points": sorted(BREADTH_POINTS),
        "minute_points": list(minute.MINUTE_POINTS),
        "decision_points": sorted(DECISION_POINTS),
        "direct_entry_authorized": False, "last_event_hash": previous_hash,
        "ledger_path": str(ledger.resolve()),
        "ledger_sha256": bridge.sha256_file(ledger),
        "warning": "monitoring public Level-1 quotes does not verify Level-2 queues, auction cancels or fills",
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture public quote evidence at all six registered morning points")
    parser.add_argument("--trade-date", default=datetime.now(SHANGHAI).strftime("%Y-%m-%d"))
    parser.add_argument("--candidates", type=Path, help="auto-discovers latest standardized full pool when omitted")
    parser.add_argument("--review", type=Path, help="auto-discovers latest standardized close review when omitted")
    parser.add_argument("--capture-only", action="store_true", help="capture evidence without building decision reports")
    parser.add_argument("--universe", type=Path, default=live.DEFAULT_UNIVERSE)
    parser.add_argument("--no-breadth", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--market-dates", type=Path, default=DEFAULT_MARKET_DATES,
                        help="archived market dates used to prove the immediately prior session")
    parser.add_argument("--verify-manifest", type=Path, help="verify one completed monitor run and exit")
    parser.add_argument("--no-absorption-forward", action="store_true",
                        help="disable the frozen research-only 10:00 shadow registry")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_manifest:
        result = verify_run(args.verify_manifest)
    else:
        candidates = args.candidates or live.discover_candidates()
        review = None if args.capture_only else (args.review or discover_review())
        universe = None if args.no_breadth else args.universe
        result = run_monitor(
            candidates, args.trade_date, args.output, universe, review_path=review,
            market_dates_path=args.market_dates,
            absorption_registry_root=(None if args.no_absorption_forward else
                                      absorption_registry.DEFAULT_REGISTRY),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
