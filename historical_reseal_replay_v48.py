#!/usr/bin/env python3
"""Causal five-minute replay for pullback/reseal entries and B+1..B+3 marks.

This is deliberately a diagnostic execution proxy.  A completed five-minute
bar can create a signal; only the following bar may supply an entry-opportunity
label.  Neither a bar nor traded volume proves queue priority or an actual fill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd

from audit_historical_5min_v33 import inspect
from t1_scenario_v44 import net_return


VERSION = "v48-causal-reseal-replay-6"
ASOF_CLOCK = "100000000"
EPSILON = 1e-6
COSTS = dict(shares=100, commission=0.0003, minimum=5.0, sell_tax=0.0005, slippage=0.001)
SELECTION_BASELINE = "EARLIEST_CONFIRMATION_THEN_HEIGHT_THEN_CODE"
CUTOFF_SELECTION_BASELINE = "STILL_SEALED_AT_1000_THEN_HEIGHT_THEN_CODE"
RECONCILED_DAILY_STATUSES = {"DAILY_OHLC_MATCH_ONLY", "DAILY_OHLC_RANGE_CONSISTENT"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconciliation_usable(status: str) -> bool:
    return status in RECONCILED_DAILY_STATUSES


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite number")
    return result


def _valid_bar(raw: dict[str, Any]) -> dict[str, Any]:
    bar = dict(raw)
    values = {key: _finite(bar[key]) for key in ("open", "high", "low", "close", "volume")}
    if (min(values[key] for key in ("open", "high", "low", "close")) <= 0
            or values["volume"] < 0
            or values["low"] > min(values["open"], values["close"])
            or values["high"] < max(values["open"], values["close"])
            or values["low"] > values["high"]):
        raise ValueError("invalid OHLCV bar")
    bar.update(values)
    bar["time"] = str(bar["time"])
    return bar


def _all_zero_no_trade_archive(raw: dict[str, Any]) -> bool:
    """Recognize a complete provider session of zero bars as non-tradable, not a price bar."""
    try:
        frame = pd.DataFrame(raw["rows"], columns=raw["fields"])
        day = str(raw["requested_date"])
        expected = [day.replace("-", "") + t.strftime("%H%M%S") + "000"
                    for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                    for t in pd.date_range(day + " " + start, day + " " + end, freq="5min")]
        if (len(frame) != 48 or raw.get("error_code") != "0"
                or not frame["code"].astype(str).eq(str(raw["code"])).all()
                or not frame["date"].astype(str).eq(day).all()
                or not frame["adjustflag"].astype(str).eq("3").all()
                or frame["time"].astype(str).tolist() != expected):
            return False
        numeric = frame[["open", "high", "low", "close", "volume", "amount"]].apply(
            pd.to_numeric, errors="coerce")
        if (not numeric.notna().all().all() or not numeric[["volume", "amount"]].eq(0).all().all()
                or not numeric[["open", "high", "low", "close"]].ge(0).all().all()):
            return False
        # BaoStock may carry the prior close in the final 15:00 bar on a
        # no-trade day.  It is still non-tradable when volume and amount are
        # zero throughout, all earlier prices are zero, and that final OHLC is flat.
        prices = numeric[["open", "high", "low", "close"]]
        nonzero = prices.ne(0).any(axis=1)
        if not nonzero.any():
            return True
        indices = list(frame.index[nonzero])
        return bool(indices == [frame.index[-1]] and prices.loc[indices[0]].nunique() == 1)
    except (KeyError, TypeError, ValueError):
        return False


def causal_reseal_signal(bars: list[dict[str, Any]], entry_date: str,
                         limit_price: float, asof_clock: str = ASOF_CLOCK) -> dict[str, Any]:
    """Return the first reseal observable by *asof_clock*, never using later bars."""
    limit_price = _finite(limit_price)
    if limit_price <= 0:
        raise ValueError("positive limit price required")
    cutoff = entry_date.replace("-", "") + asof_clock
    prefix = [_valid_bar(row) for row in bars if str(row["time"]) <= cutoff]
    times = [row["time"] for row in prefix]
    if times != sorted(set(times)):
        raise ValueError("bars must be ordered and unique")
    prior_below = False
    prior_touch_then_below = False
    for index, bar in enumerate(prefix):
        at_limit = abs(bar["close"] - limit_price) <= EPSILON
        if at_limit and prior_below:
            signal_type = "TOUCHED_PULLBACK_RESEAL" if prior_touch_then_below else "FIRST_SEAL_FROM_BELOW"
            return {
                "signal": True,
                "signal_type": signal_type,
                "confirmation_time": bar["time"],
                "confirmation_index": index,
                "observed_bars": index + 1,
                "decision": "SUBMIT_LIMIT_BUY_PROXY_NEXT_BAR",
                "entry_price_cap": limit_price,
                "asof": cutoff,
            }
        if bar["close"] < limit_price - EPSILON:
            prior_below = True
            if bar["high"] >= limit_price - EPSILON:
                prior_touch_then_below = True
    return {
        "signal": False,
        "signal_type": "NO_RESEAL_BY_CUTOFF",
        "confirmation_time": None,
        "confirmation_index": None,
        "observed_bars": len(prefix),
        "decision": "NO_ORDER",
        "entry_price_cap": limit_price,
        "asof": cutoff,
    }


def next_bar_entry_label(bars: list[dict[str, Any]], signal: dict[str, Any]) -> dict[str, Any]:
    """Outcome-only label for the bar after a completed-bar signal."""
    base = {"label_only": True, "actual_fill_verified": False, "simulated_fill": False}
    if not signal.get("signal"):
        return {**base, "state": "NO_SIGNAL"}
    validated = [_valid_bar(row) for row in bars]
    times = [row["time"] for row in validated]
    if times != sorted(set(times)):
        raise ValueError("bars must be ordered and unique")
    index = int(signal["confirmation_index"]) + 1
    if index >= len(validated):
        return {**base, "state": "NO_NEXT_BAR"}
    bar = validated[index]
    limit_price = float(signal["entry_price_cap"])
    if bar["volume"] <= 0:
        state = "NO_RECORDED_VOLUME"
    elif bar["low"] < limit_price - EPSILON:
        state = "BELOW_LIMIT_ENTRY_OPPORTUNITY_PROXY"
    elif all(abs(bar[key] - limit_price) <= EPSILON for key in ("open", "high", "low", "close")):
        state = "LIMIT_QUEUE_UNRESOLVED"
    else:
        state = "TOUCH_ONLY_QUEUE_UNRESOLVED"
    return {**base, "state": state, "next_bar": bar["time"],
            "next_bar_open": bar["open"], "next_bar_low": bar["low"],
            "next_bar_volume": bar["volume"]}


def stable_cutoff_signal(bars: list[dict[str, Any]], entry_date: str,
                         limit_price: float, asof_clock: str = ASOF_CLOCK) -> dict[str, Any]:
    """Require a causal reseal and a still-sealed completed bar at the cutoff."""
    first = causal_reseal_signal(bars, entry_date, limit_price, asof_clock)
    cutoff = entry_date.replace("-", "") + asof_clock
    prefix = [_valid_bar(row) for row in bars if str(row["time"]) <= cutoff]
    if (not first["signal"] or not prefix or prefix[-1]["time"] != cutoff
            or abs(prefix[-1]["close"] - float(limit_price)) > EPSILON):
        return {
            "signal": False,
            "signal_type": "NOT_STILL_SEALED_AT_CUTOFF",
            "confirmation_time": None,
            "confirmation_index": None,
            "observed_bars": len(prefix),
            "decision": "NO_ORDER",
            "entry_price_cap": float(limit_price),
            "asof": cutoff,
            "first_confirmation_time": first.get("confirmation_time"),
        }
    return {
        **first,
        "confirmation_time": cutoff,
        "confirmation_index": len(prefix) - 1,
        "observed_bars": len(prefix),
        "decision": "SUBMIT_LIMIT_BUY_PROXY_AFTER_STABLE_CUTOFF",
        "first_confirmation_time": first["confirmation_time"],
    }


def choose_one_per_day(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen arrival baseline.  It reads no entry or holding-period outcomes."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signals:
        if row.get("candidate_scope_eligible") and row["signal"]["signal"]:
            grouped[row["entry_date"]].append(row)
    selected = []
    for day, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (
            row["signal"]["confirmation_time"],
            -int(row.get("boards_for_comparison") or row.get("boards") or 0),
            str(row["code"]),
        ))
        winner = dict(ordered[0])
        winner["selection_baseline"] = SELECTION_BASELINE
        winner["same_day_signal_count"] = len(rows)
        selected.append(winner)
    return selected


def choose_one_at_cutoff(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in signals:
        candidate = dict(row)
        candidate["signal"] = row["cutoff_signal_1000"]
        prepared.append(candidate)
    selected = choose_one_per_day(prepared)
    for row in selected:
        row["selection_baseline"] = CUTOFF_SELECTION_BASELINE
    return selected


def select_with_coverage(signals: list[dict[str, Any]], missing: list[dict[str, Any]],
                         allow_incomplete_development: bool = False
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Fail closed when any frozen candidate lacks decision-time evidence."""
    if missing and not allow_incomplete_development:
        return [], [], "BLOCKED_INCOMPLETE_CANDIDATE_UNIVERSE"
    status = "INCOMPLETE_DEVELOPMENT_OVERRIDE" if missing else "COMPLETE_CANDIDATE_UNIVERSE"
    return choose_one_per_day(signals), choose_one_at_cutoff(signals), status


def board_group(row: dict[str, Any]) -> str:
    start_known = row.get("start_known")
    if not (start_known is True or str(start_known).strip().lower() == "true"):
        return "UNKNOWN_LEFT_CENSORED"
    boards = int(row["boards"])
    return "FIRST_BOARD" if boards == 1 else "CONSECUTIVE_BOARD" if boards >= 2 else "INVALID"


def _calendar_successors(sessions: list[str], day: str, maximum: int = 3) -> list[str | None]:
    if day not in sessions:
        return [None] * maximum
    start = sessions.index(day)
    return [sessions[start + offset] if start + offset < len(sessions) else None
            for offset in range(1, maximum + 1)]


def holding_marks(row: dict[str, Any], daily: pd.DataFrame,
                  sessions: list[str], costs: dict[str, Any] = COSTS) -> dict[str, Any]:
    """Compute independent fixed-close marks; do not call them realized exits."""
    result: dict[str, Any] = {"entry_price_proxy": row["signal"]["entry_price_cap"]}
    entry = float(result["entry_price_proxy"])
    next_days = _calendar_successors(sessions, row["entry_date"], 3)
    for offset, day in enumerate(next_days, 1):
        prefix = f"b{offset}"
        result[f"{prefix}_date"] = day
        if day is None or (row["code"], day) not in daily.index:
            result[f"{prefix}_state"] = "MISSING_DAILY_BAR"
            result[f"{prefix}_net_close_mark"] = None
            continue
        bar = daily.loc[(row["code"], day)]
        try:
            close, pre_close, volume = (_finite(bar[key]) for key in ("close", "pre_close", "volume"))
            if close <= 0 or pre_close <= 0 or volume < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            result[f"{prefix}_state"] = "INVALID_DAILY_BAR"
            result[f"{prefix}_net_close_mark"] = None
            continue
        down_limit = float((Decimal(str(pre_close)) * Decimal("0.90")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP))
        blocked = volume == 0 or close <= down_limit + EPSILON
        mark = net_return(entry, close, **costs)
        result[f"{prefix}_state"] = "CLOSE_MARK_SELL_BLOCKED_PROXY" if blocked else "CLOSE_MARK_ONLY"
        result[f"{prefix}_close"] = close
        result[f"{prefix}_net_close_mark"] = mark
        result[f"{prefix}_large_loss_mark"] = mark <= -0.07
        result[f"{prefix}_sell_blocked_proxy"] = blocked
    b2, b3 = result.get("b2_net_close_mark"), result.get("b3_net_close_mark")
    result["b2_b3_both_positive"] = None if b2 is None or b3 is None else bool(b2 > 0 and b3 > 0)
    result["realized_return"] = None
    return result


def _read_archives(roots: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    available: dict[tuple[str, str], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for root in roots:
        manifest_paths = [*root.glob("minute_*/manifest.json"), *root.glob("batch_*/manifest.json")]
        if (root / "manifest.json").exists():
            manifest_paths.append(root / "manifest.json")
        for manifest_path in sorted(set(manifest_paths)):
            hashes[str(manifest_path.resolve())] = sha256_file(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for case in manifest.get("cases", []):
                if case.get("error_code") != "0":
                    continue
                raw_path = manifest_path.parent / f"{case['code']}_{case['date']}.json"
                if not raw_path.exists() or sha256_file(raw_path) != case.get("raw_sha256"):
                    raise ValueError(f"raw archive hash mismatch: {raw_path}")
                key = (case["code"], case["date"])
                if key in available:
                    raise ValueError(f"duplicate successful archive: {key}")
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                if payload.get("code") != case["code"] or payload.get("requested_date") != case["date"]:
                    raise ValueError(f"archive identity mismatch: {raw_path}")
                available[key] = payload
                hashes[str(raw_path.resolve())] = case["raw_sha256"]
    return available, hashes


def _read_imported_archives(roots: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]],
                                                         dict[str, str]]:
    """Load strict V47 normalized five-minute imports into the replay schema."""
    available: dict[tuple[str, str], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    required = {"code", "date", "time", "open", "high", "low", "close", "volume", "amount"}
    for root in roots:
        manifest_paths = [root / "manifest.json"] if (root / "manifest.json").exists() else []
        manifest_paths.extend(path for path in root.glob("*/manifest.json") if path not in manifest_paths)
        for manifest_path in sorted(manifest_paths):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("interval") != 5 or manifest.get("adjustment_declared") != "unadjusted":
                raise ValueError(f"V48 requires unadjusted five-minute imports: {manifest_path}")
            normalized_path = manifest_path.parent / "normalized.csv"
            if (not normalized_path.exists()
                    or sha256_file(normalized_path) != manifest.get("normalized_sha256")):
                raise ValueError(f"normalized import hash mismatch: {normalized_path}")
            frame = pd.read_csv(normalized_path, dtype={"code": str, "date": str, "time": str})
            if not required.issubset(frame.columns):
                raise ValueError(f"normalized import missing columns: {normalized_path}")
            frame["code"] = frame["code"].str.zfill(6)
            hashes[str(manifest_path.resolve())] = sha256_file(manifest_path)
            hashes[str(normalized_path.resolve())] = sha256_file(normalized_path)
            coverage = {(str(row["code"]).zfill(6), str(row["date"])): row
                        for row in manifest.get("coverage", [])}
            for (code, day), group in frame.groupby(["code", "date"], sort=True):
                coverage_row = coverage.get((code, day), {})
                if len(group) != 48 or coverage_row.get("row_count_complete_only") is not True:
                    continue
                provider_code = ("sh." if code.startswith("60") else "sz.") + code
                key = (provider_code, day)
                if key in available:
                    raise ValueError(f"duplicate successful imported archive: {key}")
                group = group.sort_values("time")
                fields = ["code", "date", "time", "open", "high", "low", "close",
                          "volume", "amount", "adjustflag"]
                rows = []
                for row in group.to_dict("records"):
                    rows.append([provider_code, day, row["time"], row["open"], row["high"],
                                 row["low"], row["close"], row["volume"], row["amount"], "3"])
                available[key] = {"code": provider_code, "requested_date": day, "error_code": "0",
                                  "fields": fields, "rows": rows,
                                  "source_status": "USER_IMPORTED_NOT_INDEPENDENTLY_VERIFIED"}
    return available, hashes


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entered = [row for row in rows if row.get("entry_LABEL_ONLY", {}).get("state")
               == "BELOW_LIMIT_ENTRY_OPPORTUNITY_PROXY"]
    result: dict[str, Any] = {
        "selected_days": len(rows),
        "entry_opportunity_proxies": len(entered),
        "queue_unresolved": sum("QUEUE_UNRESOLVED" in row.get("entry_LABEL_ONLY", {}).get("state", "")
                                for row in rows),
        "actual_verified_fills": 0,
        "signal_type_counts": dict(Counter(row["signal"]["signal_type"] for row in rows)),
        "board_group_counts": dict(Counter(row.get("board_group") for row in rows)),
    }
    for offset in (1, 2, 3):
        values = [row["holding_LABEL_ONLY"].get(f"b{offset}_net_close_mark") for row in entered]
        values = [value for value in values if value is not None]
        result[f"b{offset}_observed"] = len(values)
        result[f"b{offset}_positive_rate"] = (sum(value > 0 for value in values) / len(values)) if values else None
        result[f"b{offset}_mean_net_close_mark"] = (sum(values) / len(values)) if values else None
        result[f"b{offset}_large_loss_rate"] = (sum(value <= -0.07 for value in values) / len(values)) if values else None
    persistent = [row["holding_LABEL_ONLY"].get("b2_b3_both_positive") for row in entered]
    persistent = [value for value in persistent if value is not None]
    result["b2_b3_persistence_observed"] = len(persistent)
    result["b2_b3_both_positive_rate"] = (sum(persistent) / len(persistent)) if persistent else None
    return result


def run(cohort_paths: list[Path], archive_roots: list[Path], daily_path: Path,
        calendar_path: Path, features_path: Path | None, output: Path,
        imported_roots: list[Path] | None = None,
        allow_incomplete_development: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("choose a new immutable output directory")
    output.mkdir(parents=True)
    protocol = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "signal_rule": "first completed 5-minute close at limit after an earlier completed below-limit close, by 10:00",
        "entry_label_rule": "only next 5-minute bar low below limit creates an opportunity proxy",
        "selection_baseline": SELECTION_BASELINE,
        "cutoff_selection_baseline": CUTOFF_SELECTION_BASELINE,
        "holding_evaluation": "independent B+1/B+2/B+3 close marks after stated costs",
        "cost_assumptions": COSTS,
        "threshold_tuning": False,
        "actual_fill_claim": False,
        "complete_trade_claim": False,
        "allow_incomplete_development": allow_incomplete_development,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    cohort = pd.concat([pd.read_csv(path, dtype={"code": str}) for path in cohort_paths], ignore_index=True)
    cohort["code"] = cohort["code"].str.zfill(6)
    if cohort.duplicated(["code", "entry_date"]).any():
        raise ValueError("duplicate cohort security-date")
    daily = pd.read_csv(daily_path, dtype={"code": str})
    daily["code"] = daily["code"].str.zfill(6)
    daily = daily.set_index(["code", "date"])
    if not daily.index.is_unique:
        raise ValueError("duplicate daily security-date")
    calendar_frame = pd.read_csv(calendar_path)
    sessions = calendar_frame["date"].astype(str).tolist()
    if sessions != sorted(set(sessions)):
        raise ValueError("calendar must be sorted and unique")
    features: dict[tuple[str, str], dict[str, Any]] = {}
    if features_path:
        feature_frame = pd.read_csv(features_path, dtype={"code": str})
        feature_frame["code"] = feature_frame["code"].str.zfill(6)
        features = {(row["code"], row["signal_date"]): row for row in feature_frame.to_dict("records")}
    archives, archive_hashes = _read_archives(archive_roots)
    imported, imported_hashes = _read_imported_archives(imported_roots or [])
    duplicates = set(archives).intersection(imported)
    if duplicates:
        raise ValueError(f"provider/import archive overlap; choose one source per case: {sorted(duplicates)[:3]}")
    archives.update(imported)
    archive_hashes.update(imported_hashes)

    signals = []
    daily_audits: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for candidate in cohort.to_dict("records"):
        raw = archives.get((candidate["provider_code"], candidate["entry_date"]))
        if raw is None:
            missing.append({"code": candidate["code"], "entry_date": candidate["entry_date"],
                            "signal_date": candidate["signal_date"], "boards": candidate["boards"],
                            "board_group": board_group(candidate), "state": "MISSING_ARCHIVE"})
            continue
        entry_key = (candidate["code"], candidate["entry_date"])
        signal_key = (candidate["code"], candidate["signal_date"])
        if signal_key not in daily.index:
            missing.append({"code": candidate["code"], "entry_date": candidate["entry_date"],
                            "signal_date": candidate["signal_date"], "boards": candidate["boards"],
                            "board_group": board_group(candidate),
                            "state": "MISSING_SIGNAL_DAY_DAILY_COMPARATOR"})
            continue
        prior = daily.loc[signal_key]
        limit_price = float((Decimal(str(prior["close"])) * Decimal("1.10")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP))
        context = features.get(signal_key, {})
        if entry_key not in daily.index:
            if not _all_zero_no_trade_archive(raw):
                missing.append({"code": candidate["code"], "entry_date": candidate["entry_date"],
                                "signal_date": candidate["signal_date"], "boards": candidate["boards"],
                                "board_group": board_group(candidate),
                                "state": "MISSING_ENTRY_DAY_DAILY_COMPARATOR"})
                continue
            no_trade_signal = {
                "signal": False, "signal_type": "NO_RECORDED_TRADING_ZERO_BARS",
                "confirmation_time": None, "confirmation_index": None, "observed_bars": 48,
                "decision": "NO_ORDER_NONTRADABLE", "entry_price_cap": limit_price,
                "asof": candidate["entry_date"].replace("-", "") + ASOF_CLOCK,
            }
            signals.append({
                **candidate,
                "board_group": board_group(candidate),
                "candidate_scope_eligible": False,
                "entry_evidence_status": "NO_RECORDED_TRADING_ZERO_BARS",
                "minute_source_status": raw.get(
                    "source_status", "HISTORICAL_PROVIDER_ARCHIVE_NOT_FORWARD"),
                "market_emotion_state": context.get("market_emotion_state"),
                "boards_for_comparison": context.get("boards_for_comparison", candidate.get("boards")),
                "height_gap_to_known_top": context.get("height_gap_to_known_top"),
                "sole_known_top_proxy": context.get("sole_known_top_proxy"),
                "signal": no_trade_signal,
                "cutoff_signal_1000": dict(no_trade_signal),
            })
            continue
        # This full-session comparison is outcome-side data. Retain it for
        # provenance auditing, but never expose it to the frozen signal file.
        audit = inspect(raw, daily.loc[entry_key])
        daily_audits[entry_key] = audit
        # End-of-day disagreement cannot remove a morning competitor or
        # change the selected stock. Keep it in outcome-side validation only.
        fields = raw["fields"]
        bars = [dict(zip(fields, values)) for values in raw["rows"]]
        signal = causal_reseal_signal(bars, candidate["entry_date"], limit_price)
        cutoff_signal = stable_cutoff_signal(bars, candidate["entry_date"], limit_price)
        signals.append({
            **candidate,
            "board_group": board_group(candidate),
            "candidate_scope_eligible": board_group(candidate) in {"FIRST_BOARD", "CONSECUTIVE_BOARD"},
            "entry_evidence_status": "PRICE_BARS_AVAILABLE",
            "minute_source_status": raw.get(
                "source_status", "HISTORICAL_PROVIDER_ARCHIVE_NOT_FORWARD"),
            "market_emotion_state": context.get("market_emotion_state"),
            "boards_for_comparison": context.get("boards_for_comparison", candidate.get("boards")),
            "height_gap_to_known_top": context.get("height_gap_to_known_top"),
            "sole_known_top_proxy": context.get("sole_known_top_proxy"),
            "signal": signal,
            "cutoff_signal_1000": cutoff_signal,
        })
    selections, cutoff_selections, selection_coverage_status = select_with_coverage(
        signals, missing, allow_incomplete_development)
    forbidden_outcome_fields = {"data_status", "daily_differences", "entry_LABEL_ONLY",
                                "holding_LABEL_ONLY", "realized_return"}
    for row in [*selections, *cutoff_selections]:
        leaked = forbidden_outcome_fields.intersection(row)
        if leaked:
            raise AssertionError(f"outcome fields leaked into frozen signal rows: {sorted(leaked)}")
    signal_path = output / "signals_BEFORE_OUTCOMES.json"
    signal_path.write_text(json.dumps({"version": VERSION, "earliest_confirmation_rows": selections,
                                      "stable_1000_rows": cutoff_selections, "missing": missing},
                                      ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def attach_outcomes(selected_rows: list[dict[str, Any]], policy_name: str) -> list[dict[str, Any]]:
        attached = []
        for row in selected_rows:
            raw = archives[(row["provider_code"], row["entry_date"])]
            bars = [dict(zip(raw["fields"], values)) for values in raw["rows"]]
            enriched = dict(row)
            enriched["policy"] = policy_name
            audit = daily_audits[(row["code"], row["entry_date"])]
            enriched["data_status_LABEL_ONLY"] = audit["status"]
            enriched["daily_differences_LABEL_ONLY"] = audit.get("differences", {})
            enriched["entry_LABEL_ONLY"] = next_bar_entry_label(bars, row["signal"])
            if enriched["entry_LABEL_ONLY"]["state"] == "BELOW_LIMIT_ENTRY_OPPORTUNITY_PROXY":
                enriched["holding_LABEL_ONLY"] = holding_marks(row, daily, sessions)
            else:
                enriched["holding_LABEL_ONLY"] = {
                    "state": "NOT_EVALUATED_WITHOUT_ENTRY_OPPORTUNITY",
                    "entry_price_proxy": None,
                    "realized_return": None,
                }
            attached.append(enriched)
        return attached

    rows = attach_outcomes(selections, SELECTION_BASELINE)
    cutoff_rows = attach_outcomes(cutoff_selections, CUTOFF_SELECTION_BASELINE)
    by_month = {}
    for month in sorted({row["entry_date"][:7] for row in rows}):
        by_month[month] = _summary([row for row in rows if row["entry_date"].startswith(month)])
    by_group = {}
    for group in ("FIRST_BOARD", "CONSECUTIVE_BOARD"):
        group_rows = [row for row in rows if row["board_group"] == group]
        by_group[group] = _summary(group_rows)
    by_emotion = {}
    for state in sorted({str(row.get("market_emotion_state") or "UNAVAILABLE") for row in rows}):
        by_emotion[state] = _summary([
            row for row in rows if str(row.get("market_emotion_state") or "UNAVAILABLE") == state
        ])
    by_signal_type = {}
    for signal_type in ("FIRST_SEAL_FROM_BELOW", "TOUCHED_PULLBACK_RESEAL"):
        by_signal_type[signal_type] = _summary([
            row for row in rows if row["signal"]["signal_type"] == signal_type
        ])
    cohort_by_day = Counter(str(row["entry_date"]) for row in cohort.to_dict("records"))
    usable_by_day = Counter(row["entry_date"] for row in signals)
    missing_by_day = Counter(row["entry_date"] for row in missing)
    coverage_by_entry_date = [
        {
            "entry_date": day,
            "cohort_rows": cohort_by_day[day],
            "usable_rows": usable_by_day[day],
            "missing_rows": missing_by_day[day],
            "coverage_rate": usable_by_day[day] / cohort_by_day[day],
            "complete": missing_by_day[day] == 0,
        }
        for day in sorted(cohort_by_day)
    ]
    report = {
        **protocol,
        "cohort_rows": len(cohort),
        "usable_signal_rows": len(signals),
        "missing_rows": len(missing),
        "candidate_universe_coverage_complete": not missing,
        "candidate_universe_coverage_rate": len(signals) / len(cohort) if len(cohort) else None,
        "selection_coverage_status": selection_coverage_status,
        "usable_board_group_counts": dict(Counter(row["board_group"] for row in signals)),
        "missing_board_group_counts": dict(Counter(row["board_group"] for row in missing)),
        "missing_reason_counts": dict(Counter(row["state"] for row in missing)),
        "minute_source_status_counts": dict(Counter(
            row["minute_source_status"] for row in signals)),
        "coverage_by_entry_date": coverage_by_entry_date,
        "left_censored_rows_excluded_from_selection": sum(
            row["board_group"] == "UNKNOWN_LEFT_CENSORED" for row in signals),
        "daily_reconciliation_status_counts": dict(Counter(
            audit["status"] for audit in daily_audits.values())),
        "reconciliation_issues_LABEL_ONLY": [
            {"code": code, "entry_date": day, "status": audit["status"],
             "differences": audit.get("differences", {})}
            for (code, day), audit in daily_audits.items()
            if not reconciliation_usable(audit["status"])
        ],
        "reconciled_validation_ready": bool(daily_audits) and not missing and all(
            reconciliation_usable(audit["status"]) for audit in daily_audits.values()),
        "coverage_interpretation": "archive availability only; reconciliation and verified execution are separate gates",
        "candidate_signals_by_1000": sum(row["signal"]["signal"] for row in signals),
        "nontradable_zero_bar_rows": sum(
            row.get("entry_evidence_status") == "NO_RECORDED_TRADING_ZERO_BARS" for row in signals),
        "signal_type_counts_all_candidates": dict(Counter(row["signal"]["signal_type"] for row in signals)),
        "overall": _summary(rows),
        "stable_1000_overall": _summary(cutoff_rows),
        "by_month": by_month,
        "by_board_group": by_group,
        "by_emotion_state": by_emotion,
        "by_signal_type": by_signal_type,
        "rows": rows,
        "stable_1000_rows": cutoff_rows,
        "verified_trades": 0,
        "validated_complete_trade_win_rate": None,
        "deployment_allowed": False,
        "source_hashes": {
            **{str(path.resolve()): sha256_file(path) for path in [*cohort_paths, daily_path, calendar_path]
               + ([features_path] if features_path else [])},
            **archive_hashes,
            str(Path(__file__).resolve()): sha256_file(Path(__file__)),
            str(signal_path.resolve()): sha256_file(signal_path),
        },
        "limitations": [
            "historical five-minute timestamps are assumed bar-end and have no receipt-latency evidence",
            "below-limit trades after submission are an entry-opportunity proxy, not a broker-confirmed fill",
            "range containment does not establish why extrema differ; contradictions block validation, never alter morning ranking",
            "B+1/B+2/B+3 values are fixed close marks, not verified realized exits",
            "January and February were previously inspected and are development evidence, not a fresh blind test",
            "real theme identity and auction/order-queue data are unavailable in this historical cohort",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--archive-root", type=Path, action="append", required=True)
    parser.add_argument("--imported-root", type=Path, action="append", default=[])
    parser.add_argument("--allow-incomplete-development", action="store_true",
                        help="permit a clearly labelled development ranking despite missing candidate rows")
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.cohort, args.archive_root, args.daily, args.calendar, args.features, args.output,
                 args.imported_root, args.allow_incomplete_development)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "stable_1000_rows", "source_hashes"}},
                     ensure_ascii=False, indent=2))
