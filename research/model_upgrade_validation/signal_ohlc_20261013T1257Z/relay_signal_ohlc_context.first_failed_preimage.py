"""Bind existing signal-day daily OHLC to the original relay cohort, read only.

Retrospective diagnostics only. No fits, score edits, target-day lookups, or
certification of leader identity, absorption, price-limit rules, or fills.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import archive_august_minutes_v289 as august
import archive_long_interval_v290 as long_daily
import historical_ohlc_archive_v278 as september
import model_upgrade_history as history

ROOT = Path(__file__).resolve().parent
FIELDS = ["open", "high", "low", "close"]
ARCHIVES = [
    ("data/v290/full_long_interval_20261002_v1", "unadjusted_daily_reference.csv", long_daily.daily_parse),
    ("data/v290/long_interval_recovery_20261002_v1", "daily_normalized.csv", long_daily.daily_parse),
    ("data/v289/full_august_five_minute_20261002_v1", "unadjusted_daily_reference.csv", august.daily_parse),
    ("data/v278/september_unadjusted_20261002_v3", "unadjusted_daily.csv", september.parse),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cents(value):
    value = Decimal(str(value))
    if not value.is_finite() or value <= 0 or value * 100 != (value * 100).to_integral_value():
        raise ValueError("positive exact-cent price required")
    return int(value * 100)


def qualify(signal, code, price, observations, sessions):
    """Only the exact signal/code is admissible; conflicting sources stay unknown."""
    result = {"date": signal, "code": code, "actual_one_word_daily": None,
        "strict_original_time_eligible": False, "diagnostic_eligible": False,
        "status": "SIGNAL_DAY_OHLC_MISSING", "sources": observations,
        "definition": "Exact-cent O=H=L=C with close/high equal original archived relay-pool price",
        "does_not_certify": ["independent_exchange_limit_price", "leader", "absorption", "fill"]}
    if signal not in sessions:
        raise ValueError("signal is outside official sessions")
    if not observations:
        return result
    canonical = []
    for item in observations:
        if item["date"] != signal or item["code"] != code:
            raise ValueError("target-day or other-stock substitution rejected")
        canonical.append(tuple(cents(item[k]) for k in FIELDS))
    if len(set(canonical)) != 1:
        result["status"] = "SOURCE_OHLC_CONFLICT"
        return result
    op, high, low, close = canonical[0]
    if close != cents(price):
        result["status"] = "ORIGINAL_POOL_CLOSE_CONFLICT"
        return result
    if high != close:
        result["status"] = "POOL_CLOSE_NOT_DAILY_HIGH"
        return result
    times = [item.get("available_at") for item in observations]
    # Missing receipt times remain unknown, never infer a receipt from a request.
    result.update(status="SIGNAL_DAY_OHLC_QUALIFIED_RETROSPECTIVE",
        diagnostic_eligible=True, actual_one_word_daily=(op == high == low == close),
        earliest_actual_receipt=min(times) if all(times) else None,
        source_time_mode="RETROSPECTIVE_RECEIPT_NOT_ORIGINAL_SIGNAL_CUTOFF")
    return result


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError("source hash mismatch: " + path)


def archive_rows(relative, artifact_name, parser):
    directory = ROOT / relative
    report_path = directory / "report.json"
    report = read(report_path)
    sources = {str(report_path): sha(report_path)}
    if "source_hashes" in report:
        verify(report["source_hashes"])
        sources.update(report["source_hashes"])
    elif report["source_pool_sha256"] != sha(september.POOL):
        raise ValueError("original source pool changed")
    artifact = directory / artifact_name
    if "artifact_hashes" in report and sha(artifact) != report["artifact_hashes"][artifact_name]:
        raise ValueError("normalized daily artifact hash mismatch")
    sources[str(artifact)] = sha(artifact)
    frames, metadata, seen = [], {}, set()
    entries = report["audit"]
    for item in entries:
        if "kind" in item and item["kind"] != "daily":
            continue
        symbol = item["symbol"]
        if symbol in seen:
            raise ValueError("duplicate daily source symbol")
        seen.add(symbol)
        if item["status"] not in ("RAW_PARSED", "UNADJUSTED_DAY_AVAILABLE"):
            continue
        path = Path(item["raw_path"]) if "raw_path" in item else directory / "raw" / (symbol + ".json")
        expected = item.get("raw_sha256", item.get("sha256"))
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError("raw daily path/hash mismatch")
        start, end = parser.__globals__["START"], parser.__globals__["END"]
        request = item["parameters"]["param"].split(",")
        if len(request) != 6 or request[:4] != [symbol, "day", start, end] or request[5] != "":
            raise ValueError("raw daily symbol/date/unadjusted request mismatch")
        if item.get("http_status") != 200:
            raise ValueError("successful daily receipt requires HTTP 200")
        frame = parser(read(path), symbol)
        if len(frame) != item.get("parsed_rows", item.get("rows")):
            raise ValueError("parsed daily count mismatch")
        frames.append(frame)
        sources[str(path)] = sha(path)
        available = item.get("available_at", item.get("captured_at_utc"))
        if available:
            dt = datetime.fromisoformat(available)
            if dt.tzinfo is None or dt > datetime.now(timezone.utc):
                raise ValueError("invalid actual source receipt time")
            available = dt.astimezone(timezone.utc).isoformat()
        metadata[symbol] = {"raw_path": str(path), "raw_sha256": expected,
            "report_path": str(report_path), "report_sha256": sha(report_path),
            "available_at": available, "request": item["parameters"], "source": item.get("url", item.get("source"))}
    rebuilt = pd.concat(frames, ignore_index=True)
    stored = pd.read_csv(artifact)
    if rebuilt.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate daily bars")
    keys = ["symbol", "date"]
    pd.testing.assert_frame_equal(stored.sort_values(keys).reset_index(drop=True),
        rebuilt.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-10, rtol=1e-12)
    # Prices are checked again as exact cents for every signal-day cell below.
    return rebuilt, metadata, sources


def summary(rows):
    known = [r for r in rows if r["diagnostic_eligible"]]
    return {"total": len(rows), "diagnostic_known": len(known),
        "one_word": sum(r["actual_one_word_daily"] is True for r in known),
        "non_one_word": sum(r["actual_one_word_daily"] is False for r in known),
        "unknown": len(rows) - len(known), "strict_original_time_known": 0,
        "status_counts": dict(Counter(r["status"] for r in rows))}


def build(receipt_path, calendar_path, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("unique source-layer output directory required")
    receipt, calendar = read(receipt_path), read(calendar_path)
    if calendar["status"] != "verified" or not calendar.get("source"):
        raise ValueError("verified official calendar required")
    verify(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    full = joblib.load(receipt["prepared_path"])["relay"]
    if full.duplicated(["date", "code"]).any():
        raise ValueError("duplicate original candidates")
    sessions = set(calendar["sessions"])
    cohort = set(zip(full.date, full.code))
    observations, source_bindings, archive_checks = defaultdict(list), {}, []
    for relative, artifact, parser in ARCHIVES:
        daily, metadata, sources = archive_rows(relative, artifact, parser)
        source_bindings.update(sources)
        matched = 0
        for r in daily.itertuples(index=False):
            code = r.symbol[2:]
            if (r.date, code) not in cohort:
                continue
            item = {"date": r.date, "code": code, **{k: float(getattr(r, k)) for k in FIELDS},
                **metadata[r.symbol]}
            observations[(r.date, code)].append(item)
            matched += 1
        archive_checks.append({"archive": relative, "reconstructed_daily_rows": len(daily),
            "original_signal_candidate_matches": matched, "successful_daily_responses": len(metadata)})
    rows = []
    for r in full.itertuples(index=False):
        item = qualify(r.date, r.code, r.price, observations[(r.date, r.code)], sessions)
        item.update(original_price=float(r.price), original_proxy=None if pd.isna(r.signal_one_word) else float(r.signal_one_word))
        rows.append(item)
    by_key = {(r["date"], r["code"]): r for r in rows}
    folds = {}
    for fold in history.FOLDS:
        train = history.training_rows(full, fold["start"], "relay")
        original = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if history._frame_digest(train, history.RELAY_FIELDS) != original["training_digest"]:
            raise ValueError("original training member/feature digest changed")
        selected = [by_key[k] for k in zip(train.date, train.code)]
        folds[fold["id"]] = {**summary(selected), "training_min_signal": train.date.min(),
            "training_max_signal": train.date.max(), "training_max_label": train.target_date.max(),
            "original_training_digest": original["training_digest"]}
    disagreement = [r for r in rows if r["diagnostic_eligible"] and r["original_proxy"] is not None
        and float(r["actual_one_word_daily"]) != r["original_proxy"]]
    for path in [receipt_path, calendar_path, receipt["prepared_path"], __file__, history.__file__,
                 august.__file__, long_daily.__file__, september.__file__]:
        source_bindings[str(Path(path).resolve())] = sha(path)
    report = {"formed_at": datetime.now(timezone.utc).isoformat(), "summary": summary(rows),
        "training_windows": folds, "archive_checks": archive_checks,
        "proxy_disagreement_count": len(disagreement), "proxy_disagreement_by_period":
            dict(Counter(r["date"][:4] for r in disagreement)),
        "numeric_fit_ready": False, "reason": "Earliest training window has no signal OHLC; all sources retrospective",
        "source_bindings": source_bindings, "original_fields_labels_members_preserved": True,
        "numeric_fits": 0, "new_mature_forward_samples": 0, "accuracy_gain_verified": False}
    verify(source_bindings)
    output.mkdir(parents=True)
    (output / "signal_day_ohlc_context.json").write_text(json.dumps({"formed_at": report["formed_at"],
        "track": "DIAGNOSTIC", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("sources", "does_not_certify", "definition")}
        for r in rows]).to_csv(output / "complete_original_cohort.csv", index=False, encoding="utf-8-sig")
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir()}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: report[k] for k in ("summary", "training_windows", "archive_checks", "proxy_disagreement_count", "numeric_fit_ready")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.receipt, args.calendar, args.output_dir), ensure_ascii=False))
