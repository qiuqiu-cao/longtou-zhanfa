"""Fail-closed input contract for the daily leader-relay review.

It audits a candidate table without filling missing auction, intraday, theme,
or execution fields. The contract answers whether a dataset can support
identity review, conditional relay review, or neither.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FIELD_GROUPS = {
    "basic": {
        "code", "name", "boards", "amount", "float_cap", "turnover",
    },
    "auction": {
        "auction_gap", "auction_amount", "auction_amount_ratio",
        "auction_0920_stable", "auction_0925_pct",
    },
    "intraday": {
        "first_seal_time", "failed_seal_count", "reseal_count",
        "last_seal_time", "sealed_funds", "max_drawdown", "vwap_reclaimed",
    },
    "theme": {
        "theme_group_id", "theme", "theme_evidence_status", "theme_ladder_count",
        "theme_middle_support", "theme_first_support",
    },
    "emotion": {
        "market_emotion_state", "market_limit_count", "market_failed_count",
        "market_down_limit_count", "market_promotion_mid", "market_max_board",
    },
    "execution": {
        "source_timestamp", "data_status", "fill_price", "fill_quantity",
        "actual_exit_price", "execution_status",
    },
}

REQUIRED_FOR_IDENTITY = {"code", "boards", "signal_date"}
REQUIRED_FOR_CONDITIONAL_RELAY = set().union(FIELD_GROUPS["basic"], FIELD_GROUPS["auction"],
                                              FIELD_GROUPS["intraday"], FIELD_GROUPS["theme"],
                                              FIELD_GROUPS["emotion"])


def _status(frame: pd.DataFrame, fields: set[str]) -> dict:
    result = {}
    for field in sorted(fields):
        if field not in frame.columns:
            result[field] = {"status": "UNAVAILABLE", "reason": "field_missing"}
            continue
        values = frame[field]
        known = int(values.notna().sum())
        nonempty = int(values.astype(str).str.strip().ne("").sum()) if values.dtype == object else known
        result[field] = {
            "status": "AVAILABLE" if known else "UNAVAILABLE",
            "known_rows": known,
            "coverage": known / len(frame) if len(frame) else 0.0,
            "nonempty_rows": nonempty,
        }
    return result


def audit(frame: pd.DataFrame, asof: str | None = None) -> dict:
    rows = frame.copy()
    if "code" in rows:
        rows["code"] = rows["code"].astype(str).str.zfill(6)
    if "signal_date" in rows:
        rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce")
    if "boards" in rows:
        rows["boards"] = pd.to_numeric(rows["boards"], errors="coerce")
    statuses = {group: _status(rows, fields) for group, fields in FIELD_GROUPS.items()}
    identity_ok = REQUIRED_FOR_IDENTITY <= set(rows.columns)
    date_errors = 0
    if asof and "source_timestamp" in rows:
        timestamps = pd.to_datetime(rows["source_timestamp"], errors="coerce", utc=True)
        cutoff = pd.Timestamp(asof, tz="Asia/Shanghai").tz_convert("UTC")
        date_errors = int(timestamps.gt(cutoff).sum())
    missing_relay = sorted(REQUIRED_FOR_CONDITIONAL_RELAY - set(rows.columns))
    board_rows = rows[rows.get("boards", pd.Series(index=rows.index)).fillna(0).ge(2)]
    return {
        "version": "v265-leader-strategy-input-contract-1",
        "status": "DIAGNOSTIC_ONLY",
        "rows": int(len(rows)),
        "dates": int(rows.signal_date.nunique()) if "signal_date" in rows else 0,
        "latest_signal_date": (rows.signal_date.max().strftime("%Y-%m-%d")
                               if "signal_date" in rows and rows.signal_date.notna().any() else None),
        "field_status": statuses,
        "identity_review": {
            "allowed": bool(identity_ok and date_errors == 0),
            "coverage_rows": int(len(board_rows)),
            "required_fields": sorted(REQUIRED_FOR_IDENTITY),
        },
        "conditional_relay_review": {
            "allowed": bool(not missing_relay and date_errors == 0),
            "missing_fields": missing_relay,
            "required_fields": sorted(REQUIRED_FOR_CONDITIONAL_RELAY),
        },
        "trade_authorization": {
            "allowed": False,
            "reason": "This contract audits inputs only; actual fills and live confirmation remain separate gates.",
        },
        "future_timestamp_rows": date_errors,
        "source_quality": "MISSING_OR_UNVERIFIED_UNLESS_EACH_FIELD_HAS_SOURCE_AND_TIMESTAMP",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asof")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable output already exists")
    frame = pd.read_csv(args.input, dtype={"code": str}, low_memory=False)
    result = audit(frame, args.asof)
    result["source"] = {"path": str(args.input.resolve()),
                         "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "identity_review": result["identity_review"]["allowed"],
                      "conditional_relay_review": result["conditional_relay_review"]["allowed"],
                      "trade_authorization": result["trade_authorization"]["allowed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
