#!/usr/bin/env python3
"""Audit whether V9 has enough verified intraday evidence to train safely."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import open_confirmation_v4 as opening
import session_forward_v4 as session_forward
import intraday_forward_v10


ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOTS = ROOT / "data" / "v9" / "derived_open_snapshots.csv"
DEFAULT_SHADOW = ROOT / "results" / "v9" / "intraday_shadow"
DEFAULT_SESSION = ROOT / "results" / "v4" / "session_forward"
DEFAULT_FORWARD_V10 = ROOT / "results" / "v10" / "intraday_forward"
DEFAULT_OUTPUT = ROOT / "results" / "v10" / "intraday_readiness.json"
VERSION = "v10-intraday-readiness-2"
MIN_FULL_SEQUENCES = 120
MIN_SHADOW_OBSERVATIONS = 100
MIN_SETTLED_CONFIRMATIONS = 60
MIN_CAPTURE_COVERAGE = 0.90


def snapshot_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0, "candidate_sequences": 0, "technical_complete_sequences": 0,
            "full_complete_sequences": 0, "capture_coverage": 0.0,
        }
    checked = opening.validate_snapshots(frame)
    keys = ["signal_date", "trade_date", "code"]
    technical_complete = 0
    full_complete = 0
    for _, group in checked.groupby(keys, sort=False):
        by_time = group.sort_values("asof", key=lambda x: x.map(opening.clock_seconds)).drop_duplicates("asof", keep="last").set_index("asof")
        if not {"09:25", "09:35"}.issubset(by_time.index):
            continue
        auction_fields = opening.REQUIRED_BY_ASOF["09:25"]
        open_fields = tuple(field for field in opening.REQUIRED_BY_ASOF["09:35"] if field != "theme_sync_open")
        auction_ok = all(pd.notna(by_time.loc["09:25"].get(field)) for field in auction_fields)
        open_ok = all(pd.notna(by_time.loc["09:35"].get(field)) for field in open_fields)
        verified = all(str(by_time.loc[point].get("data_status", "")).lower() in opening.VERIFIED_STATUSES for point in ("09:25", "09:35"))
        if auction_ok and open_ok and verified:
            technical_complete += 1
            if pd.notna(by_time.loc["09:35"].get("theme_sync_open")):
                full_complete += 1
    sequences = int(checked.groupby(keys).ngroups)
    return {
        "rows": int(len(checked)), "candidate_sequences": sequences,
        "technical_complete_sequences": technical_complete,
        "full_complete_sequences": full_complete,
        "capture_coverage": float(full_complete / sequences) if sequences else 0.0,
    }


def shadow_coverage(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.csv")) if directory.exists() else []
    observations: set[tuple[str, str]] = set()
    confirmations: set[tuple[str, str]] = set()
    for path in files:
        frame = pd.read_csv(path, dtype={"code": "string"})
        for _, row in frame.iterrows():
            key = (str(row.get("signal_date", "")), str(row.get("code", "")).zfill(6))
            observed = str(row.get("shadow_observation", "")).strip().lower() in {"true", "1", "1.0"}
            if observed:
                observations.add(key)
            if str(row.get("open_confirmation", "")) == "SHADOW_CONFIRM_RESEAL":
                confirmations.add(key)
    return {
        "score_files": len(files), "shadow_observations": len(observations),
        "shadow_confirmations": len(confirmations),
        "settled_confirmations": 0,
        "settlement_status": "UNAVAILABLE_NO_V9_OUTCOME_LEDGER",
    }


def audit(
    snapshots_path: Path = DEFAULT_SNAPSHOTS,
    shadow_dir: Path = DEFAULT_SHADOW,
    session_dir: Path = DEFAULT_SESSION,
    forward_v10: Path = DEFAULT_FORWARD_V10,
) -> dict[str, Any]:
    if snapshots_path.exists() and snapshots_path.stat().st_size:
        snapshots = pd.read_csv(snapshots_path, dtype={"code": "string"})
    else:
        snapshots = pd.DataFrame()
    coverage = snapshot_coverage(snapshots)
    legacy_shadow = shadow_coverage(shadow_dir)
    forward = intraday_forward_v10.monitor(forward_v10)
    shadow = {
        "registered_plans": forward["registered_plans"],
        "shadow_observations": forward["registered_candidates"],
        "shadow_confirmations": forward["shadow_reseal_confirmations"],
        "settled_confirmations": forward["executed_proxy"],
        "settlement_status": "V10_TAMPER_EVIDENT_LEDGER",
    }
    ledger = session_forward.verify_ledger(session_dir)
    gates = {
        "full_sequences": coverage["full_complete_sequences"] >= MIN_FULL_SEQUENCES,
        "capture_coverage": coverage["capture_coverage"] >= MIN_CAPTURE_COVERAGE,
        "shadow_observations": shadow["shadow_observations"] >= MIN_SHADOW_OBSERVATIONS,
        "settled_confirmations": shadow["settled_confirmations"] >= MIN_SETTLED_CONFIRMATIONS,
        "session_ledger": ledger["status"] == "PASS",
        "v10_forward_ledger": forward["status"] == "PASS",
    }
    return {
        "version": VERSION,
        "status": "TRAINING_READY" if all(gates.values()) else "DATA_COLLECTION_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "minimum_full_candidate_sequences": MIN_FULL_SEQUENCES,
            "minimum_shadow_observations": MIN_SHADOW_OBSERVATIONS,
            "minimum_settled_confirmations": MIN_SETTLED_CONFIRMATIONS,
            "minimum_capture_coverage": MIN_CAPTURE_COVERAGE,
        },
        "snapshot_coverage": coverage, "shadow_coverage": shadow,
        "legacy_unregistered_shadow_files": legacy_shadow,
        "v10_forward_ledger": forward,
        "session_ledger": ledger, "gates": gates,
        "model_training_allowed": all(gates.values()),
        "reason": (
            "verified auction, minute, theme-sync and settled T+1 evidence must reach the frozen thresholds; "
            "missing observations are not imputed as safe zeros"
        ),
    }


def run(
    snapshots_path: Path = DEFAULT_SNAPSHOTS,
    shadow_dir: Path = DEFAULT_SHADOW,
    session_dir: Path = DEFAULT_SESSION,
    forward_v10: Path = DEFAULT_FORWARD_V10,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    result = audit(snapshots_path, shadow_dir, session_dir, forward_v10)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**result, "output": str(output.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V9 intraday model readiness")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--forward-v10", type=Path, default=DEFAULT_FORWARD_V10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.snapshots, args.shadow, args.session, args.forward_v10, args.output), ensure_ascii=False, indent=2))
