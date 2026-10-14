"""V224 immutable phase settlement for V220 first-board forward snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v224-first-board-forward-settlement-1"
TZ = ZoneInfo("Asia/Shanghai")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _append_ledger(ledger: Path, event: dict) -> dict:
    existing = []
    if ledger.exists():
        existing = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    body = dict(event)
    body["sequence"] = len(existing) + 1
    body["previous_event_hash"] = existing[-1]["event_hash"] if existing else "GENESIS"
    body["event_hash"] = hashlib.sha256(_canonical(body)).hexdigest()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
    return body


def verify_ledger(ledger: str | Path) -> dict:
    path = Path(ledger)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = "GENESIS"
    errors = []
    for expected_sequence, event in enumerate(events, start=1):
        supplied_hash = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        calculated = hashlib.sha256(_canonical(body)).hexdigest()
        if event.get("sequence") != expected_sequence:
            errors.append(f"sequence mismatch at {expected_sequence}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"chain mismatch at {expected_sequence}")
        if supplied_hash != calculated:
            errors.append(f"event hash mismatch at {expected_sequence}")
        settlement = Path(str(event.get("settlement_file", "")))
        if not settlement.exists() or _sha(settlement) != event.get("settlement_sha256"):
            errors.append(f"settlement artifact mismatch at {expected_sequence}")
        previous = supplied_hash
    return {"version": VERSION, "status": "PASS" if not errors else "FAIL",
            "events": len(events), "errors": errors}


def settle(snapshot: dict, confirmation: dict | None,
           promotion_pool: pd.DataFrame | None, daily_bars: pd.DataFrame | None) -> dict:
    candidate = snapshot.get("candidate")
    report = {
        "version": VERSION, "freeze_id": snapshot.get("freeze_id"),
        "signal_date": snapshot.get("signal_date"), "candidate": candidate,
        "phases": {}, "entry_authorized": False,
        "settled_at": datetime.now(TZ).isoformat(),
    }
    if not candidate:
        report["status"] = "NO_FROZEN_CANDIDATE"
        return report
    code = str(candidate["code"]).zfill(6)
    if promotion_pool is None:
        report["phases"]["promotion"] = {"status": "PENDING_DATA"}
    else:
        required = {"code", "trade_date", "boards", "source_status"}
        if required - set(promotion_pool):
            raise ValueError(f"promotion pool fields missing: {sorted(required-set(promotion_pool))}")
        match = promotion_pool[promotion_pool.code.astype(str).str.zfill(6).eq(code)]
        verified = promotion_pool.source_status.astype(str).eq("COMPLETE_CLOSING_LIMIT_POOL").all()
        report["phases"]["promotion"] = {
            "status": "SETTLED_VERIFIED" if verified else "SETTLED_PROVIDER_OBSERVED",
            "promoted": bool((pd.to_numeric(match.boards, errors="coerce") >= 2).any()),
            "full_denominator_rows": int(len(promotion_pool)),
        }
    if confirmation is None:
        report["phases"]["open_confirmation"] = {"status": "PENDING_DATA"}
    else:
        if str((confirmation.get("candidate") or {}).get("code", "")).zfill(6) != code:
            raise ValueError("V221 confirmation candidate mismatch")
        report["phases"]["open_confirmation"] = {
            "status": "SETTLED", "decision": confirmation.get("decision"),
            "first_causal_price_opportunity": confirmation.get("first_causal_price_opportunity"),
        }
    causal = (confirmation or {}).get("first_causal_price_opportunity")
    if daily_bars is None or not causal:
        report["phases"]["payoff"] = {"status": "PENDING_DATA_OR_NO_CAUSAL_ENTRY"}
    else:
        required = {"code", "trade_date", "close", "source_status"}
        if required - set(daily_bars):
            raise ValueError(f"daily bar fields missing: {sorted(required-set(daily_bars))}")
        bars = daily_bars[daily_bars.code.astype(str).str.zfill(6).eq(code)].sort_values("trade_date")
        price = float(causal["price_proxy"])
        returns = {}
        for i, (_, row) in enumerate(bars.head(3).iterrows(), start=1):
            returns[f"B_plus_{i}"] = float(row.close) / price - 1
        complete = len(returns) == 3 and bars.source_status.astype(str).eq("VERIFIED_DAILY_BAR").all()
        report["phases"]["payoff"] = {
            "status": "SETTLED_VERIFIED" if complete else "PARTIAL_OR_PROVIDER_OBSERVED",
            "entry_price_proxy": price, "returns_before_costs": returns,
            "same_day_stop_assumed": False, "T_plus_1_constraint_applied": True,
        }
    statuses = [v["status"] for v in report["phases"].values()]
    report["status"] = "COMPLETE" if all(s.startswith("SETTLED_VERIFIED") or s == "SETTLED" for s in statuses) else "PENDING_OR_PARTIAL"
    return report


def run(snapshot_path: str, output: str, ledger: str,
        confirmation_path: str | None = None, promotion_pool_path: str | None = None,
        daily_bars_path: str | None = None) -> dict:
    source_paths = [Path(snapshot_path)]
    snapshot = json.loads(source_paths[0].read_text(encoding="utf-8-sig"))
    confirmation = None
    if confirmation_path:
        p = Path(confirmation_path); source_paths.append(p)
        confirmation = json.loads(p.read_text(encoding="utf-8-sig"))
    pool = None
    if promotion_pool_path:
        p = Path(promotion_pool_path); source_paths.append(p)
        pool = pd.read_csv(p, dtype={"code": str})
    bars = None
    if daily_bars_path:
        p = Path(daily_bars_path); source_paths.append(p)
        bars = pd.read_csv(p, dtype={"code": str})
    report = settle(snapshot, confirmation, pool, bars)
    report["source_hashes"] = {str(p.resolve()): _sha(p) for p in source_paths}
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V224 settlement")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    event = {
        "event_type": "FORWARD_SETTLEMENT", "version": VERSION,
        "freeze_id": snapshot.get("freeze_id"), "signal_date": snapshot.get("signal_date"),
        "status": report["status"], "settlement_file": str(target.resolve()),
        "settlement_sha256": _sha(target), "recorded_at": datetime.now(TZ).isoformat(),
    }
    report["ledger_event"] = _append_ledger(Path(ledger), event)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot")
    ap.add_argument("--confirmation")
    ap.add_argument("--promotion-pool")
    ap.add_argument("--daily-bars")
    ap.add_argument("--output")
    ap.add_argument("--ledger", default="data/v224/settlement_ledger.jsonl")
    ap.add_argument("--verify-ledger", action="store_true")
    ns = ap.parse_args()
    if ns.verify_ledger:
        print(json.dumps(verify_ledger(ns.ledger), ensure_ascii=False, indent=2))
    else:
        if not ns.snapshot or not ns.output:
            ap.error("--snapshot and --output are required unless --verify-ledger is used")
        print(json.dumps(run(ns.snapshot, ns.output, ns.ledger, ns.confirmation,
                             ns.promotion_pool, ns.daily_bars), ensure_ascii=False, indent=2))
