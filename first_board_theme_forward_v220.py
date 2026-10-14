"""Tamper-evident forward shadow ledger for the V218/V219 first-board path.

One close-quality and theme-supported candidate may be frozen after each close.
It never authorizes an order and never substitutes another stock when the next-
session confirmation of the frozen candidate fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v220-first-board-theme-forward-1"


def canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def append_event(path: Path, payload: dict) -> dict:
    events = read_events(path)
    event = {
        **payload, "sequence": len(events) + 1,
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "previous_event_hash": events[-1]["event_hash"] if events else "GENESIS",
    }
    event["event_hash"] = hashlib.sha256(canonical(event).encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(event) + "\n")
    return event


def verify(registry: Path) -> dict:
    events, previous, errors = read_events(registry / "ledger.jsonl"), "GENESIS", []
    for index, event in enumerate(events, 1):
        supplied = event.get("event_hash")
        expected = hashlib.sha256(canonical(
            {key: value for key, value in event.items() if key != "event_hash"}
        ).encode("utf-8")).hexdigest()
        if event.get("sequence") != index:
            errors.append(f"sequence mismatch at {index}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"chain mismatch at {index}")
        if supplied != expected:
            errors.append(f"hash mismatch at {index}")
        if event.get("snapshot_file"):
            artifact = registry / event["snapshot_file"]
            if not artifact.exists() or sha256(artifact) != event.get("snapshot_sha256"):
                errors.append(f"snapshot mismatch at {index}")
        previous = supplied
    return {"status": "PASS" if not errors else "FAIL", "events": len(events),
            "last_event_hash": previous, "errors": errors}


def freeze(registry: Path, data_end: str, artifacts: list[Path]) -> dict:
    active = registry / "active_freeze.json"
    if active.exists():
        raise FileExistsError("active V220 freeze already exists")
    policy = {
        "version": VERSION,
        "scope": "FIRST_BOARD_ONLY_LEADER_AND_RELAY_UNCHANGED",
        "selection": "theme_research_gate then highest frozen community score",
        "eligible_theme_lanes": [
            "VERIFIED_REAL_THEME", "VERIFIED_MEMBERSHIP_ONLY", "OBSERVED_THEME_ROSTER"],
        "observed_roster_is_verified_catalyst": False,
        "next_session_confirmation_required": True,
        "replacement_after_cancel": False,
        "entry_authorized": False,
        "minimum_forward_predictions": 60,
        "minimum_forward_promotions": 20,
    }
    created = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    body = {
        "version": VERSION, "created_at": created,
        "data_end": pd.Timestamp(data_end).strftime("%Y-%m-%d"),
        "policy": policy,
        "artifact_hashes": {str(path.resolve()): sha256(path) for path in artifacts},
    }
    body["freeze_id"] = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()[:16]
    registry.mkdir(parents=True, exist_ok=True)
    sealed = registry / f"freeze_{body['freeze_id']}.json"
    sealed.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    active.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(registry / "ledger.jsonl", {
        "event_type": "MODEL_FREEZE", "freeze_id": body["freeze_id"],
        "data_end": body["data_end"], "manifest_file": sealed.name,
        "manifest_sha256": sha256(sealed),
    })
    return body


def validate_freeze(registry: Path) -> dict:
    active = registry / "active_freeze.json"
    if not active.exists():
        raise ValueError("no active V220 freeze")
    manifest = json.loads(active.read_text(encoding="utf-8"))
    for value, digest in manifest["artifact_hashes"].items():
        path = Path(value)
        if not path.exists() or sha256(path) != digest:
            raise ValueError(f"frozen artifact changed or missing: {path}")
    if verify(registry)["status"] != "PASS":
        raise ValueError("forward ledger verification failed")
    return manifest


def snapshot(registry: Path, watchlist_path: Path, signal_date: str, asof: str) -> dict:
    manifest = validate_freeze(registry)
    date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    if date <= manifest["data_end"]:
        raise ValueError("snapshot date must be later than frozen label data end")
    if any(event.get("event_type") == "SIGNAL_SNAPSHOT"
           and event.get("freeze_id") == manifest["freeze_id"]
           and event.get("signal_date") == date
           for event in read_events(registry / "ledger.jsonl")):
        raise ValueError("duplicate forward snapshot date")
    rows = pd.read_csv(watchlist_path, dtype={"code": str})
    required = {
        "signal_date", "code", "name", "theme_research_gate", "close_quality_gate",
        "best_theme_lane", "best_theme_name", "community_probability_not_calibrated",
        "entry_authorized",
    }
    if required - set(rows):
        raise ValueError(f"watchlist fields missing: {sorted(required - set(rows))}")
    if set(pd.to_datetime(rows.signal_date).dt.strftime("%Y-%m-%d")) != {date}:
        raise ValueError("watchlist signal date mismatch")
    if rows.entry_authorized.astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("shadow input must not authorize entry")
    eligible = rows[
        rows.theme_research_gate.astype(str).str.lower().isin({"true", "1"})
        & rows.close_quality_gate.astype(str).str.lower().isin({"true", "1"})
    ].sort_values(["community_probability_not_calibrated", "code"], ascending=[False, True])
    selected = eligible.head(1).copy()
    candidate = None
    if len(selected):
        row = selected.iloc[0]
        candidate = {
            "code": str(row.code).zfill(6), "name": str(row["name"]),
            "theme_name": str(row.best_theme_name), "theme_lane": str(row.best_theme_lane),
            "research_score_not_probability": float(row.community_probability_not_calibrated),
            "decision": "SHADOW_TOP1_REQUIRES_NEXT_SESSION_CONFIRMATION",
            "replacement_after_cancel": False, "entry_authorized": False,
        }
    document = {
        "version": VERSION, "freeze_id": manifest["freeze_id"],
        "signal_date": date, "asof": asof,
        "watchlist_sha256": sha256(watchlist_path),
        "eligible_count": int(len(eligible)), "candidate": candidate,
        "entry_authorized": False,
    }
    target = registry / "snapshots" / manifest["freeze_id"] / f"{date}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("snapshot artifact already exists")
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    event = append_event(registry / "ledger.jsonl", {
        "event_type": "SIGNAL_SNAPSHOT", "freeze_id": manifest["freeze_id"],
        "signal_date": date, "asof": asof,
        "snapshot_file": str(target.relative_to(registry)),
        "snapshot_sha256": sha256(target), "candidate_count": int(candidate is not None),
        "entry_authorized": False,
    })
    return {"event_hash": event["event_hash"], "snapshot": str(target.resolve()),
            "candidate": candidate, "eligible_count": int(len(eligible))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--data-end", required=True)
    p_freeze.add_argument("--artifact", action="append", required=True, type=Path)
    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("--watchlist", required=True, type=Path)
    p_snapshot.add_argument("--signal-date", required=True)
    p_snapshot.add_argument("--asof", required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.registry, args.data_end, args.artifact)
    elif args.command == "snapshot":
        result = snapshot(args.registry, args.watchlist, args.signal_date, args.asof)
    else:
        result = {"freeze": validate_freeze(args.registry), "ledger": verify(args.registry)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
