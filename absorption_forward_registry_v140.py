"""Tamper-evident registry for all-candidate 10:00 absorption shadow records."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "data" / "v140" / "absorption_forward"
DEPENDENCIES = [ROOT / "absorption_forward_protocol_v139.py", ROOT / "live_decision_bridge_v1.py",
                ROOT / "public_minute_capture_v1.py", Path(__file__).resolve()]
GATES = {"SHADOW_PASS_ABOVE_OR_AT_VWAP", "SHADOW_VETO_BELOW_VWAP", "DATA_LIMITED"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_event(path, payload):
    events = read_events(path)
    event = {**payload, "sequence": len(events)+1,
             "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "previous_event_hash": events[-1]["event_hash"] if events else "GENESIS"}
    event["event_hash"] = hashlib.sha256(canonical(event).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(event)+"\n")
    return event


def initialize(registry):
    active, protocol_path = registry/"active_freeze.json", registry/"protocol.json"
    if active.exists():
        raise FileExistsError("active absorption freeze already exists")
    if not protocol_path.exists():
        raise FileNotFoundError("frozen protocol.json is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_RESEARCH_ONLY" or protocol.get("live_decision_effect") != "NONE_SHADOW_ONLY":
        raise ValueError("protocol is not a research-only frozen protocol")
    dependencies = [protocol_path, *DEPENDENCIES]
    manifest = {"version": "v140-absorption-forward-registry-1",
                "freeze_created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "assurance": "LOCAL_TAMPER_EVIDENT_ONLY_NO_EXTERNAL_TIMESTAMP",
                "dependency_hashes": {str(path.resolve()): digest(path) for path in dependencies},
                "entry_authorized": False, "actual_fill_claim": False}
    freeze_id = hashlib.sha256(canonical(manifest).encode()).hexdigest()[:16]
    manifest["freeze_id"] = freeze_id
    sealed = registry/f"freeze_{freeze_id}.json"
    sealed.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    active.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(registry/"ledger.jsonl", {"event_type": "FREEZE", "freeze_id": freeze_id,
                 "manifest_file": sealed.name, "manifest_sha256": digest(sealed),
                 "entry_authorized": False})
    return manifest


def verify(registry):
    errors, previous = [], "GENESIS"
    active = registry/"active_freeze.json"
    if not active.exists():
        return {"status": "FAIL", "errors": ["missing active freeze"], "event_count": 0}
    manifest = json.loads(active.read_text(encoding="utf-8"))
    for path_text, expected in manifest.get("dependency_hashes", {}).items():
        path = Path(path_text)
        if not path.exists() or digest(path) != expected:
            errors.append(f"frozen dependency changed or missing: {path}")
    events = read_events(registry/"ledger.jsonl")
    seen = set()
    for number, event in enumerate(events, 1):
        supplied = event.get("event_hash")
        body = {k: v for k, v in event.items() if k != "event_hash"}
        expected = hashlib.sha256(canonical(body).encode()).hexdigest()
        if event.get("sequence") != number or event.get("previous_event_hash") != previous or supplied != expected:
            errors.append(f"ledger chain/hash mismatch at {number}")
        if event.get("event_type") == "TEN_OCLOCK_SNAPSHOT":
            key = (event.get("trade_date"), event.get("asof"))
            if key in seen:
                errors.append(f"duplicate snapshot key: {key}")
            seen.add(key)
            path = registry/event.get("snapshot_file", "")
            if not path.exists() or digest(path) != event.get("snapshot_sha256"):
                errors.append(f"snapshot missing or changed at {number}")
        previous = supplied
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "event_count": len(events), "snapshot_count": len(seen), "last_event_hash": previous}


def snapshot_decision(registry, decision_path, recorded_at=None):
    audit = verify(registry)
    if audit["status"] != "PASS":
        raise ValueError(f"registry verification failed: {audit['errors']}")
    manifest = json.loads((registry/"active_freeze.json").read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    now = (datetime.now(ZoneInfo("Asia/Shanghai")) if recorded_at is None else
           datetime.fromisoformat(str(recorded_at)).astimezone(ZoneInfo("Asia/Shanghai")))
    frozen_at = datetime.fromisoformat(manifest["freeze_created_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    if decision.get("asof") != "10:00" or decision.get("direct_entry_authorized") is not False:
        raise ValueError("only non-authorizing 10:00 decisions can be registered")
    if decision.get("absorption_shadow_gate_live_effect") != "NONE_RESEARCH_ONLY":
        raise ValueError("decision must preserve shadow-only behavior")
    rows = list(decision.get("first_board_candidates", []))+list(decision.get("relay_candidates", []))
    codes = [str(row.get("code", "")).zfill(6) for row in rows]
    if not rows or len(codes) != len(set(codes)):
        raise ValueError("candidate set is empty or duplicated")
    candidates = []
    for row, code in zip(rows, codes):
        gate = row.get("absorption_shadow_gate_v139")
        if gate not in GATES:
            raise ValueError(f"candidate lacks final 10:00 shadow gate: {code}")
        candidates.append({"code": code, "name": row.get("name"), "boards": row.get("boards"),
                           "board_group": row.get("board_group"), "minute_data_status": row.get("minute_data_status"),
                           "close_to_vwap_proxy": row.get("close_to_vwap_proxy"), "shadow_gate": gate,
                           "actual_fill_verified": False, "complete_trade_result": None})
    trade_date = str(decision.get("trade_date"))
    signal_date = str(decision.get("signal_date"))
    if (now.strftime("%Y-%m-%d") != trade_date or now.strftime("%H:%M") < "10:00"
            or now <= frozen_at or signal_date >= trade_date):
        raise ValueError("snapshot is not a same-day post-10:00 forward record")
    source_modified = datetime.fromtimestamp(decision_path.stat().st_mtime, ZoneInfo("Asia/Shanghai"))
    # Allow only filesystem timestamp granularity around the freeze write.
    if source_modified < frozen_at - timedelta(seconds=2):
        raise ValueError("decision file predates the active freeze")
    duplicate = any(event.get("event_type") == "TEN_OCLOCK_SNAPSHOT" and event.get("trade_date") == trade_date
                    for event in read_events(registry/"ledger.jsonl"))
    if duplicate:
        raise ValueError("duplicate trade-date 10:00 snapshot")
    target = registry/"snapshots"/manifest["freeze_id"]
    target.mkdir(parents=True, exist_ok=True)
    path = target/f"{trade_date}_1000.json"
    if path.exists():
        raise FileExistsError("snapshot file already exists")
    document = {"freeze_id": manifest["freeze_id"], "signal_date": signal_date,
                "trade_date": trade_date, "asof": "10:00", "source_decision": str(decision_path.resolve()),
                "source_decision_sha256": digest(decision_path), "candidates": candidates,
                "entry_authorized": False, "outcomes_available_at_snapshot": False,
                "forward_recorded_at": now.isoformat()}
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    event = append_event(registry/"ledger.jsonl", {"event_type": "TEN_OCLOCK_SNAPSHOT",
                 "freeze_id": manifest["freeze_id"], "trade_date": trade_date, "asof": "10:00",
                 "snapshot_file": str(path.relative_to(registry)), "snapshot_sha256": digest(path),
                 "candidate_count": len(candidates),
                 "first_board_count": sum(row["board_group"] == "FIRST_BOARD" for row in candidates),
                 "relay_count": sum(row["board_group"] == "CONSECUTIVE_BOARD" for row in candidates),
                 "entry_authorized": False})
    return {"event_hash": event["event_hash"], "snapshot": str(path.resolve()),
            "candidate_count": len(candidates), "entry_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    capture = sub.add_parser("snapshot"); capture.add_argument("--decision", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    result = initialize(args.registry) if args.command == "init" else snapshot_decision(
        args.registry, args.decision) if args.command == "snapshot" else verify(args.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
