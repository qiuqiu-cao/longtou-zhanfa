"""Tamper-evident registry for frozen V237 relay open confirmations."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


VERSION = "v238-relay-leader-open-forward-1"
TZ = ZoneInfo("Asia/Shanghai")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _events(registry: Path) -> list[dict]:
    ledger = registry / "ledger.jsonl"
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append(registry: Path, event: dict) -> dict:
    events = _events(registry)
    body = dict(event)
    body["sequence"] = len(events) + 1
    body["previous_event_hash"] = events[-1]["event_hash"] if events else "GENESIS"
    body["recorded_at"] = datetime.now(TZ).isoformat()
    body["event_hash"] = hashlib.sha256(_canonical(body)).hexdigest()
    with (registry / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
    return body


def _active(registry: Path) -> tuple[str, Path]:
    active = json.loads((registry / "active_freeze.json").read_text(encoding="utf-8"))
    return active["freeze_id"], registry / active["manifest"]


def freeze(registry_path: str, script_path: str, baseline_report_path: str) -> dict:
    registry = Path(registry_path)
    if registry.exists():
        raise FileExistsError("refusing to overwrite V238 registry")
    script, baseline = Path(script_path), Path(baseline_report_path)
    report = json.loads(baseline.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v237-relay-leader-open-confirmation-1":
        raise ValueError("V237 report version mismatch")
    registry.mkdir(parents=True)
    manifest = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(),
        "policy": {
            "frozen_candidate_only": True, "replacement_after_failure": False,
            "first_causal_price_minute": "09:36",
            "symbolic_roster_is_real_theme": False,
            "height_anchor_break_requires_core_switch_or_cancel": True,
            "minimum_complete_forward_confirmations": 60,
            "targets": ["actual_fill", "B_plus_2_net_payoff", "B_plus_1_big_loss",
                        "maximum_drawdown"],
            "entry_authorized": False,
        },
        "artifact_hashes": {
            str(script.resolve()): _sha(script), str(baseline.resolve()): _sha(baseline),
        },
    }
    freeze_id = hashlib.sha256(_canonical(manifest)).hexdigest()[:16]
    manifest["freeze_id"] = freeze_id
    manifest_path = registry / f"freeze_{freeze_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (registry / "active_freeze.json").write_text(
        json.dumps({"freeze_id": freeze_id, "manifest": manifest_path.name},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _append(registry, {"event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
                       "manifest_file": manifest_path.name,
                       "manifest_sha256": _sha(manifest_path)})
    return manifest


def record(registry_path: str, report_path: str) -> dict:
    registry, source = Path(registry_path), Path(report_path)
    freeze_id, _ = _active(registry)
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v237-relay-leader-open-confirmation-1":
        raise ValueError("V237 report version mismatch")
    signal_date, decision_date = str(report["signal_date"]), str(report["decision_date"])
    duplicates = [event for event in _events(registry)
                  if event.get("event_type") == "OPEN_CONFIRMATION_RECORD"
                  and event.get("signal_date") == signal_date
                  and event.get("decision_date") == decision_date]
    if duplicates:
        raise FileExistsError("V238 confirmation record already exists")
    target = registry / "confirmations" / freeze_id / f"{signal_date}_for_{decision_date}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    event = _append(registry, {
        "event_type": "OPEN_CONFIRMATION_RECORD", "freeze_id": freeze_id,
        "signal_date": signal_date, "decision_date": decision_date,
        "decision": report.get("decision"),
        "confirmation_file": str(target.relative_to(registry)),
        "confirmation_sha256": _sha(target),
        "actual_fill_status": "UNAVAILABLE_UNTIL_BROKER_IMPORT",
    })
    return event


def verify(registry_path: str) -> dict:
    registry = Path(registry_path)
    events, errors, previous = _events(registry), [], "GENESIS"
    for expected, event in enumerate(events, 1):
        supplied = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("sequence") != expected:
            errors.append(f"sequence mismatch at {expected}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"chain mismatch at {expected}")
        if hashlib.sha256(_canonical(body)).hexdigest() != supplied:
            errors.append(f"event hash mismatch at {expected}")
        for file_key, hash_key in (("manifest_file", "manifest_sha256"),
                                   ("confirmation_file", "confirmation_sha256")):
            if event.get(file_key):
                path = registry / event[file_key]
                if not path.exists() or _sha(path) != event.get(hash_key):
                    errors.append(f"artifact mismatch at {expected}:{file_key}")
        previous = supplied
    if (registry / "active_freeze.json").exists():
        _, manifest_path = _active(registry)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path, digest in manifest.get("artifact_hashes", {}).items():
            artifact = Path(path)
            if not artifact.exists() or _sha(artifact) != digest:
                errors.append(f"frozen artifact mismatch:{path}")
    return {"version": VERSION, "status": "PASS" if not errors else "FAIL",
            "events": len(events), "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--script", required=True); p.add_argument("--baseline-report", required=True)
    p = sub.add_parser("record"); p.add_argument("--report", required=True)
    sub.add_parser("verify")
    ns = ap.parse_args()
    if ns.command == "freeze":
        result = freeze(ns.registry, ns.script, ns.baseline_report)
    elif ns.command == "record":
        result = record(ns.registry, ns.report)
    else:
        result = verify(ns.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
