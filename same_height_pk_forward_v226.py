"""Tamper-evident forward registry for the V225 same-height PK challenger."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v226-same-height-pk-forward-1"
TZ = ZoneInfo("Asia/Shanghai")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _records_without_nan(frame: pd.DataFrame) -> list[dict]:
    records = []
    for item in frame.to_dict("records"):
        records.append({key: (None if pd.isna(value) else value) for key, value in item.items()})
    return records


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(registry: Path, event: dict) -> dict:
    ledger = registry / "ledger.jsonl"
    events = _read_events(ledger)
    body = dict(event)
    body["sequence"] = len(events) + 1
    body["previous_event_hash"] = events[-1]["event_hash"] if events else "GENESIS"
    body["event_hash"] = hashlib.sha256(_canonical(body)).hexdigest()
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
    return body


def freeze(registry_path: str, script_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    if registry.exists():
        raise FileExistsError("refusing to overwrite V226 registry")
    registry.mkdir(parents=True)
    script, report = Path(script_path), Path(report_path)
    report_obj = json.loads(report.read_text(encoding="utf-8-sig"))
    if report_obj.get("version") != "v225-same-height-leader-pk-1":
        raise ValueError("V225 report version mismatch")
    manifest = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(),
        "policy": {
            "scope": "SAME_HEIGHT_RELAY_PK_CHALLENGER_ONLY",
            "case_derived_weights_frozen": True,
            "minimum_forward_date_clusters": 60,
            "targets": ["next_session_promotion", "B_plus_2_executable_payoff", "B_plus_1_big_loss"],
            "replacement_after_failure": False,
            "official_champion_claimed": False, "entry_authorized": False,
        },
        "artifact_hashes": {str(script.resolve()): _sha(script), str(report.resolve()): _sha(report)},
    }
    freeze_id = hashlib.sha256(_canonical(manifest)).hexdigest()[:16]
    manifest["freeze_id"] = freeze_id
    manifest_path = registry / f"freeze_{freeze_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (registry / "active_freeze.json").write_text(
        json.dumps({"freeze_id": freeze_id, "manifest": manifest_path.name}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    event = {
        "event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
        "manifest_file": manifest_path.name, "manifest_sha256": _sha(manifest_path),
        "recorded_at": datetime.now(TZ).isoformat(),
    }
    _append(registry, event)
    return manifest


def _active(registry: Path) -> tuple[str, Path]:
    active = json.loads((registry / "active_freeze.json").read_text(encoding="utf-8"))
    return active["freeze_id"], registry / active["manifest"]


def snapshot(registry_path: str, scores_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    freeze_id, manifest_path = _active(registry)
    scores_file, report_file = Path(scores_path), Path(report_path)
    scores = pd.read_csv(scores_file, dtype={"code": str})
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v225-same-height-leader-pk-1":
        raise ValueError("V225 report version mismatch")
    signal_date = str(report["signal_date"])
    groups = []
    for item in report.get("group_descriptive_leaders", []):
        members = _records_without_nan(scores[scores.height_group_id.eq(item["height_group_id"])][
            ["code", "name", "boards", "research_score_NOT_PROBABILITY",
             "board_path_status", "announcement_risk_tier", "symbolic_tags",
             "missing_evidence"]])
        groups.append({**item, "members": members})
    payload = {
        "version": VERSION, "freeze_id": freeze_id, "signal_date": signal_date,
        "frozen_at": datetime.now(TZ).isoformat(), "groups": groups,
        "source_hashes": {str(scores_file.resolve()): _sha(scores_file),
                          str(report_file.resolve()): _sha(report_file)},
        "official_champion_claimed": False, "entry_authorized": False,
    }
    target = registry / "snapshots" / freeze_id / f"{signal_date}.json"
    if target.exists():
        raise FileExistsError("daily V226 snapshot already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _append(registry, {
        "event_type": "SIGNAL_SNAPSHOT", "freeze_id": freeze_id,
        "signal_date": signal_date, "snapshot_file": str(target.relative_to(registry)),
        "snapshot_sha256": _sha(target), "recorded_at": datetime.now(TZ).isoformat(),
    })
    return payload


def settle(registry_path: str, snapshot_path: str, next_pool_path: str,
           next_manifest_path: str) -> dict:
    registry, source = Path(registry_path), Path(snapshot_path)
    pool_path, manifest_path = Path(next_pool_path), Path(next_manifest_path)
    snap = json.loads(source.read_text(encoding="utf-8-sig"))
    pool = pd.read_csv(pool_path, dtype={"code": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    complete = manifest.get("full_market_completeness") == "verified"
    closing = pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    rows = []
    for group in snap["groups"]:
        for member in group["members"]:
            code = str(member["code"]).zfill(6)
            found = closing[closing.code.eq(code)]
            promoted = bool((pd.to_numeric(found.boards, errors="coerce") >= int(member["boards"]) + 1).any())
            rows.append({
                "height_group_id": group["height_group_id"], "code": code,
                "name": member["name"], "frozen_descriptive_leader": code == group["descriptive_leader_code"],
                "promotion_outcome": promoted if complete else None,
                "outcome_status": "VERIFIED_FULL_DENOMINATOR" if complete else "UNAVAILABLE_INCOMPLETE_POOL",
            })
    outcome = pd.DataFrame(rows)
    report = {
        "version": VERSION, "freeze_id": snap["freeze_id"], "signal_date": snap["signal_date"],
        "status": "PROMOTION_SETTLED" if complete else "PENDING_COMPLETE_NEXT_SESSION_POOL",
        "promotion_rows": outcome.to_dict("records"),
        "B_plus_2_payoff_status": "PENDING_SEPARATE_EXECUTION_EVIDENCE",
        "B_plus_1_big_loss_status": "PENDING_SEPARATE_EXECUTION_EVIDENCE",
        "source_hashes": {str(source.resolve()): _sha(source), str(pool_path.resolve()): _sha(pool_path),
                          str(manifest_path.resolve()): _sha(manifest_path)},
        "entry_authorized": False,
    }
    target = registry / "settlements" / snap["freeze_id"] / f"{snap['signal_date']}.json"
    if target.exists():
        raise FileExistsError("V226 settlement already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _append(registry, {
        "event_type": "PROMOTION_SETTLEMENT", "freeze_id": snap["freeze_id"],
        "signal_date": snap["signal_date"], "settlement_file": str(target.relative_to(registry)),
        "settlement_sha256": _sha(target), "status": report["status"],
        "recorded_at": datetime.now(TZ).isoformat(),
    })
    return report


def verify(registry_path: str) -> dict:
    registry = Path(registry_path)
    events = _read_events(registry / "ledger.jsonl")
    errors, previous = [], "GENESIS"
    for expected, event in enumerate(events, start=1):
        supplied = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("sequence") != expected:
            errors.append(f"sequence mismatch at {expected}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"chain mismatch at {expected}")
        if hashlib.sha256(_canonical(body)).hexdigest() != supplied:
            errors.append(f"event hash mismatch at {expected}")
        for file_key, hash_key in (("manifest_file", "manifest_sha256"),
                                   ("snapshot_file", "snapshot_sha256"),
                                   ("settlement_file", "settlement_sha256")):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--script", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("--scores", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("settle"); p.add_argument("--snapshot", required=True); p.add_argument("--next-pool", required=True); p.add_argument("--next-manifest", required=True)
    sub.add_parser("verify")
    ns = ap.parse_args()
    if ns.command == "freeze": result = freeze(ns.registry, ns.script, ns.report)
    elif ns.command == "snapshot": result = snapshot(ns.registry, ns.scores, ns.report)
    elif ns.command == "settle": result = settle(ns.registry, ns.snapshot, ns.next_pool, ns.next_manifest)
    else: result = verify(ns.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
