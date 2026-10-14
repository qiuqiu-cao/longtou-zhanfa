"""Tamper-evident forward registry for the V229 first-board lane duel."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v230-first-board-multilane-forward-1"
TZ = ZoneInfo("Asia/Shanghai")
ROLE_KEYS = (
    "narrative_lane_candidate", "symbolic_pure_buyable_pioneer",
    "symbolic_dual_identity_candidate",
)


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


def freeze(registry_path: str, script_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    if registry.exists():
        raise FileExistsError("refusing to overwrite V230 registry")
    script, report_file = Path(script_path), Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v229-first-board-multilane-identity-1":
        raise ValueError("V229 report version mismatch")
    registry.mkdir(parents=True)
    manifest = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(),
        "policy": {
            "roles_compared": list(ROLE_KEYS), "case_derived_rules_frozen": True,
            "replacement_after_failure": False,
            "targets": ["next_session_promotion", "B_plus_2_executable_payoff",
                        "B_plus_1_big_loss"],
            "minimum_forward_date_clusters": 60,
            "official_first_board_primary": None, "entry_authorized": False,
        },
        "artifact_hashes": {
            str(script.resolve()): _sha(script), str(report_file.resolve()): _sha(report_file),
        },
    }
    freeze_id = hashlib.sha256(_canonical(manifest)).hexdigest()[:16]
    manifest["freeze_id"] = freeze_id
    manifest_path = registry / f"freeze_{freeze_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (registry / "active_freeze.json").write_text(
        json.dumps({"freeze_id": freeze_id, "manifest": manifest_path.name},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _append(registry, {
        "event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
        "manifest_file": manifest_path.name, "manifest_sha256": _sha(manifest_path),
    })
    return manifest


def _active(registry: Path) -> tuple[str, Path]:
    active = json.loads((registry / "active_freeze.json").read_text(encoding="utf-8"))
    return active["freeze_id"], registry / active["manifest"]


def snapshot(registry_path: str, candidates_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    freeze_id, _ = _active(registry)
    candidate_file, report_file = Path(candidates_path), Path(report_path)
    candidates = pd.read_csv(candidate_file, dtype={"code": str})
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v229-first-board-multilane-identity-1":
        raise ValueError("V229 report version mismatch")
    roles = []
    for role in ROLE_KEYS:
        selected = report.get(role)
        if not selected:
            roles.append({"role": role, "candidate": None})
            continue
        code = str(selected["code"]).zfill(6)
        found = candidates[candidates.code.astype(str).str.zfill(6).eq(code)]
        if len(found) != 1:
            raise ValueError(f"role candidate missing or duplicated: {role}:{code}")
        row = found.iloc[0]
        roles.append({
            "role": role,
            "candidate": {
                "code": code, "name": row["name"],
                "primary_business_theme": row.get("primary_business_theme"),
                "symbolic_theme_name": row.get("symbolic_theme_name"),
                "multilane_role": row.get("multilane_role"),
                "first_seal_minute": None if pd.isna(row.get("first_seal_minute"))
                else float(row.get("first_seal_minute")),
                "entry_authorized": False,
            },
        })
    payload = {
        "version": VERSION, "freeze_id": freeze_id,
        "signal_date": str(report["signal_date"]), "frozen_at": datetime.now(TZ).isoformat(),
        "role_candidates": roles,
        "all_first_board_candidates_hash": _sha(candidate_file),
        "source_hashes": {str(candidate_file.resolve()): _sha(candidate_file),
                          str(report_file.resolve()): _sha(report_file)},
        "official_first_board_primary": None, "entry_authorized": False,
    }
    target = registry / "snapshots" / freeze_id / f"{payload['signal_date']}.json"
    if target.exists():
        raise FileExistsError("V230 daily snapshot already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    _append(registry, {
        "event_type": "SIGNAL_SNAPSHOT", "freeze_id": freeze_id,
        "signal_date": payload["signal_date"],
        "snapshot_file": str(target.relative_to(registry)),
        "snapshot_sha256": _sha(target),
    })
    return payload


def settle(registry_path: str, snapshot_path: str, next_pool_path: str,
           next_manifest_path: str) -> dict:
    registry = Path(registry_path)
    source, pool_file, manifest_file = (Path(snapshot_path), Path(next_pool_path),
                                        Path(next_manifest_path))
    snap = json.loads(source.read_text(encoding="utf-8-sig"))
    pool = pd.read_csv(pool_file, dtype={"code": str})
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    complete = manifest.get("full_market_completeness") == "verified"
    closing = pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    outcomes = []
    for role in snap["role_candidates"]:
        candidate = role.get("candidate")
        if not candidate:
            outcomes.append({"role": role["role"], "code": None,
                             "promotion_outcome": None, "outcome_status": "NO_CANDIDATE"})
            continue
        code = str(candidate["code"]).zfill(6)
        found = closing[closing.code.eq(code)]
        promoted = bool((pd.to_numeric(found.boards, errors="coerce") >= 2).any())
        outcomes.append({
            "role": role["role"], "code": code, "name": candidate["name"],
            "promotion_outcome": promoted if complete else None,
            "outcome_status": ("VERIFIED_FULL_DENOMINATOR" if complete
                               else "UNAVAILABLE_INCOMPLETE_POOL"),
        })
    report = {
        "version": VERSION, "freeze_id": snap["freeze_id"],
        "signal_date": snap["signal_date"],
        "status": "PROMOTION_SETTLED" if complete else "PENDING_COMPLETE_NEXT_SESSION_POOL",
        "role_outcomes": outcomes,
        "B_plus_2_payoff_status": "PENDING_SEPARATE_EXECUTION_EVIDENCE",
        "B_plus_1_big_loss_status": "PENDING_SEPARATE_EXECUTION_EVIDENCE",
        "source_hashes": {str(source.resolve()): _sha(source),
                          str(pool_file.resolve()): _sha(pool_file),
                          str(manifest_file.resolve()): _sha(manifest_file)},
        "entry_authorized": False,
    }
    target = registry / "settlements" / snap["freeze_id"] / f"{snap['signal_date']}.json"
    if target.exists():
        raise FileExistsError("V230 settlement already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    _append(registry, {
        "event_type": "PROMOTION_SETTLEMENT", "freeze_id": snap["freeze_id"],
        "signal_date": snap["signal_date"],
        "settlement_file": str(target.relative_to(registry)),
        "settlement_sha256": _sha(target), "status": report["status"],
    })
    return report


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
    p = sub.add_parser("snapshot"); p.add_argument("--candidates", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("settle"); p.add_argument("--snapshot", required=True); p.add_argument("--next-pool", required=True); p.add_argument("--next-manifest", required=True)
    sub.add_parser("verify")
    ns = ap.parse_args()
    if ns.command == "freeze": result = freeze(ns.registry, ns.script, ns.report)
    elif ns.command == "snapshot": result = snapshot(ns.registry, ns.candidates, ns.report)
    elif ns.command == "settle": result = settle(ns.registry, ns.snapshot, ns.next_pool, ns.next_manifest)
    else: result = verify(ns.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
