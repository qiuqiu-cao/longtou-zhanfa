"""Tamper-evident forward registry for V242/V243 path expectations."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v244-relay-path-forward-1"
TZ = ZoneInfo("Asia/Shanghai")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _events(registry: Path) -> list[dict]:
    path = registry / "ledger.jsonl"
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append(registry: Path, event: dict) -> dict:
    events = _events(registry); body = dict(event)
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


def freeze(registry_path: str, expectation_script_path: str,
           competition_script_path: str, baseline_report_path: str) -> dict:
    registry = Path(registry_path)
    if registry.exists(): raise FileExistsError("refusing to overwrite V244 registry")
    expectation, competition, baseline = map(
        Path, (expectation_script_path, competition_script_path, baseline_report_path))
    report = json.loads(baseline.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v242-relay-path-expectation-1":
        raise ValueError("V242 report version mismatch")
    registry.mkdir(parents=True)
    manifest = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(),
        "policy": {
            "path_conditioned_expectations": True,
            "stage_chain": ["09:25_AUCTION", "09:35_OPEN_ATTACK", "10:00_SEAL_THEME", "CLOSE"],
            "full_3_4_board_denominator_required": True,
            "community_sources_are_hypotheses_only": True,
            "systemic_retreat_not_core_switch": True,
            "co_winners_allowed": True,
            "minimum_forward_date_clusters": 60,
            "validated_probability_claimed": False,
            "entry_authorized": False,
        },
        "artifact_hashes": {str(path.resolve()): _sha(path)
                            for path in (expectation, competition, baseline)},
    }
    freeze_id = hashlib.sha256(_canonical(manifest)).hexdigest()[:16]
    manifest["freeze_id"] = freeze_id
    manifest_file = registry / f"freeze_{freeze_id}.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (registry / "active_freeze.json").write_text(
        json.dumps({"freeze_id": freeze_id, "manifest": manifest_file.name},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _append(registry, {"event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
                       "manifest_file": manifest_file.name,
                       "manifest_sha256": _sha(manifest_file)})
    return manifest


def snapshot(registry_path: str, matrix_path: str, scenarios_path: str,
             report_path: str) -> dict:
    registry = Path(registry_path); freeze_id, _ = _active(registry)
    matrix_file, scenarios_file, report_file = map(Path, (matrix_path, scenarios_path, report_path))
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v242-relay-path-expectation-1":
        raise ValueError("V242 report version mismatch")
    hashes = report.get("artifact_hashes", {})
    if hashes.get(matrix_file.name) != _sha(matrix_file) or hashes.get(scenarios_file.name) != _sha(scenarios_file):
        raise ValueError("V242 artifact hash mismatch")
    matrix = pd.read_csv(matrix_file, dtype={"code": str})
    matrix["code"] = matrix.code.astype(str).str.zfill(6)
    candidates = [{"code": row.code, "name": row["name"], "boards": int(row.boards),
                   "expectation_type": row.expectation_type,
                   "normal_gap_low": None if pd.isna(row.normal_gap_low) else float(row.normal_gap_low),
                   "normal_gap_high": None if pd.isna(row.normal_gap_high) else float(row.normal_gap_high),
                   "predeclared_role": row.predeclared_role}
                  for _, row in matrix.iterrows()]
    payload = {
        "version": VERSION, "freeze_id": freeze_id, "signal_date": str(report["signal_date"]),
        "frozen_at": datetime.now(TZ).isoformat(), "full_denominator": candidates,
        "full_denominator_count": len(candidates),
        "primary_close_expectation": report.get("primary_close_expectation"),
        "source_hashes": {str(path.resolve()): _sha(path)
                          for path in (matrix_file, scenarios_file, report_file)},
        "entry_authorized": False,
    }
    target = registry / "snapshots" / freeze_id / f"{payload['signal_date']}.json"
    if target.exists(): raise FileExistsError("V244 snapshot already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    _append(registry, {"event_type": "PATH_EXPECTATION_SNAPSHOT", "freeze_id": freeze_id,
                       "signal_date": payload["signal_date"],
                       "snapshot_file": str(target.relative_to(registry)),
                       "snapshot_sha256": _sha(target)})
    return payload


def record(registry_path: str, report_path: str) -> dict:
    registry, source = Path(registry_path), Path(report_path); freeze_id, _ = _active(registry)
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v243-relay-stage-competition-1":
        raise ValueError("V243 report version mismatch")
    target = registry / "stage_records" / freeze_id / f"{report['signal_date']}_for_{report['decision_date']}.json"
    if target.exists(): raise FileExistsError("V244 stage record already exists")
    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    return _append(registry, {"event_type": "STAGE_COMPETITION_RECORD", "freeze_id": freeze_id,
                              "signal_date": str(report["signal_date"]),
                              "decision_date": str(report["decision_date"]),
                              "decision": report.get("decision"),
                              "stage_file": str(target.relative_to(registry)),
                              "stage_sha256": _sha(target)})


def settle(registry_path: str, snapshot_path: str, next_pool_path: str,
           next_manifest_path: str) -> dict:
    registry = Path(registry_path)
    snap_file, pool_file, manifest_file = map(Path, (snapshot_path, next_pool_path, next_manifest_path))
    snap = json.loads(snap_file.read_text(encoding="utf-8-sig"))
    pool = pd.read_csv(pool_file, dtype={"code": str})
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    complete = manifest.get("full_market_completeness") == "verified"
    closing = pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    outcomes = []
    for candidate in snap["full_denominator"]:
        found = closing[closing.code.eq(candidate["code"])]
        promoted = bool((pd.to_numeric(found.boards, errors="coerce")
                         >= int(candidate["boards"]) + 1).any())
        outcomes.append({**candidate, "promotion_outcome": promoted if complete else None,
                         "outcome_status": ("VERIFIED_FULL_DENOMINATOR" if complete
                                            else "UNAVAILABLE_INCOMPLETE_POOL")})
    report = {
        "version": VERSION, "freeze_id": snap["freeze_id"], "signal_date": snap["signal_date"],
        "status": "SETTLED_FULL_DENOMINATOR" if complete else "PENDING_COMPLETE_NEXT_POOL",
        "candidate_outcomes": outcomes,
        "minimum_forward_date_clusters": 60,
        "validated_probability_claimed": False,
        "source_hashes": {str(path.resolve()): _sha(path)
                          for path in (snap_file, pool_file, manifest_file)},
        "entry_authorized": False,
    }
    target = registry / "settlements" / snap["freeze_id"] / f"{snap['signal_date']}.json"
    if target.exists(): raise FileExistsError("V244 settlement already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _append(registry, {"event_type": "CLOSE_SETTLEMENT", "freeze_id": snap["freeze_id"],
                       "signal_date": snap["signal_date"], "status": report["status"],
                       "settlement_file": str(target.relative_to(registry)),
                       "settlement_sha256": _sha(target)})
    return report


def verify(registry_path: str) -> dict:
    registry = Path(registry_path); events = _events(registry); errors = []; previous = "GENESIS"
    for expected, event in enumerate(events, 1):
        supplied = event.get("event_hash")
        body = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("sequence") != expected: errors.append(f"sequence mismatch at {expected}")
        if event.get("previous_event_hash") != previous: errors.append(f"chain mismatch at {expected}")
        if hashlib.sha256(_canonical(body)).hexdigest() != supplied: errors.append(f"event hash mismatch at {expected}")
        for file_key, hash_key in (("manifest_file", "manifest_sha256"),
                                   ("snapshot_file", "snapshot_sha256"),
                                   ("stage_file", "stage_sha256"),
                                   ("settlement_file", "settlement_sha256")):
            if event.get(file_key):
                path = registry / event[file_key]
                if not path.exists() or _sha(path) != event.get(hash_key):
                    errors.append(f"artifact mismatch at {expected}:{file_key}")
        previous = supplied
    if (registry / "active_freeze.json").exists():
        _, manifest_file = _active(registry)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for path, digest in manifest.get("artifact_hashes", {}).items():
            artifact = Path(path)
            if not artifact.exists() or _sha(artifact) != digest:
                errors.append(f"frozen artifact mismatch:{path}")
    return {"version": VERSION, "status": "PASS" if not errors else "FAIL",
            "events": len(events), "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--registry", required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--expectation-script", required=True); p.add_argument("--competition-script", required=True); p.add_argument("--baseline-report", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("--matrix", required=True); p.add_argument("--scenarios", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("record"); p.add_argument("--report", required=True)
    p = sub.add_parser("settle"); p.add_argument("--snapshot", required=True); p.add_argument("--next-pool", required=True); p.add_argument("--next-manifest", required=True)
    sub.add_parser("verify"); ns = ap.parse_args()
    if ns.command == "freeze": result = freeze(ns.registry, ns.expectation_script, ns.competition_script, ns.baseline_report)
    elif ns.command == "snapshot": result = snapshot(ns.registry, ns.matrix, ns.scenarios, ns.report)
    elif ns.command == "record": result = record(ns.registry, ns.report)
    elif ns.command == "settle": result = settle(ns.registry, ns.snapshot, ns.next_pool, ns.next_manifest)
    else: result = verify(ns.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
