"""Tamper-evident forward registry for the V239/V240 3/4-board race."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


VERSION = "v241-relay-cross-theme-forward-1"
TZ = ZoneInfo("Asia/Shanghai")


def _truthy(value: object) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {
        "1", "true", "yes", "y"
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _events(registry: Path) -> list[dict]:
    path = registry / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
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


def freeze(registry_path: str, expectation_script_path: str,
           confirmation_script_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    if registry.exists():
        raise FileExistsError("refusing to overwrite V241 registry")
    expectation_script = Path(expectation_script_path)
    confirmation_script = Path(confirmation_script_path)
    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v239-relay-cross-theme-expectation-1":
        raise ValueError("V239 report version mismatch")
    registry.mkdir(parents=True)
    manifest = {
        "version": VERSION, "created_at": datetime.now(TZ).isoformat(),
        "policy": {
            "scope": "ALL_3_AND_4_BOARD_CROSS_THEME_CANDIDATES",
            "full_denominator_required": True,
            "hierarchy_frozen_before_next_session": True,
            "fallback_is_observation_not_replacement_trade": True,
            "unexpected_winner_counts_as_model_miss": True,
            "minimum_forward_date_clusters": 60,
            "higher_continuation_win_rate_claimed": False,
            "entry_authorized": False,
        },
        "artifact_hashes": {
            str(expectation_script.resolve()): _sha(expectation_script),
            str(confirmation_script.resolve()): _sha(confirmation_script),
            str(report_file.resolve()): _sha(report_file),
        },
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


def snapshot(registry_path: str, pool_path: str, tree_path: str, report_path: str) -> dict:
    registry = Path(registry_path)
    freeze_id, _ = _active(registry)
    pool_file, tree_file, report_file = Path(pool_path), Path(tree_path), Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v239-relay-cross-theme-expectation-1":
        raise ValueError("V239 report version mismatch")
    hashes = report.get("artifact_hashes", {})
    if hashes.get(pool_file.name) != _sha(pool_file) or hashes.get(tree_file.name) != _sha(tree_file):
        raise ValueError("V239 artifact hash mismatch")
    pool = pd.read_csv(pool_file, dtype={"code": str})
    tree = pd.read_csv(tree_file, dtype={"code": str})
    pool["code"] = pool.code.astype(str).str.zfill(6)
    tree["code"] = tree.code.astype(str).str.zfill(6)
    rows = []
    for _, row in pool.iterrows():
        rank = None if pd.isna(row.winner_hierarchy_rank) else int(row.winner_hierarchy_rank)
        rows.append({
            "code": row.code, "name": row["name"], "boards": int(row.boards),
            "theme_group_id": row.theme_group_id, "theme_name": row.theme_name,
            "predeclared_role": row.predeclared_role,
            "winner_hierarchy_rank": rank,
            "tradable_expectation_eligible": _truthy(row.tradable_expectation_eligible),
        })
    payload = {
        "version": VERSION, "freeze_id": freeze_id,
        "signal_date": str(report["signal_date"]),
        "frozen_at": datetime.now(TZ).isoformat(),
        "full_denominator": rows,
        "full_denominator_count": len(rows),
        "hierarchy_codes": tree.sort_values("hierarchy_rank").code.tolist(),
        "primary_expected_winner": report.get("primary_expected_winner"),
        "fallbacks": report.get("conditional_fallback_winners", []),
        "source_hashes": {str(path.resolve()): _sha(path)
                          for path in (pool_file, tree_file, report_file)},
        "entry_authorized": False,
    }
    target = registry / "snapshots" / freeze_id / f"{payload['signal_date']}.json"
    if target.exists():
        raise FileExistsError("V241 daily snapshot already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    _append(registry, {"event_type": "EXPECTATION_SNAPSHOT", "freeze_id": freeze_id,
                       "signal_date": payload["signal_date"],
                       "snapshot_file": str(target.relative_to(registry)),
                       "snapshot_sha256": _sha(target)})
    return payload


def record_confirmation(registry_path: str, report_path: str) -> dict:
    registry, source = Path(registry_path), Path(report_path)
    freeze_id, _ = _active(registry)
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    if report.get("version") != "v240-relay-cross-theme-confirmation-1":
        raise ValueError("V240 report version mismatch")
    target = registry / "confirmations" / freeze_id / (
        f"{report['signal_date']}_for_{report['decision_date']}.json")
    if target.exists():
        raise FileExistsError("V241 confirmation already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return _append(registry, {
        "event_type": "OPEN_CONFIRMATION", "freeze_id": freeze_id,
        "signal_date": str(report["signal_date"]),
        "decision_date": str(report["decision_date"]), "decision": report.get("decision"),
        "conditional_winner_code": (report.get("conditional_winner") or {}).get("code"),
        "confirmation_file": str(target.relative_to(registry)),
        "confirmation_sha256": _sha(target),
    })


def settle(registry_path: str, snapshot_path: str, next_pool_path: str,
           next_manifest_path: str, confirmation_path: str | None = None) -> dict:
    registry = Path(registry_path)
    snapshot_file, pool_file, manifest_file = (Path(snapshot_path), Path(next_pool_path),
                                                Path(next_manifest_path))
    snap = json.loads(snapshot_file.read_text(encoding="utf-8-sig"))
    next_pool = pd.read_csv(pool_file, dtype={"code": str})
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    complete = manifest.get("full_market_completeness") == "verified"
    closing = next_pool[next_pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
    closing["code"] = closing.code.astype(str).str.zfill(6)
    outcomes = []
    for item in snap["full_denominator"]:
        found = closing[closing.code.eq(str(item["code"]).zfill(6))]
        promoted = bool((pd.to_numeric(found.boards, errors="coerce")
                         >= int(item["boards"]) + 1).any())
        outcomes.append({**item,
                         "promotion_outcome": promoted if complete else None,
                         "outcome_status": ("VERIFIED_FULL_DENOMINATOR" if complete
                                            else "UNAVAILABLE_INCOMPLETE_POOL")})
    promoted_codes = [item["code"] for item in outcomes if item["promotion_outcome"] is True]
    hierarchy = set(snap["hierarchy_codes"])
    primary = str((snap.get("primary_expected_winner") or {}).get("code", "")).zfill(6)
    confirmation = None
    sources = [snapshot_file, pool_file, manifest_file]
    if confirmation_path:
        confirmation_file = Path(confirmation_path); sources.append(confirmation_file)
        confirmation = json.loads(confirmation_file.read_text(encoding="utf-8-sig"))
        if confirmation.get("version") != "v240-relay-cross-theme-confirmation-1":
            raise ValueError("V240 confirmation version mismatch")
    observed_fallback = (confirmation or {}).get("conditional_winner") or {}
    report = {
        "version": VERSION, "freeze_id": snap["freeze_id"],
        "signal_date": snap["signal_date"],
        "status": "SETTLED_FULL_DENOMINATOR" if complete else "PENDING_COMPLETE_NEXT_SESSION_POOL",
        "full_denominator_outcomes": outcomes,
        "primary_expected_winner_promoted": (primary in promoted_codes) if complete else None,
        "observed_conditional_winner": observed_fallback or None,
        "observed_conditional_winner_promoted": (
            str(observed_fallback.get("code", "")).zfill(6) in promoted_codes
            if complete and observed_fallback else None),
        "actual_promoted_identity_codes": promoted_codes if complete else [],
        "unexpected_promoted_codes": ([code for code in promoted_codes if code not in hierarchy]
                                      if complete else []),
        "tradable_and_identity_roles_separated": True,
        "higher_continuation_win_rate_claimed": False,
        "minimum_forward_date_clusters": 60,
        "source_hashes": {str(path.resolve()): _sha(path) for path in sources},
        "entry_authorized": False,
    }
    target = registry / "settlements" / snap["freeze_id"] / f"{snap['signal_date']}.json"
    if target.exists():
        raise FileExistsError("V241 settlement already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    _append(registry, {"event_type": "PROMOTION_SETTLEMENT", "freeze_id": snap["freeze_id"],
                       "signal_date": snap["signal_date"], "status": report["status"],
                       "settlement_file": str(target.relative_to(registry)),
                       "settlement_sha256": _sha(target)})
    return report


def verify(registry_path: str) -> dict:
    registry = Path(registry_path)
    errors, previous = [], "GENESIS"
    events = _events(registry)
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
                                   ("confirmation_file", "confirmation_sha256"),
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--expectation-script", required=True); p.add_argument("--confirmation-script", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("--pool", required=True); p.add_argument("--tree", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("record-confirmation"); p.add_argument("--report", required=True)
    p = sub.add_parser("settle"); p.add_argument("--snapshot", required=True); p.add_argument("--next-pool", required=True); p.add_argument("--next-manifest", required=True); p.add_argument("--confirmation")
    sub.add_parser("verify")
    ns = ap.parse_args()
    if ns.command == "freeze":
        result = freeze(ns.registry, ns.expectation_script, ns.confirmation_script, ns.report)
    elif ns.command == "snapshot":
        result = snapshot(ns.registry, ns.pool, ns.tree, ns.report)
    elif ns.command == "record-confirmation":
        result = record_confirmation(ns.registry, ns.report)
    elif ns.command == "settle":
        result = settle(ns.registry, ns.snapshot, ns.next_pool, ns.next_manifest, ns.confirmation)
    else:
        result = verify(ns.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
