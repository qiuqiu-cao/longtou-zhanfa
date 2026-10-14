"""Tamper-evident forward registry for the frozen first-board challenger.

The registry is intentionally separate from production decisions.  It records a
single daily SHADOW_TOP1 after the close, never backfills past dates, never
authorises an order, and preserves unfilled candidates instead of substituting a
later winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "v128" / "first_board_forward"
CODE_FILES = [
    ROOT / "first_board_promotion_v124.py",
    ROOT / "first_board_trade_outcomes_v125.py",
    ROOT / "first_board_profit_link_audit_v126.py",
    ROOT / "first_board_factor_profit_audit_v127.py",
    Path(__file__).resolve(),
]
POLICY = {
    "version": "first-board-shadow-policy-v128-1",
    "status": "DIAGNOSTIC_ONLY",
    "action_authorized": False,
    "candidate_scope": "Shanghai/Shenzhen main-board, non-ST, closing first boards",
    "daily_selection": "highest base-quality score after excluding thin_control_risk=1",
    "thin_control_rule": "WATCH_CONTROL_OR_LIQUIDITY_RISK; never substitute after selection",
    "execution_baseline": "next-session open only when below upper limit; otherwise unfilled",
    "evaluation": "B+1/B+2/B+3 net return after frozen costs; no retroactive replacement",
    "minimum_forward_fills": 60,
    "minimum_complete_trades_for_70pct_challenge": 100,
    "entry_signal": "none; shadow research only",
}


def canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_event(ledger: Path, payload: dict) -> dict:
    events = read_events(ledger)
    previous = events[-1]["event_hash"] if events else "GENESIS"
    event = {
        **payload, "sequence": len(events) + 1,
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "previous_event_hash": previous,
    }
    event["event_hash"] = hashlib.sha256(canonical(event).encode("utf-8")).hexdigest()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(event) + "\n")
    return event


def verify_ledger(ledger: Path, registry: Path) -> dict:
    errors, previous = [], "GENESIS"
    events = read_events(ledger)
    for number, event in enumerate(events, 1):
        supplied = event.get("event_hash")
        body = {k: v for k, v in event.items() if k != "event_hash"}
        expected = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()
        if event.get("sequence") != number:
            errors.append(f"sequence mismatch at event {number}")
        if event.get("previous_event_hash") != previous:
            errors.append(f"chain mismatch at event {number}")
        if supplied != expected:
            errors.append(f"event hash mismatch at event {number}")
        if event.get("snapshot_file"):
            snapshot = registry / event["snapshot_file"]
            if not snapshot.exists():
                errors.append(f"missing snapshot at event {number}")
            elif sha256_file(snapshot) != event.get("snapshot_sha256"):
                errors.append(f"snapshot hash mismatch at event {number}")
        previous = supplied
    return {"status": "PASS" if not errors else "FAIL", "event_count": len(events),
            "last_event_hash": previous, "errors": errors}


def freeze(registry: Path, data_end: str, artifacts: list[Path]) -> dict:
    active = registry / "active_freeze.json"
    ledger = registry / "ledger.jsonl"
    if active.exists():
        raise FileExistsError("active first-board freeze already exists; create a new version for changes")
    registry.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {str(path.resolve()): sha256_file(path) for path in artifacts}
    code_hashes = {str(path.resolve()): sha256_file(path) for path in CODE_FILES}
    created = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    basis = {"created_at": created, "data_end": pd.Timestamp(data_end).strftime("%Y-%m-%d"),
             "policy": POLICY, "artifact_hashes": artifact_hashes, "code_hashes": code_hashes}
    freeze_id = hashlib.sha256(canonical(basis).encode("utf-8")).hexdigest()[:16]
    manifest = {"freeze_id": freeze_id, **basis,
                "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
                "warning": "hashes are not an external trusted timestamp"}
    sealed = registry / f"freeze_{freeze_id}.json"
    sealed.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    active.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(ledger, {"event_type": "MODEL_FREEZE", "freeze_id": freeze_id,
                          "manifest_file": sealed.name, "manifest_sha256": sha256_file(sealed),
                          "data_end": manifest["data_end"]})
    return manifest


def validate_freeze(registry: Path) -> dict:
    active = registry / "active_freeze.json"
    if not active.exists():
        raise ValueError("no active first-board freeze")
    manifest = json.loads(active.read_text(encoding="utf-8"))
    for path_text, digest in {**manifest["artifact_hashes"], **manifest["code_hashes"]}.items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != digest:
            raise ValueError(f"frozen dependency changed or missing: {path}")
    sealed = registry / f"freeze_{manifest['freeze_id']}.json"
    events = [event for event in read_events(registry / "ledger.jsonl")
              if event.get("event_type") == "MODEL_FREEZE"
              and event.get("freeze_id") == manifest["freeze_id"]]
    if not sealed.exists() or not events or sha256_file(sealed) != events[-1].get("manifest_sha256"):
        raise ValueError("active freeze is not bound to its sealed ledger event")
    return manifest


def snapshot(registry: Path, scores_path: Path, signal_date: str, asof: str) -> dict:
    manifest = validate_freeze(registry)
    verification = verify_ledger(registry / "ledger.jsonl", registry)
    if verification["status"] != "PASS":
        raise ValueError("ledger verification failed")
    date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    if date <= manifest["data_end"]:
        raise ValueError("forward snapshots must be later than the frozen data end")
    if date < manifest["created_at"][:10]:
        raise ValueError("forward snapshots cannot predate the freeze")
    for event in read_events(registry / "ledger.jsonl"):
        if (event.get("event_type") == "SIGNAL_SNAPSHOT"
                and event.get("freeze_id") == manifest["freeze_id"]
                and event.get("signal_date") == date and event.get("asof") == asof):
            raise ValueError("duplicate freeze/date/asof snapshot")
    scores = pd.read_csv(scores_path, dtype={"code": str})
    required = {"code", "name", "signal_date", "thin_control_risk",
                "research_score_not_probability"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing fields: {sorted(missing)}")
    score_dates = set(pd.to_datetime(scores.signal_date).dt.strftime("%Y-%m-%d"))
    if score_dates != {date}:
        raise ValueError("scores must contain only the declared signal date")
    scores["code"] = scores.code.astype(str).str.zfill(6)
    scores = scores.sort_values(["thin_control_risk", "research_score_not_probability", "code"],
                                ascending=[True, False, True]).reset_index(drop=True)
    scores["shadow_rank"] = range(1, len(scores) + 1)
    scores["forward_decision"] = "WATCH"
    eligible = scores.index[scores.thin_control_risk.eq(0)]
    if len(eligible):
        scores.loc[eligible[0], "forward_decision"] = "SHADOW_TOP1"
    scores.loc[scores.thin_control_risk.eq(1), "forward_decision"] = "WATCH_CONTROL_RISK"
    scores["action_authorized"] = False
    snapshot_dir = registry / "snapshots" / manifest["freeze_id"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{date}_{asof.replace(':', '')}.json"
    if snapshot_path.exists():
        raise FileExistsError("snapshot path already exists")
    document = {"freeze_id": manifest["freeze_id"], "signal_date": date, "asof": asof,
                "scores_sha256": sha256_file(scores_path), "policy": manifest["policy"],
                "candidates": scores.to_dict("records")}
    snapshot_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    relative = snapshot_path.relative_to(registry)
    event = append_event(registry / "ledger.jsonl", {
        "event_type": "SIGNAL_SNAPSHOT", "freeze_id": manifest["freeze_id"],
        "signal_date": date, "asof": asof, "snapshot_file": str(relative),
        "snapshot_sha256": sha256_file(snapshot_path), "candidate_count": len(scores),
        "shadow_top1_count": int(scores.forward_decision.eq("SHADOW_TOP1").sum()),
        "action_authorized": False,
    })
    return {"event_hash": event["event_hash"], "snapshot": str(snapshot_path),
            "candidate_count": len(scores),
            "shadow_top1": scores[scores.forward_decision.eq("SHADOW_TOP1")]
            [["code", "name", "research_score_not_probability"]].to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--data-end", required=True)
    freeze_parser.add_argument("--artifact", type=Path, action="append", required=True)
    snapshot_parser = sub.add_parser("snapshot")
    snapshot_parser.add_argument("--scores", type=Path, required=True)
    snapshot_parser.add_argument("--signal-date", required=True)
    snapshot_parser.add_argument("--asof", default="15:10")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.registry, args.data_end, args.artifact)
    elif args.command == "snapshot":
        result = snapshot(args.registry, args.scores, args.signal_date, args.asof)
    else:
        result = {"freeze": validate_freeze(args.registry),
                  "ledger": verify_ledger(args.registry / "ledger.jsonl", args.registry)}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
