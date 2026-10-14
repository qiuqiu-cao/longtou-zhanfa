#!/usr/bin/env python3
"""Freeze and verify the once-opened V12 selective-risk blind audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "v12" / "selective_risk_gate"
MODEL_SCRIPT = ROOT / "selective_risk_gate_v12.py"
LOCK_PATH = RESULT / "blind_lock.json"
VERSION = "v12-selective-risk-lock-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def lock_id(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "lock_id"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:16]


def build_lock(result: Path = RESULT, model_script: Path = MODEL_SCRIPT, output: Path = LOCK_PATH) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError("V12 blind lock already exists and is immutable")
    card_path = result / "model_card.json"
    report_path = result / "model_card.md"
    audit_path = result / "threshold_audit.csv"
    for path in (card_path, report_path, audit_path, model_script):
        if not path.exists():
            raise FileNotFoundError(path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if not card.get("blind_opened") or not card.get("blind_reuse_prohibited"):
        raise RuntimeError("model card does not permanently seal the blind window")
    if card.get("script_sha256") != sha256_file(model_script):
        raise RuntimeError("model card script hash differs from current V12 script")
    input_path = Path(card["input"]["path"])
    if sha256_file(input_path) != card["input"]["sha256"]:
        raise RuntimeError("V12 input changed after blind audit")
    payload = {
        "version": VERSION,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": card["status"],
        "blind_opened": True,
        "blind_reuse_prohibited": True,
        "deployment_gate_passed": bool(card["deployment_gate_passed"]),
        "veto_only_gate_passed": bool(card["veto_only_gate_passed"]),
        "files": {
            "model_script": {"path": str(model_script.resolve()), "sha256": sha256_file(model_script)},
            "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
            "model_card": {"path": str(card_path.resolve()), "sha256": sha256_file(card_path)},
            "report": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
            "threshold_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }
    payload["lock_id"] = lock_id(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def verify_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    errors: list[str] = []
    if not path.exists():
        return {"status": "FAIL", "errors": ["lock missing"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("lock_id") != lock_id(payload):
        errors.append("lock id mismatch")
    for name, artifact in payload.get("files", {}).items():
        target = Path(artifact["path"])
        if not target.exists():
            errors.append(f"{name} missing")
        elif sha256_file(target) != artifact["sha256"]:
            errors.append(f"{name} hash mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "version": payload.get("version"),
        "lock_id": payload.get("lock_id"),
        "model_status": payload.get("status"),
        "blind_reuse_prohibited": payload.get("blind_reuse_prohibited"),
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or verify V12 blind audit artifacts")
    parser.add_argument("command", choices=["lock", "verify"])
    parser.add_argument("--path", type=Path, default=LOCK_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_lock(output=args.path) if args.command == "lock" else verify_lock(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
