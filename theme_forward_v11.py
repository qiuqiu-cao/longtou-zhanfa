#!/usr/bin/env python3
"""Freeze point-in-time theme evidence against a V10 shadow plan.

The binding is intentionally separate from the already frozen V10 plan.  It can
only make the plan more conservative: an unverified, broad, or undersized theme
roster cannot be upgraded later by adding hindsight members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results" / "v11" / "theme_forward"
VERSION = "v11-theme-forward-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def binding_hash(binding: dict[str, Any]) -> str:
    payload = {key: value for key, value in binding.items() if key != "binding_hash"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate_bind_time(plan: dict[str, Any], current: datetime) -> None:
    local = current.astimezone(ZoneInfo("Asia/Shanghai"))
    signal = pd.Timestamp(plan["signal_date"]).date()
    trade = pd.Timestamp(plan["trade_date"]).date()
    if local.date() < signal or (local.date() == signal and local.time() < datetime.strptime("15:05", "%H:%M").time()):
        raise RuntimeError("theme binding is allowed only after signal close")
    if local.date() > trade or (local.date() == trade and local.time() >= datetime.strptime("09:15", "%H:%M").time()):
        raise RuntimeError("theme binding must be frozen before trade day 09:15")


def evaluate_theme(plan: dict[str, Any], roster: pd.DataFrame, roles: pd.DataFrame) -> dict[str, Any]:
    code = str(plan.get("candidate_code", "")).zfill(6)
    signal = str(plan["signal_date"])
    role_rows = roles[
        roles["date"].astype(str).eq(signal) & roles["code"].astype(str).str.zfill(6).eq(code)
    ]
    if len(role_rows) != 1:
        return {
            "theme_decision": "UNVERIFIED_THEME_REJECT",
            "theme_group_id": "", "theme": "", "theme_role": "UNVERIFIED",
            "theme_status": "unavailable", "theme_ladder_verified": False,
            "expected_peer_count": 0, "required_peer_count": 2, "required_peers": [],
            "reason": "candidate has no unique point-in-time role row",
        }
    role = role_rows.iloc[0]
    group_id = str(role.get("theme_group_id", "")).strip()
    status = str(role.get("theme_status", "")).strip()
    specific = truthy(role.get("theme_specific_enough", False))
    group = roster[
        roster["signal_date"].astype(str).eq(signal)
        & roster["theme_group_id"].astype(str).eq(group_id)
    ].copy() if group_id else roster.iloc[0:0].copy()
    if not group.empty:
        group["code"] = group["code"].astype(str).str.zfill(6)
    peers = group[~group["code"].eq(code)] if "code" in group else group
    peer_rows = [
        {"code": str(row.code).zfill(6), "name": str(row.name), "boards": int(float(row.boards))}
        for row in peers.sort_values(["boards", "code"], ascending=[False, True]).itertuples(index=False)
    ]
    if status == "verified_same_day_broad_group" or not specific and group_id:
        decision = "BROAD_THEME_REJECT"
        reason = "broad theme cannot supply positive identity or open-sync evidence"
    elif status != "verified_same_day_group":
        decision = "UNVERIFIED_THEME_REJECT"
        reason = "candidate lacks usable same-day point-in-time theme membership"
    elif len(peer_rows) < 2:
        decision = "INSUFFICIENT_PEER_ROSTER"
        reason = "verified theme has fewer than two observable peers"
    else:
        decision = "THEME_SYNC_ELIGIBLE"
        reason = "at least two point-in-time verified peers are frozen for open confirmation"
    return {
        "theme_decision": decision,
        "theme_group_id": group_id,
        "theme": str(role.get("theme", "")),
        "theme_role": str(role.get("theme_role", "UNVERIFIED")),
        "theme_status": status,
        "theme_ladder_verified": truthy(role.get("theme_ladder_verified", False)),
        "expected_peer_count": len(peer_rows),
        "required_peer_count": 2,
        "required_peers": peer_rows,
        "reason": reason,
    }


def write_capture_template(path: Path, plan: dict[str, Any], evaluation: dict[str, Any]) -> None:
    rows = []
    for asof in ("09:35", "10:00"):
        for peer in evaluation["required_peers"]:
            rows.append({
                "signal_date": plan["signal_date"], "trade_date": plan["trade_date"], "asof": asof,
                "theme_group_id": evaluation["theme_group_id"], "member_code": peer["code"],
                "member_name": peer["name"], "supportive": "", "source": "",
                "source_timestamp": "", "data_status": "",
            })
    columns = [
        "signal_date", "trade_date", "asof", "theme_group_id", "member_code", "member_name",
        "supportive", "source", "source_timestamp", "data_status",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def bind_theme(
    plan_path: Path,
    roster_path: Path,
    roles_path: Path,
    output_root: Path = OUTPUT_ROOT,
    current: datetime | None = None,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    current = current or datetime.now(ZoneInfo("Asia/Shanghai"))
    validate_bind_time(plan, current)
    session_id = str(plan["session_id"])
    output_dir = output_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "theme_binding.json"
    if output_path.exists():
        raise RuntimeError("theme binding is immutable and already exists")
    roster = pd.read_csv(roster_path, dtype={"code": "string"})
    roles = pd.read_csv(roles_path, dtype={"code": "string"})
    evaluation = evaluate_theme(plan, roster, roles)
    binding = {
        "version": VERSION,
        "session_id": session_id,
        "model_lock_id": str(plan["model_lock_id"]),
        "signal_date": str(plan["signal_date"]),
        "trade_date": str(plan["trade_date"]),
        "candidate_code": str(plan.get("candidate_code", "")).zfill(6),
        "candidate_name": str(plan.get("candidate_name", "")),
        "bound_at": current.astimezone(ZoneInfo("Asia/Shanghai")).replace(microsecond=0).isoformat(),
        "plan_path": str(plan_path.resolve()), "plan_sha256": sha256_file(plan_path),
        "roster_path": str(roster_path.resolve()), "roster_sha256": sha256_file(roster_path),
        "roles_path": str(roles_path.resolve()), "roles_sha256": sha256_file(roles_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        **evaluation,
        "assurance": "LOCAL_TAMPER_EVIDENT_PREOPEN_BINDING",
    }
    binding["binding_hash"] = binding_hash(binding)
    output_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding="utf-8")
    template_path = output_dir / "theme_member_open_template.csv"
    write_capture_template(template_path, plan, evaluation)
    return {"binding": binding, "binding_path": str(output_path.resolve()), "capture_template": str(template_path.resolve())}


def verify_binding(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("binding_hash") != binding_hash(binding):
        errors.append("binding hash mismatch")
    for prefix in ("plan", "roster", "roles"):
        artifact = Path(binding[f"{prefix}_path"])
        if not artifact.exists():
            errors.append(f"{prefix} artifact missing")
        elif sha256_file(artifact) != binding[f"{prefix}_sha256"]:
            errors.append(f"{prefix} artifact hash mismatch")
    script_path = Path(__file__).resolve()
    if sha256_file(script_path) != binding.get("script_sha256"):
        errors.append("binding script hash mismatch")
    if not errors:
        plan = json.loads(Path(binding["plan_path"]).read_text(encoding="utf-8"))
        roster = pd.read_csv(binding["roster_path"], dtype={"code": "string"})
        roles = pd.read_csv(binding["roles_path"], dtype={"code": "string"})
        expected = evaluate_theme(plan, roster, roles)
        for key, value in expected.items():
            if binding.get(key) != value:
                errors.append(f"recomputed field mismatch: {key}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "version": binding.get("version"),
        "session_id": binding.get("session_id"),
        "theme_decision": binding.get("theme_decision"),
        "binding_hash": binding.get("binding_hash"),
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind point-in-time theme evidence to a V10 plan")
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--plan", type=Path, required=True)
    bind.add_argument("--roster", type=Path, required=True)
    bind.add_argument("--roles", type=Path, required=True)
    bind.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    verify = sub.add_parser("verify")
    verify.add_argument("binding", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "bind":
        result = bind_theme(args.plan, args.roster, args.roles, args.output)
    else:
        result = verify_binding(args.binding)
    print(json.dumps(result, ensure_ascii=False, indent=2))
