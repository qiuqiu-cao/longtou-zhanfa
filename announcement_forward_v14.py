#!/usr/bin/env python3
"""Point-in-time announcement-risk binding for the next-session relay plan.

This layer can only reject or downgrade a frozen candidate.  It never creates
an entry signal.  A binding is valid only when created after the signal close
and before 09:15 on the trade date from evidence collected no later than the
binding time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

import intraday_forward_v10 as v10_forward


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "v14" / "announcement_forward"
SHANGHAI = ZoneInfo("Asia/Shanghai")
VERSION = "v14-announcement-forward-1"
COMPLETE_INDEX_STATUSES = {
    "verified_index_date_only", "verified_official_full_text", "verified_no_rows",
}
REQUIRED_ANNOUNCEMENT_FIELDS = {
    "code", "name", "published_at", "risk_type", "title", "url", "risk_hard",
    "published_at_status", "source", "source_timestamp", "data_status",
}


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=json_default,
    )


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize(SHANGHAI)
    return stamp.tz_convert(SHANGHAI)


def code6(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def validate_binding_time(plan: dict[str, Any], bound_at: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = local_timestamp(bound_at)
    signal = pd.Timestamp(plan["signal_date"]).date()
    trade = pd.Timestamp(plan["trade_date"]).date()
    earliest = local_timestamp(datetime.combine(signal, time(15, 5)))
    deadline = local_timestamp(datetime.combine(trade, time(9, 15)))
    if stamp < earliest:
        raise ValueError("announcement binding must be created after signal-date 15:05")
    if stamp >= deadline:
        raise ValueError("announcement binding must be frozen before trade-date 09:15")
    planned_at = local_timestamp(plan["planned_at"])
    if planned_at > stamp:
        raise ValueError("announcement binding predates the frozen relay plan")
    return stamp


def validate_inputs(
    plan: dict[str, Any], manifest: dict[str, Any], announcements: pd.DataFrame,
    candidates: pd.DataFrame, announcement_path: Path,
) -> None:
    missing = sorted(REQUIRED_ANNOUNCEMENT_FIELDS - set(announcements.columns))
    if missing:
        raise ValueError(f"announcement archive missing fields: {missing}")
    if str(manifest.get("signal_date")) != str(plan.get("signal_date")):
        raise ValueError("announcement manifest signal date differs from plan")
    file_record = manifest.get("files", {}).get("announcements", {})
    if file_record.get("sha256") != sha256_file(announcement_path):
        raise ValueError("announcement archive hash differs from collector manifest")
    needed_candidates = {"date", "code", "name", "boards"}
    if not needed_candidates.issubset(candidates.columns):
        raise ValueError(f"candidate file missing fields: {sorted(needed_candidates - set(candidates.columns))}")
    current = candidates[candidates["date"].astype(str).eq(str(plan["signal_date"]))].copy()
    current["code"] = current["code"].map(code6)
    if plan.get("candidate_code") and code6(plan["candidate_code"]) not in set(current["code"]):
        raise ValueError("frozen plan candidate is absent from candidate archive")


def validate_registered_v10_plan(plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    verification = v10_forward.verify()
    if verification.get("status") != "PASS":
        raise ValueError("V10 forward ledger is invalid")
    session_id = str(plan.get("session_id", ""))
    matches = [
        event for event in v10_forward.read_events()
        if event.get("event_type") == "PLAN" and str(event.get("session_id")) == session_id
    ]
    if len(matches) != 1:
        raise ValueError("announcement binding requires exactly one registered V10 plan")
    event = matches[0]
    if event.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("plan file hash differs from registered V10 plan")
    if code6(event.get("candidate_code")) != code6(plan.get("candidate_code")):
        raise ValueError("registered V10 plan candidate differs from plan file")
    return {
        "status": "PASS", "model_lock_id": event.get("model_lock_id"),
        "plan_event_hash": event.get("event_hash"), "plan_sha256": event.get("plan_sha256"),
    }


def coverage_status(
    manifest: dict[str, Any], announcements: pd.DataFrame, plan: dict[str, Any], bound_at: pd.Timestamp,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    info = manifest.get("announcements", {})
    if str(info.get("status")) not in COMPLETE_INDEX_STATUSES:
        reasons.append(f"collector status is {info.get('status', 'missing')}")
    if int(info.get("failed_codes", 0) or 0) != 0:
        reasons.append("one or more candidate announcement queries failed")
    if int(info.get("candidate_codes", 0) or 0) != int(manifest.get("candidate_count", -1) or -1):
        reasons.append("collector candidate coverage count is incomplete")
    collected_at = local_timestamp(manifest.get("collected_at"))
    if collected_at > bound_at:
        reasons.append("announcement collection occurred after the binding cutoff")
    source_times = pd.to_datetime(announcements["source_timestamp"], errors="coerce")
    if source_times.isna().any():
        reasons.append("announcement source timestamp is missing")
    else:
        localized = source_times.map(local_timestamp)
        if any(stamp > bound_at for stamp in localized):
            reasons.append("announcement rows were collected after the binding cutoff")
    candidate = code6(plan.get("candidate_code")) if plan.get("candidate_code") else ""
    observed = set(announcements["code"].map(code6))
    if candidate and candidate not in observed and str(info.get("status")) != "verified_no_rows":
        reasons.append("candidate has no row in a non-empty announcement archive")
    return not reasons, reasons


def effective_rows(announcements: pd.DataFrame, bound_at: pd.Timestamp) -> pd.DataFrame:
    frame = announcements.copy()
    frame["code"] = frame["code"].map(code6)
    localized: list[Any] = []
    for value in frame["published_at"]:
        try:
            localized.append(local_timestamp(value))
        except (TypeError, ValueError):
            localized.append(pd.NaT)
    # Keep this as object dtype so archives containing both explicit offsets and
    # naive provider timestamps can be normalized without pandas silently
    # dropping timezone information.
    frame["_published_at"] = pd.Series(localized, index=frame.index, dtype="object")
    usable = frame["_published_at"].map(lambda value: pd.notna(value) and value <= bound_at)
    return frame[usable].copy()


def serialize_risks(frame: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "code", "name", "published_at", "published_at_status", "risk_type", "title",
        "url", "source", "source_timestamp", "data_status",
    ]
    return [
        {field: (None if pd.isna(row.get(field)) else row.get(field)) for field in fields}
        for _, row in frame.sort_values(["published_at", "risk_type", "title"]).iterrows()
    ]


def evaluate(
    plan: dict[str, Any], manifest: dict[str, Any], announcements: pd.DataFrame,
    candidates: pd.DataFrame, bound_at: datetime | pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    stamp = local_timestamp(bound_at)
    complete, coverage_reasons = coverage_status(manifest, announcements, plan, stamp)
    effective = effective_rows(announcements, stamp)
    hard = effective[pd.to_numeric(effective["risk_hard"], errors="coerce").eq(1)].copy()
    candidate_code = code6(plan["candidate_code"]) if plan.get("candidate_code") else ""
    candidate_hard = hard[hard["code"].eq(candidate_code)].copy()
    if not complete:
        decision = "ANNOUNCEMENT_DATA_UNAVAILABLE"
    elif not candidate_code:
        decision = "NO_CANDIDATE"
    elif not candidate_hard.empty:
        decision = "HARD_RISK_REJECT"
    else:
        decision = "NO_HARD_RISK_FOUND"

    current = candidates[candidates["date"].astype(str).eq(str(plan["signal_date"]))].copy()
    current["code"] = current["code"].map(code6)
    current["boards"] = pd.to_numeric(current["boards"], errors="coerce")
    anchors = current[current["boards"].ge(6)][["code", "name", "boards"]].copy()
    overlay_rows: list[dict[str, Any]] = []
    for _, anchor in anchors.sort_values(["boards", "code"], ascending=[False, True]).iterrows():
        rows = hard[hard["code"].eq(anchor["code"])]
        overlay_rows.append({
            "signal_date": str(plan["signal_date"]), "code": anchor["code"],
            "name": anchor["name"], "boards": int(anchor["boards"]),
            "hard_risk_count": int(len(rows)),
            "anchor_risk_decision": "ANCHOR_HARD_RISK" if len(rows) else "NO_HARD_RISK_FOUND",
            "risk_types": "；".join(dict.fromkeys(rows["risk_type"].astype(str))),
            "titles": "；".join(rows["title"].astype(str)),
            "urls": "；".join(rows["url"].astype(str)),
        })
    overlay = pd.DataFrame(overlay_rows, columns=[
        "signal_date", "code", "name", "boards", "hard_risk_count",
        "anchor_risk_decision", "risk_types", "titles", "urls",
    ])
    collector_status = str(manifest.get("announcements", {}).get("status", ""))
    assurance = (
        "OFFICIAL_INDEX_VERIFIED_NO_ROWS"
        if collector_status == "verified_no_rows" and announcements.empty
        else "OFFICIAL_FULL_TEXT_VERIFIED"
        if not announcements.empty and announcements["data_status"].astype(str).eq("verified_official_full_text").all()
        else "INDEX_DATE_VERIFIED_FULL_TEXT_UNVERIFIED"
    )
    result = {
        "decision": decision,
        "candidate_code": candidate_code or None,
        "candidate_name": plan.get("candidate_name"),
        "candidate_hard_risk_count": int(len(candidate_hard)),
        "candidate_hard_risks": serialize_risks(candidate_hard),
        "coverage_complete": complete,
        "coverage_reasons": coverage_reasons,
        "evidence_assurance": assurance,
        "policy": "announcement layer may only reject or downgrade; it can never create ENTER",
    }
    return result, candidate_hard, overlay


def bind(
    plan_path: Path, manifest_path: Path, announcement_path: Path, candidate_path: Path,
    output: Path = DEFAULT_OUTPUT, now: datetime | pd.Timestamp | None = None,
    require_registered_v10: bool = True,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    announcements = pd.read_csv(announcement_path, dtype={"code": "string"})
    candidates = pd.read_csv(candidate_path, dtype={"code": "string"})
    validate_inputs(plan, manifest, announcements, candidates, announcement_path)
    v10_registration = (
        validate_registered_v10_plan(plan_path, plan)
        if require_registered_v10 else {"status": "TEST_ONLY_UNREGISTERED"}
    )
    stamp = validate_binding_time(plan, now or datetime.now(SHANGHAI))
    evaluation, candidate_hard, overlay = evaluate(plan, manifest, announcements, candidates, stamp)
    session_id = str(plan["session_id"])
    target = output / session_id
    if target.exists():
        raise FileExistsError("immutable announcement binding already exists for session")
    target.mkdir(parents=True)
    frozen = {
        "plan": target / "frozen_plan.json",
        "manifest": target / "frozen_announcement_manifest.json",
        "announcements": target / "frozen_announcements.csv",
        "candidates": target / "frozen_candidates.csv",
    }
    for source, destination in (
        (plan_path, frozen["plan"]), (manifest_path, frozen["manifest"]),
        (announcement_path, frozen["announcements"]), (candidate_path, frozen["candidates"]),
    ):
        shutil.copy2(source, destination)
    candidate_hard.drop(columns=["_published_at"], errors="ignore").to_csv(
        target / "candidate_hard_risks.csv", index=False, encoding="utf-8-sig"
    )
    overlay.to_csv(target / "anchor_risk_overlay.csv", index=False, encoding="utf-8-sig")
    file_hashes = {key: sha256_file(path) for key, path in frozen.items()}
    file_hashes["candidate_hard_risks"] = sha256_file(target / "candidate_hard_risks.csv")
    file_hashes["anchor_risk_overlay"] = sha256_file(target / "anchor_risk_overlay.csv")
    body = {
        "version": VERSION, "session_id": session_id,
        "signal_date": plan["signal_date"], "trade_date": plan["trade_date"],
        "bound_at": stamp.isoformat(), "plan_sha256": sha256_file(plan_path),
        "v10_plan_verification": v10_registration,
        "source_file_hashes": file_hashes,
        "forward_script_sha256": sha256_file(Path(__file__)),
        **evaluation,
    }
    body["binding_hash"] = hash_value(body)
    (target / "announcement_binding.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    return body


def verify_one(target: Path) -> list[str]:
    errors: list[str] = []
    binding_path = target / "announcement_binding.json"
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{target.name}: unreadable binding ({exc})"]
    claimed = binding.get("binding_hash")
    body = {key: value for key, value in binding.items() if key != "binding_hash"}
    if claimed != hash_value(body):
        errors.append(f"{target.name}: binding content hash mismatch")
    if body.get("forward_script_sha256") != sha256_file(Path(__file__)):
        errors.append(f"{target.name}: forward script hash mismatch")
    paths = {
        "plan": target / "frozen_plan.json",
        "manifest": target / "frozen_announcement_manifest.json",
        "announcements": target / "frozen_announcements.csv",
        "candidates": target / "frozen_candidates.csv",
        "candidate_hard_risks": target / "candidate_hard_risks.csv",
        "anchor_risk_overlay": target / "anchor_risk_overlay.csv",
    }
    for key, path in paths.items():
        expected = body.get("source_file_hashes", {}).get(key)
        if not path.exists() or sha256_file(path) != expected:
            errors.append(f"{target.name}: {key} file hash mismatch")
    if errors:
        return errors
    try:
        plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        announcements = pd.read_csv(paths["announcements"], dtype={"code": "string"})
        candidates = pd.read_csv(paths["candidates"], dtype={"code": "string"})
        validate_inputs(plan, manifest, announcements, candidates, paths["announcements"])
        validate_binding_time(plan, body["bound_at"])
        evaluation, _, _ = evaluate(plan, manifest, announcements, candidates, body["bound_at"])
        for key, value in evaluation.items():
            if body.get(key) != value:
                errors.append(f"{target.name}: recomputed {key} mismatch")
        if str(plan.get("session_id")) != str(body.get("session_id")):
            errors.append(f"{target.name}: session id mismatch")
        registration = body.get("v10_plan_verification", {})
        if registration.get("status") == "PASS":
            current_registration = validate_registered_v10_plan(paths["plan"], plan)
            if current_registration != registration:
                errors.append(f"{target.name}: registered V10 plan verification mismatch")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{target.name}: recomputation failed ({exc})")
    return errors


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    targets = sorted(path.parent for path in output.glob("*/announcement_binding.json")) if output.exists() else []
    errors = [error for target in targets for error in verify_one(target)]
    return {
        "status": "PASS" if not errors else "FAIL", "version": VERSION,
        "bindings": len(targets), "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind next-session announcement risks to a frozen relay plan")
    sub = parser.add_subparsers(dest="command", required=True)
    binder = sub.add_parser("bind")
    binder.add_argument("--plan", type=Path, required=True)
    binder.add_argument("--manifest", type=Path, required=True)
    binder.add_argument("--announcements", type=Path, required=True)
    binder.add_argument("--candidates", type=Path, required=True)
    binder.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verifier = sub.add_parser("verify")
    verifier.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "bind":
        result = bind(args.plan, args.manifest, args.announcements, args.candidates, args.output)
    else:
        result = verify(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
