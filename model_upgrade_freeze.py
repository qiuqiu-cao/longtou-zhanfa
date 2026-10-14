"""Recheck an upgraded author-reviewed draft immediately before original V298 freeze.

--calendar is verified here; --output is a unique JSON execution receipt.
The original V298 freeze accepts only --plan and --registry. Its complete
context/hypothesis contracts still run in a subprocess with no shell.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import model_upgrade_apply as apply
import model_upgrade_shadow as shadow

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "hypothesis_bound_daily_runner_v298.py"
GUARD_NAME = "production_guard_BEFORE_FORWARD.json"


class FreezeGuardError(ValueError):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _conflict(application, module, reason):
    error = FreezeGuardError("STOP_NEW_MODULE_USE_BASE: " + reason)
    shadow._binding_failure(application["proof"]["protocol_path"], module, error)
    raise error


def _guard(application):
    proof = application["proof"]
    protocol = Path(proof["protocol_path"]).resolve()
    path = protocol.parent / GUARD_NAME
    guard = _read(path)
    if guard.get("protocol_sha256") != shadow.file_sha256(protocol):
        raise ValueError("production guard protocol hash conflict")
    if (shadow.stamp(guard["formed_at"]) > shadow.now_utc()
            or shadow.stamp(guard["formed_at"]) > shadow.stamp(application["knowledge_cutoff"])):
        raise ValueError("production guard unavailable at knowledge cutoff")
    sources = guard.get("sources", {})
    required = {str(Path(__file__).resolve()), str(ENTRY.resolve()),
                str(Path(shadow.__file__).resolve()), str(Path(apply.__file__).resolve())}
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise ValueError("production guard lacks required entry/dependency source bindings")
    for source, expected in sources.items():
        if not Path(source).is_absolute() or shadow.file_sha256(source) != expected:
            raise ValueError("production guard source hash conflict: " + source)
    return {"path": str(path), "sha256": shadow.file_sha256(path)}


def _projection(plan, module, application, record, pointer, current_proof):
    if (module not in ("market", "relay") or application.get("module") != module
            or application.get("version") != apply.VERSION or application.get("arm") not in ("C1", "C2")
            or application.get("arm") != pointer.get("active_arm")
            or current_proof.get("status") != "VERIFIED_ACTIVE"
            or application.get("effective_signal") != pointer.get("effective_signal")
            or application.get("model_binding") != pointer.get("model")
            or application.get("model_binding") != record.get("model_bindings", {}).get(application["arm"])):
        raise ValueError("applied active arm/model/effective-date binding conflict")
    if any(application.get(key) != plan.get(key) for key in ("signal_date", "target_date", "knowledge_cutoff")):
        raise ValueError("applied plan date/cutoff binding conflict")
    if (application.get("frozen") is not False or application.get("requires_actual_author_review") is not True
            or application.get("scores_are_calibrated_probabilities") is not False
            or application.get("automatic_entry_allowed") is not False):
        raise ValueError("applied review/probability/entry contract conflict")
    saved = application["proof"]
    for key in ("status", "registration_path", "registration_sha256", "active_path", "active_sha256",
                "protocol_path", "protocol_sha256", "receipt_path", "receipt_sha256"):
        if key not in saved or saved[key] != current_proof.get(key):
            raise ValueError("applied proof binding changed: " + key)
    # The immutable adapter supplies the same field projection, without review
    # or registration. Only its governed fields are compared, not human text.
    pending = deepcopy(plan)
    pending["daily_research"]["draft_status"] = apply.PENDING
    for key in ("author_reviewed_at", "author_review", "reviewed_at"):
        pending["daily_research"].pop(key, None)
    for key in ("author_reviewed_at", "author_review"):
        pending.pop(key, None)
    pending.get("model_upgrade_application", {}).pop(module, None)
    expected = apply._adapt_verified(pending, record, pointer, current_proof)
    if module == "market":
        for key in ("direction", "direction_basis", "calibration_status", "probability"):
            if plan["market"].get(key) != expected["market"].get(key):
                raise ValueError("applied market direction field changed: " + key)
    else:
        fields = ("boards", "actual_one_word_daily", "promotion_call", "call_basis", "model_raw_score",
                  "model_promotion_score", "model_score_calibrated", "model_score_sha256", "model_rank", "probability")
        originals = {c["code"]: c for c in plan["candidates"]}
        expected_rows = {c["code"]: c for c in expected["candidates"]}
        for code, row in originals.items():
            if row.get("actual_one_word_daily") is not None and type(row.get("actual_one_word_daily")) is not bool:
                raise ValueError("applied candidate true OHLC type changed: " + code)
            apply._score(row.get("model_raw_score"))
            display = row.get("model_promotion_score")
            if (isinstance(display, bool) or not isinstance(display, (int, float))
                    or not math.isfinite(display) or not 0 <= display <= 100
                    or type(row.get("model_rank")) is not int or row["model_rank"] < 1
                    or row.get("model_score_calibrated") is not False):
                raise ValueError("applied candidate score/rank contract changed: " + code)
            for key in fields:
                if row.get(key) != expected_rows[code].get(key):
                    raise ValueError("applied candidate field changed: " + code + ":" + key)
        for key in ("identity_primary_candidate", "non_one_word_primary_candidate", "model_rank_first_candidate",
                    "model_nonword_rank_first", "nonword_selection_status", "nonword_selection_unknown_codes"):
            if plan.get(key) != expected.get(key):
                raise ValueError("applied selection/rank field changed: " + key)
        events = {c["code"]: c for c in plan["daily_research"]["candidates"]}
        expected_events = {c["code"]: c for c in expected["daily_research"]["candidates"]}
        for code in originals:
            for event in ("promotion", "break"):
                claim = events[code]["events"][event]
                if (claim.get("call") is not expected_events[code]["events"][event]["call"]
                        or claim.get("probability") is not None):
                    raise ValueError("applied promotion/break event changed: " + code)
        primary = plan.get("primary_candidate")
        if primary is not None:
            row = originals.get(primary)
            if (not row or row["promotion_call"] != "PROMOTE" or row["actual_one_word_daily"] is not False
                    or plan.get("nonword_selection_status") != "KNOWN"):
                raise ValueError("execution primary requires positive judgment and complete true nonword facts")


def freeze_plan(plan_path, calendar_path, output, registry=None):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    registry = Path(registry or ROOT / "data/v271/expectation_registry").resolve()
    if output.exists():
        raise FileExistsError("unique new freeze receipt required")
    plan = _read(plan_path)
    if (not isinstance(plan, dict) or "plan" in plan
            or plan.get("daily_research", {}).get("draft_status") != "AUTHOR_REVIEWED"):
        raise FreezeGuardError("actual author review is required before freeze; no automatic review")
    for directory in {registry, (ROOT / "data/v271/expectation_registry").resolve()}:
        if any(directory.glob(plan["signal_date"] + "_*.json")):
            raise FreezeGuardError("already frozen signal; original forecast is immutable")
    applications = plan.get("model_upgrade_application", {})
    if not isinstance(applications, dict):
        raise FreezeGuardError("invalid model application group")
    if not set(applications).issubset({"market", "relay"}):
        raise FreezeGuardError("unknown module in model application group")
    score_markers = {"model_raw_score", "model_promotion_score", "model_score_sha256", "model_rank"}
    upgraded_markers = (apply.VERSION in str(plan.get("model_version", ""))
                        or any(score_markers.intersection(row) for row in plan.get("candidates", []) if isinstance(row, dict)))
    if not applications and upgraded_markers:
        raise FreezeGuardError("upgraded version/score fields require original model_upgrade_application binding")
    try:
        calendar = shadow.load_verified_calendar(calendar_path)
        if (plan["target_date"] != shadow.next_official_session(calendar["sessions"], plan["signal_date"])
                or plan.get("calendar") != calendar):
            raise FreezeGuardError("explicit official calendar differs from plan signal/target/calendar")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        for module, application in applications.items():
            shadow._binding_failure(application["proof"]["protocol_path"], module, exc)
        raise
    original_sha = shadow.file_sha256(plan_path)
    checks = {}
    for module, application in applications.items():
        try:
            guard = _guard(application)
            proof = application["proof"]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _conflict(application, module, str(exc))
        # Public verification already suspends its own source conflicts.
        try:
            record, pointer, current = shadow.verify_active_for_registration(proof["registration_path"], proof["active_path"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise FreezeGuardError("source-bound final verification refused: " + str(exc)) from exc
        try:
            _projection(plan, module, application, record, pointer, current)
            checks[module] = {"guard": guard, "proof": current, "arm": pointer["active_arm"]}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _conflict(application, module, str(exc))
    # Recheck sources and plan at the subprocess boundary, without altering it.
    for module, application in applications.items():
        try:
            if _guard(application) != checks[module]["guard"]:
                raise ValueError("production guard changed during verification")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _conflict(application, module, str(exc))
    if shadow.file_sha256(plan_path) != original_sha:
        raise FreezeGuardError("reviewed source plan changed during verification")
    command = [sys.executable, str(ENTRY), "freeze", "--plan", str(plan_path), "--registry", str(registry)]
    # Reserve the unique writable receipt before the original entry can register.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        receipt = {"verified_at": shadow.now_utc().isoformat(), "source_plan": str(plan_path), "source_plan_sha256": original_sha,
                   "calendar_path": str(Path(calendar_path).resolve()), "calendar_sha256": shadow.file_sha256(calendar_path),
                   "modules": checks, "command": command, "automatic_author_review": False,
                   "plan_modified": False, "production_model_formula_changed": False}
        try:
            completed = subprocess.run(command, shell=False, capture_output=True, text=True, check=False)
            receipt.update(status="ORIGINAL_V298_FREEZE_COMPLETE" if completed.returncode == 0 else "ORIGINAL_V298_FREEZE_REFUSED",
                           returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except OSError as exc:
            receipt.update(status="ORIGINAL_V298_LAUNCH_FAILED", returncode=None, stdout="", stderr=str(exc))
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {**receipt, "receipt": str(output), "receipt_sha256": shadow.file_sha256(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "calendar", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--registry", default=str(ROOT / "data/v271/expectation_registry"))
    args = parser.parse_args(argv)
    try:
        result = freeze_plan(args.plan, args.calendar, args.output, args.registry)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FREEZE_REFUSED", "error": str(exc), "original_freeze_invoked": False}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
