"""Apply an accepted module to a new, unreviewed V298 draft only.

This adapter never registers/freezes a forecast or promotes an active pointer.
Every invocation rechecks the acceptance chain; binding conflicts suspend only
the affected upgraded module through the shadow controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import model_upgrade_shadow as shadow

ROOT = Path(__file__).resolve().parent
VERSION = "model-upgrade-new-draft-apply-1"
PENDING = "NEEDS_FACTS_AND_RESEARCH_CALLS"


class UpgradeApplyError(ValueError):
    """The new module must stop; the original model and plan remain intact."""


class UpgradeApplySourceConflict(UpgradeApplyError):
    """A verified module's projected source facts are internally inconsistent."""


def _refuse(reason):
    raise UpgradeApplyError("STOP_NEW_MODULE_USE_BASE: " + reason)


def _source_conflict(reason):
    raise UpgradeApplySourceConflict("STOP_NEW_MODULE_USE_BASE: " + reason)


def _hash(value):
    return hashlib.sha256(shadow.canonical(value)).hexdigest()


def _new_draft(plan, plan_path=None):
    if not isinstance(plan, dict) or isinstance(plan.get("plan"), dict):
        _refuse("registered wrappers cannot be adapted")
    if plan_path is not None and any(p.lower() == "expectation_registry" for p in Path(plan_path).parts):
        _refuse("an original registry plan cannot be changed")
    forbidden = ("sha256", "forecast_status", "recorded_at", "registered_at", "frozen_at",
                 "registration_hash", "author_reviewed_at", "author_review")
    if any(key in plan for key in forbidden):
        _refuse("registered, frozen or author-reviewed plan markers present")
    extension = plan.get("daily_research")
    if not isinstance(extension, dict) or extension.get("draft_status") != PENDING:
        _refuse("only a new unreviewed V298 draft is accepted")
    if any(key in extension for key in ("author_reviewed_at", "author_review", "reviewed_at")):
        _refuse("author-review evidence cannot be overwritten")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        _refuse("the full original candidate group is required")
    codes = [c.get("code") for c in candidates]
    if any(not isinstance(c, str) or len(c) != 6 or not c.isdigit() for c in codes) or len(codes) != len(set(codes)):
        _refuse("invalid or duplicated original candidate group")
    extension_codes = [c.get("code") for c in extension.get("candidates", [])]
    if len(extension_codes) != len(set(extension_codes)) or set(extension_codes) != set(codes):
        _refuse("V298 extension must retain the same complete candidate group")
    if not isinstance(plan.get("market"), dict):
        _refuse("market plan is missing")


def _score(value):
    if isinstance(value, bool):
        _refuse("model score is not a numeric internal score")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _refuse("internal score unavailable")
    if not math.isfinite(number) or not 0 <= number <= 1:
        _refuse("internal score outside the frozen class-score scale")
    return number


def _check_correspondence(plan, record):
    if (plan.get("signal_date"), plan.get("target_date")) != (record.get("signal_date"), record.get("target_date")):
        _refuse("plan dates differ from the source-bound registration")
    codes = [c["code"] for c in plan["candidates"]]
    original = record.get("original_codes")
    if not isinstance(original, list) or len(original) != len(set(original)) or set(codes) != set(original):
        _refuse("plan must preserve every registered candidate code")
    if not record.get("forward") or "outcomes" in record:
        _refuse("only the original pre-outcome forward registration may be applied")
    try:
        cutoff = shadow.stamp(plan["knowledge_cutoff"])
        available = max(shadow.stamp(record["registered_at"]), shadow.stamp(record["knowledge_cutoff"]))
    except (KeyError, ValueError) as exc:
        _refuse("knowledge/availability binding missing: " + str(exc))
    if cutoff < available:
        _refuse("registration was unavailable at the plan knowledge cutoff")
    if cutoff > shadow.now_utc():
        _refuse("plan knowledge cutoff is in the future")


def _adapt_verified(plan, record, pointer, proof):
    """Internal projection; callers must first use shadow's public verifier."""
    _new_draft(plan)
    _check_correspondence(plan, record)
    arm, module = pointer.get("active_arm", "BASE"), record.get("module")
    if arm == "BASE":
        return None
    if module not in ("market", "relay") or arm not in ("C1", "C2"):
        _refuse("unknown accepted module or arm")
    if proof.get("status") != "VERIFIED_ACTIVE" or pointer.get("suspended"):
        _refuse("complete source-bound acceptance proof is unavailable")
    if plan["signal_date"] < pointer.get("effective_signal", "9999-12-31"):
        _refuse("accepted module is not effective for this new signal")
    if shadow.stamp(pointer["promoted_at"]) > shadow.stamp(plan["knowledge_cutoff"]):
        _refuse("accepted module was not yet available at the new cutoff")
    binding = record.get("model_bindings", {}).get(arm)
    if not binding or binding != pointer.get("model"):
        _source_conflict("active and registered model/source/hash bindings differ")
    forecast = record.get("arms", {}).get(arm, {})
    if forecast.get("status") != "KNOWN":
        _refuse("accepted module forecast is unknown")
    result = deepcopy(plan)
    applications = result.setdefault("model_upgrade_application", {})
    if module in applications:
        _refuse("do not overwrite a previous module application")
    if module == "market":
        predicted = forecast.get("prediction")
        if predicted not in ("IMPROVE", "CONTINUE", "WORSEN"):
            _refuse("accepted three-class market direction is unknown")
        if result["market"].get("probability") is not None:
            _refuse("old calibrated probability cannot be carried to an uncalibrated new model")
        result["market"]["direction"] = predicted
        result["market"]["direction_basis"] = VERSION+":"+arm+"; source-bound fixed forward acceptance"
        result["market"]["calibration_status"] = "NOT_CALIBRATED"
    else:
        model_rows = forecast.get("candidates")
        if not isinstance(model_rows, list):
            _source_conflict("accepted full candidate forecasts are missing")
        by_code = {c["code"]: c for c in model_rows}
        if len(by_code) != len(model_rows) or set(by_code) != set(record["original_codes"]):
            _source_conflict("accepted module lost original candidates")
        extension = {c["code"]: c for c in result["daily_research"]["candidates"]}
        scores = {}
        for candidate in result["candidates"]:
            original = by_code[candidate["code"]]
            if candidate.get("boards") != original.get("boards"):
                _source_conflict("candidate board count differs from bound signal facts")
            word = original.get("actual_one_word_daily")
            if word is not None and word not in (0, 1):
                _source_conflict("invalid bound true signal-day OHLC one-price status")
            old_word = candidate.get("actual_one_word_daily")
            if word is not None and old_word is not None and old_word != bool(word):
                _source_conflict("plan one-price fact conflicts with bound OHLC")
            call = original.get("predicted_positive")
            if not isinstance(call, bool):
                _refuse("accepted binary promotion judgment is unknown")
            if candidate.get("probability") is not None:
                _refuse("candidate probability cannot be replaced with an internal raw score")
            score = _score(original.get("raw_score"))
            if call != (score > .5):
                _source_conflict("binary judgment differs from the frozen internal-score threshold")
            scores[candidate["code"]] = score
            candidate.update(actual_one_word_daily=None if word is None else bool(word),
                             promotion_call="PROMOTE" if call else "FAIL",
                             call_basis=VERSION+":"+arm, model_raw_score=score,
                             model_promotion_score=round(100*score, 1), model_score_calibrated=False,
                             model_score_sha256=binding["model_sha256"])
            for event, value in (("promotion", call), ("break", not call)):
                claim = extension[candidate["code"]].get("events", {}).get(event)
                if not isinstance(claim, dict) or claim.get("probability") is not None:
                    _refuse("promotion/break draft event or probability evidence conflicts")
                claim["call"] = value
                claim["basis"] = [VERSION+":"+arm+"; original promotion target unchanged"]
        ranked = sorted(result["candidates"], key=lambda c: (-scores[c["code"]], c["code"]))
        for rank, candidate in enumerate(ranked, 1):
            candidate["model_rank"] = rank
        positive = [c for c in ranked if c["promotion_call"] == "PROMOTE"]
        nonword = [c for c in ranked if c["actual_one_word_daily"] is False]
        nonword_positive = [c for c in nonword if c["promotion_call"] == "PROMOTE"]
        unknown_word_codes = [c["code"] for c in ranked if c["actual_one_word_daily"] is None]
        nonword_complete = not unknown_word_codes
        result["identity_primary_candidate"] = positive[0]["code"] if positive else None
        result["non_one_word_primary_candidate"] = nonword_positive[0]["code"] if nonword_complete and nonword_positive else None
        result["model_rank_first_candidate"] = ranked[0]["code"] if ranked else None
        result["model_nonword_rank_first"] = nonword[0]["code"] if nonword_complete and nonword else None
        result["nonword_selection_status"] = "KNOWN" if nonword_complete else "UNKNOWN"
        result["nonword_selection_unknown_codes"] = unknown_word_codes
        # Cancel an invalid or unconfirmed human primary without picking a backup.
        if result.get("primary_candidate") is not None:
            selected = next((c for c in ranked if c["code"] == result["primary_candidate"]), None)
            if selected is None:
                _refuse("execution primary is outside the original candidate group")
            if not nonword_complete or selected["promotion_call"] != "PROMOTE" or selected["actual_one_word_daily"] is not False:
                reason = ("UNKNOWN_TRUE_OHLC_GROUP" if not nonword_complete else
                          "NEW_MODEL_NEGATIVE_OR_ONE_WORD")
                result["execution_primary_review"] = {"previous_primary": result["primary_candidate"],
                                                       "status": "CANCELLED_PENDING_AUTHOR_REVIEW",
                                                       "reason": reason, "automatic_backup": False}
                result["primary_candidate"] = None
                result["highest_accept_price"] = None
    application = {"version": VERSION, "module": module, "arm": arm, "proof": deepcopy(proof),
                   "model_binding": deepcopy(binding), "effective_signal": pointer["effective_signal"],
                   "signal_date": record["signal_date"], "target_date": record["target_date"],
                   "knowledge_cutoff": plan["knowledge_cutoff"], "baseline_input_hash": plan.get("input_hash"),
                   "baseline_model_version": plan.get("model_version"),
                   "requires_actual_author_review": True, "frozen": False,
                   "scores_are_calibrated_probabilities": False, "automatic_entry_allowed": False}
    applications[module] = application
    result["input_hash"] = _hash({"previous_input_hash": plan.get("input_hash"), "application": application})
    result["model_version"] = str(plan.get("model_version", ""))+"|"+VERSION+":"+module+":"+arm
    result["daily_research"]["draft_status"] = PENDING
    result["automatic_entry_allowed"] = False
    result["daily_research"]["automatic_entry_allowed"] = False
    return result


def apply_plan(plan_path, registration, active, output):
    """Verify the full live chain, then write exactly one new pending plan."""
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("unique new output required; refusing to overwrite")
    if any(p.lower() == "expectation_registry" for p in output.parts):
        _refuse("draft output cannot be written into the prediction registry")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _new_draft(plan, plan_path)
    signal = plan.get("signal_date", "")
    if any((ROOT / "data/v271/expectation_registry").glob(signal+"_*.json")):
        _refuse("an original forecast already exists for this signal; do not rewrite it")
    original_sha = shadow.file_sha256(plan_path)
    record = None
    try:
        record, pointer, proof = shadow.verify_active_for_registration(registration, active)
        _check_correspondence(plan, record)
        if pointer.get("active_arm", "BASE") == "BASE":
            return {"status": "UNCHANGED_BASE", "output": None, "plan": str(plan_path),
                    "plan_sha256": original_sha, "proof": proof, "frozen": False}
        result = _adapt_verified(plan, record, pointer, proof)
    except (ValueError, OSError, KeyError) as exc:
        if isinstance(exc, UpgradeApplySourceConflict) and record is not None:
            shadow._binding_failure(record["protocol_path"], record["module"], exc)
        if isinstance(exc, UpgradeApplyError):
            raise
        _refuse("source-bound verification failed: " + str(exc))
    if shadow.file_sha256(plan_path) != original_sha:
        _refuse("source draft changed during verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return {"status": "NEW_DRAFT_REQUIRES_AUTHOR_REVIEW", "module": record["module"],
            "arm": pointer["active_arm"], "output": str(output), "output_sha256": shadow.file_sha256(output),
            "source_plan": str(plan_path), "source_plan_sha256": original_sha,
            "frozen": False, "production_registration_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "registration", "active", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    try:
        result = apply_plan(args.plan, args.registration, args.active, args.output)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "REFUSED_STOP_NEW_MODULE", "error": str(exc),
                          "action": "STOP_NEW_MODULE_USE_BASE", "output_created": False}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
