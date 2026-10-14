"""Explicit per-module inference; immutable fixed comparison remains separate.

No training, directory discovery, promotion, rollback or registry mutation occurs
here. The CLI writes only one new prediction file. BASE is the current baseline,
not a claim that it has proved superior. Challenges require both the original
source-reconstructed 120-session gate and the additional complete-pool policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import model_upgrade_shadow as shadow
from model_upgrade_inputs import load_feature_archive
from model_upgrade_metrics import acceptance

VERSION = "explicit-module-champion-runtime-1"
SELECTION_POLICY = "complete-pool-accuracy-3pp-recall-f1-coverage-1"
ARMS = ("BASE", "C1", "C2")


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _stats(module, arm):
    stats = arm["all_candidate"] if module == "relay" else arm
    recalls = ([stats.get("complete_positive_recall")] if module == "relay"
               else [stats.get("by_class", {}).get(k, {}).get("complete_recall")
                     for k in ("IMPROVE", "CONTINUE", "WORSEN")])
    return {"accuracy": stats.get("accuracy"), "coverage": stats.get("scoring_coverage"),
            "recalls": recalls, "f1": stats.get("F1" if module == "relay" else "macro_f1"),
            "registered": stats.get("registered_rows"),
            "known": stats.get("known_outcome_rows"), "scored": stats.get("scored_rows")}


def choose_champion(native_result, field_counts, current_arm="BASE"):
    """Pure future policy, fed source-reconstructed native metrics by runtime.

    It cannot promote anything. Complete-pool accuracy is the first criterion;
    ties retain the incumbent, then use complete Recall, F1, fewer fields and ID.
    The original protocol's C1-first selection is never rewritten.
    """
    module = native_result.get("module")
    if module not in ("market", "relay") or current_arm not in ARMS:
        raise ValueError("unknown module or incumbent arm")
    if set(field_counts) != set(ARMS) or any(type(v) is not int or v <= 0 for v in field_counts.values()):
        raise ValueError("exact frozen BASE/C1/C2 field counts required")
    summary = native_result.get("summary", {})
    arms = summary.get("arms", {})
    if set(arms) != set(ARMS):
        raise ValueError("complete registered arm metrics required")
    metrics = {a: _stats(module, arms[a]) for a in ARMS}
    baseline = metrics["BASE"]
    mature = (native_result.get("official_window_size") == 120
              and summary.get("registered_dates") == 120
              and summary.get("evidence_class") == "REAL_FORWARD_SAVED_RECORDS")
    checks, eligible = {}, ["BASE"]
    for arm in ("C1", "C2"):
        native = native_result.get("gates", {}).get(arm, {})
        candidate = metrics[arm]
        values = [baseline["accuracy"], candidate["accuracy"], baseline["coverage"],
                  candidate["coverage"], baseline["f1"], candidate["f1"],
                  *baseline["recalls"], *candidate["recalls"]]
        known = all(_finite(v) for v in values)
        checks[arm] = {
            "original_120_session_all_gates": (mature and native.get("passed") is True
                and bool(native.get("checks")) and all(v is True for v in native["checks"].values())
                and native["checks"].get("mature_fixed_120_node") is True),
            "all_metrics_known": known,
            "same_complete_pool": (type(baseline["registered"]) is int and baseline["registered"] > 0
                and candidate["registered"] == baseline["registered"]
                and candidate["known"] == baseline["known"]
                and candidate["scored"] == baseline["scored"]
                and baseline["scored"] == baseline["known"] and baseline["known"] > 0),
            "complete_pool_accuracy_improves_3pp": known and candidate["accuracy"]-baseline["accuracy"] >= .03-1e-12,
            "complete_recall_not_lower": known and all(n >= b-1e-12 for n, b in zip(candidate["recalls"], baseline["recalls"])),
            "f1_not_lower": known and candidate["f1"] >= baseline["f1"]-1e-12,
            "coverage_not_lower": known and candidate["coverage"] >= baseline["coverage"]-1e-12,
        }
        if all(checks[arm].values()):
            eligible.append(arm)
    def key(arm):
        row = metrics[arm]
        return (-row["accuracy"], arm != current_arm,
                -sum(row["recalls"])/len(row["recalls"]), -row["f1"], field_counts[arm], arm)
    ranked = sorted(eligible, key=key) if len(eligible) > 1 else eligible
    return {"policy": SELECTION_POLICY, "selected_arm": ranked[0], "eligible_arms": ranked,
            "checks": checks, "metrics": metrics, "automatic_switch": False,
            "scope": "additional future policy; original fixed C1 priority unchanged"}


def _validate_binding(path, module, arm, binding):
    expected = path.parent / "models" / module / arm
    if (Path(binding["path"]).resolve() != (expected / "manifest.json").resolve()
            or Path(binding["model_path"]).resolve() != (expected / "model.joblib").resolve()):
        raise ValueError("selected model must be the exact frozen model, never a retired/arbitrary path")
    if (binding["manifest"].get("retired") is True
            or binding["manifest"].get("lifecycle_status") == "RETIRED"):
        raise ValueError("retired model cannot enter production")
    shadow._source_check({binding["path"]: binding["sha256"],
                          binding["model_path"]: binding["model_sha256"]})


def _prove_active(registration, path, module, models, pack, pointer, cutoff):
    """Read-only native evidence reconstruction, avoiding mutating error wrappers."""
    record, registered_path, protocol, calendar, root = shadow._registered(registration)
    if (registered_path != path or record["module"] != module
            or record["signal_date"] != pack["signal_date"]
            or Path(record["feature_archive"]).resolve() != Path(pack["feature_archive"]).resolve()
            or record["model_bindings"] != models):
        raise ValueError("active registration differs from the module/features/frozen model group")
    shadow.production_policy.require_allowed(path, module, "apply", shadow.now_utc())
    arm = pointer["active_arm"]
    if arm not in ("C1", "C2") or pointer.get("suspended"):
        raise ValueError("active challenge is suspended or unknown")
    if (pack["signal_date"] < pointer["effective_signal"]
            or shadow.stamp(pointer["promoted_at"]) > shadow.stamp(record["knowledge_cutoff"])
            or shadow.stamp(record["knowledge_cutoff"]) > cutoff
            or pointer["protocol_sha256"] != shadow.file_sha256(path)):
        raise ValueError("active model was not effective at the actual signal boundary")
    receipt_path = shadow._owned(pointer["receipt"], root)
    if shadow.file_sha256(receipt_path) != pointer["receipt_sha256"]:
        raise ValueError("active acceptance receipt hash conflict")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("node") != 120 or receipt.get("module") != module
            or receipt.get("protocol_sha256") != shadow.file_sha256(path)
            or shadow.stamp(receipt["reviewed_at"]) > shadow.stamp(record["knowledge_cutoff"])):
        raise ValueError("active receipt is outside the original knowledge boundary")
    shadow._receipt_binding_check(receipt)
    _, derived, verified_calendar, _, records, _ = shadow._review_inputs(path, module)
    live = acceptance(module, records, derived, verified_calendar["sessions"])
    if (live != receipt["result"] or not live.get("upgrade_eligible")
            or live.get("selected_arm") != arm or not live.get("gates", {}).get(arm, {}).get("passed")
            or pointer["model"] != models[arm] or pointer["fallback"] != models["BASE"]):
        raise ValueError("active arm lacks the original source-bound full 120-session acceptance")
    selection = choose_champion(live, {a: len(b["manifest"]["fields"]) for a, b in models.items()}, arm)
    if selection["selected_arm"] != arm:
        raise ValueError("native active arm does not win additional complete-pool policy; explicit policy adaptation required")
    return {"status": "SOURCE_REBUILT_ACTIVE_WITH_ADDITIONAL_GATES", "selection": selection,
            "receipt_path": str(receipt_path), "receipt_sha256": shadow.file_sha256(receipt_path),
            "registration_path": str(Path(registration).resolve()),
            "registration_sha256": shadow.file_sha256(registration)}


def _forecast(module, binding, pack):
    roster = pack["relay"]
    fields = binding["manifest"]["fields"]
    vectors = [pack["market"]["features"]] if module == "market" else [r["features"] for r in roster]
    observed = bool(vectors) and all(any(v.get(k) is not None for k in fields) for v in vectors)
    raw, classes = None, None
    if observed:
        classes, raw = shadow._model_probabilities(binding, vectors)
    if module == "market":
        return {"status": "KNOWN" if raw is not None else "UNKNOWN",
                "prediction": classes[int(raw[0].argmax())] if raw is not None else None,
                "raw_scores": dict(zip(classes, raw[0].tolist())) if raw is not None else {},
                "input_status": pack["market"]["status"],
                "reason": None if observed else "ALL_OBSERVED_FEATURES_MISSING"}
    positive = classes.index("1") if raw is not None else None
    candidates = [{"code": r["code"], "name": r.get("name"), "boards": r["boards"],
                   "actual_one_word_daily": r["actual_one_word_daily"],
                   "raw_score": float(raw[i, positive]) if raw is not None else None,
                   "predicted_positive": bool(raw[i, positive] > .5) if raw is not None else None}
                  for i, r in enumerate(roster)]
    ranked = sorted(candidates, key=lambda r: (-r["raw_score"], r["code"])) if raw is not None else []
    return {"status": "KNOWN" if raw is not None else "UNKNOWN", "candidates": candidates,
            "ranking": [r["code"] for r in ranked],
            "ranking_rule": "raw_score descending; code ascending; no replacement",
            "reason": None if observed else "INCOMPLETE_ROSTER_OR_FEATURE_EVIDENCE"}


def predict(protocol_path, module, features_path, registration=None):
    if module not in ("market", "relay"):
        raise ValueError("unknown production module")
    path, protocol, calendar, root = shadow._protocol(protocol_path)
    cutoff = shadow.now_utc()
    pack = load_feature_archive(features_path, cutoff=cutoff.isoformat())
    shadow.require_current_closed_signal(pack["signal_date"], calendar["sessions"], cutoff)
    shadow.validate_cutoff(pack["signal_date"], cutoff, cutoff)
    shadow._source_check(pack["sources"])
    roster = pack["relay"]
    codes = [r["code"] for r in roster]
    if len(set(codes)) != len(codes) or any(not isinstance(c, str) or len(c) != 6 or not c.isdigit() for c in codes):
        raise ValueError("complete original candidate roster required")
    if any(type(r["boards"]) is not int or r["boards"] < 2
           or r["actual_one_word_daily"] not in (0, 1, None) for r in roster):
        raise ValueError("original candidate facts invalid")
    # Validates the immutable comparison group; no estimator is deserialized here.
    models = shadow._models(path, protocol, module, path.parent / "models", cutoff)
    control = shadow._pointer(root)
    pointer = control.get("modules", {}).get(module, {"active_arm": "BASE"})
    arm = pointer.get("active_arm")
    if arm not in ARMS:
        raise ValueError("unknown active arm; no implicit BASE fallback")
    proof = {"status": "CURRENT_BASE_NOT_PROVEN_BEST", "reason": "no accepted challenge active"}
    if arm != "BASE":
        if registration is None:
            raise ValueError("native source-bound current registration required for active challenge")
        proof = _prove_active(registration, path, module, models, pack, pointer, cutoff)
    _validate_binding(path, module, arm, models[arm])
    forecast = _forecast(module, models[arm], pack)
    return {"version": VERSION, "module": module, "selected_arm": arm, "selection_proof": proof,
            "signal_date": pack["signal_date"],
            "target_date": shadow.next_official_session(calendar["sessions"], pack["signal_date"]),
            "knowledge_cutoff": cutoff.isoformat(), "feature_available_at": pack["available_at"],
            "input_status": pack["market"]["status"], "forecast": forecast,
            "model_binding": models[arm], "protocol_path": str(path),
            "protocol_sha256": shadow.file_sha256(path), "feature_archive": str(Path(features_path).resolve()),
            "feature_manifest_sha256": pack["manifest_sha256"],
            "feature_projection_sha256": hashlib.sha256(shadow.canonical(pack)).hexdigest(),
            "runtime_code_sha256": shadow.file_sha256(__file__),
            "estimators_deserialized": 1 if forecast["status"] == "KNOWN" else 0,
            "score_calibrated": False, "production_model_changed": False,
            "registry_mutated": False, "accuracy_improvement_proved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--protocol", required=True)
    p.add_argument("--module", choices=("market", "relay"), required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--registration")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    protocol_root = Path(args.protocol).resolve().parent
    if output.exists() or output.suffix != ".json":
        raise ValueError("unique new JSON output required")
    if output.is_relative_to(protocol_root) or any(v.lower() == "expectation_registry" for v in output.parts):
        raise ValueError("production prediction output cannot change fixed protocol or original registry")
    if not output.parent.is_dir():
        raise ValueError("output directory must already exist; prediction creates no directories")
    result = predict(args.protocol, args.module, args.features, args.registration)
    serialized = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            if handle.write(serialized) != len(serialized):
                raise OSError("incomplete prediction output write")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "SINGLE_EXPLICIT_MODEL_PREDICTED", "module": args.module,
                      "selected_arm": result["selected_arm"], "input_status": result["input_status"],
                      "estimators_deserialized": result["estimators_deserialized"], "output": str(output),
                      "sha256": shadow.file_sha256(output), "production_model_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
