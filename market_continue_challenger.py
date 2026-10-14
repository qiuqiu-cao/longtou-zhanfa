"""One predeclared CONTINUE-class diagnostic; no production model changes.

BASE predictions and fit membership are reused from the immutable fixed study.
Only balanced_subsample class weights differ from the original forest recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from model_upgrade_history import (FOLDS, ROOT, _frame_digest, _predict,
                                   now, prepare_datasets, sha, training_rows)
from model_upgrade_metrics import classification_metrics

FIXED = ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z"
ARM = "M1_BALANCED_SUBSAMPLE"
CLASSES = ("IMPROVE", "CONTINUE", "WORSEN")
FOREST = dict(n_estimators=160, max_depth=5, min_samples_leaf=15,
              max_features=.75, random_state=274, n_jobs=2,
              class_weight="balanced_subsample")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def make_challenger():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        RandomForestClassifier(**FOREST))


def gate(base, candidate):
    """Missing classes cannot establish a three-class improvement."""
    def not_lower(new, old):
        return new is not None and old is not None and new >= old - 1e-12
    checks = {
        "accuracy_improves_at_least3pp": base["accuracy"] is not None
            and candidate["accuracy"] is not None
            and candidate["accuracy"] - base["accuracy"] >= .03 - 1e-12,
        "label_coverage_at_least90pct": base["label_coverage"] is not None
            and base["label_coverage"] >= .9,
        "scoring_coverage_not_lower": not_lower(candidate["scoring_coverage"], base["scoring_coverage"]),
        "macro_f1_not_lower": not_lower(candidate["macro_f1"], base["macro_f1"]),
    }
    for label in CLASSES:
        checks[label + "_complete_recall_not_lower"] = not_lower(
            candidate["by_class"][label]["complete_recall"],
            base["by_class"][label]["complete_recall"])
    old = base["by_class"]["CONTINUE"]["complete_recall"]
    new = candidate["by_class"]["CONTINUE"]["complete_recall"]
    checks["CONTINUE_recall_strictly_improves"] = old is not None and new is not None and new > old + 1e-12
    return {"checks": checks, "passed": all(checks.values())}


def verify_bindings(bindings):
    for name, expected in bindings.items():
        if sha(name) != expected:
            raise ValueError("Bound source or protected file changed: " + name)


def run(output):
    started, started_at = time.monotonic(), now()
    out = Path(output).resolve()
    expected_root = (ROOT / "research/model_upgrade_validation").resolve()
    if out.parent != expected_root:
        raise ValueError("Batch output must be a unique direct child of model_upgrade_validation")
    out.mkdir(exist_ok=False)
    original_protocol_path = FIXED / "protocol_BEFORE_EFFECTS.json"
    original_protocol = json.loads(original_protocol_path.read_text(encoding="utf-8"))
    protected = dict(original_protocol["protected_sha256"])
    for p in (original_protocol_path, FIXED / "production_hold_20261007.json",
              FIXED / "production_authorization_20261008.json",
              FIXED / "market_history/predictions.csv", FIXED / "market_history/report.json",
              FIXED / "market_history/metrics.json", FIXED / "models/market/BASE/model.joblib",
              FIXED / "models/market/BASE/manifest.json"):
        protected[str(p)] = sha(p)
    protocol = {
        "research_id": ARM, "registered_at": now(), "budget_minutes": 30,
        "hypothesis": "Training-only balanced bootstrap class weights may recover CONTINUE recall without sacrificing full-date three-class accuracy or other class recall.",
        "baseline": "Saved BASE predictions and original fold-specific training membership; no BASE refit",
        "challengers": [{"id": ARM, "only_change": "RandomForestClassifier.class_weight=None -> balanced_subsample", "forest": FOREST}],
        "maximum_challengers": 1, "maximum_fold_fits": 3,
        "fields": "Original ordered 46 market fields; unchanged",
        "folds": FOLDS, "labels": "Original exact-next-official-session emotion quality delta: >0.05 IMPROVE, <-0.05 WORSEN, else CONTINUE; invalid/unknown retained",
        "preprocessing": "Original training-only median, add_indicator=True, keep_empty_features=True; market sample_weight=ones",
        "training_boundary": "Original identical members and labels; signal and target date strictly before fold start; original training digest must match before any fit",
        "historical_evidence": "PREVIOUSLY_INSPECTED_HISTORICAL_DIAGNOSTIC_NOT_BLIND_NOT_FORWARD",
        "primary": "Full same-date known-label three-class Accuracy with complete original official-grid coverage disclosed",
        "secondary": ["All three complete recalls", "All three Precision/F1", "macro_f1", "coverage", "confusion_matrix"],
        "diagnostic_gate": "2026JanAug AND 2026Sep each: Accuracy +3pp; all complete recalls, macroF1 and scoring coverage not lower; CONTINUE recall strictly improves; label coverage >=90%. Missing class/label metrics cannot pass.",
        "future_freeze": "Only if BOTH fixed later-fold gates pass, save the September estimator already fitted; zero extra fits. No production or shadow-arm replacement; prior 120-day gates still required.",
        "rejected_policy": "If either later gate does not pass, mark REJECTED_FOR_FUTURE_USE; retain predictions, metrics, failures and protocol but save no deployable challenger.",
        "brier": "No calibrated probabilities; calibrated Brier unavailable",
        "stop": "Finish this one fixed comparison or 30-minute deadline; no retuning, retry or second hypothesis",
        "protected_hashes": protected, "implementation_sha256": sha(__file__),
        "orders_enabled": False, "production_upgrade_allowed": False,
    }
    write_json(out / "protocol_BEFORE_COMPARISON.json", protocol)
    fit_logs, predictions, saved_future_model = [], [], None
    fit_count = 0
    try:
        verify_bindings(protected)
        prepared = prepare_datasets()  # Original rebuild and provenance validation; no fits.
        fields, frame = prepared["market_fields"], prepared["market"]
        old_report = json.loads((FIXED / "market_history/report.json").read_text(encoding="utf-8"))
        old_metrics = json.loads((FIXED / "market_history/metrics.json").read_text(encoding="utf-8"))
        old_predictions = pd.read_csv(FIXED / "market_history/predictions.csv")
        logs = {x["fold"]["id"]: x for x in old_report["fit_logs"] if x["arm"] == "BASE"}
        if len(fields) != 46:
            raise ValueError("Original 46-field contract differs")
        folds = []
        for fold in FOLDS:
            train = training_rows(frame, fold["start"], "market")
            test = frame[frame.date.between(fold["start"], fold["end"])].copy()
            old = old_predictions[old_predictions.fold.eq(fold["id"]) & old_predictions.arm.eq("BASE")].copy()
            if logs[fold["id"]]["fields"] != fields or _frame_digest(train, fields) != logs[fold["id"]]["training_digest"]:
                raise ValueError("Original training membership, values or ordered fields differ: " + fold["id"])
            if sorted(test.date.tolist()) != sorted(old.date.tolist()) or old.date.duplicated().any():
                raise ValueError("Original test universe differs: " + fold["id"])
            for col in ("target_date", "target", "label_valid"):
                a = test.set_index("date")[col].sort_index().fillna("<NA>").astype(str)
                b = old.set_index("date")[col].sort_index().fillna("<NA>").astype(str)
                if not a.equals(b):
                    raise ValueError("Original test labels differ: " + fold["id"] + "/" + col)
            folds.append((fold, train, test, old))
        bindings = prepared["provenance"]["source_hashes"] | prepared["provenance"]["dependency_hashes"] | protected | {
            str(Path(__file__).resolve()): sha(__file__),
            str(ROOT / "tests/test_market_continue_challenger.py"): sha(ROOT / "tests/test_market_continue_challenger.py"),
        }
        write_json(out / "implementation_BEFORE_FITS.json", {
            "formed_at": now(), "protocol_sha256": sha(out / "protocol_BEFORE_COMPARISON.json"),
            "file_hashes": bindings, "runtime": prepared["provenance"]["runtime_versions"],
            "python_version": prepared["provenance"]["python_version"],
            "provenance": prepared["provenance"],
            "original_training_match": [{"fold": f["id"], "training_digest": _frame_digest(t, fields),
                "training_rows": len(t), "max_training_label_date": str(t.target_date.max())} for f,t,_,_ in folds],
        })
        verify_bindings(bindings)
        summaries = {}
        for fold, train, test, old in folds:
            if time.monotonic() - started >= 1800:
                raise TimeoutError("30-minute batch deadline reached before next complete fit")
            fitted_at = now()
            model = make_challenger()
            fit_count += 1
            model.fit(train[fields], train.target, randomforestclassifier__sample_weight=np.ones(len(train)))
            predicted = _predict(model, test, fields, "market")
            predicted["arm"], predicted["fold"] = ARM, fold["id"]
            predictions.append(predicted)
            fit_logs.append({"fold": fold, "fitted_at": fitted_at, "completed_at": now(),
                "train_n": len(train), "training_digest": _frame_digest(train, fields),
                "training_max_label_date": str(train.target_date.max()),
                "training_classes": {str(k): int(v) for k,v in train.target.value_counts().items()},
                "fields": fields, "forest": FOREST})
            grid = pd.DataFrame({"date": [d for d in prepared["provenance"]["sessions"] if fold["start"] <= d <= fold["end"]]})
            aligned = grid.merge(test[["date", "target", "label_valid"]], how="left", on="date", validate="one_to_one")
            truth = aligned.target.where(aligned.label_valid.eq(True)).tolist()
            base_calls = grid.merge(old[["date", "predicted"]], how="left", on="date", validate="one_to_one").predicted.tolist()
            calls = grid.merge(predicted[["date", "predicted"]], how="left", on="date", validate="one_to_one").predicted.tolist()
            bm = classification_metrics(truth, base_calls, CLASSES)
            cm = classification_metrics(truth, calls, CLASSES)
            saved = old_metrics["folds"][fold["id"]]["arms"]["BASE"]
            for key in ("registered_rows", "known_outcome_rows", "scored_rows", "correct", "confusion_matrix"):
                if bm[key] != saved[key]:
                    raise ValueError("Saved BASE metrics no longer align: " + fold["id"] + "/" + key)
            summaries[fold["id"]] = {"BASE": bm, ARM: cm, "diagnostic_gate": gate(bm, cm),
                "accuracy_difference": cm["accuracy"] - bm["accuracy"] if bm["accuracy"] is not None and cm["accuracy"] is not None else None}
            if fold["id"] == "2026Sep":
                saved_future_model = model
            write_json(out / ("fold_" + fold["id"] + ".json"), summaries[fold["id"]])
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(out / "challenger_predictions.csv", index=False, encoding="utf-8-sig")
        passed = all(summaries[k]["diagnostic_gate"]["passed"] for k in ("2026JanAug", "2026Sep"))
        future = None
        if passed:
            directory = out / "future_challenger"
            directory.mkdir()
            joblib.dump(saved_future_model, directory / "model.joblib")
            future = {"model_path": str(directory / "model.joblib"), "model_sha256": sha(directory / "model.joblib"),
                "arm": ARM, "fields": fields, "formed_at": now(), "training_max_label_date": "2026-08-31",
                "reused_september_estimator": True, "no_production_authorization": True}
            write_json(directory / "manifest.json", future)
        verify_bindings(bindings)
        report = {"status": "DIAGNOSTIC_GATE_PASSED_FORWARD_VALIDATION_REQUIRED" if passed else "REJECTED_FOR_FUTURE_USE",
            "formed_at": now(), "folds": summaries, "fit_logs": fit_logs,
            "fit_count": fit_count, "future_extra_fit_count": 0,
            "future_model": future, "protected_originals_unchanged": True,
            "evidence_class": "HISTORICAL_DIAGNOSTIC_NOT_BLIND_NOT_FORWARD",
            "calibrated_Brier": None, "new_forward_samples": 0,
            "production_model_changed": False, "old_shadow_arms_changed": False,
            "limitations": ["Historical data and outcomes previously inspected", "Market target is emotion quality change, not index return", "September unknown dates and absent CONTINUE label retained", "No executable trading returns estimated"],
            "predictions_sha256": sha(out / "challenger_predictions.csv")}
        write_json(out / "report.json", report)
        receipt = {"research_id": ARM, "started_at": started_at, "finished_at": now(),
            "elapsed_seconds": time.monotonic() - started, "budget_minutes": 30,
            "status": report["status"], "protocol_sha256": sha(out / "protocol_BEFORE_COMPARISON.json"),
            "implementation_receipt_sha256": sha(out / "implementation_BEFORE_FITS.json"),
            "report_sha256": sha(out / "report.json"), "fit_count": fit_count,
            "base_refit_count": 0, "future_extra_fit_count": 0, "new_forward_samples": 0,
            "protected_originals_unchanged": True, "production_model_changed": False,
            "unique_next_step": "Reject this challenger and retain its evidence; continue existing source-bound fixed shadow comparisons without retuning this result." if not passed else "Independently preregister future shadow validation before use; original 120-day evidence gates remain required."}
        write_json(out / "batch_receipt.json", receipt)
        return receipt
    except Exception as error:
        write_json(out / "FAILED.json", {"formed_at": now(), "error": str(error), "fit_count": fit_count,
            "fit_logs": fit_logs, "base_refit_count": 0, "retry_allowed": False,
            "production_model_changed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False))
