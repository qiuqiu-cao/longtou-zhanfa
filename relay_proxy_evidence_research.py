"""Execute the existing, single proxy-evidence hypothesis once; never deploy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import relay_context_research as common
from relay_expert_research import subgroup

ARM = "PROXY_EVIDENCE_UNKNOWN"
BOUNDARY = "2026-09-01"
ATOM_FIELDS = ["first_seal_minute", "last_seal_minute", "bomb_count", "turnover"]


def ambiguous_mask(frame):
    """Three-valued AND on signal-day atoms; targets are never read."""
    atoms = frame[ATOM_FIELDS].apply(pd.to_numeric, errors="coerce")
    observed = np.isfinite(atoms)
    true = pd.DataFrame({
        "first_seal_minute": atoms.first_seal_minute.le(.02 / 240 + 1e-12),
        "last_seal_minute": atoms.last_seal_minute.le(.02 / 240 + 1e-12),
        "bomb_count": atoms.bomb_count.eq(0),
        "turnover": atoms.turnover.lt(.03),
    }, index=frame.index)
    known_false = (observed & ~true).any(axis=1)
    known_true = (observed & true).all(axis=1)
    return frame.date.lt(BOUNDARY) & ~known_false & ~known_true


def apply_mask(frame):
    result = frame.copy()
    mask = ambiguous_mask(frame)
    result.loc[mask, "signal_one_word"] = np.nan
    pd.testing.assert_frame_equal(result.drop(columns="signal_one_word"),
                                  frame.drop(columns="signal_one_word"))
    pd.testing.assert_series_equal(result.loc[~mask, "signal_one_word"],
                                   frame.loc[~mask, "signal_one_word"])
    return result, mask


def validate_definition(p):
    expected = {"algorithm": "original logistic", "C": .25, "max_iter": 2500,
        "random_state": 271, "solver": "lbfgs", "class_weight": None,
        "training_window": 252,
        "window_unit": "last252distinct original valid-labelled signal dates",
        "daily_weight": "inverse original full training candidate rows per date",
        "preprocess": "training-only median/add_indicator/keep_empty_features/scaler unchanged",
        "threshold": "raw positive score strictly >0.5"}
    if (p["version"] != "relay-proxy-evidence-fixed-1" or p["arms"] != ["BASE", ARM]
            or p["fields"] != common.RELAY_FIELDS or p["folds"] != common.history.FOLDS
            or p["parameters"] != expected
            or p["expected_improvement_and_fixed_screen"]["Accuracy_delta_each_fold_at_least"] != .03
            or p["execution"]["maximum_challenger_fits"] != 3):
        raise ValueError("pre-registered experiment definition changed")


def prepare(protocol_path, output_dir, started):
    """Freeze source/runtime/member/mask checks before any estimator fit."""
    protocol_path, root = Path(protocol_path).resolve(), Path(output_dir).resolve()
    p = common.read(protocol_path)
    validate_definition(p)
    common.verify(p["source_bindings"])
    receipt = common.read(p["prepared_source_receipt"])
    common.verify(receipt["sources"] | {p["prepared_path"]: p["prepared_sha256"]})
    for name, version in receipt["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("original runtime version changed: " + name)
    full = joblib.load(p["prepared_path"])["relay"]
    if full.duplicated(["date", "code"]).any():
        raise ValueError("duplicate original candidate")
    masked, mask = apply_mask(full)
    qualifications = common.read(p["evidence"]["qualification_path"])
    common.verify({p["evidence"]["qualification_path"]: p["evidence"]["qualification_sha256"]})
    declared_path = protocol_path.parent / "full_ambiguous_proxy_evidence_cells.csv"
    declared = pd.read_csv(declared_path, dtype={"code": str})
    if (int(mask.sum()) != 551 or set(zip(full.loc[mask, "date"], full.loc[mask, "code"]))
            != set(zip(declared.date, declared.code))):
        raise ValueError("fixed 551-cell evidence roster differs")
    checks = []
    for fold in p["folds"]:
        train = common.history.training_rows(full, fold["start"], "relay")
        expected = next(v for v in receipt["training_checks"] if v["fold"] == fold["id"])
        changed = common.history.training_rows(masked, fold["start"], "relay")
        if (common.history._frame_digest(train, common.RELAY_FIELDS) != expected["training_digest"]
                or train.target_date.max() != expected["training_max_label_date"]
                or len(train) != expected["train_n"] or train.date.nunique() != 252):
            raise ValueError("original members/inputs/maturity differ")
        pd.testing.assert_frame_equal(train.drop(columns="signal_one_word"),
                                      changed.drop(columns="signal_one_word"))
        n = int(ambiguous_mask(train).sum())
        if n != qualifications["per_original_training_window"][fold["id"]]["ambiguous_proxy_n"]:
            raise ValueError("original training mask count differs")
        checks.append({**expected, "masked_cells": n,
            "challenger_digest": common.history._frame_digest(changed, common.RELAY_FIELDS),
            "members_labels_daily_weights_equal": True})
    recovery = Path("research/model_upgrade_validation/proxy_evidence_20261013T122155Z/source_recovery_and_roundtrip_verification.json").resolve()
    recovered = common.read(recovery)
    preimage = recovered["preimage"]
    common.verify({preimage["preimage_path"]: preimage["old_sha256"],
                   preimage["new_distinct_module_path"]: preimage["new_module_sha256"]})
    if not recovered["new_complete18and10_rows_equal"]:
        raise ValueError("source recovery incomplete")
    sources = dict(p["source_bindings"])
    for filename in [protocol_path, declared_path, recovery, __file__, "relay_context_research.py",
                     "relay_expert_research.py", "model_upgrade_history.py", "relay_class_balance_research.py",
                     "cycle_regime_diagnostics.py", "rolling_forecast_challenger_v274.py",
                     "next_session_forecast_lab_v271.py", "tests/test_relay_proxy_evidence_research.py"]:
        path = Path(filename).resolve(); sources[str(path)] = common.sha(path)
    evidence = {"formed_at": common.now(), "started_at": started, "budget_minutes": 30,
        "protocol": str(protocol_path), "protocol_sha256": common.sha(protocol_path),
        "sources": sources, "runtime_versions": receipt["runtime_versions"], "checks": checks,
        "total_masked_cells": int(mask.sum()), "BASE_refits": 0, "new_forward_samples": 0,
        "original_source_recovery_reused": str(recovery), "fit_count": 0}
    common.write_new(root / "implementation_BEFORE_FITS.json", evidence)
    return evidence


def compare(receipt_path):
    e = common.read(receipt_path); root = Path(receipt_path).resolve().parent
    common.verify(e["sources"])
    p = common.read(e["protocol"]); validate_definition(p)
    receipt = common.read(p["prepared_source_receipt"])
    for name, version in e["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("runtime drift")
    if (root / "attempt_STARTED.json").exists():
        raise ValueError("comparison already attempted; no repeat fitting")
    # Exclusive shared lock follows the immutable hypothesis across directories.
    lock = Path(e["protocol"]).parent / "PROXY_EVIDENCE_ATTEMPT_STARTED.json"
    common.write_new(lock, {"formed_at": common.now(), "implementation_receipt": str(Path(receipt_path).resolve()),
                            "receipt_sha256": common.sha(receipt_path)})
    common.write_new(root / "attempt_STARTED.json", {"formed_at": common.now(), "shared_lock": str(lock)})
    full = joblib.load(p["prepared_path"])["relay"]
    masked, _ = apply_mask(full)
    saved = pd.read_csv(p["baseline_predictions"], dtype={"code": str}, low_memory=False)
    comparisons, rows, logs = {}, [], []
    survivor = None
    for fold in p["folds"]:
        if (datetime.now(timezone.utc) - datetime.fromisoformat(e["started_at"])).total_seconds() >= 1800:
            raise ValueError("30-minute batch budget reached; retain completed folds")
        common.verify(e["sources"])
        train = common.history.training_rows(masked, fold["start"], "relay")
        test = masked[masked.date.between(fold["start"], fold["end"])].copy()
        original_test = full.loc[test.index]
        base = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])]
        if len(base) != len(test) or set(zip(base.date, base.code)) != set(zip(test.date, test.code)):
            raise ValueError("full evaluation roster changed")
        br = common.project(base, "BASE", receipt["sessions"])
        bm = common.score(br, receipt["sessions"], fold)
        common.write_new(root / (fold["id"] + "_fit_STARTED.json"), {
            "formed_at": common.now(), "fit_attempts": 1, "train_members": len(train),
            "training_digest": common.history._frame_digest(train, common.RELAY_FIELDS)})
        try:
            model = common.history._fit(train, common.RELAY_FIELDS, "relay")
            pred = common.history._predict(model, test, common.RELAY_FIELDS, "relay")
            buffer = BytesIO(); joblib.dump(model, buffer); buffer.seek(0)
            loaded = common.history._predict(joblib.load(buffer), test, common.RELAY_FIELDS, "relay")
            reproduction = common.history.compare_reproduction(loaded, pred, "relay")
        except Exception as exc:
            common.write_new(root / "FIT_FAILURE_STOP.json", {"formed_at": common.now(), "fold": fold["id"],
                "error": type(exc).__name__ + ": " + str(exc), "no_retry": True})
            raise
        cr = common.project(pred, ARM, receipt["sessions"])
        truths = {(r["signal_date"], r["code"]): r["truth"] for r in br}
        if any(truths[(r["signal_date"], r["code"])] != r["truth"] for r in cr):
            raise ValueError("paired target labels changed")
        cm = common.score(cr, receipt["sessions"], fold)
        groups = {"BASE": subgroup(br, original_test), ARM: subgroup(cr, original_test)}
        guards = common.guards(bm, cm)
        bg, cg = groups["BASE"]["true_nonword"], groups[ARM]["true_nonword"]
        if fold["id"] == "2026Sep":
            guards["September_actual_nonword_Recall"] = cg["complete_positive_Recall"] >= bg["complete_positive_Recall"]
        delta = cm["Accuracy"] - bm["Accuracy"]
        comparisons[fold["id"]] = {"BASE": bm, ARM: cm, "Accuracy_delta": delta,
            "guards": guards, "passed": delta >= .03 and all(guards.values()), "subgroups": groups}
        logs.append({"fold": fold["id"], "fit_attempts": 1, "train_members": len(train),
                     "save_load_reproduction": reproduction})
        rows.extend([{**r, "fold": fold["id"]} for r in br + cr])
        pd.DataFrame(rows).to_csv(root / "full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        common.write_new(root / (fold["id"] + "_checkpoint.json"), comparisons[fold["id"]])
        if fold["id"] == "2026Sep": survivor = model
        print(fold["id"], bm["Accuracy"], cm["Accuracy"], "passed", comparisons[fold["id"]]["passed"], flush=True)
    passed = all(v["passed"] for v in comparisons.values())
    artifact = None
    if passed:
        artifact = root / "research_only_survivor.joblib"; joblib.dump(survivor, artifact)
    common.verify(e["sources"])
    result = {"formed_at": common.now(), "status": "HISTORICAL_SURVIVOR_FORWARD_GATES_REQUIRED" if passed else "REJECTED_PROXY_EVIDENCE_UNKNOWN",
        "comparisons": comparisons, "fit_logs": logs, "fit_count": len(logs), "BASE_refits": 0,
        "model_path": str(artifact) if artifact else None, "deployment_changed": False,
        "new_mature_forward_samples": 0, "historical_is_blind": False, "calibrated_Brier": None,
        "receipt_sha256": common.sha(receipt_path), "predictions_sha256": common.sha(root / "full_candidate_comparison.csv")}
    common.write_new(root / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for field in ("protocol", "output-dir", "started"): prep.add_argument("--" + field, required=True)
    run = sub.add_parser("compare"); run.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = prepare(args.protocol, args.output_dir, args.started) if args.command == "prepare" else compare(args.receipt)
    print(result.get("status", "SOURCE_MEMBER_MASK_RECEIPT_READY"))


if __name__ == "__main__":
    main()
