"""Single preregistered retrospective price-shape test; no production apply."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import relay_context_research as common

ARM = "SIGNAL_PRICE_SHAPE"
HYPOTHESIS_ROOT = Path(__file__).resolve().parent / "research/model_upgrade_validation/early_signal_ohlc_20261013T1303Z"
PARAMETERS = {"algorithm": "original logistic", "C": .25, "max_iter": 2500,
    "random_state": 271, "solver": "lbfgs", "class_weight": None,
    "training_window": 252, "window_unit": "last252distinct original valid-labelled signal dates",
    "daily_weight": "inverse original full training candidate rows per date",
    "preprocess": "training-only median/add_indicator/keep_empty_features/scaler unchanged",
    "threshold": "raw positive score strictly >0.5"}
SCREEN = {"Accuracy_delta_each_fold_at_least": .03,
    "complete_positive_Recall_each_fold_not_lower": True, "Precision_each_fold_not_lower": True,
    "F1_each_fold_not_lower": True, "original_prediction_and_label_scoring_coverage_unchanged": True,
    "price_shape_nonword_Recall_and_F1_each_defined_fold_not_lower": True,
    "positive_call_coverage_all_original_fold_not_lower": True,
    "unknowns_remain_in_full_original_denominator": True, "all_fold_conditions_required": True,
    "undefined_required_metric_cannot_certify_pass": True}


def shape(row):
    """Recheck price pattern from source facts, never future outcomes or proxies."""
    values = []
    for source in row["sources"]:
        if (source["date"], source["code"]) != (row["date"], row["code"]):
            raise ValueError("target-day/code source substitution")
        prices = tuple(Decimal(str(source[k])) for k in ("open", "high", "low", "close"))
        if any(not v.is_finite() or v <= 0 or v * 100 != (v * 100).to_integral_value() for v in prices):
            return None
        op, high, low, close = prices
        if not low <= op <= high or not low <= close <= high:
            return None
        values.append(prices)
    if not values or len(set(values)) != 1:
        return None
    op, high, low, close = values[0]
    if high != close or close != Decimal(str(row["original_price"])):
        return None
    return op == high == low == close


def attach(full, context):
    rows = context["rows"]
    index = {(r["date"], r["code"]): r for r in rows}
    if len(index) != len(rows) or set(index) != set(zip(full.date, full.code)):
        raise ValueError("complete original cohort required; missing/duplicate candidate")
    values = []
    for r in full.itertuples():
        item = index[(r.date, r.code)]
        if Decimal(str(item["original_price"])) != Decimal(str(r.price)):
            raise ValueError("original pool price changed")
        value = shape(item)
        if value is not item["provisional_one_price_session"] or item["training_truth_authorized"] is not False:
            raise ValueError("provisional price pattern or truth authorization changed")
        values.append(np.nan if value is None else float(value))
    result = full.copy()
    result["signal_one_word"] = values
    pd.testing.assert_frame_equal(full.drop(columns="signal_one_word"), result.drop(columns="signal_one_word"))
    return result


def paired_targets_equal(left, right):
    """Missing equals missing for validation; original dates/labels are untouched."""
    def token(value):
        return None if value is None or pd.isna(value) else value
    expected = {(r["signal_date"], r["code"]): (token(r["target_date"]), r["truth"]) for r in left}
    observed = {(r["signal_date"], r["code"]): (token(r["target_date"]), r["truth"]) for r in right}
    return len(expected) == len(left) and len(observed) == len(right) and expected == observed


def load(protocol_path):
    p = common.read(protocol_path)
    if (p["version"] != "relay-signal-price-shape-fixed-1" or p["arms"] != ["BASE", ARM]
            or p["fields"] != common.RELAY_FIELDS or p["folds"] != common.history.FOLDS
            or p["parameters"] != PARAMETERS or p["fixed_screen_before_effects"] != SCREEN
            or p["execution"]["maximum_challenger_fits"] != 3 or p["production_application"]["allowed"] is not False
            or p["hypothesis_id"] != "SIGNAL_PRICE_SHAPE_REPLACEMENT"
            or Path(p["context_path"]).resolve() != HYPOTHESIS_ROOT / "complete9245_provisional_price_context_DELIVERY.json"):
        raise ValueError("fixed definition changed")
    common.verify(p["source_bindings"])
    policy = common.read(p["source_policy"])
    if policy["retrospective_price_feature_test_data_qualified"] is not True or policy["deployment_qualified"] is not False:
        raise ValueError("price-only diagnostic policy required")
    receipt = common.read(p["prepared_source_receipt"])
    common.verify(receipt["sources"] | {p["prepared_path"]: p["prepared_sha256"]})
    qpath = Path(p["context_path"]).parent / "qualification_result.json"
    common.verify(common.read(qpath)["source_bindings"])
    for package, version in receipt["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("original runtime version changed")
    return p, receipt


def prepare(protocol_path, output_dir, started):
    p, receipt = load(protocol_path)
    full = joblib.load(p["prepared_path"])["relay"]
    changed = attach(full, common.read(p["context_path"]))
    saved = pd.read_csv(p["baseline_predictions"], dtype={"code": str}, low_memory=False)
    checks = []
    for fold in p["folds"]:
        original = common.history.training_rows(full, fold["start"], "relay")
        candidate = common.history.training_rows(changed, fold["start"], "relay")
        expected = next(c for c in p["training_checks"] if c["fold"] == fold["id"])
        if (common.history._frame_digest(original, p["fields"]) != expected["training_digest"]
                or common.history._frame_digest(candidate, p["fields"]) != expected["challenger_digest"]
                or len(original) != expected["train_n"] or original.date.nunique() != 252
                or original.target_date.max() != expected["training_max_label_date"]):
            raise ValueError("original training window or frozen price values changed")
        pd.testing.assert_frame_equal(original.drop(columns="signal_one_word"), candidate.drop(columns="signal_one_word"))
        baseline = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])]
        inputs = changed[changed.date.between(fold["start"], fold["end"])]
        if not paired_targets_equal(common.project(baseline, "BASE", receipt["sessions"]),
                                    common.project(inputs, ARM, receipt["sessions"])):
            raise ValueError("paired target/label mismatch before fitting")
        checks.append(expected)
    sources = dict(p["source_bindings"])
    sources.update(receipt["sources"])
    for path in [protocol_path, __file__, "tests/test_relay_price_shape_research.py", "rolling_forecast_challenger_v274.py",
                 "cycle_regime_diagnostics.py", "current_model_diagnostics.py"]:
        sources[str(Path(path).resolve())] = common.sha(path)
    e = {"formed_at": common.now(), "started_at": started, "protocol": str(Path(protocol_path).resolve()),
        "sources": sources, "checks": checks, "fit_count": 0, "BASE_refits": 0,
        "budget_minutes": 30, "source_track": "RETROSPECTIVE_PRICE_FEATURE_ONLY",
        "new_mature_forward_samples": 0}
    common.write_new(Path(output_dir) / "implementation_BEFORE_FITS.json", e)
    return e


def subgroup(rows, index):
    parts = {"single_price_shape": [], "non_single_price_shape": [], "unknown_price_shape": []}
    for row in rows:
        v = index[(row["signal_date"], row["code"])]["provisional_one_price_session"]
        parts["single_price_shape" if v is True else "non_single_price_shape" if v is False else "unknown_price_shape"].append(row)
    return {name: common.metrics(values) for name, values in parts.items()}


def claim_once(protocol_path, receipt_path, root):
    # Shared immutable lock prevents moving the same hypothesis to a new output.
    lock = Path(common.read(protocol_path)["context_path"]).parent / "SIGNAL_PRICE_SHAPE_ATTEMPT_STARTED.json"
    common.write_new(lock, {"formed_at": common.now(), "receipt": str(Path(receipt_path).resolve()),
                            "receipt_sha256": common.sha(receipt_path)})
    common.write_new(root / "attempt_STARTED.json", {"formed_at": common.now(), "shared_lock": str(lock)})


def compare(receipt_path):
    e = common.read(receipt_path)
    root = Path(receipt_path).parent
    if (root / "attempt_STARTED.json").exists() or (HYPOTHESIS_ROOT / "SIGNAL_PRICE_SHAPE_ATTEMPT_STARTED.json").exists():
        raise ValueError("comparison attempted; no repeat fitting")
    common.verify(e["sources"])
    p, receipt = load(e["protocol"])
    full = joblib.load(p["prepared_path"])["relay"]
    context = common.read(p["context_path"])
    changed = attach(full, context)
    index = {(r["date"], r["code"]): r for r in context["rows"]}
    saved = pd.read_csv(p["baseline_predictions"], dtype={"code": str}, low_memory=False)
    claim_once(e["protocol"], receipt_path, root)
    comparisons, rows, logs = {}, [], []
    survivor = None
    for fold in p["folds"]:
        if (datetime.now(timezone.utc) - datetime.fromisoformat(e["started_at"])).total_seconds() >= 1800:
            raise ValueError("30-minute batch budget reached; keep checkpoints")
        common.verify(e["sources"])
        train = common.history.training_rows(changed, fold["start"], "relay")
        test = changed[changed.date.between(fold["start"], fold["end"])].copy()
        base = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])]
        if len(base) != len(test) or set(zip(base.date, base.code)) != set(zip(test.date, test.code)):
            raise ValueError("full original evaluation cohort changed")
        br = common.project(base, "BASE", receipt["sessions"])
        common.write_new(root / (fold["id"] + "_fit_STARTED.json"), {
            "formed_at": common.now(), "fit_attempts": 1, "training_digest":
                common.history._frame_digest(train, p["fields"]), "train_n": len(train)})
        try:
            model = common.history._fit(train, p["fields"], "relay")
            pred = common.history._predict(model, test, p["fields"], "relay")
            pred.to_csv(root / (fold["id"] + "_predictions_BEFORE_VALIDATION.csv"), index=False, encoding="utf-8-sig")
            memory = BytesIO()
            joblib.dump(model, memory)
            memory.seek(0)
            loaded = common.history._predict(joblib.load(memory), test, p["fields"], "relay")
            reproduction = common.history.compare_reproduction(loaded, pred, "relay")
        except Exception as error:
            common.write_new(root / "FIT_FAILURE_STOP.json", {"formed_at": common.now(),
                "fold": fold["id"], "error": type(error).__name__ + ": " + str(error), "no_retry": True})
            raise
        cr = common.project(pred, ARM, receipt["sessions"])
        if not paired_targets_equal(br, cr):
            raise ValueError("paired truth or target changed")
        bm, cm = common.score(br, receipt["sessions"], fold), common.score(cr, receipt["sessions"], fold)
        groups = {"BASE": subgroup(br, index), ARM: subgroup(cr, index)}
        guards = {k: cm[k] is not None and bm[k] is not None and cm[k] >= bm[k] - 1e-12
                  for k in ("complete_positive_Recall", "Precision", "F1")}
        guards["coverage_equal"] = cm["coverage"] == bm["coverage"]
        guards["positive_call_coverage"] = sum(r["prediction"] is True for r in cr) >= sum(r["prediction"] is True for r in br)
        bg, cg = groups["BASE"]["non_single_price_shape"], groups[ARM]["non_single_price_shape"]
        for metric in ("complete_positive_Recall", "F1"):
            guards["non_single_price_" + metric] = (cg[metric] is not None and bg[metric] is not None
                and cg[metric] >= bg[metric] - 1e-12)
        delta = cm["Accuracy"] - bm["Accuracy"]
        comparisons[fold["id"]] = {"BASE": bm, ARM: cm, "Accuracy_delta": delta,
            "guards": guards, "passed": delta >= .03 and all(guards.values()), "price_shape_subgroups": groups}
        rows.extend([{**r, "fold": fold["id"]} for r in br + cr])
        pd.DataFrame(rows).to_csv(root / "full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        common.write_new(root / (fold["id"] + "_checkpoint.json"), comparisons[fold["id"]])
        logs.append({"fold": fold["id"], "fit_attempts": 1, "train_n": len(train), "save_load": reproduction})
        survivor = model
        print(fold["id"], "BASE", bm["Accuracy"], "shape", cm["Accuracy"], "pass", comparisons[fold["id"]]["passed"], flush=True)
    passed = all(c["passed"] for c in comparisons.values())
    artifact = None
    if passed:
        artifact = root / "research_only_survivor.joblib"
        joblib.dump(survivor, artifact)
    common.verify(e["sources"])
    result = {"formed_at": common.now(), "status": "HISTORICAL_PRICE_SURVIVOR_FORWARD_GATES_REQUIRED" if passed else "REJECTED_SIGNAL_PRICE_SHAPE",
        "comparisons": comparisons, "fit_logs": logs, "fit_count": len(logs), "BASE_refits": 0,
        "model_path": str(artifact) if artifact else None, "deployment_changed": False,
        "new_mature_forward_samples": 0, "historical_is_blind": False, "calibrated_Brier": None,
        "source_scope": "PRICE_SHAPE_DIAGNOSTIC_NOT_CERTIFIED_IDENTITY", "predictions_sha256": common.sha(root / "full_candidate_comparison.csv")}
    common.write_new(root / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for field in ("protocol", "output-dir", "started"):
        prep.add_argument("--" + field, required=True)
    run = sub.add_parser("compare")
    run.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = prepare(args.protocol, args.output_dir, args.started) if args.command == "prepare" else compare(args.receipt)
    print(result.get("status", "PREPARED_SOURCE_AND_MEMBER_RECEIPT"))
