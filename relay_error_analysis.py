"""Source-bound stock explanations and one fixed nonlinear relay comparison.

Independent of production and the original three-arm experiment. No score edits,
threshold search, retrospective forecast registration, or fits at import time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from types import FunctionType

import joblib
import numpy as np
import pandas as pd

import model_upgrade_history as history
from current_model_diagnostics import RELAY_FIELDS, _read, sha
from relay_class_balance_research import project, statistics

PARAMETERS = dict(n_estimators=160, max_depth=5, min_samples_leaf=15,
                  max_features=.75, random_state=271, n_jobs=2,
                  class_weight=None, criterion="gini", bootstrap=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify_sources(sources):
    for name, expected in sources.items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError("source/program hash conflict: " + name)


def components(estimator, inputs, fields):
    """Exact standardized-logit terms, not causal effects or score increments."""
    if fields != RELAY_FIELDS or list(estimator.classes_) != [0, 1]:
        raise ValueError("original ordered relay fields/classes required")
    x = inputs[fields].apply(pd.to_numeric, errors="coerce")
    if np.isinf(x.to_numpy(float)).any() or not x.notna().any(axis=1).all():
        raise ValueError("nonfinite or completely unknown explanatory input")
    imputer, scaler, classifier = list(estimator.named_steps.values())
    names = list(imputer.get_feature_names_out(fields))
    imputed = imputer.transform(x)
    transformed = scaler.transform(imputed)
    terms = transformed * classifier.coef_[0]
    logits = classifier.intercept_[0] + terms.sum(axis=1)
    raw = estimator.predict_proba(x)[:, 1]
    reconstructed = 1 / (1 + np.exp(-logits))
    if not np.allclose(raw, reconstructed, atol=1e-12, rtol=0):
        raise ValueError("explanation does not reproduce the original estimator")
    return names, terms, logits, raw, imputed, scaler.mean_, float(classifier.intercept_[0])


def replay_feature_archive(path, cutoff, policy_path):
    """Explicit read-only adapter for one documented protected-runner change.

    The original numeric function and all its actual dependencies stay intact.
    A local function namespace supplies an explicit verifier; no global monkey
    patch, fabricated hash, on-disk program change or production bypass occurs.
    """
    import model_upgrade_inputs as inputs
    policy = _read(policy_path)
    verify_sources(policy["authorization_receipts"])
    verify_sources({policy["exception_path"]: policy["current_sha256"],
                    policy["preserved_original"]["path"]: policy["original_sha256"],
                    policy["original_protocol_path"]: policy["original_protocol_sha256"]})
    authorization = [_read(p) for p in policy["authorization_receipts"]]
    if not any(a.get("source_hashes", {}).get(policy["exception_path"]) == policy["current_sha256"] for a in authorization):
        raise ValueError("current protected runner lacks its actual G1 source receipt")
    def verify_original_protocol(protocol_path):
        if Path(protocol_path).resolve() != Path(policy["original_protocol_path"]).resolve():
            raise ValueError("read-only adapter cannot change protocols")
        p = _read(protocol_path)
        for group in ("source_hashes", "protected_sha256"):
            for name, expected in p[group].items():
                if group == "protected_sha256" and name == policy["exception_path"]:
                    if expected != policy["original_sha256"]:
                        raise ValueError("protected original/preimage hash conflict")
                else:
                    verify_sources({name: expected})
        if (p["version"] != "model-upgrade-fixed-1" or p["freeze_cutoff"] != "2026-09-01"
                or p["forward_sessions"] != 120 or p["preferred_arm"] != "C1"):
            raise ValueError("original model protocol semantics changed")
        receipt = _read(Path(protocol_path).parent/"implementation_BEFORE_FITS.json")
        if receipt["protocol_sha256"] != sha(protocol_path):
            raise ValueError("original protocol receipt conflict")
        return {**p, "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": sha(protocol_path)}
    directory = Path(path); manifest = _read(directory/"manifest.json")
    verify_sources(manifest["source_hashes"] | {str(directory/"features.json"): manifest["features_sha256"]})
    if (manifest["version"] != inputs.VERSION or inputs.stamp(manifest["knowledge_cutoff"]) > inputs.stamp(manifest["formed_at"])
            or inputs.stamp(manifest["formed_at"]) > inputs.stamp(cutoff)):
        raise ValueError("original feature formation/cutoff conflict")
    derive = FunctionType(inputs._derive.__code__, {**inputs._derive.__globals__, "verify_protocol": verify_original_protocol},
                          inputs._derive.__name__, inputs._derive.__defaults__)
    rebuilt = derive(manifest["protocol_path"], manifest["source_archive"], manifest["calendar_path"],
                     manifest["history_archives"], manifest["knowledge_cutoff"])
    saved = _read(directory/"features.json"); rebuilt["sources"] = saved["sources"]
    if rebuilt != saved:
        raise ValueError("read-only replay differs from original raw-source feature values")
    return {**saved, "manifest_sha256": sha(directory/"manifest.json")}


def explain(base_path, output, history_path=None, mainline_path=None, reader_policy=None):
    b = _read(base_path)
    binding = b["model_bindings"]["relay"]
    verify_sources({binding["path"]: binding["sha256"],
                    binding["model_path"]: binding["model_sha256"]})
    manifest = _read(binding["path"])
    if manifest["fields"] != RELAY_FIELDS or manifest["sha256"] != binding["model_sha256"]:
        raise ValueError("model manifest/input semantics conflict")
    if not reader_policy:
        from model_upgrade_inputs import load_feature_archive
        features = load_feature_archive(b["feature_archive"], b["knowledge_cutoff"])
    else:
        features = replay_feature_archive(b["feature_archive"], b["knowledge_cutoff"], reader_policy)
    if (features["signal_date"], features["target_date"]) != (b["signal_date"], b["target_date"]):
        raise ValueError("original dates conflict")
    observed = {r["code"]: r for r in features["relay"]}
    calls = b["relay"]["candidates"]
    if set(observed) != {r["code"] for r in calls} or len(observed) != len(calls):
        raise ValueError("full original candidate roster mismatch")
    estimator = joblib.load(binding["model_path"])
    x = pd.DataFrame([observed[r["code"]]["features"] for r in calls])
    names, terms, logits, raw, imputed, means, intercept = components(estimator, x, RELAY_FIELDS)
    prior = pd.read_csv(history_path, dtype={"code": str}, low_memory=False) if history_path else None
    mainline = _read(mainline_path) if mainline_path else None
    if mainline and mainline["original_base_prediction_binding"]["sha256"] != sha(base_path):
        raise ValueError("mainline is not bound to the same original forecast")
    rows = []
    for i, call in enumerate(calls):
        if abs(raw[i] - call["raw_score"]) > 1e-8 or bool(raw[i] > .5) != call["predicted_positive"]:
            raise ValueError("saved original class/score does not reproduce")
        detail = [{"field": name, "signal_value": None if name not in RELAY_FIELDS or pd.isna(x.iloc[i][name]) else float(x.iloc[i][name]),
                   "imputed_value": float(imputed[i, j]), "training_mean": float(means[j]),
                   "logit_contribution": float(terms[i, j])} for j, name in enumerate(names)]
        delta = None
        if prior is not None:
            candidates = prior[(prior.arm == "BASE") & (prior.code == call["code"]) & (prior.date < b["signal_date"])]
            if not candidates.empty:
                previous = candidates.sort_values("date").iloc[-1]
                _, pt, pl, ps, _, _, _ = components(estimator, pd.DataFrame([previous]), RELAY_FIELDS)
                delta = {"prior_signal_date": previous.date, "replayed_with_same_frozen_model": True,
                         "same_model_prior_score": float(ps[0]), "original_saved_prior_score": float(previous.raw_score),
                         "logit_change": float(logits[i] - pl[0]),
                         "terms": sorted([{"field": name, "logit_change": float(terms[i,j]-pt[0,j])}
                                          for j, name in enumerate(names)], key=lambda r: r["logit_change"])}
        supported = [t["theme_name"] for t in mainline.get("candidates", [])
                     if t.get("structure_qualified") and call["code"] in t["relay_codes"]] if mainline else []
        rows.append({**call, "logit": float(logits[i]), "intercept": intercept,
                     "terms": sorted(detail, key=lambda r: r["logit_contribution"]), "prior_signal_comparison": delta,
                     "missing_fields": [name for name in RELAY_FIELDS if pd.isna(x.iloc[i][name])],
                     "theme_structure_conflict": bool(supported and not call["predicted_positive"]),
                     "independent_supported_themes": supported,
                     "theme_used_in_original_numeric_model": False, "target_truth": None})
    result = {"status": "SOURCE_REBUILT_EXACT_ORIGINAL_LOGITS", "formed_at": now(),
              "signal_date": b["signal_date"], "target_date": b["target_date"],
              "original_knowledge_cutoff": b["knowledge_cutoff"], "full_candidate_count": len(rows),
              "source_bindings": {str(Path(base_path).resolve()): sha(base_path),
                  binding["path"]: binding["sha256"], binding["model_path"]: binding["model_sha256"],
                  str(Path(b["feature_archive"])/"manifest.json"): features["manifest_sha256"],
                  **({str(Path(reader_policy).resolve()): sha(reader_policy), str(Path(__file__).resolve()): sha(__file__)} if reader_policy else {}),
                  **({str(Path(history_path).resolve()): sha(history_path)} if history_path else {}),
                  **({str(Path(mainline_path).resolve()): sha(mainline_path)} if mainline_path else {})},
              "mainline_annex_actual_cutoff": mainline.get("knowledge_cutoff") if mainline else None,
              "interpretation": "Terms are standardized contributions relative to the training mean plus intercept. Coefficients are conditional, correlated and not causal; they are not points to add to the score. Prior replay uses the same current frozen model, not an earlier author's amended prediction. Independent theme annex formed later does not become an original numeric feature.",
              "production_changed": False, "calibrated_probability": False, "rows": rows}
    write_new(output, result)
    return result


def fit_nonlinear(train, fields):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if fields != RELAY_FIELDS or train.empty or set(train.target.unique()) != {0., 1.}:
        raise ValueError("exact original training fields and both label classes required")
    estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), RandomForestClassifier(**PARAMETERS))
    weights = 1 / train.groupby("date").date.transform("size")
    estimator.fit(train[fields], train.target.astype(int), randomforestclassifier__sample_weight=weights)
    return estimator


def compare(protocol_path):
    p = _read(protocol_path); root = Path(protocol_path).parent
    if p["version"] != "relay-nonlinear-fixed-1" or p["arms"] != ["BASE", "R2"] or p["folds"] != history.FOLDS:
        raise ValueError("fixed experiment protocol conflict")
    if p["parameters"] != {**PARAMETERS, "window": 252, "threshold": .5}:
        raise ValueError("fixed nonlinear parameters changed")
    verify_sources(p["sources"])
    receipt = _read(p["cached_receipt"])
    verify_sources(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    if receipt["fields"] != RELAY_FIELDS or any(importlib.metadata.version(k) != v for k, v in receipt["runtime_versions"].items()):
        raise ValueError("input fields/runtime conflict")
    if (root/"attempt_STARTED.json").exists():
        raise ValueError("fixed comparison already attempted; no repeat or parameter search")
    implementation = _read(root/"implementation_BEFORE_FITS.json")
    correction_path = root/"implementation_correction_BEFORE_FITS.json"
    if correction_path.exists():
        correction = _read(correction_path)
        if correction["original_receipt_sha256"] != sha(root/"implementation_BEFORE_FITS.json") or correction["numeric_fits"] != 0:
            raise ValueError("pre-fit reader correction receipt conflict")
        verify_sources(correction["preserved_originals"])
        implementation = correction
    verify_sources(implementation["sources"])
    write_new(root/"attempt_STARTED.json", {"formed_at": now(), "protocol_sha256": sha(protocol_path), "challenge": "R2"})
    full = joblib.load(receipt["prepared_path"])
    saved = pd.read_csv(Path(p["base_root"])/"relay_history/predictions.csv", dtype={"code": str}, low_memory=False)
    comparisons, rows, logs, failed = {}, [], [], False
    for fold in p["folds"]:
        train = history.training_rows(full, fold["start"], "relay")
        log = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if history._frame_digest(train, RELAY_FIELDS) != log["training_digest"]:
            raise ValueError("training values/members differ from original BASE")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        baseline = saved[(saved.arm == "BASE") & (saved.fold == fold["id"])].copy()
        if set(zip(test.date, test.code)) != set(zip(baseline.date, baseline.code)) or len(test) != len(baseline):
            raise ValueError("full same-date candidate roster differs")
        error = None
        try:
            if failed or (datetime.now(timezone.utc)-datetime.fromisoformat(p["batch_started_at"])).total_seconds() >= 1800:
                raise ValueError("prior fit failed or fixed batch budget reached")
            estimator = fit_nonlinear(train, RELAY_FIELDS)
            predicted = history._predict(estimator, test, RELAY_FIELDS, "relay")
        except Exception as exc:
            failed = True; error = type(exc).__name__ + ": " + str(exc)
            predicted = test.copy(); predicted["status"] = "FIT_FAILED_UNKNOWN"
            predicted["predicted"] = None; predicted["raw_score"] = np.nan
        br, cr = project(baseline, "BASE", receipt["sessions"]), project(predicted, "R2", receipt["sessions"])
        indexed = {(r["signal_date"], r["code"]): r for r in br}
        if any(r["truth"] != indexed[(r["signal_date"], r["code"])]["truth"] for r in cr):
            raise ValueError("challenger/baseline label conflict")
        bm, cm = statistics(br, receipt["sessions"], fold), statistics(cr, receipt["sessions"], fold)
        delta = cm["Accuracy"]-bm["Accuracy"] if cm["Accuracy"] is not None and bm["Accuracy"] is not None else None
        guards = {k: cm[k] is not None and bm[k] is not None and cm[k] >= bm[k]-1e-12
                  for k in ("Precision", "complete_positive_Recall", "F1", "coverage")}
        bt, ct = bm["true_nonword_rank_first"], cm["true_nonword_rank_first"]
        guards["true_nonword_rank_first"] = bt["rate"] is None or (ct["rate"] is not None and ct["rate"] >= bt["rate"] and ct["scoreable_days"] >= bt["scoreable_days"])
        comparisons[fold["id"]] = {"BASE": bm, "R2": cm, "Accuracy_delta": delta, "guards": guards,
            "historical_screen_passed": bool(not error and delta is not None and delta >= .03 and all(guards.values())), "fit_failure": error}
        rows.extend([{**r, "fold": fold["id"]} for r in br + cr])
        logs.append({**log, "fitted_once": error is None, "fit_failure": error, "model_saved_for_deployment": False})
        if error is None:
            importance = estimator.named_steps["randomforestclassifier"].feature_importances_
            logs[-1]["feature_importance_diagnostic"] = dict(zip(estimator.named_steps["simpleimputer"].get_feature_names_out(RELAY_FIELDS), map(float, importance)))
    output = root/"full_candidate_comparison.csv"
    if output.exists():
        raise ValueError("cannot overwrite previous comparison")
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    passed = all(c["historical_screen_passed"] for c in comparisons.values())
    result = {"status": "HISTORICAL_SCREEN_PASSED_NEW_FORWARD_REQUIRED" if passed else "REJECTED_FIXED_NONLINEAR_CHALLENGE",
              "formed_at": now(), "protocol_sha256": sha(protocol_path), "comparisons": comparisons,
              "fit_logs": logs, "source_sha256": sha(output), "production_model_changed": False,
              "new_mature_forward_samples": 0, "deployment_weights_saved": False,
              "calibrated_probability_Brier": None, "history_is_blind_or_forward": False}
    write_new(root/"comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("explain", "compare"))
    parser.add_argument("--protocol"); parser.add_argument("--base"); parser.add_argument("--output")
    parser.add_argument("--history"); parser.add_argument("--mainline")
    parser.add_argument("--reader-policy")
    args = parser.parse_args()
    result = compare(args.protocol) if args.command == "compare" else explain(args.base, args.output, args.history, args.mainline, args.reader_policy)
    print(json.dumps({"status": result["status"], "formed_at": result["formed_at"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
