"""One preregistered training-only relay balance challenge; never production."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

import model_upgrade_history as history
from current_model_diagnostics import RELAY_FIELDS, sha, _read, _write, _boolean, _number
from cycle_regime_diagnostics import metrics


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol(path):
    p = _read(path)
    if (p["version"] != "relay-training-class-balance-fixed-1" or p["arms"] != ["BASE", "R1"]
            or p["parameters"] != {"C": .25, "max_iter": 2500, "random_state": 271, "solver": "lbfgs", "window": 252, "threshold": .5}
            or p["folds"] != history.FOLDS):
        raise ValueError("fixed relay balance protocol changed")
    for key in ("baseline_sources", "production_originals"):
        for name, wanted in p[key].items():
            if not Path(name).is_file() or sha(name) != wanted:
                raise ValueError("original source changed: " + name)
    return p


def training_weights(train):
    if train.empty or set(train.target.unique()) != {0., 1.}:
        raise ValueError("both known original training classes required")
    weights = 1/train.groupby("date").date.transform("size")
    masses = {c: float(weights[train.target.eq(c)].sum()) for c in (0, 1)}
    total = float(weights.sum())
    factors = {c: total/(2*masses[c]) for c in (0, 1)}
    if not all(np.isfinite(v) and v > 0 for v in factors.values()):
        raise ValueError("invalid training-only class factors")
    adjusted = weights * train.target.map(factors)
    if not np.isclose(adjusted.sum(), total, atol=1e-10, rtol=0):
        raise ValueError("class balance must preserve total daily weight mass")
    return weights, factors, masses


def fit_once(train, fields):
    from rolling_forecast_challenger_v274 import make
    from sklearn.exceptions import ConvergenceWarning
    if set(fields)-set(RELAY_FIELDS) or not train[fields].notna().any().any():
        raise ValueError("unregistered future fields or completely unknown training inputs")
    weights, factors, masses = training_weights(train)
    estimator = make("logistic")
    step = estimator.steps[-1][0]
    estimator.set_params(**{step+"__class_weight": factors})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(train[fields], train.target.astype(int), **{step+"__sample_weight": weights})
    return estimator, {"class_factors": {str(k): v for k, v in factors.items()},
        "original_class_weight_mass": {str(k): v for k, v in masses.items()}, "original_daily_weight_total": float(weights.sum())}


def prepare(path):
    p = protocol(path); root = Path(path).parent
    if (root/"implementation_BEFORE_FITS.json").exists():
        raise ValueError("pre-fit inputs already frozen; do not reconstruct or overwrite")
    data = history.prepare_datasets()  # Original source reconstruction only; no fits.
    fields = data["relay_fields"]
    manifest = _read(Path(p["base_root"])/"models/relay/BASE/manifest.json")
    if fields != manifest["fields"] or fields != RELAY_FIELDS:
        raise ValueError("original 29-field order changed")
    logs = _read(Path(p["base_root"])/"relay_history/report.json")["fit_logs"]
    checks = []
    for fold in p["folds"]:
        train = history.training_rows(data["relay"], fold["start"], "relay")
        saved = next(log for log in logs if log["arm"] == "BASE" and log["fold"] == fold)
        fingerprint = history._frame_digest(train, fields)
        if (fingerprint != saved["training_digest"] or len(train) != saved["train_n"]
                or train.date.nunique() != saved["training_dates"]
                or train.target_date.max() != saved["training_max_label_date"]):
            raise ValueError("training members/values/label cutoff differ from frozen BASE")
        checks.append({"fold": fold["id"], "training_digest": fingerprint, "train_n": len(train),
            "training_dates": int(train.date.nunique()), "training_max_label_date": train.target_date.max()})
    joblib.dump(data["relay"], root/"prepared_relay.joblib")
    _write(root/"provenance_BEFORE_FITS.json", {"formed_at": now(), "protocol_sha256": sha(path),
        "provenance": data["provenance"], "training_checks": checks})
    sources = data["provenance"]["source_hashes"] | data["provenance"]["dependency_hashes"]
    sources[str(Path(__file__).resolve())] = sha(__file__)
    sources[str(Path("cycle_regime_diagnostics.py").resolve())] = sha("cycle_regime_diagnostics.py")
    sources[str(Path("current_model_diagnostics.py").resolve())] = sha("current_model_diagnostics.py")
    _write(root/"implementation_BEFORE_FITS.json", {"formed_at": now(), "protocol_sha256": sha(path),
        "sources": sources, "prepared_path": str((root/"prepared_relay.joblib").resolve()),
        "prepared_sha256": sha(root/"prepared_relay.joblib"), "fields": fields,
        "training_checks": checks, "sessions": data["provenance"]["sessions"],
        "runtime_versions": data["provenance"]["runtime_versions"], "python_version": sys.version})
    return {"status": "BEFORE_FITS_INPUTS_FROZEN", "training_checks": checks}


def project(frame, arm, sessions):
    successor = dict(zip(sessions[:-1], sessions[1:]))
    rows = []
    for raw in frame.to_dict("records"):
        valid = _boolean(raw.get("label_valid")) is True and raw.get("target_date") == successor.get(raw["date"])
        target = _boolean(raw.get("target")) if valid else None
        call = _boolean(raw.get("predicted")) if raw.get("status") == "PREDICTED_UNCALIBRATED" else None
        rows.append({"arm": arm, "signal_date": raw["date"], "target_date": raw.get("target_date"),
            "code": raw["code"], "name": raw.get("name"), "boards": int(raw["boards"]),
            "prediction": call, "truth": target, "raw_score": _number(raw.get("raw_score")),
            "actual_one_word_daily": _number(raw.get("actual_one_word_daily")),
            "truth_status": "KNOWN" if target is not None else "UNKNOWN_LABEL_OR_NONEXACT_TARGET"})
    return rows


def statistics(rows, sessions, fold):
    result = metrics(rows)
    result["by_boards"] = {str(b): metrics([r for r in rows if r["boards"] == b]) for b in sorted({r["boards"] for r in rows})}
    scored = [r for r in rows if r["truth"] is not None and r["raw_score"] is not None]
    result["raw_score_squared_loss_diagnostic"] = sum((r["raw_score"]-int(r["truth"]))**2 for r in scored)/len(scored) if scored else None
    result["raw_score_loss_is_calibrated_probability_Brier"] = False
    days = [d for d in sessions if fold["start"] <= d <= fold["end"]]
    daily = []
    for day in days:
        cohort = [r for r in rows if r["signal_date"] == day]
        state, selected = "NO_SAVED_CANDIDATES", None
        if cohort:
            if any(r["raw_score"] is None for r in cohort): state = "UNKNOWN_SCORE"
            elif any(r["actual_one_word_daily"] not in (0, 1) for r in cohort): state = "UNKNOWN_OHLC_IDENTITY"
            else:
                ordered = sorted(cohort, key=lambda r: (-r["raw_score"], r["code"]))
                selected = next((r for r in ordered if r["actual_one_word_daily"] == 0), None)
                state = "NO_NONWORD" if not selected else "MISSING_LABEL" if selected["truth"] is None else "SCORED"
        daily.append({"signal_date": day, "status": state, "selected_code": selected["code"] if selected else None,
            "original_prediction": selected["prediction"] if selected else None, "truth": selected["truth"] if selected else None})
    known = [r for r in daily if r["status"] == "SCORED"]
    result["true_nonword_rank_first"] = {"official_days": len(days), "scoreable_days": len(known),
        "promotions": sum(r["truth"] is True for r in known),
        "rate": sum(r["truth"] is True for r in known)/len(known) if known else None,
        "daily_rows": daily, "not_trading_return": True}
    return result


def run(path):
    p = protocol(path); root = Path(path).parent
    if (root/"attempt_STARTED.json").exists():
        raise ValueError("fixed experiment already attempted; no repeated weighting/parameter search")
    receipt = _read(root/"implementation_BEFORE_FITS.json")
    if receipt["protocol_sha256"] != sha(path) or sha(receipt["prepared_path"]) != receipt["prepared_sha256"]:
        raise ValueError("pre-fit input/protocol hash conflict")
    for name, wanted in receipt["sources"].items():
        if not Path(name).is_file() or sha(name) != wanted:
            raise ValueError("pre-fit implementation/source hash conflict: "+name)
    if any(importlib.metadata.version(k) != v for k, v in receipt["runtime_versions"].items()):
        raise ValueError("runtime version differs from frozen preparation")
    _write(root/"attempt_STARTED.json", {"formed_at": now(), "protocol_sha256": sha(path), "fixed_challenge": "R1"})
    full = joblib.load(receipt["prepared_path"])
    saved = pd.read_csv(Path(p["base_root"])/"relay_history/predictions.csv", dtype={"code": str})
    fields, sessions = receipt["fields"], receipt["sessions"]
    comparisons, output_rows, fit_logs = {}, [], []
    for fold in p["folds"]:
        train = history.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        baseline = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])].copy()
        left = set(zip(test.date, test.code)); right = set(zip(baseline.date, baseline.code))
        if left != right or len(test) != len(baseline):
            raise ValueError("fixed same-candidate comparison roster differs")
        log = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if history._frame_digest(train, fields) != log["training_digest"]:
            raise ValueError("frozen training members changed before fit")
        failure = None
        try:
            estimator, details = fit_once(train, fields)
            predicted = history._predict(estimator, test, fields, "relay")
        except Exception as error:
            failure = type(error).__name__+": "+str(error)
            details = {"failure": failure}; predicted = test.copy()
            predicted["status"] = "FIT_FAILED_UNKNOWN"; predicted["predicted"] = None; predicted["raw_score"] = np.nan
        br, cr = project(baseline, "BASE", sessions), project(predicted, "R1", sessions)
        bm, cm = statistics(br, sessions, fold), statistics(cr, sessions, fold)
        bkeys = {(r["signal_date"], r["code"]): r for r in br}
        for row in cr:
            if row["truth"] != bkeys[(row["signal_date"], row["code"])]["truth"]:
                raise ValueError("challenger and baseline labels differ")
        accuracy_delta = cm["Accuracy"]-bm["Accuracy"] if cm["Accuracy"] is not None and bm["Accuracy"] is not None else None
        not_down = lambda key: bm[key] is not None and cm[key] is not None and cm[key] >= bm[key]-1e-12
        btop, ctop = bm["true_nonword_rank_first"], cm["true_nonword_rank_first"]
        top_guard = btop["rate"] is None or (ctop["rate"] is not None and ctop["rate"] >= btop["rate"] and ctop["scoreable_days"] >= btop["scoreable_days"])
        passed = not failure and accuracy_delta is not None and accuracy_delta >= .03 and all(not_down(k) for k in ("complete_positive_Recall", "F1", "coverage")) and top_guard
        comparisons[fold["id"]] = {"BASE": bm, "R1": cm, "Accuracy_delta": accuracy_delta,
            "historical_screen_passed": bool(passed), "complete_original_candidates": len(test), "fit_failure": failure}
        output_rows.extend([{**r, "fold": fold["id"]} for r in br+cr])
        fit_logs.append({**log, **details, "fitted_once": failure is None, "fields": fields,
            "model_saved_for_deployment": False})
        if failure:
            break  # Retain the failed attempt; no alternate fit or optimizer.
    pd.DataFrame(output_rows).to_csv(root/"full_candidate_comparison.csv", index=False, encoding="utf-8-sig")
    all_pass = len(comparisons) == 3 and all(v["historical_screen_passed"] for v in comparisons.values())
    result = {"status": "HISTORICAL_SCREEN_PASSED_STILL_NEEDS_NEW_FORWARD_PROTOCOL" if all_pass else "REJECTED_FIXED_CHALLENGE_NO_PRODUCTION_CHANGE",
        "formed_at": now(), "protocol_sha256": sha(path), "implementation_sha256": sha(root/"implementation_BEFORE_FITS.json"),
        "comparisons": comparisons, "fit_logs": fit_logs, "production_model_changed": False,
        "deployment_weights_saved": False, "new_mature_forward_samples": 0,
        "evidence_class": "PREVIOUSLY_INSPECTED_RETROSPECTIVE_NOT_BLIND_NOT_FORWARD",
        "predictions_sha256": sha(root/"full_candidate_comparison.csv"),
        "calibrated_probability_Brier": None, "fixed_threshold_unchanged": True}
    _write(root/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run")); parser.add_argument("--protocol", required=True)
    args = parser.parse_args(); result = prepare(args.protocol) if args.command == "prepare" else run(args.protocol)
    print(json.dumps({"status": result["status"], "output": str(Path(args.protocol).parent)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
