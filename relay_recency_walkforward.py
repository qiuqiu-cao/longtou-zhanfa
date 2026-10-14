"""Preregistered monthly relay recency comparison on existing historical data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import relay_context_research as common
from rolling_forecast_challenger_v274 import make

ROOT = Path(__file__).resolve().parent / "research/model_upgrade_validation/recency_walkforward_20261013T134351Z"
ARMS = {"ROLL252": None, "HL63": 63, "HL126": 126}


def weights(train, cutoff, sessions, half_life):
    """Equal full-cohort day weights, optionally decayed; same total penalty scale."""
    if half_life is not None and half_life not in (63, 126):
        raise ValueError("unregistered half life")
    if (train.date.ge(cutoff).any() or train.target_date.ge(cutoff).any() or train.target_date.isna().any()
            or train.target.isna().any() or not train.label_valid.all()):
        raise ValueError("future or unresolved training outcome")
    index = {day: i for i, day in enumerate(sessions)}
    prior = [day for day in sessions if day < cutoff]
    if not prior or any(day not in index for day in train.date):
        raise ValueError("official training calendar missing")
    days = sorted(train.date.unique())
    decay = {day: 1. if half_life is None else float(np.exp2(-(index[prior[-1]] - index[day]) / half_life)) for day in days}
    scale = len(days) / sum(decay.values())
    result = train.date.map(decay) * scale / train.groupby("date").date.transform("size")
    if not np.isfinite(result).all() or not result.gt(0).all() or abs(result.sum() - len(days)) > 1e-8:
        raise ValueError("training weight mass changed")
    return result


def target_signature(rows):
    def token(value):
        return None if value is None or pd.isna(value) else value
    out = {(r["signal_date"], r["code"]): (token(r["target_date"]), r["truth"]) for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate original cohort")
    return out


def load(protocol_path):
    p = common.read(protocol_path)
    if (p["version"] != "relay-fixed-monthly-recency-1" or p["arms"] != ARMS
            or p["fields"] != common.RELAY_FIELDS or p["folds"] != common.history.FOLDS
            or p["monthly_grid"] != [str(m) for m in pd.period_range("2025-07", "2026-09", freq="M")]
            or Path(p["experiment_root"]) != ROOT or p["execution"]["maximum_fits"] != 45
            or p["fixed_acceptance"]["Accuracy_delta_each_original_fold_min"] != .03):
        raise ValueError("fixed recency protocol changed")
    common.verify(p["source_bindings"])
    receipt = common.read(p["prepared_receipt"])
    common.verify(receipt["sources"] | {p["prepared_path"]: p["prepared_sha256"]})
    for package, version in p["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("source runtime changed")
    return p, receipt


def prepare(protocol_path):
    p, receipt = load(protocol_path)
    full = joblib.load(p["prepared_path"])["relay"]
    sessions = common.read(p["calendar"])["sessions"]
    if full.duplicated(["date", "code"]).any():
        raise ValueError("duplicate original candidates")
    checks = []
    for expected in p["training_checks"]:
        month, cutoff = expected["month"], expected["cutoff"]
        train = common.history.training_rows(full, cutoff, "relay")
        test = full[full.date.str[:7].eq(month)]
        if (common.history._frame_digest(train, p["fields"]) != expected["training_digest"]
                or len(train) != expected["training_n"] or train.date.nunique() != 252 or len(test) != expected["test_n"]):
            raise ValueError("monthly members or inputs changed")
        w = {arm: weights(train, cutoff, sessions, life) for arm, life in ARMS.items()}
        original = 1 / train.groupby("date").date.transform("size")
        np.testing.assert_allclose(w["ROLL252"], original, atol=0, rtol=0)
        checks.append({**expected, "weight_sum": {arm: float(v.sum()) for arm, v in w.items()},
            "weight_digest": {arm: common.history.digest(v.tolist()) for arm, v in w.items()}})
        target_signature(common.project(test, "INPUT", sessions))
    sources = dict(p["source_bindings"])
    sources.update(receipt["sources"])
    for filename in [protocol_path, __file__, "tests/test_relay_recency_walkforward.py", "current_model_diagnostics.py", "relay_class_balance_research.py"]:
        path = Path(filename).resolve()
        sources[str(path)] = common.sha(path)
    e = {"formed_at": common.now(), "started_at": p["execution"]["started_at"],
        "protocol_path": str(Path(protocol_path).resolve()), "sources": sources, "checks": checks,
        "production_model_refits": 0, "actual_research_fits": 0, "new_mature_forward_samples": 0}
    common.write_new(ROOT / "implementation_BEFORE_FITS.json", e)
    return e


def fit_once(train, fields, w):
    model = make("logistic")
    params = model.steps[-1][1].get_params()
    if (params["C"], params["max_iter"], params["random_state"], params["class_weight"], params["solver"]) != (.25, 2500, 271, None, "lbfgs"):
        raise ValueError("original estimator changed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train[fields], train.target.astype(int), **{model.steps[-1][0] + "__sample_weight": w})
    return model


def groups(rows):
    parts = {
        "proxy0": [r for r in rows if r["legacy_proxy"] == 0.],
        "proxy1": [r for r in rows if r["legacy_proxy"] == 1.],
        "actual_nonword": [r for r in rows if r["actual_one_word_daily"] == 0.],
        "actual_oneword": [r for r in rows if r["actual_one_word_daily"] == 1.],
        "actual_identity_unknown": [r for r in rows if r["actual_one_word_daily"] not in (0., 1.)],
    }
    for state in sorted({r["emotion_state"] for r in rows}):
        parts["emotion:" + state] = [r for r in rows if r["emotion_state"] == state]
    for height in sorted({r["boards"] for r in rows}):
        parts["boards:" + str(height)] = [r for r in rows if r["boards"] == height]
    return {name: common.metrics(value) for name, value in parts.items()}


def paired_interval(base, challenge, sessions, fold, policy):
    days = [day for day in sessions if fold["start"] <= day <= fold["end"]]
    mapping = {(r["signal_date"], r["code"]): r for r in challenge}
    counts = {day: [0, 0, 0] for day in days}
    for row in base:
        other = mapping[(row["signal_date"], row["code"])]
        if row["truth"] is None or row["prediction"] is None or other["prediction"] is None:
            continue
        cell = counts[row["signal_date"]]
        cell[0] += 1
        cell[1] += row["prediction"] == row["truth"]
        cell[2] += other["prediction"] == row["truth"]
    values = np.asarray(list(counts.values()), dtype=float)
    rng = np.random.default_rng(policy["seed"])
    size, block = len(days), policy["official_session_block"]
    starts = rng.integers(0, size, size=(policy["draws"], (size + block - 1) // block))
    indices = ((starts[:, :, None] + np.arange(block)) % size).reshape(policy["draws"], -1)[:, :size]
    samples = values[indices].sum(axis=1)
    undefined = int((samples[:, 0] == 0).sum())
    if undefined:
        return {"lower": None, "upper": None, "undefined_draws": undefined, "draws": policy["draws"]}
    differences = (samples[:, 2] - samples[:, 1]) / samples[:, 0]
    return {"lower": float(np.quantile(differences, policy["lower_tail"])),
        "upper": float(np.quantile(differences, policy["upper_tail"])), "undefined_draws": 0,
        "draws": policy["draws"], "official_days": size, "block_days": block,
        "method": "paired circular official-session blocks; family-adjusted tails fixed before fits"}


def compare(receipt_path):
    if (ROOT / "attempt_STARTED.json").exists():
        raise ValueError("fixed historical recency comparison already attempted; no repeat")
    e = common.read(receipt_path)
    common.verify(e["sources"])
    p, receipt = load(e["protocol_path"])
    full = joblib.load(p["prepared_path"])["relay"]
    sessions = common.read(p["calendar"])["sessions"]
    common.write_new(ROOT / "attempt_STARTED.json", {"formed_at": common.now(), "receipt_sha256": common.sha(receipt_path)})
    all_rows, logs = [], []
    for expected in e["checks"]:
        month, cutoff = expected["month"], expected["cutoff"]
        train = common.history.training_rows(full, cutoff, "relay")
        test = full[full.date.str[:7].eq(month)].copy()
        truth = target_signature(common.project(test, "INPUT", sessions))
        for arm, half_life in ARMS.items():
            if (datetime.now(timezone.utc) - datetime.fromisoformat(e["started_at"])).total_seconds() >= 1800:
                raise ValueError("30-minute limit reached; preserve monthly outputs")
            common.verify(e["sources"])
            w = weights(train, cutoff, sessions, half_life)
            if common.history.digest(w.tolist()) != expected["weight_digest"][arm]:
                raise ValueError("frozen training weights changed")
            prefix = month + "_" + arm
            common.write_new(ROOT / (prefix + "_fit_STARTED.json"), {"formed_at": common.now(),
                "training_digest": expected["training_digest"], "weight_digest": expected["weight_digest"][arm],
                "training_max_target": train.target_date.max(), "cutoff": cutoff, "fit_attempts": 1})
            try:
                model = fit_once(train, p["fields"], w)
                pred = common.history._predict(model, test, p["fields"], "relay")
                columns = ["date", "code", "name", "boards", "target_date", "target", "label_valid", "status", "predicted", "raw_score", "actual_one_word_daily", "signal_one_word", "emotion_state"]
                raw_path = ROOT / (prefix + "_predictions_BEFORE_VALIDATION.csv")
                pred[columns].to_csv(raw_path, index=False, encoding="utf-8-sig")
                rows = common.project(pred, arm, sessions)
                if target_signature(rows) != truth:
                    raise ValueError("monthly complete cohort or targets changed")
                attributes = test.set_index(["date", "code"])
                for row in rows:
                    original = attributes.loc[(row["signal_date"], row["code"])]
                    row.update(month=month, emotion_state="UNKNOWN" if pd.isna(original.emotion_state) else str(original.emotion_state),
                        legacy_proxy=common.safe(original.signal_one_word), model_cutoff=cutoff,
                        train_n=len(train), training_max_target=train.target_date.max())
                all_rows.extend(rows)
                common.write_new(ROOT / (prefix + "_checkpoint.json"), {"formed_at": common.now(),
                    "full_candidates": len(rows), "metrics": common.metrics(rows), "prediction_sha256": common.sha(raw_path)})
                logs.append({"month": month, "arm": arm, "actual_fits": 1, "train_n": len(train), "prediction_sha256": common.sha(raw_path)})
            except Exception as error:
                common.write_new(ROOT / "FAILURE_STOP.json", {"formed_at": common.now(), "month": month,
                    "arm": arm, "error": type(error).__name__ + ": " + str(error), "repeat_fitting_allowed": False})
                raise
        pd.DataFrame(all_rows).to_csv(ROOT / "full_candidate_predictions.csv", index=False, encoding="utf-8-sig")
        print(month, "fits completed", len(logs), flush=True)
    comparisons = {}
    for fold in p["folds"]:
        parts = {arm: [r for r in all_rows if r["arm"] == arm and fold["start"] <= r["signal_date"] <= fold["end"]] for arm in ARMS}
        metrics = {arm: common.metrics(rows) for arm, rows in parts.items()}
        subgroup = {arm: groups(rows) for arm, rows in parts.items()}
        verdicts = {}
        for arm in ("HL63", "HL126"):
            bm, cm = metrics["ROLL252"], metrics[arm]
            guards = {key: bm[key] is not None and cm[key] is not None and cm[key] >= bm[key] - 1e-12
                for key in ("Precision", "complete_positive_Recall", "F1")}
            guards["same_coverage"] = bm["coverage"] == cm["coverage"]
            guards["positive_call_coverage"] = sum(r["prediction"] is True for r in parts[arm]) >= sum(r["prediction"] is True for r in parts["ROLL252"])
            for key in ("complete_positive_Recall", "F1"):
                bg, cg = subgroup["ROLL252"]["proxy0"], subgroup[arm]["proxy0"]
                guards["proxy0_" + key] = bg[key] is not None and cg[key] is not None and cg[key] >= bg[key] - 1e-12
                if fold["id"] == "2026Sep":
                    bg, cg = subgroup["ROLL252"]["actual_nonword"], subgroup[arm]["actual_nonword"]
                    guards["actual_nonword_" + key] = bg[key] is not None and cg[key] is not None and cg[key] >= bg[key] - 1e-12
            interval = paired_interval(parts["ROLL252"], parts[arm], sessions, fold, p["fixed_acceptance"]["bootstrap"])
            guards["interval_lower_positive"] = interval["lower"] is not None and interval["lower"] > 0
            delta = cm["Accuracy"] - bm["Accuracy"]
            verdicts[arm] = {"Accuracy_delta": delta, "guards": guards, "paired_interval": interval,
                "passed": delta >= .03 and all(guards.values())}
        comparisons[fold["id"]] = {"metrics": metrics, "subgroups": subgroup, "verdicts": verdicts}
    survivors = [arm for arm in ("HL63", "HL126") if all(v["verdicts"][arm]["passed"] for v in comparisons.values())]
    common.verify(e["sources"])
    result = {"formed_at": common.now(), "status": "HISTORICAL_RECENCY_SURVIVORS_FORWARD_GATES_REQUIRED" if survivors else "REJECTED_FIXED_RECENCY_CHALLENGES",
        "comparisons": comparisons, "survivors": survivors, "fit_logs": logs, "actual_research_fits": len(logs),
        "fair_rolling_baseline_fits": 15, "production_model_refits": 0, "deployment_changed": False,
        "historical_is_blind_or_forward": False, "new_mature_forward_samples": 0, "calibrated_Brier": None,
        "all_candidate_predictions_sha256": common.sha(ROOT / "full_candidate_predictions.csv")}
    common.write_new(ROOT / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", required=True)
    run = sub.add_parser("compare")
    run.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = prepare(args.protocol) if args.command == "prepare" else compare(args.receipt)
    print(result.get("status", "PREPARED_FIXED_MONTHLY_RECEIPT"))
