"""Predeclared sequential direction studies; inspected history, never production."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_environment_regression import CHANNELS, digest, verify_bindings, write_json

ARMS = ("LR120", "KNN120", "LR120_NEGATIVE2", "LR252")
KNOWN = {"KNOWN_DIRECTION", "ONE_CLASS_TRAIN_PRIOR", "SAVED_RIDGE_SIGN"}
TARGET = "yesterday_premium"


def direction_forecasts(obs, sessions, positions, arms):
    """Every training label ends no later than the current signal session."""
    if sessions != sorted(set(sessions)) or obs.index.tolist() != sessions:
        raise ValueError("complete ordered official date grid required")
    if list(obs.columns) != CHANNELS or np.isinf(obs.to_numpy(float)).any():
        raise ValueError("exact eight finite-or-missing channels required")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("unregistered research arm")
    rows = []
    for pos in positions:
        if not 0 <= pos < len(sessions):
            raise ValueError("signal position outside official grid")
        for arm in arms:
            window = 252 if arm == "LR252" else 120
            start = max(0, pos + 1 - window)
            ends = np.arange(max(1, start), pos + 1)
            labels = obs[TARGET].iloc[ends].to_numpy(float)
            sources = obs.iloc[ends - 1]
            keep = np.isfinite(labels) & sources.notna().any(axis=1).to_numpy()
            training = sources.iloc[np.flatnonzero(keep)].copy()
            # A missing outcome is excluded BEFORE turning a finite outcome into a class.
            y = labels[keep] > 0
            end_dates = [sessions[i] for i in ends[keep]]
            active = training.columns[training.notna().any(axis=0)].tolist()
            dropped = [name for name in CHANNELS if name not in active]
            status = ("UNKNOWN_CURRENT_LEVELS" if not obs.iloc[pos].notna().any()
                      else "INSUFFICIENT_HISTORY" if len(y) < 60 or not active
                      else "KNOWN_DIRECTION")
            raw, decision, neighbors, distances, failure = np.nan, np.nan, [], [], None
            if status == "KNOWN_DIRECTION":
                if arm.startswith("LR") and len(np.unique(y)) == 1:
                    status = "ONE_CLASS_TRAIN_PRIOR"
                    raw = (int(y.sum()) + 1) / (len(y) + 2)
                else:
                    transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                              StandardScaler())
                    x = transform.fit_transform(training[active])
                    query = transform.transform(obs.iloc[[pos]][active])
                    if arm == "KNN120":
                        squared = ((x - query[0]) ** 2).sum(axis=1)
                        chosen = np.lexsort((np.asarray(end_dates), squared))[:21]
                        raw = float(y[chosen].mean())
                        neighbors = [end_dates[i] for i in chosen]
                        distances = [float(squared[i]) for i in chosen]
                    else:
                        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                                   random_state=20261007,
                                                   class_weight={False: 2.0, True: 1.0}
                                                   if arm == "LR120_NEGATIVE2" else None)
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always", ConvergenceWarning)
                                model.fit(x, y)
                            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                                status = "FIT_NONCONVERGENCE"
                                failure = "; ".join(str(w.message) for w in caught)
                            else:
                                raw = float(model.predict_proba(query)[0, list(model.classes_).index(True)])
                        except (ValueError, FloatingPointError) as exc:
                            status = "FIT_FAILED"
                            failure = str(exc)
                if np.isfinite(raw):
                    decision = float(raw > 0.5)
            binding = training.copy()
            binding["FINITE_TRAINING_OUTCOME"] = labels[keep]
            if end_dates and max(end_dates) > sessions[pos]:
                raise AssertionError("future training label")
            rows.append({"signal_date": sessions[pos],
                         "target_date": sessions[pos + 1] if pos + 1 < len(sessions) else None,
                         "arm": arm, "status": status, "positive_probability": raw,
                         "predicted_positive": decision, "train_n": len(y),
                         "max_training_target_date": max(end_dates) if end_dates else None,
                         "window_start": sessions[start], "window_official_sessions": window,
                         "training_digest": hashlib.sha256(binding.to_csv().encode()).hexdigest(),
                         "dropped_all_missing_features": json.dumps(dropped),
                         "neighbor_end_dates_json": json.dumps(neighbors),
                         "neighbor_squared_distances_json": json.dumps(distances),
                         "fit_failure": failure, "score_calibrated": False,
                         "forward_evidence": False})
    return pd.DataFrame(rows)


def label_only(predictions, obs):
    result = predictions.copy()
    result["actual_value"] = [obs.at[d, TARGET] if d in obs.index else np.nan
                              for d in result.target_date]
    result["actual_positive"] = np.where(np.isfinite(result.actual_value),
                                         (result.actual_value > 0).astype(float), np.nan)
    valid = result.status.isin(KNOWN) & result.predicted_positive.notna()
    result["evaluated"] = valid & result.actual_positive.notna()
    result["correct"] = result.predicted_positive.eq(result.actual_positive).astype(float).where(result.evaluated)
    result["false_positive"] = (result.predicted_positive.eq(1) & result.actual_positive.eq(0)).astype(float).where(result.evaluated)
    return result


def metrics(group):
    known = group[group.evaluated]
    y = known.actual_positive.eq(1).to_numpy()
    p = known.predicted_positive.eq(1).to_numpy()
    n = len(known)
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    positives, negatives = int(y.sum()), int((~y).sum())
    all_known_positives = int(group.actual_positive.eq(1).sum())
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    raw_known = known[known.positive_probability.notna()]
    score = raw_known.positive_probability.to_numpy(float)
    truth = raw_known.actual_positive.to_numpy(float)
    clipped = np.clip(score, 1e-15, 1 - 1e-15)
    total = len(group)
    known_truth = int(group.actual_positive.notna().sum())
    return {"registered_dates": total, "known_target_dates": known_truth,
            "predicted_dates": int(group.status.isin(KNOWN).sum()), "scored_dates": n,
            "unknown_target_dates": total - known_truth,
            "unknown_prediction_dates": int((~group.status.isin(KNOWN)).sum()),
            "scoring_coverage": n / total if total else None,
            "label_coverage": known_truth / total if total else None,
            "valid_label_scoring_coverage": n / known_truth if known_truth else None,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "positive_count": positives, "nonpositive_count": negatives,
            "called_positive": int(p.sum()), "positive_call_fraction": float(p.mean()) if n else None,
            "accuracy": (tp + tn) / n if n else None,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "scored_positive_recall": recall,
            "complete_positive_recall": tp / all_known_positives if all_known_positives else None,
            "nonpositive_recall": specificity, "FPR": fp / negatives if negatives else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "all_positive_accuracy": positives / n if n else None,
            "all_nonpositive_accuracy": negatives / n if n else None,
            "raw_score_dates": len(raw_known),
            "raw_score_Brier": float(((score - truth) ** 2).mean()) if len(score) else None,
            "raw_score_logloss": float(-(truth * np.log(clipped) + (1-truth) * np.log(1-clipped)).mean()) if len(score) else None,
            "probability_calibration": "NOT_CALIBRATED_INTERNAL_SCORE" if len(score) else "UNAVAILABLE_RIDGE_SIGN"}


def block_interval(values, protocol):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).any():
        return None
    spec = protocol["bootstrap"]
    rng = np.random.default_rng(spec["seed"])
    block = spec["block_official_sessions"]
    samples = []
    for _ in range(spec["draws"]):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indexes = np.concatenate([(s + np.arange(block)) % len(values) for s in starts])[:len(values)]
        sampled = values[indexes]
        if np.isfinite(sampled).any():
            samples.append(float(np.nanmean(sampled)))
    return np.quantile(samples, spec["tails"]).tolist()


def paired_metrics(part, baseline_arm, arm, protocol, uncertainty=True):
    base = part[part.arm.eq(baseline_arm)].set_index("signal_date")
    challenger = part[part.arm.eq(arm)].set_index("signal_date")
    if base.index.tolist() != challenger.index.tolist():
        raise ValueError("paired comparison lost original signal dates")
    common = base.evaluated & challenger.evaluated
    b, c = base.loc[common], challenger.loc[common]
    b_metrics, c_metrics = metrics(b), metrics(c)
    grid = base.index.tolist()
    cost = protocol["stage"] == 2
    # Retain positive days and missing days in the calendar grid; only eligible
    # nonpositive outcomes contribute to the cost denominator after resampling.
    eligible = b.actual_positive.eq(0) if cost else pd.Series(True, index=b.index)
    change = ((c.false_positive - b.false_positive) if cost else (c.correct - b.correct)).where(eligible).reindex(grid)
    interval = block_interval(change, protocol) if uncertainty else None
    return {"arm": arm, "baseline_arm": baseline_arm,
            "registered_dates": len(grid), "paired_dates": len(b),
            "pair_coverage": len(b) / len(grid) if grid else None,
            "paired_nonpositive_dates": int(b.actual_positive.eq(0).sum()),
            "baseline": b_metrics, "challenger": c_metrics,
            "Accuracy_difference": c_metrics["accuracy"] - b_metrics["accuracy"]
            if b_metrics["accuracy"] is not None and c_metrics["accuracy"] is not None else None,
            "FPR_difference": c_metrics["FPR"] - b_metrics["FPR"]
            if b_metrics["FPR"] is not None and c_metrics["FPR"] is not None else None,
            "primary_difference": float(change.mean()) if change.notna().any() else None,
            "family_adjusted_primary_interval": interval}


def summarize(scores, baseline_arm, protocol):
    summaries, pairs, monthly = [], [], []
    for period, (start, end) in protocol["periods"].items():
        part = scores[scores.signal_date.between(start, end)]
        for arm in [baseline_arm] + protocol["arms"]:
            summaries.append({"period": period, "arm": arm, **metrics(part[part.arm.eq(arm)])})
        for arm in protocol["arms"]:
            pairs.append({"period": period, **paired_metrics(part, baseline_arm, arm, protocol)})
    for month in sorted(scores.signal_date.str[:7].unique()):
        part = scores[scores.signal_date.str.startswith(month)]
        for arm in protocol["arms"]:
            monthly.append({"month": month, **paired_metrics(part, baseline_arm, arm, protocol, False)})
    gates = {}
    for arm in protocol["arms"]:
        ev = next(p for p in pairs if p["period"] == "EVALUATION_DIAGNOSTIC" and p["arm"] == arm)
        dev = next(p for p in pairs if p["period"] == "DEVELOPMENT_DIAGNOSTIC" and p["arm"] == arm)
        b, c = ev["baseline"], ev["challenger"]
        interval = ev["family_adjusted_primary_interval"]
        bs = next(p for p in summaries if p["period"] == "EVALUATION_DIAGNOSTIC" and p["arm"] == baseline_arm)
        cs = next(p for p in summaries if p["period"] == "EVALUATION_DIAGNOSTIC" and p["arm"] == arm)
        checks = {"paired_sample_at_least60": ev["paired_dates"] >= 60,
                  "coverage_not_lower": cs["scored_dates"] >= bs["scored_dates"],
                  "balanced_accuracy_not_lower": c["balanced_accuracy"] is not None and b["balanced_accuracy"] is not None and c["balanced_accuracy"] >= b["balanced_accuracy"]}
        if protocol["stage"] == 2:
            checks.update({"nonpositive_sample_at_least20": ev["paired_nonpositive_dates"] >= 20,
                           "adjusted_FPR_interval_below0": interval is not None and interval[1] < 0,
                           "accuracy_not_lower": c["accuracy"] is not None and b["accuracy"] is not None and c["accuracy"] >= b["accuracy"],
                           "development_FPR_not_higher": dev["FPR_difference"] is not None and dev["FPR_difference"] <= 0})
        else:
            checks.update({"adjusted_accuracy_interval_above0": interval is not None and interval[0] > 0,
                           "nonpositive_recall_not_lower": c["nonpositive_recall"] is not None and b["nonpositive_recall"] is not None and c["nonpositive_recall"] >= b["nonpositive_recall"],
                           "development_accuracy_not_lower": dev["Accuracy_difference"] is not None and dev["Accuracy_difference"] >= 0})
            if protocol["stage"] == 3:
                months = [p for p in monthly if p["arm"] == arm and "2026-03" <= p["month"] <= "2026-09"]
                checks["at_least4_eligible_months_improve"] = sum(p["paired_dates"] >= 10 and p["Accuracy_difference"] is not None and p["Accuracy_difference"] > 0 for p in months) >= 4
        gates[arm] = {"checks": checks, "historical_candidate_gate_passed": all(checks.values()),
                      "production_upgrade_allowed": False}
    return summaries, pairs, monthly, gates


def _ridge_baseline(path, dates):
    old = pd.read_csv(path)
    old = old[old.arm.eq("RIDGE_LEVEL") & old.metric.eq(TARGET)].copy()
    if old.signal_date.tolist() != dates or old.signal_date.duplicated().any():
        raise ValueError("saved Ridge baseline date set mismatch")
    valid = old.status.eq("KNOWN_DIAGNOSTIC_POINT") & old.prediction.notna()
    old["predicted_positive"] = np.where(valid, (old.prediction > 0).astype(float), np.nan)
    old["positive_probability"] = np.nan
    old["neighbor_end_dates_json"] = "[]"
    old["status"] = np.where(valid, "SAVED_RIDGE_SIGN", old.status)
    old["arm"] = "RIDGE120_SIGN_SAVED"
    old["saved_numeric_prediction"] = old.prediction
    old["score_calibrated"] = False
    return old.drop(columns=["prediction", "metric"])


def validate_protocol(protocol):
    arms = {1: ["LR120", "KNN120"], 2: ["LR120_NEGATIVE2"], 3: ["LR252"]}
    if (protocol["input_channels"] != CHANNELS or protocol["arms"] != arms.get(protocol["stage"])
            or protocol["LR"] != {"C": 1.0, "solver": "lbfgs", "max_iter": 2000,
                                   "random_state": 20261007, "threshold": "raw positive class score >0.5; tie nonpositive"}
            or protocol["KNN"]["neighbors"] != 21
            or protocol["bootstrap"] != {"draws": 2000, "seed": 20261007, "block_official_sessions": 5,
                                         "sequence_primary_challenges": 4, "tails": [0.00625, 0.99375],
                                         "missing_dates": "retain full official grid before circular block resampling"}
            or protocol["production_upgrade_allowed"] is not False):
        raise ValueError("protocol differs from the fixed implemented study")
    if digest(protocol["sequence_path"]) != protocol["sequence_sha256"]:
        raise ValueError("sequence preregistration changed")
    sequence = json.loads(Path(protocol["sequence_path"]).read_text(encoding="utf-8"))
    spec = next(s for s in sequence["stages"] if s["stage"] == protocol["stage"])
    for key, value in spec.items():
        if protocol.get(key) != value:
            raise ValueError("stage contradicts preregistered sequence: " + key)


def run(protocol_path):
    started = datetime.now(timezone.utc)
    path = Path(protocol_path).resolve()
    out, root = path.parent, Path(__file__).resolve().parent
    if (out / "predictions_NO_LABELS_NOT_FORWARD.csv").exists() or (out / "run_started.json").exists():
        raise FileExistsError("fixed batch already attempted; preserve results and do not retune/refit")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    verify_bindings(protocol, root)
    source = next(Path(p) for p in protocol["input_hashes"] if Path(p).name == "observations_DIAGNOSTIC.csv")
    calendar = next(Path(p) for p in protocol["input_hashes"] if Path(p).name == "calendar.json")
    sessions = json.loads(calendar.read_text(encoding="utf-8"))["sessions"]
    obs = pd.read_csv(source, index_col=0)
    obs.index = obs.index.astype(str)
    positions = [i for i, day in enumerate(sessions) if protocol["signal_dates"][0] <= day <= protocol["signal_dates"][1]]
    dates = [sessions[i] for i in positions]
    baseline_binding = {}
    if protocol["stage"] == 1:
        baseline_path = next(Path(p) for p in protocol["input_hashes"] if Path(p).name == "predictions_NO_LABELS_NOT_FORWARD.csv")
        baseline = _ridge_baseline(baseline_path, dates)
        baseline_arm = "RIDGE120_SIGN_SAVED"
    else:
        previous = out.parent / ("stage" + str(protocol["stage"] - 1)) / "report.json"
        if not previous.exists():
            raise ValueError("sequential earlier batch not completed")
        baseline_path = out.parent / "stage1" / "predictions_NO_LABELS_NOT_FORWARD.csv"
        receipt_path = out.parent / "stage1" / "prediction_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["prediction_sha256"] != digest(baseline_path):
            raise ValueError("stage1 baseline prediction artifact changed")
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[baseline.arm.eq("LR120")].copy()
        baseline_arm = "LR120"
        if baseline.signal_date.tolist() != dates:
            raise ValueError("LR120 saved baseline date set mismatch")
        baseline_binding[str(receipt_path)] = digest(receipt_path)
    baseline_binding[str(baseline_path)] = digest(baseline_path)
    write_json(out / "run_started.json", {"started_at": started.isoformat(),
               "stage": protocol["stage"], "protocol_sha256": digest(path),
               "implementation_sha256": digest(__file__), "baseline_bindings": baseline_binding})
    predictions = direction_forecasts(obs, sessions, positions, protocol["arms"])
    predictions = pd.concat([baseline, predictions], ignore_index=True, sort=False)
    if len(predictions) != len(dates) * (1 + len(protocol["arms"])):
        raise AssertionError("full original date set lost")
    for row in predictions.itertuples():
        pos = sessions.index(row.signal_date)
        if row.target_date != sessions[pos + 1]:
            raise AssertionError("not exact next official session")
        if isinstance(row.max_training_target_date, str) and row.max_training_target_date > row.signal_date:
            raise AssertionError("future label leakage")
    predicted_path = out / "predictions_NO_LABELS_NOT_FORWARD.csv"
    predictions.to_csv(predicted_path, index=False)
    write_json(out / "prediction_receipt.json", {"saved_at": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": digest(path), "sequence_sha256": protocol["sequence_sha256"],
               "implementation_sha256": digest(__file__), "prediction_sha256": digest(predicted_path),
               "baseline_bindings": baseline_binding, "new_forward_samples": 0,
               "status": "SAVED_BEFORE_EVALUATION_NOT_FORWARD"})
    scores = label_only(predictions, obs)
    scores.to_csv(out / "scores_LABEL_ONLY.csv", index=False)
    summaries, pairs, monthly, gates = summarize(scores, baseline_arm, protocol)
    pd.DataFrame(summaries).to_csv(out / "metric_summary.csv", index=False)
    verify_bindings(protocol, root)
    for filename, expected in baseline_binding.items():
        if digest(filename) != expected:
            raise ValueError("saved baseline changed during scoring")
    ended = datetime.now(timezone.utc)
    write_json(out / "report.json", {"stage": protocol["stage"], "id": protocol["id"],
               "started_at": started.isoformat(), "ended_at": ended.isoformat(),
               "elapsed_seconds": (ended - started).total_seconds(), "baseline_arm": baseline_arm,
               "protocol_sha256": digest(path), "implementation_sha256": digest(__file__),
               "artifacts": {p.name: digest(p) for p in out.iterdir() if p.suffix == ".csv"},
               "summaries": summaries, "paired_comparisons": pairs, "monthly_comparisons": monthly,
               "gates": gates, "new_forward_samples": 0, "production_upgrade_allowed": False,
               "limits": ["Previously inspected historical diagnosis; not an independent blind test",
                          "Provider pool scope, not official full-A certification",
                          "Historical knowledge-time availability not certified",
                          "Internal scores uncalibrated; no executable returns estimated",
                          "Shared history across all batches; four-challenge family adjustment",
                          "Only next-session pool-median sign, not entire market forecast accuracy"],
               "status": "FIXED_COMPARISON_COMPLETE_NOT_CORE_UPGRADE"})
    print(json.dumps({"stage": protocol["stage"], "status": "complete", "gates": gates,
                      "evaluation": [s for s in summaries if s["period"] == "EVALUATION_DIAGNOSTIC"],
                      "elapsed_seconds": (ended - started).total_seconds()}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    run(parser.parse_args().protocol)
