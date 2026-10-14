"""Nested chronological market/relay research; never a live win-rate claim.

Models and confidence gates are chosen on validation dates only. All candidate
predictions are retained before outcome availability is checked. The historical
archive has been inspected before and is explicitly not a pristine blind set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from relay_promotion_holdout_v256 import IDENTITY, EMOTION, PRICE_VOLUME, wilson
from emotion_three_regime_challenger_v86 import FEATURES as MARKET_BASE, STATE_TO_CLASS

VERSION = "v271-nested-next-session-research-1"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2"
FOLDS = [
    {"name": "2025H2", "validation_start": "2025-01-01", "test_start": "2025-07-01", "test_end": "2025-12-31"},
    {"name": "2026YTD", "validation_start": "2025-07-01", "test_start": "2026-01-01", "test_end": "2026-08-31"},
]
KINDS = ("logistic", "boosting")
THRESHOLDS = (0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
MIN_COVERAGE = 0.25
MIN_VALIDATION_CALLS = 20
CLASSES = ["REPAIR", "RISK_OFF", "RISK_ON"]
FORBIDDEN = {"promoted_next_day", "target", "target_date", "next_open_gap", "net_return",
             "trade_open", "trade_close", "trade_high", "trade_low", "terminal_board", "additional_boards"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model(kind):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=0.25, max_iter=2500, random_state=271))
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         HistGradientBoostingClassifier(learning_rate=.04, max_iter=120,
                             max_leaf_nodes=7, min_samples_leaf=35, l2_regularization=4,
                             early_stopping=False, random_state=271))


def metrics(y, predicted, confidence=None, probability=None):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if len(y) == 0:
        return {"n": 0, "accuracy": None, "wilson95": None}
    correct = y == predicted
    result = {"n": len(y), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
              "wilson95": wilson(int(correct.sum()), len(y)),
              "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
              "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
              "by_class": {str(k): {"support": int((y == k).sum()),
                  "called": int((predicted == k).sum()),
                  "recall": float((predicted[y == k] == k).mean())} for k in np.unique(y)}}
    if probability is not None:
        p = np.asarray(probability)
        result["brier"] = float(np.mean((p - y.astype(float)) ** 2))
        result["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    if confidence is not None:
        conf = np.asarray(confidence)
        result["confidence_ece"] = sum(float(mask.mean() * abs(conf[mask].mean()-correct[mask].mean()))
            for lo in np.arange(0, 1, .1)
            if (mask := ((conf >= lo) & (conf < lo+.1 if lo < .9 else conf <= 1))).any())
    return result


def rate(rows, column="correct"):
    values = rows[column].to_numpy(dtype=int)
    k, n = int(values.sum()), len(values)
    return {"n": n, "successes": k, "accuracy": float(k/n) if n else None, "wilson95": wilson(k, n)}


def top1(rows, score="probability"):
    # Stable code tie-break; never an outcome or a next-day field.
    return rows.sort_values(["date", score, "boards", "code"],
                            ascending=[True, False, False, True], kind="stable").drop_duplicates("date")


def choose_gate(rows, score="confidence"):
    grid = []
    for threshold in THRESHOLDS:
        selected = rows[rows[score] >= threshold]
        r = rate(selected)
        r.update(threshold=threshold, coverage=len(selected)/len(rows) if len(rows) else 0)
        r["qualified"] = bool(r["n"] >= MIN_VALIDATION_CALLS and r["coverage"] >= MIN_COVERAGE
                              and r["accuracy"] > .6)
        grid.append(r)
    qualified = [r for r in grid if r["qualified"]]
    # Prefer widest coverage, not the most flattering tiny subset.
    winner = max(qualified, key=lambda r: (r["coverage"], r["accuracy"], -r["threshold"])) if qualified else None
    return {"threshold": winner["threshold"] if winner else .6,
            "validation_qualified": winner is not None, "validation_grid": grid}


def acceptance(summary, coverage, validation_qualified):
    interval = summary.get("wilson95")
    return {"point_above_60": summary.get("accuracy") is not None and summary["accuracy"] > .6,
            "at_least_60_independent_dates": summary["n"] >= 60,
            "coverage_at_least_25pct": coverage >= MIN_COVERAGE,
            "lower_95_above_60": bool(interval and interval[0] > .6),
            "validation_qualified": bool(validation_qualified),
            "pristine_forward_evidence": False, "live_accuracy_claim_allowed": False}


def prepare_market(path):
    frame = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    if frame.date.duplicated().any():
        raise ValueError("duplicate market dates")
    missing = set(MARKET_BASE + ["emotion_state", "date", "data_status"]) - set(frame)
    if missing:
        raise ValueError(f"missing market fields: {sorted(missing)}")
    frame["target_date"] = frame.date.shift(-1)
    frame["target"] = frame.emotion_state.shift(-1).map(STATE_TO_CLASS)
    frame["label_valid"] = frame.data_status.eq("verified") & frame.data_status.shift(-1).eq("verified")
    frame["current_class"] = frame.emotion_state.map(STATE_TO_CLASS)
    features = list(MARKET_BASE)
    for field in ["emotion_quality_score", "promotion_mid", "yesterday_premium", "failed_rate", "down_limit_count"]:
        values = pd.to_numeric(frame[field], errors="coerce")
        for lag in [1, 3]:
            name = f"{field}_delta_{lag}"
            frame[name] = values - values.shift(lag)
            features.append(name)
        name = f"{field}_mean_5"
        frame[name] = values.rolling(5, min_periods=3).mean()
        features.append(name)
    for key in CLASSES:
        name = f"current_{key}"
        frame[name] = frame.current_class.eq(key).astype(int)
        features.append(name)
    return frame, features


def prepare_relay(path, sessions):
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame = frame[frame.boards.ge(2)].copy()
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("duplicate relay date/code")
    frame["code"] = frame.code.str.zfill(6)
    frame["target_date"] = frame.trade_date
    successor = dict(zip(sessions[:-1], sessions[1:]))
    frame["calendar_matched"] = frame.date.map(successor).eq(frame.target_date)
    frame["label_valid"] = (frame.promotion_label_status.eq("verified_from_next_daily_limit_close")
                            & frame.promoted_next_day.isin([0, 1]) & frame.calendar_matched)
    frame["target"] = frame.promoted_next_day
    features = list(dict.fromkeys(IDENTITY + EMOTION + PRICE_VOLUME))
    # At-close price/volume and cross-sectional relative positions are observable.
    for field in ["price", "signal_amount", "turnover", "turnover_vs_prev"]:
        values = pd.to_numeric(frame[field], errors="coerce")
        name = f"{field}_same_height_percentile"
        frame[name] = values.groupby([frame.date, frame.boards]).rank(pct=True)
        features.append(name)
    frame["signal_one_word"] = frame.one_word_proxy
    frame["log_signal_amount"] = np.log1p(pd.to_numeric(frame.signal_amount, errors="coerce").clip(lower=0))
    features += ["signal_one_word", "log_signal_amount"]
    return frame, features


def split(frame, fold):
    valid = frame.label_valid & frame.target.notna()
    train = frame[(frame.date < fold["validation_start"]) & valid
                  & (frame.target_date < fold["validation_start"])].copy()
    validation = frame[(frame.date >= fold["validation_start"]) & (frame.date < fold["test_start"])
                       & valid & (frame.target_date < fold["test_start"])].copy()
    test = frame[frame.date.between(fold["test_start"], fold["test_end"])].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("empty chronological slice")
    return train, validation, test


def fitted(train, fields, kind, channel):
    if FORBIDDEN & set(fields):
        raise ValueError("future field in feature allowlist")
    mdl = model(kind)
    y = train.target.astype(int) if channel == "relay" else train.target
    weights = (1 / train.groupby("date").date.transform("size")) if channel == "relay" else np.ones(len(train))
    mdl.fit(train[fields], y, **{f"{mdl.steps[-1][0]}__sample_weight": weights})
    return mdl


def prediction(mdl, frame, fields, channel):
    p = mdl.predict_proba(frame[fields])
    classes = np.asarray(mdl.classes_)
    result = frame.copy()
    result["predicted"] = classes[p.argmax(axis=1)]
    result["confidence"] = p.max(axis=1)
    if channel == "relay":
        result["probability"] = p[:, list(classes).index(1)]
    result["correct"] = result.predicted.eq(result.target).astype(int)
    return result


def evaluate_channel(frame, fields, channel):
    reports, outputs = {}, []
    for fold in FOLDS:
        train, validation, test = split(frame, fold)
        choices = []
        for kind in KINDS:
            mdl = fitted(train, fields, kind, channel)
            val = prediction(mdl, validation, fields, channel)
            if channel == "relay":
                val = top1(val[val.one_word_proxy.eq(0)])
                val["correct"] = val.target.astype(int)  # Positive selection precision, not negative accuracy.
                val["confidence"] = val.probability
            scored = rate(val)
            choices.append((scored["accuracy"], kind, mdl, val))
        _, chosen_kind, mdl, val = max(choices, key=lambda c: (c[0], c[1] == "logistic"))
        gate = choose_gate(val)
        pred = prediction(mdl, test, fields, channel)
        pred["fold"] = fold["name"]
        pred["model_kind"] = chosen_kind
        pred["threshold"] = gate["threshold"]
        complete = pred.groupby("date").label_valid.all()
        evaluation = pred[pred.date.isin(complete[complete].index) & pred.target.notna()].copy()
        report = {"fold": fold, "chosen_model": chosen_kind, "selection_uses_test_labels": False,
                  "train_dates": int(train.date.nunique()), "validation_dates": int(validation.date.nunique()),
                  "max_train_target_date": train.target_date.max(), "max_validation_target_date": validation.target_date.max(),
                  "validation_models": {kind: {"accuracy": acc} for acc, kind, _, _ in choices},
                  "gate": gate, "test_dates_total": int(test.date.nunique()),
                  "test_dates_complete": int(evaluation.date.nunique()),
                  "test_dates_unresolved": int((~complete).sum())}
        if channel == "market":
            summary = metrics(evaluation.target, evaluation.predicted, evaluation.confidence)
            selective = evaluation[evaluation.confidence >= gate["threshold"]]
            baseline_class = train.target.value_counts().index[0]
            report["baseline_majority"] = metrics(evaluation.target, [baseline_class]*len(evaluation))
            report["baseline_persistence"] = metrics(evaluation.target, evaluation.current_class)
            report["all_predictions"] = summary
            report["selective_predictions"] = rate(selective)
            report["coverage"] = len(selective)/len(evaluation)
            report["acceptance"] = acceptance(rate(selective), report["coverage"], gate["validation_qualified"])
        else:
            report["all_candidate_binary"] = metrics(evaluation.target.astype(int), evaluation.predicted,
                evaluation.confidence, evaluation.probability)
            report["all_negative_baseline"] = metrics(evaluation.target.astype(int), np.zeros(len(evaluation), dtype=int))
            for pool, mask in [("all_identity", evaluation.boards.ge(2)), ("non_one_word", evaluation.one_word_proxy.eq(0)),
                               ("two_board_non_one_word", evaluation.boards.eq(2) & evaluation.one_word_proxy.eq(0)),
                               ("three_plus_non_one_word", evaluation.boards.ge(3) & evaluation.one_word_proxy.eq(0))]:
                candidates = evaluation[mask]
                selected = top1(candidates).copy()
                selected["correct"] = selected.target.astype(int)
                selective = selected[selected.probability >= gate["threshold"]]
                baseline = candidates.sort_values(["date", "boards", "code"], ascending=[True, False, True]).drop_duplicates("date").copy()
                baseline["correct"] = baseline.target.astype(int)
                paired = selected[["date", "correct"]].merge(baseline[["date", "correct"]], on="date", suffixes=("_model", "_baseline"))
                coverage = len(selective)/len(selected) if len(selected) else 0
                report[pool] = {"top1": rate(selected), "selective_top1": rate(selective), "coverage": coverage,
                    "height_baseline_same_dates": rate(baseline),
                    "paired_mean_difference": float((paired.correct_model-paired.correct_baseline).mean()) if len(paired) else None,
                    "acceptance": acceptance(rate(selective), coverage, gate["validation_qualified"])}
        reports[fold["name"]] = report
        outputs.append(pred)
    return reports, pd.concat(outputs, ignore_index=True)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable research output exists")
    market_path, relay_path = DATA / "market_emotion.csv", DATA / "leader_candidates.csv"
    market, market_fields = prepare_market(market_path)
    relay, relay_fields = prepare_relay(relay_path, market.date.tolist())
    market_reports, market_pred = evaluate_channel(market, market_fields, "market")
    relay_reports, relay_pred = evaluate_channel(relay, relay_fields, "relay")
    report = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "market": market_reports, "relay": relay_reports,
        "targets": {"market": "next_session_three_regimes_REPAIR_RISK_OFF_RISK_ON",
                    "relay": "next_session_closing_next_board; positive Top1 precision is primary"},
        "scope": "mainboard non-ST reconstructed >=2-board pool; first boards are a separate model",
        "source_limitations": ["Previously inspected history: not pristine blind/forward evidence",
            "Observed market sessions cross-check successor only; independent full historical exchange calendar not provided",
            "Reconstructed theme, minute, auction, queue and actual-fill evidence unavailable",
            "Point-in-time historical universe can retain survivorship risk",
            "No economic-return or live-entry claim; confidence is uncalibrated research output"],
        "source_audit": {"market_dates": len(market), "relay_rows": len(relay),
                         "relay_calendar_mismatches": int((~relay.calendar_matched & relay.target_date.notna()).sum())},
        "fields": {"market": market_fields, "relay": relay_fields},
        "specification": {"folds": FOLDS, "models": list(KINDS), "confidence_thresholds": list(THRESHOLDS),
                          "min_coverage": MIN_COVERAGE, "min_validation_calls": MIN_VALIDATION_CALLS},
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [market_path, relay_path, Path(__file__)]},
        "goal_verified": False, "trade_authorized": False}
    output.mkdir(parents=True)
    market_pred.to_csv(output / "market_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    relay_pred.to_csv(output / "relay_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    r = run(args.output)
    print(json.dumps({"status": r["status"], "market": {k:v["all_predictions"] for k,v in r["market"].items()},
                      "relay": {k:v["non_one_word"] for k,v in r["relay"].items()}}, ensure_ascii=True))
