"""Monthly prequential refit challenger, with immutable research output.

Validation labels predate every forecast month. Select a fixed model family and
history window on the immediately preceding 63 known-label dates, then refit
with all available pre-month labels. Reused history is not a pristine blind test.
No post-signal opening, seal-path placeholder, or actual outcome is a feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from next_session_forecast_lab_v271 import (
    DATA, ROOT, FORBIDDEN, model, prediction, prepare_relay, top1, metrics, rate, sha,
)
from market_expectation_extensions_v272 import prepare, direction

VERSION = "v274-monthly-prequential-refit-1"
KINDS = ("logistic", "boosting", "forest")
WINDOWS = (252, None)
VALIDATION_DATES = 63
START, END = "2025-07-01", "2026-08-31"


def make(kind):
    if kind != "forest":
        return model(kind)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=15,
            max_features=.75, random_state=274, n_jobs=2))


def fit(frame, fields, kind, channel):
    if set(fields) & FORBIDDEN:
        raise ValueError("forbidden future feature")
    mdl = make(kind)
    target = frame.target.astype(int) if channel == "relay" else frame.target
    weights = 1/frame.groupby("date").date.transform("size") if channel == "relay" else np.ones(len(frame))
    mdl.fit(frame[fields], target, **{mdl.steps[-1][0]+"__sample_weight": weights})
    return mdl


def known_before(frame, cutoff):
    return frame[(frame.date < cutoff) & (frame.target_date < cutoff)
                 & frame.label_valid & frame.target.notna()].copy()


def trailing(frame, window):
    if window is None:
        return frame.copy()
    dates = sorted(frame.date.unique())[-window:]
    return frame[frame.date.isin(dates)].copy()


def selection_metric(pred, channel):
    if channel == "relay":
        selected = top1(pred[pred.one_word_proxy.eq(0)])
        return float(selected.target.mean()), int(len(selected))
    return float(pred.correct.mean()), int(len(pred))


def summary(pred, channel):
    complete = pred.groupby("date").label_valid.all()
    evaluation = pred[pred.date.isin(complete[complete].index) & pred.target.notna()].copy()
    if channel == "market":
        return {"all": metrics(evaluation.target, evaluation.predicted, evaluation.confidence),
                "unresolved_dates": int((~complete).sum())}
    result = {"all_candidate_binary": metrics(evaluation.target.astype(int), evaluation.predicted,
                                              evaluation.confidence, evaluation.probability),
              "unresolved_dates": int((~complete).sum())}
    for key, mask in [("all_identity", evaluation.boards.ge(2)), ("non_one_word", evaluation.one_word_proxy.eq(0)),
                      ("two_board", evaluation.one_word_proxy.eq(0) & evaluation.boards.eq(2)),
                      ("three_plus", evaluation.one_word_proxy.eq(0) & evaluation.boards.ge(3))]:
        selected = top1(evaluation[mask]).copy()
        selected["correct"] = selected.target.astype(int)
        result[key] = rate(selected)
    return result


def evaluate(frame, fields, channel):
    forecasts, logs = [], []
    for period in pd.period_range(START[:7], END[:7], freq="M"):
        cutoff, month_end = str(period.start_time.date()), str(period.end_time.date())
        test = frame[frame.date.between(cutoff, min(month_end, END))].copy()
        if test.empty:
            continue
        known = known_before(frame, cutoff)
        dates = sorted(known.date.unique())
        validation_start = dates[-VALIDATION_DATES]
        train = known[(known.date < validation_start) & (known.target_date < validation_start)]
        validation = known[known.date >= validation_start]
        candidates = []
        for kind in KINDS:
            for window in WINDOWS:
                subset = trailing(train, window)
                if subset.date.nunique() < 100:
                    continue
                mdl = fit(subset, fields, kind, channel)
                val = prediction(mdl, validation, fields, channel)
                score, n = selection_metric(val, channel)
                candidates.append({"kind": kind, "window": window, "validation_accuracy": score, "validation_dates": n})
        if not candidates:
            raise ValueError("not enough causal validation history")
        best = sorted(candidates, key=lambda r: (-r["validation_accuracy"], KINDS.index(r["kind"]),
                                                 -(r["window"] or 100000)))[0]
        refit = trailing(known, best["window"])
        mdl = fit(refit, fields, best["kind"], channel)
        pred = prediction(mdl, test, fields, channel)
        pred["model_kind"], pred["history_window"], pred["model_freeze"] = best["kind"], str(best["window"]), cutoff
        pred["fold"] = "2025H2" if cutoff < "2026-01-01" else "2026YTD"
        forecasts.append(pred)
        logs.append({"month": str(period), "chosen": best, "validation_candidates": candidates,
                     "max_train_label_date": train.target_date.max(), "validation_start": validation_start,
                     "max_validation_label_date": validation.target_date.max(),
                     "refit_max_label_date": refit.target_date.max(), "refit_dates": int(refit.date.nunique()),
                     "test_dates": int(test.date.nunique()), "test_label_selection_allowed": False,
                     "confidence_status": "UNCALIBRATED_AFTER_REFIT"})
    all_pred = pd.concat(forecasts, ignore_index=True)
    reports = {fold: summary(rows, channel) for fold, rows in all_pred.groupby("fold")}
    reports["combined"] = summary(all_pred, channel)
    return reports, all_pred, logs


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable output directory exists")
    market, fields = prepare()
    market["target"] = direction(market.emotion_quality_score.shift(-1)-market.emotion_quality_score, .05)
    market_reports, market_pred, market_log = evaluate(market, fields, "market")
    relay, relay_fields = prepare_relay(DATA / "leader_candidates.csv", market.date.tolist())
    relay_reports, relay_pred, relay_log = evaluate(relay, relay_fields, "relay")
    report = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "goal_verified": False,
              "market": market_reports, "relay": relay_reports,
              "selection_logs": {"market": market_log, "relay": relay_log},
              "specification": {"model_families": list(KINDS), "history_windows": list(WINDOWS),
                                "validation_dates": VALIDATION_DATES, "update_schedule": "monthly",
                                "market_target": "quality-score change outside +/-0.05 vs continue",
                                "relay_target": "next-session closing next-board promotion; positive non-one-word Top1 precision"},
              "fields": {"market": fields, "relay": relay_fields},
              "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                  ROOT / "next_session_forecast_lab_v271.py", ROOT / "market_expectation_extensions_v272.py",
                  DATA / "market_emotion.csv", DATA / "leader_candidates.csv"]},
              "limitations": ["reused historical dates; not pristine blind or forward evidence",
                  "monthly prequential validation may use completed outcomes of earlier test months, never current/future month",
                  "refitted-model scores are uncalibrated; do not display as live probabilities",
                  "historical calendar, themes, minute path and actual fills remain insufficient"],
              "trade_authorized": False}
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
    print(json.dumps({"market": r["market"], "relay_top1": {k:v['non_one_word'] for k,v in r['relay'].items()}}, ensure_ascii=True))
