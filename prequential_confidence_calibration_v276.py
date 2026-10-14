"""Calibrate forecast confidence using earlier completed forecast dates only.

This changes confidence, never the frozen market direction or selected stock.
Market top-call reliability and relay positive-selection probability are
different calibration targets. Early dates without 60 prior calls stay unknown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from next_session_forecast_lab_v271 import ROOT, top1, metrics, rate, sha

VERSION = "v276-prequential-call-confidence-1"
SOURCE = ROOT / "results/v274/monthly_refit_20261002_v1"
MINIMUM, WINDOW = 60, 120


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1-1e-6)
    return np.log(values/(1-values)).reshape(-1, 1)


def calibration_metrics(truth, p):
    truth, p = np.asarray(truth, dtype=float), np.asarray(p, dtype=float)
    ece = 0.
    for lower in np.arange(0, 1, .1):
        mask = (p >= lower) & (p < lower+.1 if lower < .9 else p <= 1)
        if mask.any():
            ece += float(mask.mean()*abs(p[mask].mean()-truth[mask].mean()))
    return {"n": len(truth), "actual_success_rate": float(truth.mean()) if len(truth) else None,
            "mean_forecast_confidence": float(p.mean()) if len(p) else None,
            "brier": float(np.mean((p-truth)**2)) if len(p) else None, "ece_10bin": ece}


def calibrate(rows, score):
    rows = rows.sort_values("date").copy()
    if rows.date.duplicated().any():
        raise ValueError("confidence calibration requires one frozen call per date")
    rows["calibrated_confidence"] = np.nan
    logs = []
    for month, current in rows.groupby(rows.date.str[:7], sort=True):
        cutoff = month+"-01"
        known = rows[(rows.date < cutoff) & (rows.target_date < cutoff)
                     & rows.label_valid & rows.correct.notna()].tail(WINDOW)
        if len(known) < MINIMUM or known.correct.nunique() < 2:
            logs.append({"month": month, "status": "INSUFFICIENT_PRIOR_CALLS", "n": len(known)})
            continue
        mdl = LogisticRegression(C=.5, random_state=276, max_iter=1000)
        mdl.fit(logit(known[score]), known.correct.astype(int))
        p = mdl.predict_proba(logit(current[score]))[:, list(mdl.classes_).index(1)]
        rows.loc[current.index, "calibrated_confidence"] = p
        logs.append({"month": month, "status": "CAUSAL_PRIOR_CALL_CALIBRATION", "n": len(known),
                     "max_calibration_label_date": known.target_date.max(), "coefficient": float(mdl.coef_[0, 0])})
    evaluated = rows[rows.label_valid & rows.correct.notna() & rows.calibrated_confidence.notna()].copy()
    summaries = {}
    for fold, frame in [("combined", evaluated), *list(evaluated.groupby("fold"))]:
        above60 = frame[frame.calibrated_confidence >= .6]
        summaries[fold] = {"raw": calibration_metrics(frame.correct, frame[score]),
                           "calibrated": calibration_metrics(frame.correct, frame.calibrated_confidence),
                           "confidence_at_least_60": {**rate(above60), "coverage_of_calibrated_calls": len(above60)/len(frame) if len(frame) else 0}}
    return rows, summaries, logs


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("immutable calibration output exists")
    market_source, relay_source = SOURCE/"market_predictions_LABEL_ONLY.csv", SOURCE/"relay_predictions_LABEL_ONLY.csv"
    market = pd.read_csv(market_source)
    # Preserve unknown outcomes, which must not be converted into incorrect calls.
    market.loc[~market.label_valid | market.target.isna(), "correct"] = np.nan
    relay = pd.read_csv(relay_source, dtype={"code": str})
    channels = {"market_direction_top_call": (market, "confidence")}
    for name, mask in [("relay_identity_top1", relay.boards.ge(2)), ("relay_non_one_word_top1", relay.one_word_proxy.eq(0))]:
        rows = top1(relay[mask]).copy()
        complete = relay.groupby("date").label_valid.all()
        rows["label_valid"] = rows.date.map(complete)
        rows["correct"] = rows.target.where(rows.label_valid)
        channels[name] = (rows, "probability")
    reports, predictions, logs = {}, {}, {}
    for name, (rows, score) in channels.items():
        predictions[name], reports[name], logs[name] = calibrate(rows, score)
    report = {"version": VERSION, "status": "DIAGNOSTIC_ONLY", "channels": reports,
        "calibration_logs": logs, "specification": {"minimum_prior_calls": MINIMUM, "trailing_calls": WINDOW,
                                                     "selection_changed": False, "update": "monthly"},
        "target_definitions": {"market_direction_top_call": "reliability of the called improve/continue/worsen direction",
            "relay_identity_top1": "promotion probability of the fixed identity Top1",
            "relay_non_one_word_top1": "promotion probability of the fixed non-one-word Top1"},
        "limitations": ["Prior-output calibration does not create new information or increase raw prediction hits",
            "Model-family changes may shift the relation between raw score and correctness",
            "No pristine forward validation; calibration confidence may not be presented as a guaranteed win rate",
            "Market-direction reliability may not replace three-regime or index probability"],
        "goal_verified": False, "live_probability_display_authorized": False,
        "source_hashes": {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), market_source, relay_source]}}
    output.mkdir(parents=True)
    for name, rows in predictions.items():
        rows.to_csv(output/(name+"_LABEL_ONLY.csv"),index=False,encoding="utf-8-sig")
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    r=run(args.output)
    print(json.dumps({k:v['combined'] for k,v in r['channels'].items()},ensure_ascii=True))
