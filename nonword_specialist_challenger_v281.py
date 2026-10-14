"""One fixed ablation: train only on signal-day true non-one-word candidates.

Same feature list, monthly schedule, model/window grid and evaluation dates as
V279 CONTROL. Reused dates are diagnostic, not a new blind test. No live plan
or default model is changed. All rows are retained, including unknown outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from next_session_forecast_lab_v271 import ROOT, DATA, prepare_relay, prediction, top1, rate, sha, choose_gate
from rolling_forecast_challenger_v274 import KINDS, WINDOWS, VALIDATION_DATES, START, END, known_before, trailing, fit
from relay_daily_path_challenger_v279 import CLUES
from relay_promotion_holdout_v256 import _paired

VERSION = "v281-nonword-training-cohort-ablation-1"
CONTROL = ROOT / "results/v279/daily_path_paired_20261002_v2"


def splits(frame, cutoff):
    known = known_before(frame, cutoff)
    dates = sorted(known.date.unique())
    if len(dates) <= VALIDATION_DATES:
        raise ValueError("insufficient pre-cutoff validation history")
    start = dates[-VALIDATION_DATES]
    train = known[(known.date < start) & (known.target_date < start) & known.daily_true_one_word.eq(0)]
    val = known[known.date.ge(start) & known.daily_true_one_word.eq(0)]
    refit = known[known.daily_true_one_word.eq(0)]
    if train.empty or val.empty or refit.target_date.max() >= cutoff:
        raise ValueError("empty cohort or post-cutoff labels")
    return train.copy(), val.copy(), refit.copy(), start


def frozen(frame, fields, cutoff):
    train, val, refit_all, validation_start = splits(frame, cutoff)
    choices = []
    predictions = {}
    for kind in KINDS:
        for window in WINDOWS:
            subset = trailing(train, window)
            if subset.date.nunique() < 100 or subset.target.nunique() != 2:
                continue
            mdl = fit(subset, fields, kind, "relay")
            p = top1(prediction(mdl, val, fields, "relay"))
            predictions[(kind, window)] = p
            choices.append(dict(kind=kind, window=window,
                validation_accuracy=float(p.target.mean()), validation_dates=len(p)))
    if not choices:
        raise ValueError("insufficient nonword training dates/classes")
    chosen = sorted(choices, key=lambda r: (-r["validation_accuracy"], KINDS.index(r["kind"]),
                                          -(r["window"] or 100000)))[0]
    validation_calls = predictions[(chosen["kind"], chosen["window"])].copy()
    validation_calls["correct"] = validation_calls.target.astype(int)
    gate = choose_gate(validation_calls, score="probability")
    refit = trailing(refit_all, chosen["window"])
    mdl = fit(refit, fields, chosen["kind"], "relay")
    return mdl, dict(cutoff=cutoff, chosen=chosen, validation_candidates=choices,
        validation_start=validation_start, max_train_label_date=train.target_date.max(),
        max_validation_label_date=val.target_date.max(), max_refit_label_date=refit.target_date.max(),
        refit_rows=len(refit), refit_dates=int(refit.date.nunique()),
        refit_word_rows=int(refit.daily_true_one_word.ne(0).sum()),
        refit_positive_fraction=float(refit.target.mean()), gate=gate, test_labels_used=False)


def forecast(mdl, rows, fields, log):
    result = rows.copy()
    # Explicit domain abstention, without removing any candidate or its label.
    for col in ("predicted", "confidence", "probability"):
        result[col] = np.nan
    eligible = result.daily_true_one_word.eq(0)
    if eligible.any():
        p = prediction(mdl, result[eligible], fields, "relay")
        result.loc[p.index, ["predicted", "confidence", "probability"]] = p[["predicted", "confidence", "probability"]]
    result["domain"] = np.where(eligible, "SIGNAL_DAY_TRUE_NONWORD", "OUTSIDE_SPECIALIST_DOMAIN")
    result["binary_call"] = np.where(~eligible, "UNCERTAIN", np.where(result.predicted.eq(1), "PROMOTE", "FAIL"))
    result["ranking_first"] = False
    result["gated_ranking_call"] = False
    picked = top1(result[eligible])
    result.loc[picked.index, "ranking_first"] = True
    if log["gate"]["validation_qualified"]:
        accepted = picked[picked.probability.ge(log["gate"]["threshold"])]
        result.loc[accepted.index, "gated_ranking_call"] = True
    result["model_freeze"] = log["cutoff"]
    return result


def complete_rows(frame):
    complete = frame.groupby("date").label_valid.all()
    return frame[frame.date.isin(complete[complete].index) & frame.target.notna()].copy()


def selected(frame):
    eligible = complete_rows(frame)
    picked = top1(eligible[eligible.daily_true_one_word.eq(0)]).copy()
    picked["correct"] = picked.target.astype(int)
    return picked


def summarize(frame):
    picked = selected(frame)
    gated = picked[picked.gated_ranking_call]
    eligible = complete_rows(frame)
    eligible = eligible[eligible.daily_true_one_word.eq(0)]
    positive = eligible[eligible.predicted.eq(1)].copy()
    positive["correct"] = positive.target.astype(int)
    return dict(dates_total=int(frame.date.nunique()), complete_dates=int(complete_rows(frame).date.nunique()),
        full_nonword_top1=rate(picked), gated_nonword_top1=rate(gated),
        gated_coverage=len(gated)/len(picked) if len(picked) else 0,
        positive_binary_precision=rate(positive), actual_promotions=int(eligible.target.sum()),
        positive_binary_recall=float(positive.target.sum()/eligible.target.sum()) if eligible.target.sum() else None,
        nonword_binary_accuracy=float(eligible.predicted.eq(eligible.target).mean()) if len(eligible) else None,
        uncertainty_rows=int(frame.domain.ne("SIGNAL_DAY_TRUE_NONWORD").sum()))


def run(output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("immutable specialist output already exists")
    control = pd.read_csv(CONTROL / "CONTROL_predictions_LABEL_ONLY.csv", dtype={"code": str})
    checkpoint = json.loads((CONTROL / "CONTROL_checkpoint.json").read_text(encoding="utf-8"))
    if sha(CONTROL / "CONTROL_predictions_LABEL_ONLY.csv") != checkpoint["prediction_sha256"]:
        raise ValueError("control prediction checkpoint hash mismatch")
    for path, wanted in checkpoint["source_hashes"].items():
        if sha(ROOT / path) != wanted:
            raise ValueError("control input or source hash changed: " + path)
    calendar = pd.read_csv(DATA / "market_emotion.csv").date.tolist()
    base, fields = prepare_relay(DATA / "leader_candidates.csv", calendar)
    hp = pd.read_csv(CLUES / "historical_relay_paths_LABEL_ONLY.csv", dtype={"code": str}, low_memory=False)
    base = base.merge(hp[["date", "code", "daily_true_one_word"]], on=["date", "code"],
                      how="left", validate="one_to_one")
    september = pd.read_csv(CLUES / "september_relay_paths_LABEL_ONLY.csv", dtype={"code": str}, low_memory=False)
    target.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, status="DIAGNOSTIC_ONLY", goal_verified=False, trade_authorized=False,
        specification=dict(changed_variable="train/refit population: signal-day true non-one-word only",
            fields=fields, model_kinds=list(KINDS), windows=list(WINDOWS), validation_dates=VALIDATION_DATES,
            update="monthly", gate_grid="Unchanged V271 fixed grid; no call if validation unqualified",
            nonword_top1_target="Exact next-session promotion of every eligible-date ranking first",
            separate_metrics=["unfiltered daily ranking", "prior-validation gated ranking", "binary calls"]),
        limitations=["Previously viewed historical and September labels; no new blind evidence",
            "Scores after refit are uncalibrated and cannot be shown as live probabilities",
            "Domain flag is signal-day daily proxy, not minute path, theme or actual fill",
            "Selective accuracy cannot replace full-date accuracy; no default or frozen-plan update"],
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / "rolling_forecast_challenger_v274.py", ROOT / "next_session_forecast_lab_v271.py",
            ROOT / "relay_daily_path_challenger_v279.py", DATA / "leader_candidates.csv", DATA / "market_emotion.csv",
            CLUES / "historical_relay_paths_LABEL_ONLY.csv", CLUES / "september_relay_paths_LABEL_ONLY.csv",
            CONTROL / "CONTROL_predictions_LABEL_ONLY.csv", CONTROL / "CONTROL_checkpoint.json"]}, selection_logs=[])
    outputs = []
    periods = [(str(m.start_time.date()), min(str(m.end_time.date()), END))
               for m in pd.period_range(START[:7], END[:7], freq="M")]
    periods.append(("2026-09-01", "2026-09-30"))
    for cutoff, end in periods:
        test = september.copy() if cutoff == "2026-09-01" else base[base.date.between(cutoff, end)].copy()
        mdl, log = frozen(base, fields, cutoff)
        p = forecast(mdl, test, fields, log)
        p["fold"] = "SEPTEMBER" if cutoff == "2026-09-01" else "2025H2" if cutoff < "2026-01-01" else "2026YTD"
        outputs.append(p)
        report["selection_logs"].append(log)
        p.to_csv(target / (cutoff + "_predictions_LABEL_ONLY.csv"), index=False, encoding="utf-8-sig")
        (target / "checkpoint.json").write_text(json.dumps(dict(status="RUNNING", completed_months=len(outputs),
            selection_logs=report["selection_logs"], source_hashes=report["source_hashes"]),
            ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(dict(month_finished=cutoff, selected=log["chosen"], gate_qualified=log["gate"]["validation_qualified"])), flush=True)
    pred = pd.concat(outputs, ignore_index=True)
    if pred.duplicated(["date", "code"]).any():
        raise ValueError("duplicate prediction denominator")
    alignment = control[["date", "code", "target", "label_valid", "daily_true_one_word"]].merge(
        pred[["date", "code", "target", "label_valid", "daily_true_one_word"]], on=["date", "code"],
        how="outer", indicator=True, validate="one_to_one", suffixes=("_control", "_specialist"))
    if not alignment._merge.eq("both").all():
        raise ValueError("full candidate denominator differs from control")
    for field in ("target", "label_valid", "daily_true_one_word"):
        left, right = alignment[field + "_control"], alignment[field + "_specialist"]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError("control label/domain differs: " + field)
    report["denominator_alignment_rows"] = len(alignment)
    report["summaries"] = {fold: summarize(g) for fold, g in pred.groupby("fold")}
    report["summaries"]["combined_pre_september"] = summarize(pred[pred.fold.ne("SEPTEMBER")])
    report["paired"] = {}
    for fold in ("2025H2", "2026YTD", "SEPTEMBER"):
        c, s = selected(control[control.fold.eq(fold)]), selected(pred[pred.fold.eq(fold)])
        def pair_input(rows):
            return rows[["date", "target"]].rename(columns={"target": "promoted_next_day"})
        report["paired"][fold] = _paired(pair_input(s), pair_input(c), seed=281)
    pred.to_csv(target / "specialist_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report["prediction_sha256"] = sha(target / "specialist_predictions_LABEL_ONLY.csv")
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "checkpoint.json").write_text(json.dumps(dict(status="COMPLETED", report_sha256=sha(target / "report.json")), indent=2), encoding="utf-8")
    return dict(status=report["status"], summaries=report["summaries"], paired=report["paired"], goal_verified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=True))
