"""Fixed next experiment: replace binary loss with within-day pairwise ranking.

Only prior known-label nonword candidates are used. Same fields/window choices
as V281, fixed C grid, prior 63-date monthly selection. No confidence gating or
post-test feature changes. Rank scores are not promotion probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from next_session_forecast_lab_v271 import ROOT, DATA, FORBIDDEN, prepare_relay, top1, rate, sha
from rolling_forecast_challenger_v274 import WINDOWS, START, END
from nonword_specialist_challenger_v281 import splits, complete_rows, CLUES, CONTROL
from relay_promotion_holdout_v256 import _paired

VERSION = "v282-causal-nonword-pairwise-rank-1"
CS = (.25, 1., 4.)


class Ranker:
    def __init__(self, fields, strength):
        if set(fields) & FORBIDDEN:
            raise ValueError("future feature forbidden")
        self.fields = list(fields)
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.scale = StandardScaler()
        self.model = LogisticRegression(C=strength, fit_intercept=False, max_iter=2500, random_state=282)

    def fit(self, rows):
        if not rows.daily_true_one_word.eq(0).all() or rows.target.isna().any():
            raise ValueError("ranking train requires known nonword rows")
        rows = rows.reset_index(drop=True)
        z = self.scale.fit_transform(self.imputer.fit_transform(rows[self.fields]))
        features, labels, weights = [], [], []
        mixed_dates = 0
        for _, group in rows.groupby("date", sort=True):
            positive = group.index[group.target.eq(1)].to_numpy()
            negative = group.index[group.target.eq(0)].to_numpy()
            if not len(positive) or not len(negative):
                continue
            differences = (z[positive, None, :] - z[negative, :]).reshape(-1, z.shape[1])
            n = len(differences)
            features.extend((differences, -differences))
            labels.extend((np.ones(n, dtype=int), np.zeros(n, dtype=int)))
            weights.extend((np.full(n, .5/n), np.full(n, .5/n)))
            mixed_dates += 1
        if mixed_dates < 100:
            raise ValueError("insufficient prior mixed-outcome ranking dates")
        x, y, weight = np.concatenate(features), np.concatenate(labels), np.concatenate(weights)
        self.model.fit(x, y, sample_weight=weight)
        self.training_audit = dict(rows=len(rows), dates=int(rows.date.nunique()), mixed_dates=mixed_dates,
            symmetric_pair_rows=len(y), max_label_date=rows.target_date.max(), one_word_rows=0)
        return self

    def score(self, rows):
        return self.model.decision_function(self.scale.transform(self.imputer.transform(rows[self.fields])))


def predict(ranker, rows):
    result = rows.copy()
    result["ranking_score"] = np.nan
    eligible = result.daily_true_one_word.eq(0)
    if eligible.any():
        result.loc[eligible, "ranking_score"] = ranker.score(result[eligible])
    result["ranking_first"] = False
    picked = top1(result[eligible], score="ranking_score")
    result.loc[picked.index, "ranking_first"] = True
    result["ranking_call"] = np.where(result.ranking_first, "PROMOTE", "NO_SELECTION")
    result["binary_call"] = "NOT_ESTIMATED_BY_RANKING_MODEL"
    result["score_semantics"] = "UNCALIBRATED_RELATIVE_UTILITY_NOT_PROMOTION_PROBABILITY"
    # Do not inherit the old model's probabilities as this model's output.
    return result.drop(columns=[c for c in ("probability", "confidence", "predicted", "correct") if c in result])


def selected(frame, score="ranking_score"):
    rows = complete_rows(frame)
    picks = top1(rows[rows.daily_true_one_word.eq(0)], score=score).copy()
    picks["correct"] = picks.target.astype(int)
    return picks


def frozen(frame, fields, cutoff):
    train, val, refit, start = splits(frame, cutoff)
    choices = []
    for strength in CS:
        for window in WINDOWS:
            dates = sorted(train.date.unique())
            subset = train if window is None else train[train.date.isin(dates[-window:])]
            ranker = Ranker(fields, strength).fit(subset)
            p = predict(ranker, val)
            picks = top1(p, score="ranking_score")
            choices.append(dict(C=strength, window=window, validation_accuracy=float(picks.target.mean()),
                                validation_dates=len(picks)))
    chosen = sorted(choices, key=lambda r: (-r["validation_accuracy"], CS.index(r["C"]),
                                          -(r["window"] or 100000)))[0]
    dates = sorted(refit.date.unique())
    subset = refit if chosen["window"] is None else refit[refit.date.isin(dates[-chosen["window"]:])]
    ranker = Ranker(fields, chosen["C"]).fit(subset)
    log = dict(cutoff=cutoff, chosen=chosen, validation_choices=choices, validation_start=start,
        max_train_label_date=train.target_date.max(), max_validation_label_date=val.target_date.max(),
        max_refit_label_date=subset.target_date.max(), training=ranker.training_audit, test_labels_used=False)
    return ranker, log


def run(output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("immutable ranking output exists")
    prior = ROOT / "results/v281/nonword_specialist_20261002_v1"
    prior_report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    for path, wanted in prior_report["source_hashes"].items():
        if sha(ROOT / path) != wanted:
            raise ValueError("prior experiment input hash changed")
    if sha(prior / "specialist_predictions_LABEL_ONLY.csv") != prior_report["prediction_sha256"]:
        raise ValueError("prior prediction hash changed")
    control = pd.read_csv(CONTROL / "CONTROL_predictions_LABEL_ONLY.csv", dtype={"code": str})
    specialist = pd.read_csv(prior / "specialist_predictions_LABEL_ONLY.csv", dtype={"code": str})
    calendar = pd.read_csv(DATA / "market_emotion.csv").date.tolist()
    base, fields = prepare_relay(DATA / "leader_candidates.csv", calendar)
    if fields != prior_report["specification"]["fields"]:
        raise ValueError("comparison fields changed")
    hp = pd.read_csv(CLUES / "historical_relay_paths_LABEL_ONLY.csv", dtype={"code": str}, low_memory=False)
    base = base.merge(hp[["date", "code", "daily_true_one_word"]], on=["date", "code"],
                      how="left", validate="one_to_one")
    september = pd.read_csv(CLUES / "september_relay_paths_LABEL_ONLY.csv", dtype={"code": str}, low_memory=False)
    target.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
        ROOT / "nonword_specialist_challenger_v281.py", ROOT / "next_session_forecast_lab_v271.py",
        ROOT / "relay_promotion_holdout_v256.py", prior / "report.json", prior / "specialist_predictions_LABEL_ONLY.csv",
        DATA / "leader_candidates.csv", DATA / "market_emotion.csv",
        CLUES / "historical_relay_paths_LABEL_ONLY.csv", CLUES / "september_relay_paths_LABEL_ONLY.csv",
        CONTROL / "CONTROL_predictions_LABEL_ONLY.csv"]}
    specification = dict(version=VERSION, fields=fields, C_grid=list(CS), windows=list(WINDOWS),
        validation_dates=63, update="monthly", domain="signal-day true nonword", gating="NONE",
        selection_metric="every eligible date positive Top1 promotion",
        changed_variable="within-date pairwise loss instead of binary classifier loss",
        tie_break="prior validation accuracy, then smaller C, then expanding window")
    (target / "specification.json").write_text(json.dumps(dict(specification=specification, source_hashes=source_hashes),
        ensure_ascii=False, indent=2), encoding="utf-8")
    periods = [(str(m.start_time.date()), min(str(m.end_time.date()), END))
               for m in pd.period_range(START[:7], END[:7], freq="M")]
    periods.append(("2026-09-01", "2026-09-30"))
    predictions, logs = [], []
    for cutoff, end in periods:
        test = september.copy() if cutoff == "2026-09-01" else base[base.date.between(cutoff, end)].copy()
        ranker, log = frozen(base, fields, cutoff)
        p = predict(ranker, test)
        p["model_freeze"] = cutoff
        p["fold"] = "SEPTEMBER" if cutoff == "2026-09-01" else "2025H2" if cutoff < "2026-01-01" else "2026YTD"
        predictions.append(p); logs.append(log)
        p.to_csv(target / (cutoff+"_predictions_LABEL_ONLY.csv"), index=False, encoding="utf-8-sig")
        (target / "checkpoint.json").write_text(json.dumps(dict(status="RUNNING", completed_months=len(logs),
            selection_logs=logs, source_hashes=source_hashes), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(month_finished=cutoff, chosen=log["chosen"])), flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    alignment = control[["date", "code", "target", "label_valid", "daily_true_one_word"]].merge(
        pred[["date", "code", "target", "label_valid", "daily_true_one_word"]], on=["date", "code"], how="outer",
        indicator=True, validate="one_to_one", suffixes=("_control", "_ranker"))
    if not alignment._merge.eq("both").all():
        raise ValueError("ranker changed full denominator")
    for field in ("target", "label_valid", "daily_true_one_word"):
        a, b = alignment[field+"_control"], alignment[field+"_ranker"]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError("ranker changed outcome/domain labels")
    summary = {fold: rate(selected(g)) for fold, g in pred.groupby("fold")}
    summary["combined_pre_september"] = rate(selected(pred[pred.fold.ne("SEPTEMBER")]))
    paired = {}
    for fold in ("2025H2", "2026YTD", "SEPTEMBER"):
        s = selected(pred[pred.fold.eq(fold)])
        paired[fold] = {}
        for name, other in (("CONTROL", control), ("NONWORD_SPECIALIST", specialist)):
            c = selected(other[other.fold.eq(fold)], score="probability")
            def pair_input(rows):
                return rows[["date", "target"]].rename(columns={"target": "promoted_next_day"})
            paired[fold][name] = _paired(pair_input(s), pair_input(c), seed=282)
    pred.to_csv(target / "ranker_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report = dict(version=VERSION, status="DIAGNOSTIC_ONLY", specification=specification,
        source_hashes=source_hashes, selection_logs=logs, summaries=summary, paired=paired,
        denominator_alignment_rows=len(alignment), prediction_sha256=sha(target / "ranker_predictions_LABEL_ONLY.csv"),
        goal_verified=False, trade_authorized=False, default_model_changed=False,
        limitations=["Reused already viewed history and September: no new blind or forward proof",
            "Relative ranking utility is not promotion probability or leader identity verification",
            "No theme, minute absorption/reseal, announcement completeness or fill evidence added",
            "Dates without a nonword promoter remain failures in unfiltered daily positive Top1"])
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "checkpoint.json").write_text(json.dumps(dict(status="COMPLETED", report_sha256=sha(target / "report.json"))), encoding="utf-8")
    return dict(status=report["status"], summaries=summary, paired=paired, goal_verified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=True))
