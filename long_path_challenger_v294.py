"""Fixed chronological minute-path ablation; viewed history, no live upgrade."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from long_path_features_v291 import BASE_FIELDS, PRICE_FIELDS, VOLUME_FIELDS
from next_session_forecast_lab_v271 import ROOT, FORBIDDEN, sha
from relay_promotion_holdout_v256 import wilson, _paired

VERSION = "v294-fixed-long-path-calibrated-diagnostic-1"
ARMS = {"BASE": BASE_FIELDS, "PRICE_PATH": BASE_FIELDS+PRICE_FIELDS,
        "PRICE_VOLUME_PATH": BASE_FIELDS+PRICE_FIELDS+VOLUME_FIELDS}
SPEC = {"train_start":"2025-10-01", "train_end_exclusive":"2026-01-01",
        "calibration_start":"2026-01-01", "test_start":"2026-03-01", "test_end":"2026-09-30",
        "model":"median imputation with missing indicators, standardization, logistic C=0.25",
        "calibration":"logistic C=1 on prior calibration logits, no refit after calibration",
        "binary_threshold":0.5, "arms":ARMS, "selection":"none; report all three fixed arms",
        "ranking":"full eligible-date Top1, descending calibrated score, boards, ascending code",
        "research_status":"PREVIOUSLY_VIEWED_HISTORICAL_DIAGNOSTIC_ONLY",
        "pressure":"same predictions; six original positive contradictions assumed zero separately"}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def inputs(directory):
    directory = Path(directory)
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    for name, wanted in report["source_hashes"].items():
        if sha(Path(name)) != wanted:
            raise ValueError("prepared source changed: "+name)
    for name, wanted in report["artifact_hashes"].items():
        if sha(directory/name) != wanted:
            raise ValueError("prepared artifact changed: "+name)
    features = pd.read_csv(directory/"features_NO_NEXT_SESSION_LABELS.csv", dtype={"code":str})
    labels = pd.read_csv(directory/"labels_ORIGINAL_AND_PRESSURE_ONLY.csv", dtype={"code":str})
    legacy = pd.read_csv(directory/"legacy_predictions_NO_LABELS.csv", dtype={"code":str})
    expected = {"date", "code", "daily_true_one_word", *BASE_FIELDS, *PRICE_FIELDS, *VOLUME_FIELDS}
    if set(features) != expected or set(features)&FORBIDDEN:
        raise ValueError("feature allowlist changed or contains outcome fields")
    keys = features[["date", "code"]]
    for frame in [features, labels, legacy]:
        if frame.duplicated(["date", "code"]).any() or set(map(tuple, frame[["date", "code"]].to_numpy())) != set(map(tuple, keys.to_numpy())):
            raise ValueError("candidate denominator changed")
    if len(features) != report["candidate_rows"] or features.date.nunique() != report["candidate_dates"]:
        raise ValueError("prepared counts changed")
    if labels.label_valid.dtype != bool or not labels.loc[labels.label_valid, "target_original"].isin([0,1]).all():
        raise ValueError("invalid original label flags")
    return features, labels, legacy, report


def chronological(features, labels):
    frame = features.merge(labels[["date", "code", "target_date", "target_original", "label_valid"]], on=["date", "code"], validate="one_to_one")
    known = frame.label_valid & frame.target_original.notna()
    train = frame[known & frame.date.ge(SPEC["train_start"]) & frame.date.lt(SPEC["train_end_exclusive"]) & frame.target_date.lt(SPEC["train_end_exclusive"])].copy()
    calibration = frame[known & frame.date.ge(SPEC["calibration_start"]) & frame.date.lt(SPEC["test_start"]) & frame.target_date.lt(SPEC["test_start"])].copy()
    test = features[features.date.between(SPEC["test_start"], SPEC["test_end"])].copy()
    for part, minimum in [(train,30), (calibration,20)]:
        if part.date.nunique() < minimum or len(part) < 60 or part.target_original.nunique() != 2:
            raise ValueError("insufficient prior training/calibration history")
        if not part.target_date.gt(part.date).all():
            raise ValueError("invalid next-session label chronology")
    if test.empty:
        raise ValueError("empty test")
    return train, calibration, test


def logit(values):
    p = np.clip(np.asarray(values,dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def fit_forecast(features, labels, fields):
    if not set(fields).issubset(set(BASE_FIELDS+PRICE_FIELDS+VOLUME_FIELDS)) or set(fields)&FORBIDDEN:
        raise ValueError("unapproved model fields")
    train, calibration, test = chronological(features, labels)
    mdl = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(),
        LogisticRegression(C=.25, max_iter=2500, random_state=294))
    mdl.fit(train[fields], train.target_original.astype(int))
    prior_score = mdl.predict_proba(calibration[fields])[:,1]
    calibrator = LogisticRegression(C=1., max_iter=2500, random_state=294)
    calibrator.fit(logit(prior_score), calibration.target_original.astype(int))
    raw = mdl.predict_proba(test[fields])[:,1]
    calibrated = calibrator.predict_proba(logit(raw))[:,1]
    predicted = test[["date", "code", "boards", "daily_true_one_word"]].copy()
    predicted["raw_score"] = raw
    predicted["calibrated_score"] = calibrated
    predicted["predicted"] = calibrated >= SPEC["binary_threshold"]
    log = {"train_rows":len(train), "train_dates":int(train.date.nunique()), "calibration_rows":len(calibration),
        "calibration_dates":int(calibration.date.nunique()), "max_train_target_date":train.target_date.max(),
        "max_calibration_target_date":calibration.target_date.max(), "test_rows":len(test), "test_dates":int(test.date.nunique()),
        "refit_after_calibration":False, "test_labels_used":False, "production_probability_authorized":False,
        "calibrator_intercept":float(calibrator.intercept_[0]), "calibrator_slope":float(calibrator.coef_[0,0])}
    return predicted, log


def binary(rows, target="target_original"):
    known = rows.label_valid & rows[target].notna()
    selected = rows[known]
    y = selected[target].to_numpy(dtype=int)
    call = selected.predicted.to_numpy(dtype=bool)
    score = selected.calibrated_score.to_numpy(dtype=float)
    tp,fp,fn,tn = [int(mask.sum()) for mask in [call&(y==1), call&(y==0), ~call&(y==1), ~call&(y==0)]]
    n = len(y)
    bins = []
    for i in range(10):
        mask = (score>=i/10)&(score<(i+1)/10 if i<9 else score<=1)
        bins.append({"lower":i/10, "upper":(i+1)/10, "n":int(mask.sum()),
            "mean_probability":float(score[mask].mean()) if mask.any() else None,
            "actual_rate":float(y[mask].mean()) if mask.any() else None})
    ece = sum(b["n"]/n*abs(b["mean_probability"]-b["actual_rate"]) for b in bins if b["n"]) if n else None
    return {"total_frozen_rows":len(rows), "known":n, "unknown":int((~known).sum()), "called":len(rows),
        "tp":tp,"fp":fp,"fn":fn,"tn":tn, "correct":tp+tn,
        "accuracy":(tp+tn)/n if n else None, "accuracy_wilson95_descriptive":wilson(tp+tn,n),
        "precision":tp/(tp+fp) if tp+fp else None, "recall":tp/(tp+fn) if tp+fn else None,
        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "all_fail_accuracy":tn/n+fp/n if n else None,
        "brier":float(np.mean((score-y)**2)) if n else None,
        "raw_brier":float(np.mean((selected.raw_score.to_numpy(dtype=float)-y)**2)) if n else None,
        "ece":ece, "calibration_bins":bins, "date_cluster_warning":True}


def ranking(rows, target="target_original", nonword=True):
    # Choose on every frozen date first. Missing labels never change the ranking.
    pool = rows[rows.daily_true_one_word.eq(0)] if nonword else rows
    picked = pool.sort_values(["date","calibrated_score","boards","code"], ascending=[True,False,False,True], kind="stable").drop_duplicates("date").copy()
    complete = rows.groupby("date").apply(lambda g:bool(g.label_valid.all() and g[target].notna().all()), include_groups=False)
    scored = picked[picked.date.map(complete).eq(True)]
    hits = int(scored[target].sum())
    return {"frozen_eligible_dates":len(picked), "scored_complete_dates":len(scored), "unknown_dates":len(picked)-len(scored),
        "successes":hits,"rate":hits/len(scored) if len(scored) else None,"wilson95":wilson(hits,len(scored))}, scored[["date",target]].rename(columns={target:"promoted_next_day"})


def summarize(rows, target):
    nonword, picks = ranking(rows,target)
    tiers = {str(h)+"进"+str(h+1):binary(rows[rows.boards.eq(h)],target) for h in [2,3,4,5]}
    tiers["5板以上"] = binary(rows[rows.boards.ge(5)],target)
    return {"full_candidate":binary(rows,target),"nonword_candidate":binary(rows[rows.daily_true_one_word.eq(0)],target),
        "nonword_top1":nonword,"all_top1":ranking(rows,target,False)[0], "tiers":tiers,
        "months":{m:{"binary":binary(g,target),"nonword_top1":ranking(g,target)[0]} for m,g in rows.groupby(rows.date.str[:7])}}, picks


def run(prepared, output):
    features, labels, legacy, prepared_report = inputs(prepared)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"specification_BEFORE_MODEL_TEST_SCORING.json", {"recorded_at":datetime.now(timezone.utc).isoformat(), **SPEC})
    forecasts,logs,artifact_hashes = {},{},{}
    for arm, fields in ARMS.items():
        pred,log = fit_forecast(features, labels, fields)
        path = output/(arm+"_predictions_BEFORE_LABEL_JOIN.csv")
        pred.to_csv(path,index=False,encoding="utf-8-sig")
        forecasts[arm] = pred
        logs[arm] = log
        artifact_hashes[path.name] = sha(path)
        print(json.dumps({"arm_forecast_saved":arm,"rows":len(pred),"chronology":log}),flush=True)
    results,paired,picks = {},{},{}
    for arm,pred in forecasts.items():
        scored = pred.merge(labels,on=["date","code"],how="left",validate="one_to_one")
        path = output/(arm+"_scored_LABEL_ONLY.csv")
        scored.to_csv(path,index=False,encoding="utf-8-sig")
        artifact_hashes[path.name] = sha(path)
        results[arm] = {}
        for target in ["target_original","target_pressure_zero"]:
            results[arm][target], picks[arm,target] = summarize(scored,target)
    # Same-period old predictions are a contextual control with a different training history.
    legacy_scored = legacy.merge(features[["date","code","boards","daily_true_one_word"]],on=["date","code"],validate="one_to_one")
    legacy_scored = legacy_scored[legacy_scored.date.between(SPEC["test_start"],SPEC["test_end"])].rename(columns={"probability":"calibrated_score"})
    legacy_scored = legacy_scored.merge(labels,on=["date","code"],validate="one_to_one")
    legacy_context = {}
    for target in ["target_original","target_pressure_zero"]:
        legacy_context[target] = ranking(legacy_scored,target)[0]
        paired[target] = {arm:_paired(picks[arm,target],picks["BASE",target],seed=294) for arm in ARMS if arm!="BASE"}
    paths = [Path(__file__),ROOT/"long_path_features_v291.py",ROOT/"next_session_forecast_lab_v271.py",ROOT/"relay_promotion_holdout_v256.py",
        Path(prepared)/"report.json",*[Path(prepared)/name for name in prepared_report["artifact_hashes"]]]
    record = {"version":VERSION,"status":"FIXED_CHRONOLOGICAL_ABLATION_COMPLETE_DIAGNOSTIC_ONLY", "recorded_at":datetime.now(timezone.utc).isoformat(),
        "specification":SPEC,"fit_logs":logs,"arms":results,"paired_nonword_top1_against_same_training_base":paired,
        "legacy_context_different_training_history":legacy_context,
        "all_candidates_preserved":True,"historical_source_certification_complete":False,"pristine_blind":False,
        "original_predictions_rewritten":False,"core_model_upgrade":False,"stable_60_verified":False,
        "actual_fill_count":0,"max_drawdown":None,"profit_loss_ratio":None,"trade_authorized":False,
        "limitations":["Previously viewed history and retrospective retrieval; diagnostic only, no new forward evidence",
            "Original historical labels, provider universe and official limit/security states remain partially unverified",
            "Pressure assumption is separate sensitivity, not corrected or certified truth",
            "30-minute ranges cannot establish exact seal/reseal, auction, causal theme influence or executable fills",
            "Historical interval VWAP uses interval means; not exact trade-by-trade VWAP",
            "Fixed probabilities calibrated only on preceding Jan-Feb; may drift afterwards",
            "Wilson intervals ignore within-day dependence; paired ranking comparison includes five-day block bootstrap",
            "Legacy context has different training history and is not a clean minute-feature ablation"],
        "feature_coverage":prepared_report["coverage"], "source_hashes":{str(p.resolve()):sha(p) for p in paths},
        "artifact_hashes":{**artifact_hashes,"specification_BEFORE_MODEL_TEST_SCORING.json":sha(output/"specification_BEFORE_MODEL_TEST_SCORING.json")}}
    write_json(output/"report.json",record)
    return {"status":record["status"],"nonword_top1":{arm:results[arm]["target_original"]["nonword_top1"] for arm in ARMS}, "stable_60_verified":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.prepared,args.output),ensure_ascii=True))
