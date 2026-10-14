"""One fixed causal previous-session identity/turnover-phase relay challenge."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import model_upgrade_history as h
from current_model_diagnostics import RELAY_FIELDS
from relay_error_analysis import verify_sources, write_new, now
from relay_class_balance_research import project, statistics
from model_upgrade_research import read, sha

EXTRA = ["prior_official_signal_one_word_proxy", "one_word_to_turnover_proxy"]


def add_phase(frame, sessions):
    if frame.duplicated(["date", "code"]).any() or sessions != sorted(set(sessions)):
        raise ValueError("unique full original cohort and official sessions required")
    out = frame.copy(); predecessor = dict(zip(sessions[1:], sessions[:-1]))
    if not set(out.date).issubset(sessions):
        raise ValueError("signal date outside official calendar")
    lookup = {(r.date, r.code): (r.signal_one_word, r.boards)
              for r in out[["date", "code", "signal_one_word", "boards"]].itertuples(index=False)}
    previous, phase = [], []
    for r in out[["date", "code", "signal_one_word", "boards"]].itertuples(index=False):
        before, board = lookup.get((predecessor.get(r.date), r.code), (np.nan, np.nan))
        if pd.isna(before) or pd.isna(board) or before not in (0, 1) or board+1 != r.boards:
            previous.append(np.nan); phase.append(np.nan)
        else:
            previous.append(float(before))
            phase.append(float((1-r.signal_one_word)*before) if pd.notna(r.signal_one_word) and r.signal_one_word in (0, 1) else np.nan)
    out[EXTRA[0]], out[EXTRA[1]] = previous, phase
    return out


def fit_once(train, fields):
    from rolling_forecast_challenger_v274 import make
    from sklearn.exceptions import ConvergenceWarning
    if fields != RELAY_FIELDS+EXTRA or set(train.target.unique()) != {0., 1.}:
        raise ValueError("registered fields and both known original training classes required")
    estimator = make("logistic")
    weight = 1/train.groupby("date").date.transform("size")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(train[fields], train.target.astype(int), logisticregression__sample_weight=weight)
    return estimator


def run(path):
    path = Path(path); p = read(path); root = path.parent
    if (p["version"] != "relay-oneword-to-turnover-fixed-1" or p["arms"] != ["BASE", "R3"]
            or p["fields"] != EXTRA or p["folds"] != h.FOLDS
            or p["parameters"] != {"C":.25, "max_iter":2500, "random_state":271, "solver":"lbfgs", "class_weight":None, "window":252, "threshold":.5}):
        raise ValueError("fixed phase protocol changed")
    verify_sources(p["sources"]); receipt = read(p["cached_receipt"])
    verify_sources(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    if receipt["fields"] != RELAY_FIELDS or any(importlib.metadata.version(k) != v for k,v in receipt["runtime_versions"].items()):
        raise ValueError("original field/runtime conflict")
    if (root/"attempt_STARTED.json").exists():
        raise ValueError("fixed phase comparison already attempted; no retuning")
    before = read(root/"implementation_BEFORE_FITS.json")
    if before["protocol_sha256"] != sha(path):
        raise ValueError("pre-fit protocol binding conflict")
    verify_sources(before["sources"])
    raw = joblib.load(receipt["prepared_path"]); full = add_phase(raw, receipt["sessions"])
    saved = pd.read_csv(Path(p["base_root"])/"relay_history/predictions.csv", dtype={"code":str}, low_memory=False)
    checks = []
    for fold in p["folds"]:
        train = h.training_rows(full, fold["start"], "relay")
        expected = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if h._frame_digest(train, RELAY_FIELDS) != expected["training_digest"] or train.target_date.max() != expected["training_max_label_date"]:
            raise ValueError("original training members/inputs/cutoff changed")
        checks.append({**expected, "phase_training_digest": h._frame_digest(train, RELAY_FIELDS+EXTRA),
                       "phase_known_training_rows": int(train[EXTRA[1]].notna().sum()),
                       "transition_training_rows": int(train[EXTRA[1]].eq(1).sum())})
    write_new(root/"training_checks_BEFORE_FITS.json", {"formed_at":now(), "checks":checks})
    write_new(root/"attempt_STARTED.json", {"formed_at":now(), "protocol_sha256":sha(path)})
    comparisons, rows, logs, failed = {}, [], [], False
    for fold in p["folds"]:
        train = h.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        baseline = saved[(saved.arm=="BASE") & (saved.fold==fold["id"])].copy()
        if set(zip(test.date,test.code)) != set(zip(baseline.date,baseline.code)) or len(test)!=len(baseline):
            raise ValueError("full original candidate roster differs")
        error = None
        try:
            if failed or (datetime.now(timezone.utc)-datetime.fromisoformat(p["batch_started_at"])).total_seconds()>=1800:
                raise ValueError("prior failure or fixed batch budget reached")
            estimator = fit_once(train, RELAY_FIELDS+EXTRA)
            predicted = h._predict(estimator, test, RELAY_FIELDS+EXTRA, "relay")
        except Exception as exc:
            failed=True;error=type(exc).__name__+": "+str(exc);predicted=test.copy()
            predicted["status"]="FIT_FAILED_UNKNOWN";predicted["predicted"]=None;predicted["raw_score"]=np.nan
        br,cr = project(baseline,"BASE",receipt["sessions"]),project(predicted,"R3",receipt["sessions"])
        index={(r["signal_date"],r["code"]):r for r in br}
        if any(r["truth"] != index[(r["signal_date"],r["code"])]["truth"] for r in cr):
            raise ValueError("baseline/challenge labels differ")
        bm,cm = statistics(br,receipt["sessions"],fold),statistics(cr,receipt["sessions"],fold)
        delta = cm["Accuracy"]-bm["Accuracy"] if cm["Accuracy"] is not None and bm["Accuracy"] is not None else None
        guards={k: cm[k] is not None and bm[k] is not None and cm[k]>=bm[k]-1e-12
                for k in ("Precision","complete_positive_Recall","F1","coverage")}
        bt,ct=bm["true_nonword_rank_first"],cm["true_nonword_rank_first"]
        guards["true_nonword_rank_first"]=bt["rate"] is None or (ct["rate"] is not None and ct["rate"]>=bt["rate"] and ct["scoreable_days"]>=bt["scoreable_days"])
        subgroup=[]
        for arm, group in (("BASE",br),("R3",cr)):
            linked={(r.date,r.code):r for r in test.itertuples()}
            from cycle_regime_diagnostics import metrics
            selected=[r for r in group if getattr(linked[(r["signal_date"],r["code"])],EXTRA[1])==1]
            subgroup.append({"arm":arm,"metrics":metrics(selected)})
        comparisons[fold["id"]]={"BASE":bm,"R3":cm,"Accuracy_delta":delta,"guards":guards,
            "historical_screen_passed":bool(not error and delta is not None and delta>=.03 and all(guards.values())),
            "transition_subgroup_diagnostic_only":subgroup,"fit_failure":error,
            "phase_known_test_rows":int(test[EXTRA[1]].notna().sum()),"test_transition_rows":int(test[EXTRA[1]].eq(1).sum())}
        rows.extend([{**r,"fold":fold["id"]} for r in br+cr]);logs.append({**next(c for c in checks if c["fold"]==fold["id"]),"fitted_once":error is None,"fit_failure":error,"deployment_weights_saved":False})
    output=root/"full_candidate_comparison.csv"
    pd.DataFrame(rows).to_csv(output,index=False,encoding="utf-8-sig")
    result={"status":"HISTORICAL_SCREEN_PASSED_NEW_FORWARD_REQUIRED" if all(c["historical_screen_passed"] for c in comparisons.values()) else "REJECTED_FIXED_PHASE_CHALLENGE",
        "formed_at":now(),"comparisons":comparisons,"fit_logs":logs,"protocol_sha256":sha(path),"predictions_sha256":sha(output),
        "production_changed":False,"new_mature_forward_samples":0,"calibrated_probability_Brier":None,"past_history_is_blind_or_forward":False}
    write_new(root/"report.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--protocol",required=True)
    args=parser.parse_args();result=run(args.protocol);print(json.dumps({"status":result["status"],"formed_at":result["formed_at"]}))
