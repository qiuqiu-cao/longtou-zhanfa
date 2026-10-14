"""One fixed same-hypothesis source-coverage repair comparison, never deploys."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import relay_context_research as common
from relay_curvature_research import paired_signature
from relay_expert_research import subgroup
from relay_recency_walkforward import paired_interval

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "research/model_upgrade_validation/vwap_source_repair_20261014T172716Z"
AUDIT = ROOT / "research/model_upgrade_validation/cap_followup_20261014T170949Z"
ARM = "DAILY_VWAP_SOURCE_COVERAGE_REPAIR"
FIELD = "daily_close_vs_vwap"
FIELDS = common.RELAY_FIELDS + [FIELD]
PARAMS = {"C": .25, "max_iter": 2500, "solver": "lbfgs", "random_state": 271, "class_weight": None}


def make():
    branch = lambda: Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                               ("scaler", StandardScaler())])
    # Original transformed dimensions are an unchanged first block.
    preprocessing = ColumnTransformer([("original", branch(), common.RELAY_FIELDS), ("daily_vwap", branch(), [FIELD])])
    return Pipeline([("preprocessing", preprocessing), ("logisticregression", LogisticRegression(**PARAMS))])


def spec():
    return {"version": "daily-vwap-source-coverage-repair-1", "arms": ["BASE", ARM], "fields": FIELDS,
            "parameters": PARAMS, "training_window": 252, "weight": "inverse original training rows per signal date",
            "threshold": "original positive score strictly>0.5", "maximum_fits": 3, "BASE_refits": 0,
            "fixed_accuracy_increment_each_fold": .03, "fixed_guards": common.GATE["guards"],
            "other_guards": "original true-nonword rank-first/positive-first coverage; September actual nonword Recall",
            "block_interval": {"seed": 20261007, "official_session_block": 5, "draws": 2000, "lower_tail": .00625, "upper_tail": .99375},
            "budget_seconds": 1800, "folds": common.history.FOLDS,
            "hypothesis_status": "Same close/VWAP economic hypothesis as V294; source/history coverage repair, not new factor",
            "source_policy": "CURRENT_KNOWLEDGE_RETROSPECTIVE_SOURCE_GRADED_DIAGNOSTIC; strict original-time source qualification NOT passed",
            "source_grades": "8920 directSina shares;182 Tencent100-share raw-volume inference;143 UNKNOWN; no proxy fill",
            "source_gate_scope": "date/code/sourceSHA/originalprice_amount reconciliation, declared Sina units/OHLC and explicitly provisional Sept unit; no certification of original HTTP receipt, share effective date, live trading or untouched test",
            "deployment_authorized": False, "old_V294_failure_preserved": True,
            "no_effect_selected_definition": True, "stop": "one actual fit per fold; no retries/retuning after failure or budget"}


def read_inputs(protocol):
    common.verify(protocol["sources"])
    for package, version in protocol["runtime_versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("runtime changed")
    full = joblib.load(protocol["prepared_path"])["relay"]
    layer = pd.read_csv(protocol["vwap_layer"], dtype={"code": str}, float_precision="round_trip")
    if layer.duplicated(["date", "code"]).any() or set(zip(full.date, full.code)) != set(zip(layer.date, layer.code)):
        raise ValueError("complete original9245 source layer required")
    frame = full.merge(layer[["date", "code", FIELD, "daily_measurement_policy"]], on=["date", "code"], validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(frame[full.columns], full.reset_index(drop=True))
    if len(frame) != 9245 or frame[FIELD].notna().sum() != 9102:
        raise ValueError("fixed source-covered and UNKNOWN pool changed")
    baseline = pd.read_csv(protocol["baseline_path"], dtype={"code": str}, low_memory=False)
    return frame, baseline


def freeze(started):
    if (EXP / "protocol_BEFORE_ANY_FIT.json").exists():
        raise ValueError("source repair protocol already frozen")
    elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(started)).total_seconds()
    if not 0 <= elapsed < 1800:
        raise ValueError("actual bounded timezone-aware start required")
    receipt_path = ROOT / "research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json"
    receipt = common.read(receipt_path)
    common.verify(receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"]})
    layer = AUDIT / "cap_information_agent/full9245_v294_vs_daily_vwap_NO_OUTCOMES.csv"
    baseline = ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/relay_history/predictions.csv"
    sources = receipt["sources"] | {receipt["prepared_path"]: receipt["prepared_sha256"], str(receipt_path): common.sha(receipt_path),
                                    str(layer): common.sha(layer), str(baseline): common.sha(baseline)}
    extras = [AUDIT / "cap_source_agent/cap_source_findings.json", AUDIT / "cap_information_agent/v294_daily_vwap_crosscheck_results.json",
              AUDIT / "cap_information_agent/v294_daily_crosscheck_verification.json", Path(__file__), ROOT / "tests/test_relay_vwap_source_repair_research.py",
              ROOT / "relay_context_research.py", ROOT / "relay_curvature_research.py", ROOT / "relay_expert_research.py", ROOT / "relay_recency_walkforward.py",
              ROOT / "model_upgrade_history.py", ROOT / "data/independent_stock_judgement/registry.jsonl",
              ROOT / "results/score_update_20261014T1153Z/supplemental_BASE_scores.json"]
    for filename in extras:
        sources[str(filename.resolve())] = common.sha(filename)
    p = {"definition": spec(), "formed_at": common.now(), "batch_started_at": started,
         "authorization": "user继续深入研究; existing user permits historical research and fixed source-repair comparison without pretending original-time source gates passed",
         "prepared_path": receipt["prepared_path"], "vwap_layer": str(layer), "baseline_path": str(baseline),
         "sources": sources, "sessions": receipt["sessions"], "runtime_versions": receipt["runtime_versions"],
         "previous_trial": "V294 bundled factor failed; not overwritten, retried, or declared successful",
         "distinct_source_evidence": "shared1913 values allwithin1e-4;973 newly observed inold2888 pool;6216 direct-observed outsideoldpool;source coverage, not measurement formula repair"}
    frame, saved = read_inputs(p)
    checks = []
    for fold in spec()["folds"]:
        train = common.history.training_rows(frame, fold["start"], "relay")
        original = next(c for c in receipt["training_checks"] if c["fold"] == fold["id"])
        if common.history._frame_digest(train, common.RELAY_FIELDS) != original["training_digest"]:
            raise ValueError("original training inputs/members differ")
        test = frame[frame.date.between(fold["start"], fold["end"])]
        base = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])]
        if paired_signature(common.project(test, ARM, p["sessions"])) != paired_signature(common.project(base, "BASE", p["sessions"])):
            raise ValueError("same original complete pool and truth required before fitting")
        checks.append({**original, "newfield_train_known": int(train[FIELD].notna().sum()),
                       "test_n": len(test), "newfield_test_known": int(test[FIELD].notna().sum()),
                       "test_source_grades": test.daily_measurement_policy.value_counts(dropna=False).to_dict()})
    p["training_checks"] = checks
    common.write_new(EXP / "protocol_BEFORE_ANY_FIT.json", p)
    return {"status": "FROZEN_SOURCE_COVERAGE_REPAIR_NOT_STRICT_SOURCE_PASS", "checks": checks}


def compare():
    if (EXP / "attempt_STARTED.json").exists():
        raise ValueError("same source-repair comparison already attempted; no retry")
    path = EXP / "protocol_BEFORE_ANY_FIT.json"
    p = common.read(path)
    if p["definition"] != spec():
        raise ValueError("fixed definition changed")
    full, saved = read_inputs(p)
    common.write_new(EXP / "attempt_STARTED.json", {"formed_at": common.now(), "protocol_sha256": common.sha(path)})
    output, comparisons, logs = [], {}, []
    for fold in spec()["folds"]:
        if (datetime.now(timezone.utc)-datetime.fromisoformat(p["batch_started_at"])).total_seconds() >= 1800:
            common.write_new(EXP / "BUDGET_STOP.json", {"formed_at": common.now(), "fits": len(logs), "no_retry": True})
            raise ValueError("fixed budget ended")
        train = common.history.training_rows(full, fold["start"], "relay")
        test = full[full.date.between(fold["start"], fold["end"])].copy()
        base = saved[saved.arm.eq("BASE") & saved.fold.eq(fold["id"])].copy()
        wanted = next(c for c in p["training_checks"] if c["fold"] == fold["id"])
        if common.history._frame_digest(train, common.RELAY_FIELDS) != wanted["training_digest"]:
            raise ValueError("source/membership drift")
        common.write_new(EXP / (fold["id"]+"_fit_STARTED.json"), {"formed_at": common.now(), "fit_count": 1, "training_digest": wanted["training_digest"]})
        try:
            model = make()
            with warnings.catch_warnings(), threadpool_limits(limits=2):
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train[FIELDS], train.target.astype(int), logisticregression__sample_weight=1/train.groupby("date").date.transform("size"))
                pred = common.history._predict(model, test, FIELDS, "relay")
            columns = list(dict.fromkeys(["date", "target_date", "code", "name", "boards", "target", "label_valid", "status", "predicted", "raw_score", "actual_one_word_daily", "daily_measurement_policy"]+FIELDS))
            with (EXP / (fold["id"]+"_predictions_BEFORE_VALIDATION.csv")).open("x", encoding="utf-8-sig", newline="") as f:
                pred[columns].to_csv(f,index=False)
            artifact = EXP / (fold["id"]+"_research_only.joblib")
            joblib.dump(model,artifact)
            common.history.compare_reproduction(common.history._predict(joblib.load(artifact),test,FIELDS,"relay"),pred,"relay")
            br, cr = common.project(base,"BASE",p["sessions"]), common.project(pred,ARM,p["sessions"])
            if paired_signature(br) != paired_signature(cr):
                raise ValueError("complete original truth changed")
            bm,cm = common.score(br,p["sessions"],fold),common.score(cr,p["sessions"],fold)
            groups={"BASE":subgroup(br,test),ARM:subgroup(cr,test)}
            guards=common.guards(bm,cm)
            if fold["id"]=="2026Sep":
                guards["September_actual_nonword_Recall"] = groups[ARM]["true_nonword"]["complete_positive_Recall"] >= groups["BASE"]["true_nonword"]["complete_positive_Recall"]
            delta=cm["Accuracy"]-bm["Accuracy"]
            interval=paired_interval(br,cr,p["sessions"],fold,spec()["block_interval"])
            comparisons[fold["id"]]={"BASE":bm,ARM:cm,"Accuracy_delta":delta,"guards":guards,"subgroups":groups,"fixed_interval":interval,
                "passed":delta>=.03 and all(guards.values()) and interval["lower"] is not None and interval["lower"]>0}
            logs.append({"fold":fold["id"],"fit_count":1,"model_sha256":common.sha(artifact)})
            output.extend([{**r,"fold":fold["id"]} for r in br+cr])
            pd.DataFrame(output).to_csv(EXP/"full_candidate_comparison.csv",index=False,encoding="utf-8-sig")
            common.write_new(EXP/(fold["id"]+"_checkpoint.json"),comparisons[fold["id"]])
            print(fold["id"],"BASE",bm["Accuracy"],"REPAIR",cm["Accuracy"],"delta",delta,flush=True)
        except Exception as exc:
            common.write_new(EXP/"FAILURE_STOP_NO_RETRY.json",{"formed_at":common.now(),"fold":fold["id"],"error":type(exc).__name__+": "+str(exc),"fit_receipts":len(logs)+1})
            raise
    common.verify(p["sources"])
    report={"formed_at":common.now(),"status":"HISTORICAL_SURVIVOR_REQUIRES_STRICT_SOURCE_AND_FORWARD" if all(c["passed"] for c in comparisons.values()) else "REJECTED_DAILY_VWAP_SOURCE_COVERAGE_REPAIR",
            "comparisons":comparisons,"fit_logs":logs,"actual_predictive_fits":len(logs),"BASE_refits":0,"paired_rows":len(output),"new_mature_forward_samples":0,
            "strict_original_source_gate_passed":False,"untouched_blind_test":False,"deployment_authorized":False,"production_changed":False,
            "protocol_sha256":common.sha(path),"predictions_sha256":common.sha(EXP/"full_candidate_comparison.csv")}
    common.write_new(EXP/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["freeze","compare"])
    parser.add_argument("--started")
    args=parser.parse_args()
    if args.command=="freeze" and not args.started:parser.error("freeze requires actual --started")
    print((freeze(args.started) if args.command=="freeze" else compare())["status"])
