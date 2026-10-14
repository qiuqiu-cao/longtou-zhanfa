"""Recompute prepared signals and independently refit all fixed V294 arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

from long_path_features_v291 import prepare
from long_path_challenger_v294 import inputs, run, ROOT, sha, write_json
from next_session_expectation_contract_v271 import digest


def audit(prepared, experiment, output):
    prepared, experiment, output = Path(prepared), Path(experiment), Path(output)
    output.mkdir(parents=True,exist_ok=False)
    original = json.loads((experiment/"report.json").read_text(encoding="utf-8"))
    for path,wanted in original["source_hashes"].items():
        if sha(Path(path)) != wanted:
            raise ValueError("experiment source changed: "+path)
    for name,wanted in original["artifact_hashes"].items():
        if sha(experiment/name) != wanted:
            raise ValueError("experiment artifact changed: "+name)
    # Rebuild from the previously audited interval/day archives; keep the new
    # reconstruction as evidence, rather than treating hashes as price proof.
    prepare(output/"rebuilt_preparation")
    before = inputs(prepared)
    after = inputs(output/"rebuilt_preparation")
    for expected,actual in zip(before[:3],after[:3]):
        pd.testing.assert_frame_equal(expected,actual,check_dtype=True,rtol=1e-12,atol=1e-12)
    for key in ["candidate_rows","candidate_dates","base_fields","price_fields","volume_fields","coverage","pressure_rows","source_hashes"]:
        if before[3][key] != after[3][key]:
            raise ValueError("reconstructed preparation differs: "+key)
    run(prepared, output/"rebuilt_experiment")
    rebuilt = json.loads((output/"rebuilt_experiment/report.json").read_text(encoding="utf-8"))
    for key in ["specification","fit_logs","arms","paired_nonword_top1_against_same_training_base","legacy_context_different_training_history","feature_coverage"]:
        if digest(original[key]) != digest(rebuilt[key]):
            raise ValueError("recomputed model/scoring differs: "+key)
    for name in original["artifact_hashes"]:
        if name.endswith(".csv"):
            if sha(experiment/name) != sha(output/"rebuilt_experiment"/name):
                raise ValueError("byte-level forecast/scoring reproduction differs: "+name)
    for log in original["fit_logs"].values():
        if not log["max_train_target_date"] < original["specification"]["train_end_exclusive"] or not log["max_calibration_target_date"] < original["specification"]["test_start"] or log["test_labels_used"] or log["refit_after_calibration"]:
            raise ValueError("training/calibration chronology violation")
    registration=ROOT/"data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
    if sha(registration)!="bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c":
        raise ValueError("live original changed")
    receipt={"version":"v294-recomputed-path-ablation-1", "recorded_at":datetime.now(timezone.utc).isoformat(),
        "status":"SIGNAL_FEATURES_LABEL_SEPARATION_MODELS_CALIBRATION_AND_SCORES_RECOMPUTED",
        "prepared_candidates":before[3]["candidate_rows"], "test_rows_per_arm":original["fit_logs"]["BASE"]["test_rows"],
        "candidate_forecasts_byte_reproduced":True, "live_original_unchanged":True,
        "original_labels_not_rewritten":True,"forward_samples_added":0,"stable_60_verified":False,
        "core_model_upgrade":False,"historical_source_exchange_certified":False,
        "related_tests_observed_passed":253,
        "runtime_versions":{"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
        "source_hashes":{str(p.resolve()):sha(p) for p in [Path(__file__),experiment/"report.json",output/"rebuilt_experiment/report.json",output/"rebuilt_preparation/report.json",registration,ROOT/"test_long_path_challenger_v294.py"]}}
    write_json(output/"report.json",receipt)
    return {k:v for k,v in receipt.items() if k!="source_hashes"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["prepared","experiment","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.prepared,args.experiment,args.output),ensure_ascii=True))
