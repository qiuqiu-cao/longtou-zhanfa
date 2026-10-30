"""Read-only replay of saved joint models and complete validation outcomes; never fit."""
import json
from datetime import datetime
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_joint_long_high_model as model

out = Path(__file__).resolve().parent
freeze = model.read_freeze()
pending = json.loads((out / "PROTOCOL_PENDING_ACK_SNAPSHOT.json").read_text("utf-8"))
for key in pending:
    if key != "status":
        assert freeze[key] == pending[key], key
assert set(freeze) - set(pending) <= {"agreed_at", "acknowledgment_note"}
rows = model.load_rows()
original = pd.read_csv(model.prior.FACTS, dtype={"code": str}, float_precision="round_trip")
original = original[original.stage.eq(1)].reset_index(drop=True)
pd.testing.assert_frame_equal(rows[original.columns], original)
assert len(rows) == 28503 and rows[model.stage.TARGET].isna().sum() == 126
val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
baseline = model.risk.attach_frozen(rows, model.prior.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
assert len(val) == 4724 and val[model.stage.TARGET].isna().sum() == 10
wide = val[model.KEYS + [model.stage.TARGET, "label_end", "one_price", "research_selection_allowed"] + model.features.FIELDS].copy()
wide = wide.merge(baseline[model.KEYS + ["model_score", "calibrated_score"]].rename(columns={"model_score":"BASE_raw", "calibrated_score":"BASE_calibrated"}), on=model.KEYS, validate="one_to_one", sort=False)
paired = []
verified = {}
for arm, fields in model.ARMS.items():
    directory = out / arm
    report = json.loads((directory / "report.json").read_text("utf-8"))
    before = json.loads((directory / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    assert before["freeze_sha256"] == model.base.sha(out / "JOINT_FROZEN_PROTOCOL.json")
    receipts = list((directory / "models").glob("*_BEFORE_FIT.json"))
    assert len(receipts) == 2
    for receipt_path in receipts:
        actual_start=json.loads(receipt_path.read_text("utf-8"))["started_at"]
        assert datetime.fromisoformat(actual_start)>=datetime.fromisoformat(freeze["agreed_at"])
    assert not list((directory / "models").glob("*REPAIR*"))
    bundle = joblib.load(directory / "models" / ("validation_" + arm + "_bundle.joblib"))
    assert bundle["features"] == fields and bundle["stage"] == 1
    for scope in ("full", "calibration_prefix"):
        name = "validation_" + arm + "_" + scope
        completed = json.loads((directory / "models" / (name + "_COMPLETED.json")).read_text("utf-8"))
        assert model.base.sha(directory / "models" / (name + ".joblib")) == completed["sha256"]
        saved = pd.read_csv(directory / "models" / (name + "_members.csv"), dtype={"code": str})
        reference = pd.read_csv(model.prior.REFERENCE / "models" / ("validation_S1_STAGE_CYCLE_GENE_" + scope + "_members.csv"), dtype={"code": str})
        columns = model.KEYS + ["label_end", model.stage.TARGET]
        assert set(map(tuple, saved[columns].to_numpy())) == set(map(tuple, reference[columns].to_numpy()))
        estimator = joblib.load(directory / "models" / (name + ".joblib"))
        actual = estimator.named_steps["hist"].get_params()
        expected = model.base.make_model("hist_gradient").get_params()
        assert actual == expected
    scored = pd.read_csv(directory / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
    assert scored[model.KEYS].equals(val[model.KEYS].reset_index(drop=True))
    np.testing.assert_allclose(scored[fields + [model.stage.TARGET]], val[fields + [model.stage.TARGET]], atol=0, rtol=0, equal_nan=True)
    replay = model.stage.predict(bundle, val)
    for field in ("model_score", "calibrated_score", "train_prior"):
        np.testing.assert_allclose(scored[field], replay[field], atol=1e-14, rtol=1e-14, equal_nan=True)
    for scope, group, reference in (("primary", scored, baseline),
                ("non_one_price", scored[scored.one_price.eq(0)], baseline[baseline.one_price.eq(0)])):
        metric = model.risk.screen_metric(group)
        assert model.base.clean(metric) == report[scope]
        pick = model.base.selected(group[group.research_selection_allowed], "model_score", 1)
        ref = model.base.selected(reference[reference.research_selection_allowed], "model_score", 1)
        pair = pick[["signal_date", "code", model.stage.TARGET]].rename(columns={"code":"challenger_code",model.stage.TARGET:"challenger_outcome"}).merge(
            ref[["signal_date", "code", model.stage.TARGET]].rename(columns={"code":"baseline_code", model.stage.TARGET:"baseline_outcome"}), on="signal_date", validate="one_to_one")
        assert len(pair) == 117
        pair["arm"] = arm
        pair["scope"] = scope
        paired.append(pair)
    assert report["pass_validation"] == bool(report["primary"]["hits"] > 52 and report["non_one_price"]["hits"] >= 48 and report["primary"]["decision_date_coverage"] >= .95)
    wide = wide.merge(scored[model.KEYS + ["model_score", "calibrated_score"]].rename(columns={"model_score":arm + "_raw", "calibrated_score":arm + "_calibrated"}), on=model.KEYS, validate="one_to_one", sort=False)
    verified[arm] = dict(rows=len(scored), all_saved_scores_replayed=True, full_and_prefix_members_match=True,
                         exact_fixed_estimator_params=True, primary=report["primary"], non_one_price=report["non_one_price"],
                         pass_validation=report["pass_validation"], trainer=before["trainer"], core_fits=2, temporal_calibrator_fits=1)
wide.to_csv(out / "COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv", index=False, encoding="utf-8-sig")
pairs = pd.concat(paired, ignore_index=True)
pairs.to_csv(out / "COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv", index=False, encoding="utf-8-sig")
drift = []
for (arm, scope), group in pairs.groupby(["arm", "scope"]):
    drift.append(dict(arm=arm, scope=scope, dates=len(group), same_code=int(group.challenger_code.eq(group.baseline_code).sum()),
        both_hit=int((group.challenger_outcome.eq(1) & group.baseline_outcome.eq(1)).sum()),
        gained_hits=int((group.challenger_outcome.eq(1) & group.baseline_outcome.eq(0)).sum()),
        lost_hits=int((group.challenger_outcome.eq(0) & group.baseline_outcome.eq(1)).sum()),
        both_failed=int((group.challenger_outcome.eq(0) & group.baseline_outcome.eq(0)).sum()),
        unknown_outcomes=int((group.challenger_outcome.isna() | group.baseline_outcome.isna()).sum())))
for path in model.base.INPUTS.values():
    assert model.base.sha(path) == freeze["bindings"][str(path.resolve())]
decision=json.loads((out / "JOINT_SELECTION_BEFORE_EVALUATION.json").read_text("utf-8"))
passing=[arm for arm in model.ARMS if verified[arm]["pass_validation"]]
expected_winner=sorted(passing,key=lambda arm:(-verified[arm]["primary"]["hits"],len(model.ARMS[arm]),arm))[0] if passing else None
assert decision["selected"]==expected_winner
monthly_replay=None
if expected_winner is not None:
    final_report=json.loads((out / "report.json").read_text("utf-8"))
    assert final_report["monthly_fits"]==14
    evaluation=out / "evaluation"
    all_scored=pd.read_csv(evaluation / "ALL14_MONTHS_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    expected=rows[rows.signal_date.ge("2025-07-01")]
    assert all_scored[model.KEYS].equals(expected[model.KEYS].reset_index(drop=True))
    np.testing.assert_allclose(all_scored[model.ARMS[expected_winner]+[model.stage.TARGET]],expected[model.ARMS[expected_winner]+[model.stage.TARGET]],rtol=0,atol=0,equal_nan=True)
    for month,group in all_scored.groupby("model_month",sort=True):
        bundle=joblib.load(evaluation / "models" / (month+"_"+expected_winner+"_bundle.joblib"))
        assert bundle["train_max_label_end"]<month+"-01"
        replay=model.stage.predict(bundle,group)
        np.testing.assert_allclose(group[["model_score","calibrated_score","train_prior"]],replay[["model_score","calibrated_score","train_prior"]],rtol=1e-14,atol=1e-14,equal_nan=True)
        saved=pd.read_csv(evaluation / "models" / (month+"_"+expected_winner+"_full_members.csv"),dtype={"code":str})
        reference=pd.read_csv(model.prior.REFERENCE / "models" / (month+"_S1_STAGE_CYCLE_GENE_full_members.csv"),dtype={"code":str})
        cols=model.KEYS+["label_end",model.stage.TARGET]
        assert set(map(tuple,saved[cols].to_numpy()))==set(map(tuple,reference[cols].to_numpy()))
        for scope in ("full","calibration_prefix"):
            name=month+"_"+expected_winner+"_"+scope
            receipt=json.loads((evaluation / "models" / (name+"_COMPLETED.json")).read_text("utf-8"))
            assert model.base.sha(evaluation / "models" / (name+".joblib"))==receipt["sha256"]
            estimator=joblib.load(evaluation / "models" / (name+".joblib"))
            assert estimator.named_steps["hist"].get_params()==model.base.make_model("hist_gradient").get_params()
            left=pd.read_csv(evaluation / "models" / (name+"_members.csv"),dtype={"code":str})
            right=pd.read_csv(model.prior.REFERENCE / "models" / (month+"_S1_STAGE_CYCLE_GENE_"+scope+"_members.csv"),dtype={"code":str})
            assert set(map(tuple,left[cols].to_numpy()))==set(map(tuple,right[cols].to_numpy()))
    frozen_evaluation=model.risk.attach_frozen(rows,model.prior.REFERENCE / "S1_ALL_PREDICTIONS.csv")
    metrics=model.history.period_metrics(all_scored,frozen_evaluation)
    assert model.base.clean(metrics)==final_report["evaluation"]
    actual=metrics["all"]["new_screened"]
    achieved=bool(actual["hit_rate"]>=.60 and actual["known"]>=200 and actual["outcome_coverage"]>=.95 and actual["abstention_dates"]==0 and all(metrics[period]["new_screened"]["hit_rate"]>=.55 for period in ("2025H2","2026")))
    assert final_report["goal_60_achieved"]==achieved
    saved_second=pd.read_csv(model.prior.REFERENCE / "S2_ALL_PREDICTIONS.csv",dtype={"code":str},float_precision="round_trip")
    chain,chain_metric=model.stage.chain_recheck(all_scored[all_scored.research_selection_allowed],saved_second)
    assert model.base.clean(chain_metric)==final_report["same_stock_second_recheck"]
    saved_chain=pd.read_csv(evaluation / "SAME_STOCK_SECOND_RECHECK.csv",dtype={"code":str},float_precision="round_trip")
    pd.testing.assert_frame_equal(chain.reset_index(drop=True),saved_chain,check_dtype=False,check_exact=False,atol=1e-14,rtol=1e-14)
    monthly_pairs=[]
    for scope,left,right in (("primary",all_scored,frozen_evaluation),("non_one_price",all_scored[all_scored.one_price.eq(0)],frozen_evaluation[frozen_evaluation.one_price.eq(0)])):
        take=model.base.selected(left[left.research_selection_allowed],"model_score",1)
        ref=model.base.selected(right[right.research_selection_allowed],"model_score",1)
        pair=take[["signal_date","code",model.stage.TARGET]].rename(columns={"code":"challenger_code",model.stage.TARGET:"challenger_outcome"}).merge(ref[["signal_date","code",model.stage.TARGET]].rename(columns={"code":"baseline_code",model.stage.TARGET:"baseline_outcome"}),on="signal_date",validate="one_to_one")
        assert len(pair)==286
        pair["scope"]=scope
        monthly_pairs.append(pair)
    pd.concat(monthly_pairs,ignore_index=True).to_csv(out / "EVALUATION286_DAILY_PAIRED_SELECTIONS.csv",index=False,encoding="utf-8-sig")
    evaluation_wide=all_scored[model.KEYS+[model.stage.TARGET,"label_end","one_price","research_selection_allowed","model_month","model_score","calibrated_score"]+model.features.FIELDS].copy()
    evaluation_wide=evaluation_wide.rename(columns={"model_score":"NEW_raw","calibrated_score":"NEW_calibrated"}).merge(frozen_evaluation[model.KEYS+["model_score","calibrated_score"]].rename(columns={"model_score":"BASE_raw","calibrated_score":"BASE_calibrated"}),on=model.KEYS,validate="one_to_one",sort=False)
    evaluation_wide.to_csv(out / "EVALUATION_ALL13149_SELECTED_ARM_WITH_FROZEN_BASE.csv",index=False,encoding="utf-8-sig")
    monthly_replay=dict(months=14,rows=len(all_scored),all_scores_and_members_replayed=True,all28_core_params_and_hashes=True,metrics=metrics,second_recheck=chain_metric,goal_60_achieved=achieved)
summary = dict(at=model.base.now(), status="PASS_SAVED_VALIDATION_REPLAY", model_fits=0,
    full28503_original_fields_labels_screen_unchanged=True, all_original_daily_calendar_master_hashes=True,
    unchanged_pending_protocol_except_peer_ack=True, full4724_scores_per_arm=True,
    unknown126_and_validation10_retained=True, arms=verified, drift=drift,
    no_actor_or_causal_interpretation=True, production_changes=False,monthly_replay=monthly_replay)
with (out / "JOINT_SAVED_RESULT_VERIFICATION.json").open("x", encoding="utf-8") as file:
    json.dump(model.base.clean(summary), file, ensure_ascii=False, indent=2)
print(json.dumps(dict(status=summary["status"],model_fits=0,validation_arms={arm:dict(primary=info["primary"]["hits"],non_one_price=info["non_one_price"]["hits"]) for arm,info in verified.items()},monthly_replay=None if monthly_replay is None else dict(months=14,rows=monthly_replay["rows"],known=monthly_replay["metrics"]["all"]["new_screened"]["known"],hits=monthly_replay["metrics"]["all"]["new_screened"]["hits"],baseline_hits=monthly_replay["metrics"]["all"]["frozen_screened"]["hits"],second_recheck=monthly_replay["second_recheck"],goal_60_achieved=monthly_replay["goal_60_achieved"])),ensure_ascii=True))
