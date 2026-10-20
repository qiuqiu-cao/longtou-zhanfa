"""Independent membership/group/score audit. Never calls fit."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_feasible_rank as experiment

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"]}.items():
    assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(experiment.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[experiment.FIELDS+[stage.TARGET]],original[experiment.FIELDS+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
assert rows.label_end.fillna("UNKNOWN").tolist()==original.label_end.fillna("UNKNOWN").tolist()
screen=pd.read_csv(experiment.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str})
assert rows.research_selection_allowed.tolist()==screen.research_selection_allowed.tolist()
assert rows.exclusion_reason.fillna("").tolist()==screen.exclusion_reason.fillna("").tolist()
base_validation=risk.attach_frozen(rows,experiment.REF/"validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")

def direct_pick(frame):
    admissible=frame[frame.research_selection_allowed]
    indices=[]
    for day,group in admissible.groupby("signal_date",sort=True):
        order=np.lexsort((group.code.to_numpy(),-group.model_score.to_numpy()))
        indices.append(group.index[order[0]])
    return admissible.loc[indices]

frozen=direct_pick(base_validation)
assert frozen[stage.TARGET].eq(1).sum()==52
checks=[]
paired=frozen[["signal_date","code","name",stage.TARGET]].copy()
paired=paired.rename(columns={c:c+"_BASE" for c in ("code","name",stage.TARGET)})
for arm in experiment.ARMS:
    name="validation_"+arm
    bundle=joblib.load(OUT/"models"/(name+"_bundle.joblib"))
    pred=pd.read_csv(OUT/(name+"_ALL_BEFORE_METRICS.csv"),dtype={"code":str},float_precision="round_trip")
    assert set(zip(pred.signal_date,pred.code))==set(zip(base_validation.signal_date,base_validation.code))
    assert len(pred)==4724 and pred[stage.TARGET].isna().sum()==10
    x=rows.merge(pred[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    if arm==experiment.ARMS[0]:
        raw=bundle["model"].predict_proba(x[experiment.FIELDS])[:,1]
        calibration_input=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))
    else:
        assert bundle["dependency_version"]=="4.7.0"
        raw=bundle["model"].predict(x[bundle["observed_fields"]])
        calibration_input=raw
        model_params=bundle["model"].get_params()
        for k,v in experiment.RANK_PARAMS.items():assert model_params[k]==v,(k,model_params[k],v)
    np.testing.assert_allclose(raw,pred.model_score,atol=1e-12,rtol=1e-12)
    if bundle["calibrator"] is not None:
        probability=bundle["calibrator"].predict_proba(calibration_input.reshape(-1,1))[:,1]
        np.testing.assert_allclose(probability,pred.calibrated_score,atol=1e-12,rtol=1e-12)
    else:assert pred.calibrated_score.isna().all()
    cal_info=json.loads((OUT/"models"/(name+"_calibration_receipt.json")).read_text("utf-8"))
    cutoff=cal_info["cutoff"]
    assert cal_info["inner_max_label_end"]<cutoff
    mature=rows[rows[stage.TARGET].notna()&rows.signal_date.lt("2025-01-01")&rows.label_end.lt("2025-01-01")&rows.research_selection_allowed]
    dates=sorted(mature.signal_date.unique())
    assert cutoff==dates[-42]
    cal=pd.read_csv(OUT/"models"/(name+"_calibration_rows.csv"),dtype={"code":str},float_precision="round_trip")
    expected_cal=mature[mature.signal_date.ge(cutoff)]
    assert set(zip(cal.signal_date,cal.code))==set(zip(expected_cal.signal_date,expected_cal.code))
    prefix=joblib.load(OUT/"models"/(name+"_calibration_prefix.joblib"))
    cx=rows.merge(cal[["signal_date","code"]],on=["signal_date","code"],validate="one_to_one")
    if arm==experiment.ARMS[0]:
        values=prefix.predict_proba(cx[experiment.FIELDS])[:,1]
        np.testing.assert_allclose(values,cal.uncalibrated_oof,atol=1e-12,rtol=1e-12)
    else:
        info=json.loads((OUT/"models"/(name+"_calibration_prefix_COMPLETED.json")).read_text("utf-8"))
        values=prefix.predict(cx[info["observed_fields"]])
        np.testing.assert_allclose(values,cal.raw_rank_oof,atol=1e-12,rtol=1e-12)
    for suffix in ("_calibration_prefix","_full"):
        core_name=name+suffix
        receipt=json.loads((OUT/"models"/(core_name+"_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(OUT/"models"/(core_name+".joblib"))==receipt["sha256"]
        members=pd.read_csv(OUT/"models"/(core_name+"_members.csv"),dtype={"code":str},float_precision="round_trip")
        expected=mature[mature.label_end.lt(cutoff)] if suffix=="_calibration_prefix" else mature
        assert set(zip(members.signal_date,members.code))==set(zip(expected.signal_date,expected.code))
        assert not members.duplicated(["signal_date","code"]).any()
        assert members.label_end.max()<"2025-01-01"
        if suffix=="_calibration_prefix":assert members.label_end.max()<cutoff
        compare=members.merge(original[["signal_date","code",stage.TARGET]],on=["signal_date","code"],validate="one_to_one",suffixes=("_fit","_original"))
        np.testing.assert_allclose(compare[stage.TARGET+"_fit"],compare[stage.TARGET+"_original"],atol=0,rtol=0)
        if arm==experiment.ARMS[1]:
            ordered=members.sort_values(["signal_date","code"])
            assert list(zip(ordered.signal_date,ordered.code))==list(zip(members.signal_date,members.code))
            sizes=members.groupby("signal_date",sort=False).size()
            expected_weights=np.array([len(members)/(len(sizes)*sizes[d]) for d in members.signal_date])
            np.testing.assert_allclose(members.sample_weight,expected_weights,atol=1e-12,rtol=1e-12)
            ledger=pd.read_csv(OUT/"models"/(core_name+"_queries.csv"),float_precision="round_trip").set_index("signal_date")
            assert int(ledger.query_size.sum())==len(members) and ledger.query_size.to_dict()==sizes.to_dict()
            for day,group in members.groupby("signal_date",sort=False):
                assert ledger.loc[day,"positive_labels"]==group[stage.TARGET].sum()
                np.testing.assert_allclose(ledger.loc[day,"weight_sum"],group.sample_weight.sum(),atol=1e-11,rtol=1e-11)
            assert receipt["queries_all_negative"]==int(ledger.positive_labels.eq(0).sum())
    picked=direct_pick(pred)
    non=direct_pick(pred[pred.one_price.eq(0)])
    m=report["decision"]["arms"][arm]["primary"]
    assert len(picked)==m["selected"]==117
    assert picked[stage.TARGET].notna().sum()==m["known"]
    assert picked[stage.TARGET].isna().sum()==m["unknown"]
    assert picked[stage.TARGET].eq(1).sum()==m["hits"]
    assert picked[stage.TARGET].eq(0).sum()==m["false_positives"]
    assert pred[stage.TARGET].eq(1).sum()-m["hits"]==m["false_negatives"]
    assert int(non[stage.TARGET].eq(1).sum())==report["decision"]["arms"][arm]["non_one_price"]["hits"]
    changed=picked[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_"+arm for c in ("code","name",stage.TARGET)})
    paired=paired.merge(changed,on="signal_date",validate="one_to_one")
    picked[picked[stage.TARGET].isna()].to_csv(OUT/(arm+"_SELECTED_UNKNOWN_PRESERVED.csv"),index=False,encoding="utf-8-sig")
    old_y=paired[stage.TARGET+"_BASE"];new_y=paired[stage.TARGET+"_"+arm]
    checks.append(dict(arm=arm,complete_rows=len(pred),training_rows=len(mature),calibration_rows=len(cal),
        top1_hits=int(new_y.eq(1).sum()),known=int(new_y.notna().sum()),unknown=int(new_y.isna().sum()),
        gain_known=int((old_y.eq(0)&new_y.eq(1)).sum()),loss_known=int((old_y.eq(1)&new_y.eq(0)).sum()),
        baseline_hit_to_unknown=int((old_y.eq(1)&new_y.isna()).sum())))
paired.to_csv(OUT/"COMPLETE117_BASE_AND_BOTH_ARMS_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
assert report["decision"]["selected"] is None and report["goal_60_passed"] is False
assert report["core_fit_count"]==len(list((OUT/"models").glob("*_COMPLETED.json")))==4
assert not list((OUT/"models").glob("2025-*.joblib"))
repair=json.loads((OUT/"PREFIT_DEPENDENCY_PATH_FAILURE.json").read_text("utf-8"))
assert repair["fit_started"] is False and repair["completed_core_fits"]==0
result=dict(status="PASS",verified_at=base.now(),unchanged_full_rows=len(rows),unchanged62_fields_labels_screen=True,
    replayed_core_models=4,calibration_heads=2,daily_rank_query_members_weights_and_labels_independently_verified=True,
    validation=checks,pre_fit_path_failure_preserved=True,monthly_challenge_fits=0,new_fits_during_verification=0,
    actual_trades=0,new_forward_results=0,goal_60_passed=False,source_sha256=base.sha(experiment.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
