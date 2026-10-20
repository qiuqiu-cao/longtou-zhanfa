"""Independently check complete-pool replay, membership, Top1 and rank AUC."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_generalization_gap as audit
import first_board_matured_event_model as prior

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_REPLAY.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
assert base.sha(audit.__file__)==protocol["source_sha256"]
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
screen=pd.read_csv(prior.SCREEN/"COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv",dtype={"code":str})
unit_checks=0;replayed_rows=0;head_checks=[]
for name,spec in audit.SPECS.items():
    source=pd.read_csv(spec["root"]/spec["data"],dtype={"code":str},float_precision="round_trip")
    source=source[source.stage.eq(1)].copy()
    if "research_selection_allowed" not in source:
        source=source.merge(screen[["signal_date","code","research_selection_allowed","exclusion_reason"]],on=["signal_date","code"],validate="one_to_one")
    directory=spec["root"]/"models";stem=spec["stem"]
    full_members=pd.read_csv(directory/(stem+"_full_members.csv"),dtype={"code":str})
    prefix_members=pd.read_csv(directory/(stem+"_calibration_prefix_members.csv"),dtype={"code":str})
    dates=sorted(full_members.signal_date.unique());cal_dates=dates[-42:]
    bundle=joblib.load(directory/(stem+"_bundle.joblib"))
    prefix=joblib.load(directory/(stem+"_calibration_prefix.joblib"))
    for kind in protocol["windows"]:
        pred=pd.read_csv(OUT/(name+"_"+kind+"_FULL_POOL.csv"),dtype={"code":str},float_precision="round_trip")
        chosen_dates=dates if kind=="TRAIN_FULL_OPTIMISTIC" else (cal_dates if kind.startswith("CAL42") else sorted(source.loc[source.signal_date.between("2025-01-01","2025-06-30"),"signal_date"].unique()))
        expected=source[source.signal_date.isin(chosen_dates)].copy()
        assert set(zip(pred.signal_date,pred.code))==set(zip(expected.signal_date,expected.code))
        compare=pred.merge(expected[["signal_date","code",stage.TARGET,"label_end","research_selection_allowed"]],on=["signal_date","code"],validate="one_to_one",suffixes=("_replay","_source"))
        np.testing.assert_allclose(compare[stage.TARGET+"_replay"],compare[stage.TARGET+"_source"],atol=0,rtol=0,equal_nan=True)
        assert compare.label_end_replay.tolist()==compare.label_end_source.tolist()
        assert compare.research_selection_allowed_replay.tolist()==compare.research_selection_allowed_source.tolist()
        prefix_kind=kind=="CAL42_PREFIX_HELDOUT"
        model=prefix if prefix_kind else bundle["model"]
        members=prefix_members if prefix_kind else full_members
        inputs=expected.set_index(["signal_date","code"]).loc[pd.MultiIndex.from_frame(pred[["signal_date","code"]])]
        raw=model.predict_proba(inputs[bundle["features"]])[:,1]
        np.testing.assert_allclose(raw,pred.model_score,atol=1e-12,rtol=1e-12)
        member_keys=set(zip(members.signal_date,members.code))
        flags=[(d,c) in member_keys for d,c in zip(pred.signal_date,pred.code)]
        assert flags==pred.is_model_training_member.tolist()
        if kind in ("CAL42_PREFIX_HELDOUT","VALIDATION2025H1_FINAL"):
            assert not any(flags)
            assert members.label_end.max()<pred.signal_date.min()
        picked=[]
        for day,g in pred[pred.research_selection_allowed].groupby("signal_date",sort=True):
            order=np.lexsort((g.code.to_numpy(),-g.model_score.to_numpy()))
            picked.append(g.iloc[order[0]])
            unit_checks+=1
        selected=pd.DataFrame(picked)
        m=report["heads"][name][kind]
        assert len(pred)==m["pool_rows"] and pred[stage.TARGET].isna().sum()==m["pool_unknown"]
        assert len(selected)==m["selected"] and selected[stage.TARGET].notna().sum()==m["known"]
        assert selected[stage.TARGET].isna().sum()==m["unknown"]
        assert selected[stage.TARGET].eq(1).sum()==m["hits"]
        assert selected[stage.TARGET].eq(0).sum()==m["false_positives"]
        assert pred[stage.TARGET].eq(1).sum()-m["hits"]==m["false_negatives"]
        assert selected.is_model_training_member.sum()==m["selected_training_members"]
        known=pred[pred[stage.TARGET].notna()]
        ranks=known.model_score.rank(method="average").to_numpy()
        y=known[stage.TARGET].to_numpy();n1=int((y==1).sum());n0=int((y==0).sum())
        auc=(ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
        np.testing.assert_allclose(m["rank_auc"],auc,atol=1e-12,rtol=1e-12)
        if kind=="VALIDATION2025H1_FINAL":
            for month,group in selected.groupby(selected.signal_date.str[:7]):
                mm=report["heads"][name]["monthly_validation"][month]
                assert group[stage.TARGET].eq(1).sum()==mm["hits"]
                assert group[stage.TARGET].notna().sum()==mm["known"]
        replayed_rows+=len(pred)
    optimistic=pd.read_csv(OUT/(name+"_CAL42_FULL_OPTIMISTIC_FULL_POOL.csv"),dtype={"code":str})
    heldout=pd.read_csv(OUT/(name+"_CAL42_PREFIX_HELDOUT_FULL_POOL.csv"),dtype={"code":str})
    assert list(zip(optimistic.signal_date,optimistic.code))==list(zip(heldout.signal_date,heldout.code))
    assert len(heldout)==2428 and heldout[stage.TARGET].isna().sum()==6
    old_cal=pd.read_csv(directory/(stem+"_calibration_rows.csv"),dtype={"code":str})
    assert len(old_cal)==2422
    assert set(zip(old_cal.signal_date,old_cal.code))==set(zip(heldout.loc[heldout[stage.TARGET].notna(),"signal_date"],heldout.loc[heldout[stage.TARGET].notna(),"code"]))
    head_checks.append(dict(head=name,training_members=len(full_members),prefix_members=len(prefix_members),complete_calibration_rows=len(heldout),calibration_unknown=6))
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest
result=dict(status="PASS",verified_at=base.now(),heads=head_checks,head_window_combinations=12,
    independently_checked_top1_date_windows=unit_checks,replayed_rows=replayed_rows,
    same_cal42_dates_and_full_candidate_pools=True,known_only_calibration_pool_bias_prevented=True,
    raw_score_auc_independently_checked=True,all_saved_source_and_model_hashes_unchanged=True,
    fits=0,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(audit.__file__),verifier_sha256=base.sha(__file__))
(OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=True,indent=2))
