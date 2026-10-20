"""Independently recompute the two fixed ensemble rules and all Top1 picks."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_equal_ensemble as experiment

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_COMPARISON.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in {**protocol["inputs"],**protocol["protected"],**protocol["raw_source_hashes"]}.items():assert base.sha(path)==digest,path
assert base.sha(experiment.__file__)==protocol["source_sha256"]
rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FACTS_LABELS_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
original=pd.read_csv(experiment.REF/"all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",dtype={"code":str},float_precision="round_trip")
original=original[original.stage.eq(1)].reset_index(drop=True)
assert list(zip(rows.signal_date,rows.code))==list(zip(original.signal_date,original.code))
np.testing.assert_allclose(rows[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],original[stage.VARIANTS["STAGE_CYCLE_GENE"]+[stage.TARGET]],atol=0,rtol=0,equal_nan=True)
merged=pd.read_csv(OUT/"VALIDATION_THREE_FROZEN_COMPONENT_SCORES.csv",dtype={"code":str},float_precision="round_trip")
assert len(merged)==4724 and merged.signal_date.nunique()==117 and merged[stage.TARGET].isna().sum()==10
for head,file in experiment.VALIDATION.items():
    component=pd.read_csv(file,dtype={"code":str},float_precision="round_trip").set_index(["signal_date","code"])
    aligned=component.loc[pd.MultiIndex.from_frame(merged[["signal_date","code"]])]
    np.testing.assert_allclose(merged["component_"+head+"_raw"],aligned.model_score,atol=0,rtol=0)
    np.testing.assert_allclose(merged["component_"+head+"_calibrated"],aligned.calibrated_score,atol=0,rtol=0)
    np.testing.assert_allclose(merged[stage.TARGET],aligned[stage.TARGET],atol=0,rtol=0,equal_nan=True)
    np.testing.assert_allclose(merged.train_prior,aligned.train_prior,atol=1e-12,rtol=1e-12)

def picks(frame,non_one=False):
    allowed=frame.research_selection_allowed&(frame.one_price.eq(0) if non_one else True)
    indices=[]
    for day,g in frame[allowed].groupby("signal_date",sort=True):
        indices.append(g.index[np.lexsort((g.code.to_numpy(),-g.model_score.to_numpy()))[0]])
    return frame.loc[indices]

reference=risk.attach_frozen(rows,experiment.VALIDATION["BASE"])
old=picks(reference)
paired=old[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_BASE" for c in ("code","name",stage.TARGET)})
checks=[]
for arm in experiment.ARMS:
    saved=pd.read_csv(OUT/("validation_"+arm+"_ALL_BEFORE_METRICS.csv"),dtype={"code":str},float_precision="round_trip")
    assert list(zip(saved.signal_date,saved.code))==list(zip(merged.signal_date,merged.code))
    pd.testing.assert_frame_equal(saved[merged.columns],merged)
    if arm==experiment.ARMS[0]:
        expected=sum(merged["component_"+head+"_calibrated"].to_numpy() for head in experiment.HEADS)/3
        np.testing.assert_allclose(saved.calibrated_score,expected,atol=1e-12,rtol=1e-12)
    else:
        expected=np.full(len(merged),np.nan)
        for day,g in merged.groupby("signal_date",sort=True):
            total=np.zeros(len(g))
            for head in experiment.HEADS:
                total+=rankdata(g["component_"+head+"_raw"].to_numpy(),method="average")/len(g)
            expected[g.index]=total/3
        assert saved.calibrated_score.isna().all()
    np.testing.assert_allclose(saved.model_score,expected,atol=1e-12,rtol=1e-12)
    chosen=picks(saved);non=picks(saved,True)
    m=report["decision"]["arms"][arm]["primary"]
    assert len(chosen)==m["selected"]==117 and chosen[stage.TARGET].notna().all()
    assert chosen[stage.TARGET].eq(1).sum()==m["hits"]
    assert chosen[stage.TARGET].eq(0).sum()==m["false_positives"]
    assert saved[stage.TARGET].eq(1).sum()==m["full_positive_pool"]==863
    assert 863-m["hits"]==m["false_negatives"]
    assert non[stage.TARGET].eq(1).sum()==report["decision"]["arms"][arm]["non_one_price"]["hits"]
    assert report["decision"]["arms"][arm]["pass_validation"] is False
    pair=chosen[["signal_date","code","name",stage.TARGET]].rename(columns={c:c+"_"+arm for c in ("code","name",stage.TARGET)})
    paired=paired.merge(pair,on="signal_date",validate="one_to_one")
    a=paired[stage.TARGET+"_BASE"];b=paired[stage.TARGET+"_"+arm]
    checks.append(dict(arm=arm,hits=int(b.eq(1).sum()),selected=117,unknown_selected=0,
        gain=int((a.eq(0)&b.eq(1)).sum()),loss=int((a.eq(1)&b.eq(0)).sum()),changed_picks=int(paired.code_BASE.ne(paired["code_"+arm]).sum())))
paired.to_csv(OUT/"COMPLETE117_BASE_AND_BOTH_ENSEMBLES_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
assert report["decision"]["selected"] is None and report["new_component_core_fits"]==0
assert not (OUT/"components").exists() and not (OUT/"models").exists()
assert report["weights_fitted"]==report["validation_component_refits"]==0
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
result=dict(status="PASS",verified_at=base.now(),complete_validation_rows=4724,validation_unknown_pool=10,
    all_frozen_component_scores_labels_and_priors_exact=True,equal_probability_mean_independently_recomputed=True,
    equal_daily_rank_mean_independent_scipy_recomputed=True,all117daily_selections_and_metrics_checked=True,
    arms=checks,original_source_rows=len(rows),new_component_fits=0,new_weight_fits=0,monthly_challenge_fits=0,
    new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(experiment.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
