"""Independent prefix score/group/outcome audit. No new fitting."""
from pathlib import Path
import itertools
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_prefix_diversity as audit

OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/"protocol_BEFORE_REPLAY.json").read_text("utf-8"))
report=json.loads((OUT/"report.json").read_text("utf-8"))
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
assert base.sha(audit.__file__)==protocol["source_sha256"]
rows=pd.read_csv(OUT/"FULL2428_PREFIX_SCORES_AND_ONLY_PRE2025_LABELS.csv",dtype={"code":str},float_precision="round_trip")
saved_top=pd.read_csv(OUT/"ALL42_THREE_PREFIX_TOP1S_LABEL_ONLY.csv",dtype={"code_"+head:str for head in audit.HEADS})
assert len(rows)==2428 and rows.signal_date.nunique()==42
assert rows.signal_date.min()=="2024-11-01" and rows.signal_date.max()=="2024-12-30"
labels=pd.read_csv(audit.REF/"models/validation_S1_STAGE_CYCLE_GENE_calibration_rows.csv",usecols=["signal_date","code",stage.TARGET],dtype={"code":str},float_precision="round_trip")
joined=rows.merge(labels,on=["signal_date","code"],how="left",validate="one_to_one",suffixes=("_scored","_source"))
np.testing.assert_allclose(joined[stage.TARGET+"_scored"],joined[stage.TARGET+"_source"],atol=0,rtol=0,equal_nan=True)
assert rows[stage.TARGET].isna().sum()==6 and rows[stage.TARGET].eq(1).sum()==530
assert rows.research_selection_allowed.sum()==1800
chosen={};members_reference=None
for head,(root,stem) in audit.HEADS.items():
    model=joblib.load(root/"models"/(stem+"_calibration_prefix.joblib"))
    receipt=json.loads((root/"models"/(stem+"_calibration_prefix_COMPLETED.json")).read_text("utf-8"))
    assert base.sha(root/"models"/(stem+"_calibration_prefix.joblib"))==receipt["sha256"]
    fields=receipt.get("observed_fields",receipt.get("features"))
    np.testing.assert_allclose(model.predict_proba(rows[fields])[:,1],rows["score_"+head],atol=1e-12,rtol=1e-12)
    members=pd.read_csv(root/"models"/(stem+"_calibration_prefix_members.csv"),usecols=["signal_date","code","stage","label_end",stage.TARGET],dtype={"code":str})
    members=members.sort_values(["signal_date","code"]).reset_index(drop=True)
    assert len(members)==8075 and members.label_end.max()<rows.signal_date.min()
    if members_reference is None:members_reference=members
    else:pd.testing.assert_frame_equal(members_reference,members)
    picked=[]
    for day,g in rows[rows.research_selection_allowed].groupby("signal_date",sort=True):
        order=np.lexsort((g.code.to_numpy(),-g["score_"+head].to_numpy()))
        picked.append(g.iloc[order[0]])
    selected=pd.DataFrame(picked).set_index("signal_date").sort_index()
    old=saved_top.set_index("signal_date").sort_index()
    assert selected.code.tolist()==old["code_"+head].tolist()
    np.testing.assert_allclose(selected[stage.TARGET],old["outcome_"+head],atol=0,rtol=0,equal_nan=True)
    m=report["heads"][head]
    assert len(selected)==m["selected"]==42 and selected[stage.TARGET].notna().sum()==m["known"]==42
    assert selected[stage.TARGET].eq(1).sum()==m["hits"]
    assert selected[stage.TARGET].eq(0).sum()==m["false_positives"]
    assert 530-m["hits"]==m["false_negatives"]
    chosen[head]=selected
for a,b in itertools.combinations(chosen,2):
    left,right=chosen[a],chosen[b];m=report["pairwise"][a+"__"+b]
    assert left.code.eq(right.code).sum()==m["same_code_days"]
    l,r=left[stage.TARGET],right[stage.TARGET]
    assert (l.eq(1)&r.eq(1)).sum()==m["both_known_hit"]
    assert (l.eq(0)&r.eq(0)).sum()==m["both_known_failure"]
    assert (l.eq(1)&r.eq(0)).sum()==m["first_only_known_hit"]
    assert (l.eq(0)&r.eq(1)).sum()==m["second_only_known_hit"]
    assert (l.isna()|r.isna()).sum()==m["at_least_one_unknown"]
oracle=0
for day in chosen["BASE"].index:
    codes={selected.loc[day,"code"] for selected in chosen.values()}
    outcomes=[selected.loc[day,stage.TARGET] for selected in chosen.values()]
    if 1 in outcomes:oracle+=1
    saved=saved_top[saved_top.signal_date.eq(day)].iloc[0]
    assert saved.unique_top1_codes==len(codes)
    assert saved.known_hits_across_heads==sum(x==1 for x in outcomes)
    assert saved.known_failures_across_heads==sum(x==0 for x in outcomes)
    assert saved.unknown_head_outcomes==sum(pd.isna(x) for x in outcomes)
assert oracle==report["union_oracle_any_known_winner_days_LABEL_ONLY"]==28
assert report["combined_ranker_scored"] is False and report["model_fits"]==0
for path,digest in protocol["inputs"].items():assert base.sha(path)==digest,path
result=dict(status="PASS",verified_at=base.now(),prefix_heads=3,matched_training_members=8075,full_candidate_pool=2428,
    known_pool=2422,unknown_pool=6,complete_dates=42,all_model_scores_replayed=True,all_pairwise_contingencies_recomputed=True,
    union_oracle_only=28,union_oracle_not_primary_success=True,combined_ranker_scored=False,models_fitted=0,
    post2024_outcome_columns_used=False,new_forward_results=0,actual_trades=0,goal_60_passed=False,
    source_sha256=base.sha(audit.__file__),verifier_sha256=base.sha(__file__))
base.write_json(OUT/"verification.json",result)
print(json.dumps(result,ensure_ascii=True,indent=2))
