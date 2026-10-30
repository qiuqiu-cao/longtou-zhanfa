"""Qualify the new emotion trajectories without fitting or consuming outcome fields."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_ladder_emotion_features as feature
OUT=Path(__file__).resolve().parent
FACTS=base.ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
if (OUT/"LADDER_EMOTION_QUALIFICATION.json").exists():raise FileExistsError("No repeated completed feature qualification")
before=base.sha(FACTS)
original=stage.VARIANTS["STAGE_CYCLE_GENE"]
candidate=pd.read_csv(FACTS,usecols=feature.KEYS+["stage"]+original,dtype={"code":str},float_precision="round_trip")
candidate=candidate[candidate.stage.eq(1)].reset_index(drop=True)
assert len(candidate)==28503
base.write_json(OUT/"FEATURE_QUALIFICATION_SCOPE_BEFORE_COMPUTE.json",dict(at=base.now(),model_fits=0,
    labels_read=False,source_sha256=base.sha(feature.__file__),original_facts_sha256=before,
    fields=feature.FIELDS,dropped_from_candidate_inputs=[feature.DIAGNOSTIC_HIGH_SURVIVAL],
    reason="Previous4plus survival overlaps existing cycle_high_survival;retain as diagnostic with explicit unknown handling",
    feedback="Full previous official-day cohort;rate is NaN forzero members orany unknown current outcome;do not turn missing into failures",
    history_path="Current feedback minus arithmetic mean of preceding3official dates,all3 required;no future stage label",
    source_bindings={str(path):base.sha(path) for path in base.INPUTS.values()}))
states,sessions=feature.load_states()
context=feature.market_context(states,sessions)
new=feature.attach(candidate[feature.KEYS],context)
assert len(new)==28503 and new.signal_date.nunique()==668
new.to_csv(OUT/"COMPLETE28503_LADDER_EMOTION_FIELDS_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
context.to_csv(OUT/"COMPLETE_DAILY_LADDER_AND_EMOTION_CONTEXT_NO_TARGETS.csv",index=False,encoding="utf-8-sig")
periods={}
for name,mask in [("development",candidate.signal_date.lt("2025-01-01")),
    ("validation2025H1",candidate.signal_date.between("2025-01-01","2025-06-30")),
    ("evaluation14months",candidate.signal_date.between("2025-07-01","2026-08-31"))]:
    periods[name]=dict(rows=int(mask.sum()),dates=candidate[mask].signal_date.nunique(),coverage=new.loc[mask,feature.FIELDS].notna().mean().to_dict())
duplicate_fields={}
for field in feature.FIELDS:
    matches=[]
    for old in original:
        a=new[field].to_numpy(float);b=candidate[old].to_numpy(float)
        if np.array_equal(np.isnan(a),np.isnan(b)) and np.allclose(a,b,rtol=0,atol=0,equal_nan=True):matches.append(old)
    duplicate_fields[field]=matches
prefix=[]
for cutoff in ["2025-06-30","2025-12-31"]:
    restricted,_=feature.load_states(cutoff=cutoff)
    selected=candidate[candidate.signal_date.le(cutoff)][feature.KEYS].reset_index(drop=True)
    check=feature.attach(selected,feature.market_context(restricted,sessions))
    compare=new[new.signal_date.le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(check,compare,check_exact=True)
    prefix.append(dict(cutoff=cutoff,rows=len(check),all_fields_exact=True))
assert base.sha(FACTS)==before
base.write_json(OUT/"LADDER_EMOTION_QUALIFICATION.json",dict(at=base.now(),model_fits=0,labels_read=False,
    full_rows=28503,full_dates=668,daily_context_rows=len(context),fields=feature.FIELDS,
    original_targets_prices_or_features_changed=False,periods=periods,exact_original_column_duplicates=duplicate_fields,
    prefix_checks=prefix,diagnostic_high_survival_not_a_new_model_input=True,
    source_sha256=base.sha(feature.__file__),nominal_price_and_surviving_universe_limits_preserved=True,
    actual_emotion_or_fund_identity_gold_labels=False,
    result="FEATURES_AVAILABLE_FOR_ACTUAL_PEER_DEBATE_NOT_TRAINING_APPROVAL"))
print("FULL28503_LADDER_EMOTION_TARGET_FREE_QUALIFIED_WITH_COMPLETE_PREFIX_CHECKS_NO_FITS")
