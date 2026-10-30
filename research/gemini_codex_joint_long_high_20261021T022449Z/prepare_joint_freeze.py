"""Bind the independently discussed nominal-price hypothesis before any model fit."""
from pathlib import Path
import sys
import json
import shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_joint_long_high_model as m
out=m.OUTPUT
proposal=out/'gemini/GEMINI_LONG_HIGH_PROPOSAL.md'
assert proposal.exists() and not any((out/arm).exists() for arm in m.ARMS)
qualification=json.loads((out/'qualification.json').read_text('utf-8'))
assert all(qualification['qualified_fields'].values())
assert qualification['periods']['inner_prefix']['fields']['reference_chain_log_distance_high250_DIAGNOSTIC_ONLY']['known']==0
adoption=json.loads((out/'PRICE_BASIS_ADOPTION_BEFORE_FITS.json').read_text('utf-8'))
assert adoption['all28503_feature_values_unchanged'] and adoption['reference_values_different_from_prior_close']==0
rows=m.load_rows()
original=pd.read_csv(m.prior.FACTS,dtype={'code':str},float_precision='round_trip')
original=original[original.stage.eq(1)].reset_index(drop=True)
assert len(rows)==28503 and rows.signal_date.nunique()==668 and rows[m.stage.TARGET].isna().sum()==126
pd.testing.assert_frame_equal(rows[original.columns],original)
train=m.members.train_members(rows,'2025-01-01')
previous=pd.read_csv(m.prior.REFERENCE/'models/validation_S1_STAGE_CYCLE_GENE_full_members.csv',dtype={'code':str})
columns=m.KEYS+['label_end',m.stage.TARGET]
assert set(map(tuple,train[columns].to_numpy()))==set(map(tuple,previous[columns].to_numpy()))
dates=sorted(train.signal_date.unique());cutoff=dates[-42]
inner=train[train.label_end.lt(cutoff)];cal=train[train.signal_date.ge(cutoff)]
for field in m.features.FIELDS:
    assert inner[field].notna().sum()>=1000 and inner.loc[inner[field].notna(),'signal_date'].nunique()>=60
    assert cal[field].notna().sum()>=300 and cal.loc[cal[field].notna(),'signal_date'].nunique()>=30
val=rows[rows.signal_date.between('2025-01-01','2025-06-30')]
assert len(val)==4724 and val.signal_date.nunique()==117 and val[m.stage.TARGET].isna().sum()==10
frozen=m.risk.attach_frozen(rows,m.prior.REFERENCE/'validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv')
assert m.risk.screen_metric(frozen)['hits']==52 and m.risk.screen_metric(frozen[frozen.one_price.eq(0)])['hits']==48
base_freeze=json.loads((m.prior.JOINT/'JOINT_FROZEN_PROTOCOL.json').read_text('utf-8'))
bindings=base_freeze['bindings'].copy()
for path in [Path(m.__file__),Path(m.features.__file__),Path(__file__),m.NEW_FIELDS_FILE,
             out/'qualification.json',out/'qualify_long_high.py',out/'PROTOCOL_BEFORE_FEATURE_AUDIT.json',
             out/'PRICE_BASIS_ADOPTION_BEFORE_FITS.json',out/'COMPLETE28503_LONG_HIGH_FEATURES_NO_TARGETS.csv',
             out/'FEATURE_SOURCE_BEFORE_PROXY_LABEL_CORRECTION.py',proposal,
             ROOT/'tests/test_first_board_long_high_features.py',ROOT/'leader_system_v3.py']:
    bindings[str(path.resolve())]=m.base.sha(path)
for path,digest in bindings.items():assert m.base.sha(path)==digest,path
preflight=dict(at=m.base.now(),fits=0,rows=28503,unknown126_preserved=True,validation4724_unknown10=True,
    exact_original_fields_labels_screen=True,exact_train_members=True,baseline_hits=52,baseline_non_one_hits=48,
    source_tests='5passed:prioronly,full120latesttie,reference-splitfixture,prefix,missing/suspension/duplicates',
    actual_train_support={f:dict(inner_rows=int(inner[f].notna().sum()),inner_dates=int(inner.loc[inner[f].notna(),'signal_date'].nunique()),calibration_rows=int(cal[f].notna().sum()),calibration_dates=int(cal.loc[cal[f].notna(),'signal_date'].nunique())) for f in m.features.FIELDS})
m.base.write_json(out/'CODEX_PREFLIGHT_NO_FITS.json',preflight)
freeze=dict(at=m.base.now(),status='AWAIT_GEMINI_ACK_NO_FITS',batch_started_at='2026-10-21T02:24:49+00:00',
    batch_deadline_utc='2026-10-21T02:54:49+00:00',budget_minutes=30,
    counterpart='ActualAntigravityGemini3.8FlashHigh;userexplicitlyrequiresjointdiscussionandtraining',
    source_sha256=m.base.sha(m.__file__),qualification_sha256=m.base.sha(out/'qualification.json'),
    hypothesis='Previouslyunused120-sessionarchive nominal-highdistance andlatest-highage improvefull-datefirst-close1to2Top1beyondoriginal62. Not aretrialofold20dayvolume/reopen/sealfailedarms.',
    arms={arm:dict(trainer=m.OWNERS[arm],fields=fields) for arm,fields in m.ARMS.items()},
    source_meaning='Bothagentsindependentlyfoundstoredpre_closeequalspriorrawclose. Codexfull1982253adjacentpairs0differences;current22,036known120windows0referencechanges. Formulachainthereforedegeneratesexactlytonominalunadjustedrawhighdistance. Admittedonlyasarchivepriceproxy;NOTexrights-adjusted,truecostbasis/resistanceorfundcause. No newrawdataretrieval. Oldlateacquisitionanduniverse-survivorshipremain.',
    numeric_corrections_to_peer_proposal='Sourcequalification,notplaincalendaravailability,iscontrolling:120development5523/10630=51.9567%,val4470/4724=94.6232%,eval12043/13149=91.5887%;250development669/10630=6.2935%,inner0,cal641/2428,val4140/4724=87.6376%.16/265is6.04%signaldatecountand146/26555.09%,neitherstock-rowcoverage. No100%knownclaim. Predatesourcehistoryofteninsufficientorgapped;nozero-fill.',
    field_correction='PurecolumnrenamingrecordedinPRICE_BASIS_ADOPTION_BEFORE_FITS;all28503numericalvaluesunchanged,oldsourceandqualificationpreserved. Qualifiedagedistances[1,120],currentS-high excluded,equaltiesmostrecent.250diagnosticonly,neveramodelinput.',
    fixed_rules={key:base_freeze[key] for key in ('training','validation','evaluation','acceptance')},
    selection='Eacharmonce;primary>52ANDnonone>=48ANDdatecoverage>=95%.Onlypassingarmwithmostvalidationhitsenters14monthlywindows;63winschallengerties,BASEwinsties.Noresult-basedparameters,thresholds,horizonchangesorretry.',
    bindings=bindings,commands={arm:f'D:/python/python.exe first_board_joint_long_high_model.py train --arm {arm} --trainer {m.OWNERS[arm]}' for arm in m.ARMS},
    evaluation_owner='Codexonceafterbothreports,winneronly,nobaselineorS2refits',
    acknowledgment='GeminimayONLYchangestatustoAGREED_BEFORE_NEW_FITSandaddagreed_at/acknowledgment_note;mayseparatelyappendcorrectiontoproposalcoveragenumerics.Nosilentfield/hashchange. Ifobjectswriteobjection,donotfit.',
    production_deployment_allowed=False,actual_trades=0,new_forward_results=0)
with (out/'JOINT_FROZEN_PROTOCOL.json').open('x',encoding='utf-8') as f:json.dump(freeze,f,ensure_ascii=False,indent=2)
shutil.copy2(out/'JOINT_FROZEN_PROTOCOL.json',out/'PROTOCOL_PENDING_ACK_SNAPSHOT.json')
print(json.dumps(preflight,ensure_ascii=True))
