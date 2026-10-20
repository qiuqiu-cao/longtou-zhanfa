"""Append a read-only audit of both saved stages; fitting is forbidden here."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_second_stage_model as stage
import first_board_leader_model as base

OUT=Path(__file__).resolve().parent
report=json.loads((OUT/'report.json').read_text('utf-8'))
protocol=json.loads((OUT/'protocol_BEFORE_ANY_FIT.json').read_text('utf-8'))
for path,value in protocol['source_hashes'].items():assert base.sha(path)==value
assert base.sha(stage.__file__)==report['code_sha256']
assert base.sha(OUT/'first_second_stage_bundle.joblib')==report['bundle_sha256']
rows=pd.read_csv(OUT/'all_first_and_second_boards_WITH_SEPARATE_LABELS.csv',dtype={'code':str},float_precision='round_trip')
pred={s:pd.read_csv(OUT/f'S{s}_ALL_PREDICTIONS.csv',dtype={'code':str},float_precision='round_trip') for s in (1,2)}
rebuilt,bars,sessions,audit=stage.build_rows()
keys=['signal_date','stage','code']
a=rows.set_index(keys).sort_index();b=rebuilt.set_index(keys).sort_index()
assert a.index.equals(b.index)
np.testing.assert_allclose(a[stage.VARIANTS['STAGE_CYCLE_GENE']+[stage.TARGET]],b[stage.VARIANTS['STAGE_CYCLE_GENE']+[stage.TARGET]],rtol=1e-12,atol=1e-12,equal_nan=True)

label_lookup=bars.set_index(['p','code'])[['open','high','low','close','pre_close','eligible','valid_price']]
index=pd.MultiIndex.from_arrays([rows.p.to_numpy()+1,rows.code.to_numpy()],names=['p','code'])
future=label_lookup.reindex(index)
cap=np.floor(future.pre_close*110+.5+1e-8);floor=np.floor(future.pre_close*90+.5+1e-8)
known=future.eligible.fillna(False)&future.valid_price.fillna(False)&(future.high*100).round().le(cap)&(future.low*100).round().ge(floor)
label=np.where(known,(future.close*100).round().eq(cap).astype(float),np.nan)
np.testing.assert_allclose(label,rows[stage.TARGET],equal_nan=True)
previous_index=pd.MultiIndex.from_arrays([rows.p.to_numpy()-1,rows.code.to_numpy()],names=['p','code'])
previous_first=bars.set_index(['p','code'])['first'].reindex(previous_index).fillna(False).to_numpy(dtype=bool)
assert previous_first[rows.stage.to_numpy()==2].all()
master=pd.read_csv(base.INPUTS['security_intervals'],dtype={'code':str})
interval=rows[keys].merge(master,on='code',how='left')
interval=interval[interval.signal_date.ge(interval.valid_from)&interval.signal_date.le(interval.valid_to)]
assert len(interval)==len(rows) and not interval.duplicated(keys).any()
assert (interval.board.eq('MAINBOARD')&interval.listing_status.eq('LISTED')&interval.st_status.eq('NORMAL')&interval.price_limit_regime.eq('MAINBOARD_10')).all()
del bars,rebuilt,label_lookup,a,b

cores=[];head_names=set()
for receipt_path in sorted((OUT/'models').glob('*_COMPLETED.json')):
    receipt=json.loads(receipt_path.read_text('utf-8'));name=receipt['name']
    prefix=name.endswith('_calibration_prefix')
    head=name.removesuffix('_calibration_prefix') if prefix else name.removesuffix('_full')
    head_names.add(head)
    s=1 if '_S1' in head else 2
    pool=rows[rows.stage.eq(s)&rows[stage.TARGET].notna()]
    if head.startswith('validation_'):
        boundary='2025-01-01';expected=pool[pool.signal_date.lt(boundary)&pool.label_end.lt(boundary)]
    else:
        boundary=audit['last_source_session'] if head.startswith('latest_') else head[:7]+'-01'
        p=int(np.searchsorted(sessions,boundary));expected=pool[pool.p.ge(p-252)&pool.label_end.lt(boundary)]
    if prefix:
        split=sorted(expected.signal_date.unique())[-42]
        expected=expected[expected.label_end.lt(split)]
        scored=pd.read_csv(OUT/'models'/f'{head}_calibration_rows.csv',dtype={'code':str},float_precision='round_trip')
        scored=scored.merge(rows[keys+receipt['features']],on=keys,how='left',validate='one_to_one')
        expected_score=scored.uncalibrated_oof
        assert expected.label_end.max()<split
    else:
        if head.startswith('validation_'):
            scored=pd.read_csv(OUT/f'{head}_PREDICTIONS.csv',dtype={'code':str},float_precision='round_trip')
        elif head.startswith('latest_'):
            scored=rows[rows.stage.eq(s)&rows.signal_date.eq(audit['last_source_session'])]
        else:scored=pred[s][pred[s].model_month.eq(head[:7])]
        expected_score=scored.model_score if 'model_score' in scored else None
    members=pd.read_csv(OUT/'models'/f'{name}_members.csv',dtype={'code':str},float_precision='round_trip')
    assert set(map(tuple,members[keys].to_numpy()))==set(map(tuple,expected[keys].to_numpy())),name
    assert members.label_end.lt(boundary).all(),name
    dropped=receipt.get('all_missing_columns_dropped_from_estimator_only',[])
    assert dropped==sorted([f for f in receipt['features'] if not expected[f].notna().any()]),name
    model_path=OUT/'models'/f'{name}.joblib';assert base.sha(model_path)==receipt['sha256']
    model=joblib.load(model_path)
    score=model.predict_proba(scored[receipt['features']])[:,1]
    if expected_score is not None:np.testing.assert_allclose(score,expected_score,rtol=1e-12,atol=1e-12)
    cores.append(dict(name=name,rows=len(members),max_label_end=members.label_end.max(),cutoff=boundary,prediction_rows=len(scored),empty_columns=dropped))

for head in sorted(head_names):
    bundle=joblib.load(OUT/'models'/f'{head}_bundle.joblib')
    s=bundle['stage']
    if head.startswith('latest_'):continue
    if head.startswith('validation_'):
        scored=pd.read_csv(OUT/f'{head}_PREDICTIONS.csv',dtype={'code':str},float_precision='round_trip')
    else:scored=pred[s][pred[s].model_month.eq(head[:7])]
    reproduced=stage.predict(bundle,scored)
    np.testing.assert_allclose(reproduced.model_score,scored.model_score,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(reproduced.calibrated_score,scored.calibrated_score,rtol=1e-12,atol=1e-12,equal_nan=True)

for s in (1,2):
    for period,mask in [('all',pd.Series(True,index=pred[s].index)),('2025H2',pred[s].signal_date.lt('2026-01-01')),('2026',pred[s].signal_date.ge('2026-01-01')),('non_one_price',pred[s].one_price.eq(0))]:
        metric=stage.evaluate(pred[s][mask]);original=report['evaluation'][str(s)][period]
        for k in ('1','3','5'):
            for name in ('known','unknown','selected','hits','false_positives','false_negatives'):
                assert metric['top'][k][name]==original['top'][k][name],(s,period,k,name)
        np.testing.assert_allclose(metric['calibration']['brier'],original['calibration']['brier'])

chain,chain_info=stage.chain_recheck(pred[1],pred[2])
saved=pd.read_csv(OUT/'FIRST_PICK_TO_SECOND_RECHECK_ALL.csv',dtype={'code':str},float_precision='round_trip')
assert chain_info==report['chain']
assert list(zip(chain.first_signal_date,chain.code))==list(zip(saved.first_signal_date,saved.code))
assert chain.loc[chain.first_promotion_LABEL_ONLY.ne(1),'second_model_score'].isna().all()
assert chain.second_signal_date.gt(chain.first_signal_date).all()

prefix=[]
for cutoff in ('2025-06-30','2025-12-31'):
    rebuilt,temporary,_,_=stage.build_rows(feature_cutoff=cutoff)
    a=rows[rows.signal_date.le(cutoff)].set_index(keys).sort_index();b=rebuilt.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[stage.VARIANTS['STAGE_CYCLE_GENE']],b[stage.VARIANTS['STAGE_CYCLE_GENE']],rtol=1e-12,atol=1e-12,equal_nan=True)
    prefix.append(dict(cutoff=cutoff,rows=len(a),all62_features_causal=True))
    del rebuilt,temporary,a,b

curve=pd.read_csv(OUT/'S1_COST_SCENARIO_EQUITY.csv',float_precision='round_trip')
trades=pd.read_csv(OUT/'S1_COST_SCENARIO_TRADES.csv',dtype={'code':str},float_precision='round_trip')
closed=trades[trades.status.eq('SCENARIO_CLOSED')]
unresolved=trades[trades.status.eq('OPEN_UNRESOLVED')]
wealth=1e6+closed.net_pnl.sum()+sum((r['shares']*r['mark']-r['entry_cost']) for r in unresolved.to_dict('records'))
np.testing.assert_allclose(wealth,curve.equity.iloc[-1],rtol=1e-12,atol=1e-6)
assert ((pd.to_datetime(closed.exit_date)-pd.to_datetime(closed.entry_date)).dt.days>=2).all()
gate=report['evaluation']['1']['all']['top']['1']
computed=bool(gate['hit_rate']>=.6 and gate['known']>=200 and gate['outcome_coverage']>=.95 and all(report['evaluation']['1'][p]['top']['1']['hit_rate']>=.55 for p in ('2025H2','2026')))
assert computed==report['goal_60_passed']
proof=dict(status='PASS',verified_at=base.now(),rows=len(rows),all_next_official_labels_independently_equal=True,all_interval_statuses_match=len(interval),
           saved_core_models=cores,heads_replayed=len(head_names),all_calibrated_scores_replayed=True,prefix_causality=prefix,
           all_period_topk_counts_equal=True,chain_same_stock_and_no_retroactive_second_stage=True,equity_reconciled=True,
           sixty_target_passed=computed,additional_fits=0,source_hashes_verified=True,old_first_board_bundle_untouched=True)
base.write_json(OUT/'verification.json',proof)
print(json.dumps(dict(status='PASS',cores=len(cores),heads=len(head_names),rows=len(rows),goal60=computed,additional_fits=0),ensure_ascii=False))
