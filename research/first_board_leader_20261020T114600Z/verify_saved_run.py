"""Read-only fit/label replay; append verified delivery, never retrain."""
from pathlib import Path
import copy
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import first_board_leader_model as fb

OUT=Path(__file__).resolve().parent
report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
assert fb.sha(OUT/'source_used_for_17_fits.py')==report['code_sha256']
for path,value in report['source_hashes'].items():
    assert fb.sha(path)==value,(path,'source hash mismatch')
rows=pd.read_csv(OUT/'all_first_boards_features_and_LABEL_ONLY.csv',dtype={'code':str},float_precision='round_trip')
pred=pd.read_csv(OUT/'evaluation_ALL_PREDICTIONS.csv',dtype={'code':str},float_precision='round_trip')
rebuilt,bars,sessions,audit=fb.build_dataset()
keys=['signal_date','code']
x=rows.set_index(keys).sort_index()
y=rebuilt.set_index(keys).sort_index()
assert x.index.equals(y.index)
np.testing.assert_allclose(x[fb.FEATURES].to_numpy(dtype=float),y[fb.FEATURES].to_numpy(dtype=float),rtol=1e-12,atol=1e-12,equal_nan=True)
known=rows[rows[fb.TARGET].notna()]
receipts=[]
for receipt_path in sorted((OUT/'models').glob('*_COMPLETED.json')):
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    name=receipt['name']
    membership=pd.read_csv(OUT/'models'/f'{name}_membership.csv',dtype={'code':str})
    if name.startswith('validation_'):
        expected=known[(known.signal_date<'2025-01-01')&(known.label_end<'2025-01-01')]
        scored=pd.read_csv(OUT/f'validation_{receipt["family"]}_ALL_PREDICTIONS.csv',dtype={'code':str},float_precision='round_trip')
        cutoff='2025-01-01'
    elif name=='latest_research':
        cutoff=audit['last_source_session'];pos=sessions.index(cutoff)
        expected=known[(known.label_end<cutoff)&(known.p>=pos-252)]
        scored=pd.read_csv(OUT/'last_archived_session_RESEARCH_RANKING.csv',dtype={'code':str},float_precision='round_trip')
    else:
        cutoff=name+'-01';pos=int(np.searchsorted(sessions,cutoff))
        expected=known[(known.label_end<cutoff)&(known.p>=pos-252)]
        scored=pred[pred.model_month.eq(name)]
    actual_keys=set(map(tuple,membership[keys].to_numpy()))
    expected_keys=set(map(tuple,expected[keys].to_numpy()))
    assert actual_keys==expected_keys,(name,'training members')
    assert membership.label_end.lt(cutoff).all(),name
    model_path=OUT/'models'/f'{name}.joblib'
    assert fb.sha(model_path)==receipt['model_sha256']
    model=joblib.load(model_path)
    recalculated=model.predict_proba(scored[fb.FEATURES])[:,1]
    np.testing.assert_allclose(recalculated,scored.model_score.to_numpy(),rtol=1e-12,atol=1e-12)
    receipts.append(dict(name=name,rows=len(membership),predictions=len(scored),max_label_end=membership.label_end.max(),cutoff=cutoff,replay_max_abs_error=float(np.max(np.abs(recalculated-scored.model_score.to_numpy())))))

# Independently recalculate exact official-session targets from raw OHLC.
look=bars.set_index(['p','code'])[['open','high','low','close','pre_close','valid_price','eligible']]
alive=np.ones(len(rows),dtype=bool);all_known=np.ones(len(rows),dtype=bool);height=np.ones(len(rows),dtype=int)
paths=[]
for h in range(1,6):
    index=pd.MultiIndex.from_arrays([rows.p.to_numpy()+h,rows.code.to_numpy()],names=['p','code'])
    future=look.reindex(index)
    known_h=future.valid_price.fillna(False).to_numpy(dtype=bool)&future.eligible.fillna(False).to_numpy(dtype=bool)
    price=(future.close*100).round()
    cap=np.floor(future.pre_close*110+.5+1e-8)
    regular=(future.high*100).round().le(cap)&(future.low*100).round().ge(np.floor(future.pre_close*90+.5+1e-8))
    all_known &= known_h
    alive &= known_h & price.eq(cap).to_numpy() & regular.to_numpy()
    height+=alive
    paths.append(alive.copy())
all_known &= rows.p.to_numpy()+5 <= int(bars.p.max())
for h in (2,3,4,5):
    direct=np.where(all_known,paths[h-2].astype(float),np.nan)
    np.testing.assert_allclose(direct,rows[f'reach{h}_LABEL_ONLY'],equal_nan=True)
np.testing.assert_allclose(np.where(all_known,height,np.nan),rows.max_height_5_LABEL_ONLY,equal_nan=True)
master=pd.read_csv(fb.INPUTS['security_intervals'],dtype={'code':str})
state=rows[keys].merge(master,on='code',how='left')
state=state[state.signal_date.ge(state.valid_from)&state.signal_date.le(state.valid_to)]
assert len(state)==len(rows) and not state.duplicated(keys).any()
assert (state.board.eq('MAINBOARD')&state.listing_status.eq('LISTED')&state.st_status.eq('NORMAL')&state.price_limit_regime.eq('MAINBOARD_10')).all()
del bars,look,rebuilt

# Two real prefix rebuilds ensure future observations do not alter any input.
prefix_checks=[]
for cutoff in ('2025-06-30','2025-12-31'):
    prefix,temporary,_,_=fb.build_dataset(feature_cutoff=cutoff)
    a=rows[rows.signal_date.le(cutoff)].set_index(keys).sort_index()
    b=prefix.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a[fb.FEATURES].to_numpy(dtype=float),b[fb.FEATURES].to_numpy(dtype=float),rtol=1e-12,atol=1e-12,equal_nan=True)
    prefix_checks.append(dict(cutoff=cutoff,candidates=len(a),all44_features_equal=True))
    del prefix,temporary,a,b

delivery=OUT/'verified_delivery';delivery.mkdir(exist_ok=False)
corrected=copy.deepcopy(report)
for period,mask in [('all',pd.Series(True,index=pred.index)),('2025H2',pred.signal_date.lt('2026-01-01')),('2026',pred.signal_date.ge('2026-01-01')),('non_one_price',pred.one_price.eq(0))]:
    part=pred[mask]
    for score,key in [('model_score','model'),('baseline_score','baseline')]:
        new=fb.evaluate(part,score)
        for k in (1,3,5):
            for h in (2,3,4,5):
                old=report['evaluation'][period][key][f'top{k}'][f'reach{h}']
                assert new[f'top{k}'][f'reach{h}']['hits']==old['hits']
                assert new[f'top{k}'][f'reach{h}']['known']==old['known']
        corrected['evaluation'][period][key]=new
    interval=fb.paired_interval(part)
    np.testing.assert_allclose(interval['ci95'],report['evaluation'][period]['top3_reach4_paired']['ci95'])
for score in ('model_score','baseline_score'):
    curve=pd.read_csv(OUT/f'{score}_SCENARIO_EQUITY.csv')
    trades=pd.read_csv(OUT/f'{score}_SCENARIO_TRADES.csv',dtype={'code':str})
    closed=trades[trades.status.eq('SCENARIO_CLOSED')]
    assert ((pd.to_datetime(closed.exit_date)-pd.to_datetime(closed.entry_date)).dt.days>=2).all()
    assert int(closed.shares.mod(100).ne(0).sum())==0
    corrected['scenario'][score]['max_drawdown']=float((curve.equity/curve.equity.cummax().clip(lower=1_000_000.)-1).min())
    np.testing.assert_allclose(curve.equity.iloc[-1],1e6+closed.net_pnl.sum()+sum((float(t['shares'])*float(t['mark'])-float(t['entry_cost'])) for t in trades[trades.status.eq('OPEN_UNRESOLVED')].to_dict('records')),rtol=1e-12,atol=1e-6)
    corrected['scenario'][score]['unfilled_entries_by_reason']=trades[~trades.status.isin(['SCENARIO_CLOSED','OPEN_UNRESOLVED'])].status.value_counts().to_dict()
corrected.update(verified_at=fb.now(),immutable_original_report='../report.json',delivery_code_sha256=fb.sha(fb.__file__),additional_fits_during_verification=0,
                 technical_corrections=['Artifact CSV inputs must use float_precision=round_trip: default pandas parsing changed 5/818 July tree scores by up to .06134639; exact parsing reproduces original scores without fitting. Initial rejection script retained.','Daily hit rates now separate fully known days from all-registered-date confirmed-hit fractions; all ranking hits, recall and precision unchanged.','Drawdown peak includes initial capital; model scores and trades unchanged.','All-cancelled synthetic scenario exposed empty-closed-table AttributeError, fixed without changing original trade events.','Listing age is explicitly left-censored at official archive start.'])
fb.write_json(delivery/'report.json',corrected)
fb.write_report(delivery,corrected)
verification=dict(status='PASS',verified_at=fb.now(),source_hashes_verified=len(report['source_hashes']),all_first_board_rows=len(rows),all44_features_rebuilt=True,
                  independent_interval_status_matches=len(state),independent_all_four_horizon_labels_equal=True,all_saved_model_replays=receipts,
                  prefix_causality=prefix_checks,all_ranking_counts_unchanged=True,all_confidence_intervals_reproduced=True,equity_cash_pnl_reconciled=True,
                  original_report_preserved=True,additional_fits=0,model_bundle_sha256=fb.sha(OUT/'first_board_leader_bundle.joblib'))
fb.write_json(delivery/'verification.json',verification)
print(json.dumps(dict(status='PASS',models=len(receipts),prefix_checks=prefix_checks,delivery=str(delivery),additional_fits=0),ensure_ascii=False,indent=2))
