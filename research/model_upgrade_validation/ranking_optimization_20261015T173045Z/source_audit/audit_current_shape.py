"""Read-only root-cause replay of the exact bound daily() qualification.

Writes only beneath this directory. No model fits or source mutations.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import short_term_thinking_dataset_builder as b

OUT = Path(__file__).resolve().parent
CONTEXT = ROOT / 'research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z/context/review_scope_context.joblib'
PREDS = ROOT / 'research/model_upgrade_validation/expectation_345_20261015T163226Z/evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv'


def equal_missing(left, right):
    return (left is None or pd.isna(left)) and (right is None or pd.isna(right)) or left == right


def inspect_one(day, code, prices, pools, sessions, indices, bar_columns):
    """Exact production ordering plus explicit detailed invalidity flags."""
    pos = indices[day]
    previous_date = sessions[pos-1] if pos else None
    values = prices.get((day, code))
    previous_values = prices.get((previous_date, code)) if previous_date else None
    pool_previous = pools.get((previous_date, code)) if previous_date else None
    previous = dict(zip(bar_columns, previous_values)) if previous_values is not None else pool_previous
    raw = dict(zip(bar_columns, values)) if values is not None else {}
    pc = b.finite(previous.get('close')) if previous else None
    ref = b.finite(raw.get('pre_close'))
    shape = b.price_state(raw)
    vol = b.finite(raw.get('volume'))
    defects=[]
    if values is None: defects.append('MISSING_CURRENT_DAILY')
    if pc is None or pc <= 0: defects.append('MISSING_OR_NONPOSITIVE_ADJACENT_PREVIOUS_CLOSE')
    if shape['one_word'] is None: defects.append(shape['status'])
    if ref is not None and pc is not None and abs(ref-pc)>.011: defects.append('EXPLICIT_PRE_CLOSE_CONFLICT')
    if vol is None or vol <= 0: defects.append('MISSING_OR_NONPOSITIVE_VOLUME')
    cap = b.cap_cents(pc) if pc is not None and pc>0 else None
    if cap is not None and shape['one_word'] is not None:
        if b.cents(raw['high'])>cap: defects.append('HIGH_ABOVE_EXACT_10PCT_CAP')
        if b.cents(raw['close'])!=cap: defects.append('NOT_EXACT_10PCT_CLOSING_LIMIT')
    flag=raw.get('is_suspended')
    suspended=(isinstance(flag,(bool,np.bool_)) and bool(flag)) or str(raw.get('trade_status','')).upper()=='SUSPENDED'
    if suspended: defects.append('EXPLICIT_SUSPENSION')
    if values is None: status='UNKNOWN_MISSING_DAILY'; closed=None; one=None
    elif suspended: status='UNKNOWN_SUSPENDED_EXPLICIT'; closed=None; one=None
    elif pc is None or pc<=0 or shape['one_word'] is None: status='UNKNOWN_ADJACENT_REFERENCE_OR_SHAPE'; closed=None; one=None
    elif ref is not None and abs(ref-pc)>.011: status='UNKNOWN_REFERENCE_CONFLICT'; closed=None; one=None
    elif vol is None or vol<=0: status='UNKNOWN_NONTRADING_OR_VOLUME'; closed=None; one=None
    elif b.cents(raw['high'])>cap: status='UNKNOWN_10PCT_SCOPE_CONFLICT'; closed=None; one=None
    else:
        status='OBSERVED_10PCT_PRICE_PATH_PROXY'; closed=b.cents(raw['close'])==cap; one=shape['one_word'] if closed else None
    cause=status if status!='OBSERVED_10PCT_PRICE_PATH_PROXY' else 'OBSERVED_NONCLOSING_10PCT_PRICE_PATH' if not closed else 'OBSERVED_CLOSING_ONE_PRICE' if one else 'OBSERVED_CLOSING_NONONE_PRICE'
    record=dict(date=day,code=code,previous_official_date=previous_date,
        current_daily_present=values is not None,previous_daily_present=previous_values is not None,
        adjacent_reference_source='BOUND_DAILY_BAR' if previous_values is not None else 'BOUND_CLOSING_POOL_FALLBACK' if pool_previous else 'MISSING',
        previous_close=pc,previous_volume=previous.get('volume') if previous else None,
        exact_10pct_cap_cents=cap,exact_10pct_cap=cap/100 if cap else None,
        explicit_pre_close_difference=ref-pc if ref is not None and pc is not None else None,
        replay_daily_status=status,replay_closed=closed,replay_current_price_shape=one,
        price_shape_only_status=shape['status'],raw_shape_only_one_word=shape['one_word'],
        primary_root_cause=cause,all_defects=defects)
    for k in bar_columns:record['raw_'+k]=raw.get(k)
    record['current_engine_pointer']=f"bars[(date=='{day}') & (code=='{code}')]"
    record['previous_engine_pointer']=f"bars[(date=='{previous_date}') & (code=='{code}')]" if previous_date else None
    record['fallback_pool_pointer']=f"attempts[(signal_date=='{previous_date}') & (code=='{code}') & (pool_assignment=='CLOSING_LIMIT_UP')]" if pool_previous else None
    return record


def main():
    artifact=joblib.load(CONTEXT);engine_path=Path(artifact['engine_path']);engine=joblib.load(engine_path)
    rows=artifact['rows'];bars=engine['bars'];sessions=list(engine['sessions']);indices={d:i for i,d in enumerate(sessions)}
    assert len(rows)==9315 and not rows.duplicated(['date','code']).any()
    needed=set(zip(rows.date,rows.code))
    needed|={(sessions[indices[d]-1],c) for d,c in needed.copy() if indices[d]}
    columns=[x for x in ['open','high','low','close','pre_close','volume','turnover','is_suspended','trade_status'] if x in bars]
    # DataFrame merge selects only ~15k keys; no 2m-entry dictionary or row-dicts.
    keys=pd.DataFrame(sorted(needed),columns=['date','code'])
    selected=keys.merge(bars[['date','code']+columns],on=['date','code'],how='inner',validate='one_to_one')
    prices={(r[0],r[1]):r[2:] for r in selected.itertuples(index=False,name=None)}
    pa=engine['attempts'];pa=pa[pa.pool_assignment.eq('CLOSING_LIMIT_UP')]
    pools={(r.signal_date,r.code):{'close':r.close} for r in pa[['signal_date','code','close']].itertuples(index=False)}
    records=[]
    for r in rows.itertuples(index=False):
        x=inspect_one(r.date,r.code,prices,pools,sessions,indices,columns)
        x.update(name=r.name,boards=r.boards,stored_current_price_shape=r.current_price_shape,stored_current_price_path_status=r.current_price_path_status,
            replay_matches_stored_shape=bool(equal_missing(x['replay_current_price_shape'],r.current_price_shape)))
        records.append(x)
    all_rows=pd.DataFrame(records)
    if not all_rows.replay_matches_stored_shape.all(): raise AssertionError('Qualification replay differs from stored context')
    preds=pd.read_csv(PREDS,dtype={'code':str})
    sep=preds[preds.arm.eq('CORRECTED_CAUSAL') & preds.date.ge('2026-09-01') & preds.date.le('2026-09-30') & preds.daily_one_word.eq(0) & preds.current_price_shape.isna()]
    assert len(sep)==35 and not sep.duplicated(['date','code']).any()
    audit35=sep.merge(all_rows,on=['date','code'],validate='one_to_one',suffixes=('_prediction',''))
    audit35['raw_score_call']=np.where(audit35.ranking_score.gt(.5),'PROMOTE','FAIL')
    audit35['known_target']=audit35.label_valid & audit35.target.notna()
    audit35['withheld_raw_TP']=audit35.known_target & audit35.target.eq(1) & audit35.raw_score_call.eq('PROMOTE')
    audit35['withheld_raw_FP']=audit35.known_target & audit35.target.eq(0) & audit35.raw_score_call.eq('PROMOTE')
    audit35['withheld_raw_TN']=audit35.known_target & audit35.target.eq(0) & audit35.raw_score_call.eq('FAIL')
    audit35['withheld_raw_FN']=audit35.known_target & audit35.target.eq(1) & audit35.raw_score_call.eq('FAIL')
    assert audit35.withheld_raw_TP.sum()==6 and audit35.withheld_raw_FP.sum()==5
    for frame,name in [(all_rows,'all9315_exact_daily_qualification.csv'),(audit35,'september35_unknown_exact_root_causes.csv'),(audit35[audit35.withheld_raw_TP|audit35.withheld_raw_FP],'september11_positive_scores_withheld_6TP_5FP.csv')]:
        copy=frame.copy();copy['all_defects']=copy.all_defects.map(lambda x:json.dumps(x,ensure_ascii=False))
        copy.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    # Exact engine/projection lineage is retained; raw source files are indexed separately.
    sources={str(CONTEXT.resolve()):b.sha(CONTEXT),str(engine_path.resolve()):b.sha(engine_path),str(PREDS.resolve()):b.sha(PREDS),str(Path(__file__).resolve()):b.sha(__file__),str((ROOT/'relay_causal_review_context.py').resolve()):b.sha(ROOT/'relay_causal_review_context.py'),str(Path(b.__file__).resolve()):b.sha(b.__file__)}
    relevant_sources={p:h for p,h in engine['sources'].items() if p.endswith('daily-bars.csv') or 'daily' in p.lower() or p.endswith('current_extension_receipt.json')}
    b.write_new(OUT/'bound_price_source_index.json',dict(formed_at=b.now(),source_bindings=relevant_sources,engine_sources_count=len(engine['sources']),note='Index of already-bound price-source candidates; per-date raw pointer matching follows separately. These clocks are not original-time authentication.'))
    summary=dict(formed_at=b.now(),rows=len(all_rows),selected_daily_rows=len(selected),stored_shape_reproduced=int(all_rows.replay_matches_stored_shape.sum()),
        all9315_root_causes=all_rows.primary_root_cause.value_counts().to_dict(),
        all9315_unknown_shape_root_causes=all_rows[all_rows.stored_current_price_shape.isna()].primary_root_cause.value_counts().to_dict(),
        september35_root_causes=audit35.primary_root_cause.value_counts().to_dict(),
        september35_known=int(audit35.known_target.sum()),september35_known_positives=int((audit35.known_target&audit35.target.eq(1)).sum()),
        withheld_raw_TP=int(audit35.withheld_raw_TP.sum()),withheld_raw_FP=int(audit35.withheld_raw_FP.sum()),
        withheld_raw_TN=int(audit35.withheld_raw_TN.sum()),withheld_raw_FN=int(audit35.withheld_raw_FN.sum()),
        all35_calls_unknown=bool(audit35.call.eq('UNKNOWN').all()),
        source_bindings=sources,output_bindings={str(p.resolve()):b.sha(p) for p in OUT.glob('*.csv')})
    b.write_new(OUT/'exact_daily_qualification_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
