"""Rebuild live feature/outcome archives from immutable V293 raw receipts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_upgrade_research import ROOT, OLD, read, sha, verify_protocol, write_new
from public_outcome_sources_v293 import reconstruct
from public_outcome_bridge_v293 import candidate_truth
from next_session_expectation_contract_v271 import stamp

VERSION = 'model-upgrade-feature-archive-1'


def _now():
    return datetime.now(timezone.utc)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.bool_,bool)): return bool(value)
    if isinstance(value, (np.integer,int)): return int(value)
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT: return None
    return value


def _closed(day):
    return _now() >= stamp(day+'T15:00:00+08:00')


def _source_bindings(directory, manifest):
    folder = Path(directory).resolve()
    result = {str(folder/'manifest.json'): sha(folder/'manifest.json'),
              str(folder/'normalized.json'): sha(folder/'normalized.json')}
    for item in manifest['requests']:
        if item.get('raw_path'):
            result[str(Path(item['raw_path']).resolve())] = sha(item['raw_path'])
    result.update(manifest.get('source_hashes', {}))
    return result


def _pool_frame(pool, daily, day):
    rows=[]
    for r in pool['rows']:
        item=daily.get(r['code'], {})
        bar=item.get('days',{}).get(day)
        agreed=bar is not None and bar['close']==r['price_cents']
        one=float(bar['high']==bar['low']) if agreed else np.nan
        turnover=pd.to_numeric(r.get('turnover_pct'), errors='coerce')
        amount=pd.to_numeric(r.get('amount'), errors='coerce')
        field_errors=[]
        for key,value in [('turnover',turnover),('amount',amount)]:
            if pd.notna(value) and (not np.isfinite(value) or value<0):
                field_errors.append(key)
        turnover=turnover if pd.notna(turnover) and np.isfinite(turnover) and turnover>=0 else np.nan
        amount=amount if pd.notna(amount) and np.isfinite(amount) and amount>=0 else np.nan
        prior=[d for d in item.get('days',{}) if d<day]
        prev=item['days'][max(prior)]['close'] if prior else np.nan
        amplitude=(bar['high']-bar['low'])/prev*100 if agreed and prev>0 else np.nan
        rows.append({'date':day,'code':r['code'],'name':r['name'],'boards':r['boards'],
                     'price':r['price_cents']/100,'amount':amount,
                     'turnover':turnover,'one_word_proxy':one*float(turnover<3) if pd.notna(turnover) else np.nan,
                     'actual_one_word_daily':one,'amplitude':amplitude,'target_date':None,
                     'promoted_next_day':np.nan,'ohlc_status':'CENT_EXACT_UNADJUSTED' if agreed else 'UNKNOWN',
                     'captured_at':pool['available_at'],'field_errors':field_errors})
    return pd.DataFrame(rows,columns=['date','code','name','boards','price','amount','turnover',
        'one_word_proxy','actual_one_word_daily','amplitude','target_date','promoted_next_day','ohlc_status','captured_at','field_errors'])


def _old_pool(day):
    from september_forecast_holdout_v278 import ARCHIVE
    f=pd.read_csv(ARCHIVE,dtype={'code':str})
    f=f[f.date.eq(day)].copy()
    return f if len(f) else None


def _market_row(day,pool,previous,previous_market,history,pools):
    from leader_system_v3 import promotion_rates,rolling_emotion_context,emotion_dimensions,classify_emotion
    ladder=pool[pool.boards.ge(2)]
    height=int(pool.boards.max()) if len(pool) else np.nan
    failed=pools.get((day,'failed_board')); down=pools.get((day,'limit_down'))
    feedback=pools.get((day,'previous_limit'))
    complete=previous is not None and feedback is not None
    expected=set(previous.code) if previous is not None else set()
    supplied={r['code']:r for r in feedback['rows']} if feedback else {}
    if expected-set(supplied): complete=False
    returns=pd.Series([pd.to_numeric(supplied[c].get('pct'),errors='coerce')/100 for c in sorted(expected & set(supplied))],dtype=float)
    if not np.isfinite(returns.to_numpy()).all(): complete=False
    fail=float(len(failed['rows'])) if failed else np.nan
    dn=float(len(down['rows'])) if down else np.nan
    row={'date':day,'limit_count':float(len(pool)),'consecutive_count':len(ladder),'max_board':height,
         'failed_count':fail,'down_limit_count':dn,
         'seal_rate':len(pool)/(len(pool)+fail) if len(pool)+fail>0 else np.nan,
         'failed_rate':fail/(len(pool)+fail) if len(pool)+fail>0 else np.nan,
         'height_change':height-previous_market.get('max_board',np.nan),
         'limit_change':len(pool)-previous_market.get('limit_count',np.nan),
         'middle_density':ladder.boards.between(2,4).sum()/max(1,len(ladder)),
         'ladder_gap_count':sum(b not in set(ladder.boards) for b in range(2,int(height)+1)) if pd.notna(height) else np.nan,
         'unique_space':float(ladder.boards.eq(height).sum()==1),
         'one_word_share':float(ladder.one_word_proxy.mean()) if len(ladder) and ladder.one_word_proxy.notna().all() else np.nan,
         'yesterday_premium':float(returns.median()) if complete and len(returns) else np.nan,
         'previous_limit_tail_loss':float(returns.quantile(.1)) if complete and len(returns) else np.nan,
         'feedback_cohort_status':'EXACT_PREVIOUS_CLOSING_POOL_COVERED' if complete else 'UNKNOWN',
         'previous_limit_feedback_coverage':len(expected&set(supplied))/len(expected) if expected else np.nan,
         **promotion_rates(previous,pool)}
    row['space_feedback_3d']=float(pd.Series([r.get('space_feedback',np.nan) for r in history[-2:]]+[row['space_feedback']]).mean())
    row['breadth_height_divergence']=float(row['height_change']>0 and row['limit_change']<-20) if pd.notna(row['height_change']) and pd.notna(row['limit_change']) else np.nan
    row.update(rolling_emotion_context(history,row)); row.update(emotion_dimensions(row))
    row['emotion_score']=row['emotion_quality_score']
    row['emotion_state']=classify_emotion(row,previous_market.get('emotion_state'))
    row['data_status']='verified' if (row['emotion_data_completeness']>=.9 and previous is not None and complete
        and failed is not None and down is not None and pd.notna(row['height_change']) and pd.notna(row['limit_change'])
        and not any(pool.get('field_errors', pd.Series([],dtype=object)))) else 'partial'
    row['scope_status']='PROVIDER_REPORTED_NOT_EXCHANGE_CERTIFIED'
    return row


def _derive(protocol_path,archive_path,calendar_path,history_archives,cutoff,require_next=True):
    from model_upgrade_history import prepare_datasets,interactions
    from next_session_forecast_lab_v271 import prepare_market
    from september_forecast_holdout_v278 import relay_features
    protocol=verify_protocol(protocol_path)
    calendar=read(calendar_path)
    original=read(protocol['calendar_path'])
    sessions=calendar.get('sessions',[])
    if calendar.get('status')!='verified' or not calendar.get('source') or sessions!=sorted(set(sessions)):
        raise ValueError('verified official calendar required')
    if sessions[:len(original['sessions'])]!=original['sessions']:
        raise ValueError('calendar extension changed original official prefix')
    archives=list(dict.fromkeys([str(Path(p).resolve()) for p in history_archives]+[str(Path(archive_path).resolve())]))
    all_pools, all_daily, bindings={}, {}, {}
    current_manifest=None
    for path in archives:
        m,pools,daily,failures=reconstruct(path)
        if stamp(m['recorded_at'])>stamp(cutoff): raise ValueError('raw archive formed after knowledge cutoff')
        for key,pool in pools.items():
            if stamp(pool['available_at'])>stamp(cutoff): raise ValueError('pool receipt beyond cutoff')
            if key in all_pools and clean(all_pools[key]['rows'])!=clean(pool['rows']):
                raise ValueError('archived same-date pool responses conflict')
            all_pools[key]=pool
        for code,item in daily.items():
            if stamp(item['available_at'])>stamp(cutoff): raise ValueError('daily receipt beyond cutoff')
            previous=all_daily.get(code)
            if previous:
                common=set(previous['days'])&set(item['days'])
                if any(previous['days'][d]!=item['days'][d] for d in common):
                    raise ValueError('unadjusted daily revisions conflict with saved source')
                item={**item,'days':{**previous['days'],**item['days']}}
            all_daily[code]=item
        bindings.update(_source_bindings(path,m))
        if Path(path)==Path(archive_path).resolve(): current_manifest=m
    signal=current_manifest['target_date']
    if signal not in sessions or not _closed(signal): raise ValueError('completed official signal required')
    pos=sessions.index(signal)
    if require_next and pos+1>=len(sessions): raise ValueError('extend official calendar before assigning next target')
    if (signal,'limit_up') not in all_pools: raise ValueError('complete returned-date signal pool unavailable')
    prepared=prepare_datasets()
    bindings.update(prepared['provenance']['source_hashes']); bindings.update(prepared['provenance']['dependency_hashes'])
    bindings[str(Path(calendar_path).resolve())]=sha(calendar_path)
    bindings[str(Path(protocol_path).resolve())]=sha(protocol_path)
    raw=pd.read_csv(OLD/'combined_market_input.csv')
    raw=raw[raw.date.lt(signal)].copy()
    latest=raw.date.max()
    market_records=raw.to_dict('records')
    fresh_pools=[]
    for day in [d for d in sessions if latest<d<=signal]:
        pool=all_pools.get((day,'limit_up'))
        if not pool:
            market_records.append({'date':day,'data_status':'unavailable','emotion_state':None})
            continue
        f=_pool_frame(pool,all_daily,day); fresh_pools.append(f)
        prev_day=sessions[sessions.index(day)-1] if sessions.index(day)>0 else None
        pp=all_pools.get((prev_day,'limit_up'))
        previous=_pool_frame(pp,all_daily,prev_day) if pp else _old_pool(prev_day)
        prior=next((r for r in reversed(market_records) if r['date']==prev_day),{})
        row=_market_row(day,f,previous,prior,market_records,all_pools)
        market_records.append(row)
    rebuilt=pd.DataFrame(market_records).sort_values('date').reset_index(drop=True)
    # prepare_market reads a CSV; keep the feature calculation identical without
    # manufacturing an on-disk staging file: supply an in-memory text stream.
    from io import StringIO
    frame,fields=prepare_market(StringIO(rebuilt.to_csv(index=False)))
    old_index=prepared['market'][['date','sh_a_return_pct','sz_a_return_pct']]
    frame=frame.merge(old_index,on='date',how='left',validate='one_to_one')
    for key in ['sh_a_return_pct','sz_a_return_pct']:
        fields.append(key)
        for lag in (1,3):
            name=key+f'_lag_{lag}';frame[name]=frame[key].shift(lag);fields.append(name)
    market=interactions(frame,'market',sessions)
    current=market[market.date.eq(signal)]
    if len(current)!=1: raise ValueError('one current market row required')
    all_market_fields=list(fields)+['nonpositive_premium_promotion_down','nonpositive_premium_failure_up']
    mf={k:current.iloc[0][k] for k in all_market_fields}
    fresh=pd.concat(fresh_pools,ignore_index=True) if fresh_pools else pd.DataFrame()
    rf=relay_features(fresh,rebuilt,prepared['relay']) if len(fresh) else pd.DataFrame()
    if len(rf): rf=interactions(rf,'relay')
    all_relay_fields=list(prepared['relay_fields'])+['amount_rank_competition','turnover_rank_competition']
    candidates=[]
    for row in rf[rf.date.eq(signal)].to_dict('records') if len(rf) else []:
        candidates.append({'code':row['code'],'name':row.get('name'),'boards':int(row['boards']),
                           'actual_one_word_daily':row.get('actual_one_word_daily'),
                           'features':{k:row.get(k,np.nan) for k in all_relay_fields}})
    expected={r['code']:r['boards'] for r in all_pools[(signal,'limit_up')]['rows'] if r['boards']>=2}
    if {r['code']:r['boards'] for r in candidates}!=expected: raise ValueError('full original relay pool lost')
    baseline=float(current.iloc[0].emotion_quality_score) if pd.notna(current.iloc[0].emotion_quality_score) else None
    prev_day=sessions[pos-1] if pos else None
    prev=frame[frame.date.eq(prev_day)]
    persistence=None
    if baseline is not None and len(prev)==1 and pd.notna(prev.iloc[0].emotion_quality_score):
        delta=baseline-float(prev.iloc[0].emotion_quality_score)
        persistence='IMPROVE' if delta>.05 else 'WORSEN' if delta<-.05 else 'CONTINUE'
    return clean({'signal_date':signal,'target_date':sessions[pos+1] if pos+1<len(sessions) else None,'calendar':calendar,
                  'scope':'PROVIDER_REPORTED_MAINBOARD_NAME_NON_ST_NOT_EXCHANGE_CERTIFIED',
                  'market':{'features':mf,'quality':baseline,'status':current.iloc[0].data_status,'persistence':persistence,
                            'unknown_fields':[k for k,v in mf.items() if pd.isna(v)],
                            'index_scope':'Live index quotes absent unless separately source-bound; historical index rows never substituted for today'},
                  'relay':sorted(candidates,key=lambda r:r['code']),
                  'original_pool':all_pools[(signal,'limit_up')],'sources':bindings})


def build_feature_archive(protocol_path,archive_path,calendar_path,output,history_archives=()):
    out=Path(output).resolve()
    if out.exists(): raise FileExistsError('new immutable feature archive required')
    cutoff=_now().isoformat()
    payload=_derive(protocol_path,archive_path,calendar_path,history_archives,cutoff)
    for filename in ['model_upgrade_inputs.py','model_upgrade_history.py','model_upgrade_research.py']:
        payload['sources'][str(ROOT/filename)]=sha(ROOT/filename)
    out.mkdir(parents=True)
    write_new(out/'features.json',payload)
    manifest={'version':VERSION,'formed_at':_now().isoformat(),'knowledge_cutoff':cutoff,
              'signal_date':payload['signal_date'],'protocol_path':str(Path(protocol_path).resolve()),
              'source_archive':str(Path(archive_path).resolve()),'calendar_path':str(Path(calendar_path).resolve()),
              'history_archives':[str(Path(p).resolve()) for p in history_archives],
              'features_sha256':sha(out/'features.json'),'source_hashes':payload['sources']}
    write_new(out/'manifest.json',manifest)
    return {'status':'SOURCE_REBUILT_FEATURES_NOT_FORECAST','feature_archive':str(out),
            'signal_date':payload['signal_date'],'candidates':len(payload['relay'])}


def load_feature_archive(path,cutoff=None):
    out=Path(path).resolve(); m=read(out/'manifest.json')
    if m.get('version')!=VERSION or sha(out/'features.json')!=m['features_sha256']:
        raise ValueError('feature archive schema/hash mismatch')
    for filename,wanted in m['source_hashes'].items():
        if not Path(filename).is_file() or sha(filename)!=wanted: raise ValueError('feature source/program changed')
    limit=cutoff or _now().isoformat()
    if stamp(m['knowledge_cutoff'])>stamp(m['formed_at']):
        raise ValueError('feature formation precedes original knowledge cutoff')
    if stamp(m['formed_at'])>stamp(limit): raise ValueError('complete feature archive unavailable at forecast cutoff')
    rebuilt=_derive(m['protocol_path'],m['source_archive'],m['calendar_path'],m['history_archives'],m['knowledge_cutoff'])
    saved=read(out/'features.json')
    # Extra program bindings do not change observed values.
    rebuilt['sources']=saved['sources']
    if rebuilt!=saved: raise ValueError('feature values differ from raw-source reconstruction')
    return {**saved,'available_at':m['formed_at'],'feature_archive':str(out),'manifest_sha256':sha(out/'manifest.json')}


def build_outcomes(registration,source_archive):
    path=registration.get('feature_archive') or registration.get('feature_archive_path')
    if not path: raise ValueError('original source-bound feature archive missing')
    original=load_feature_archive(path,registration.get('knowledge_cutoff'))
    m,pools,daily,failures=reconstruct(source_archive)
    signal,target=registration['signal_date'],registration['target_date']
    if (m['signal_date'],m['target_date'])!=(signal,target) or not _closed(target):
        raise ValueError('only exact matured original target can settle')
    current=pools.get((signal,'limit_up'))
    if current:
        projection=lambda p:sorted((r['code'],r['boards'],r['price_cents']) for r in p['rows'])
        if projection(current)!=projection(original['original_pool']): raise ValueError('reacquired signal pool contradicts original')
    truths={}
    for c in original['relay']:
        truth=candidate_truth(c,original['original_pool'],pools.get((target,'limit_up')),
                              pools.get((target,'previous_limit')),daily,signal,target)
        truths[c['code']]=truth['promoted'] if truth['status']=='verified' else None
    fm=read(Path(path)/'manifest.json')
    target_features=_derive(fm['protocol_path'],source_archive,fm['calendar_path'],
                            fm['history_archives']+[fm['source_archive']],_now().isoformat(),require_next=False)
    actual=None
    a,b=original['market'],target_features['market']
    if a['status']=='verified' and b['status']=='verified' and a['quality'] is not None and b['quality'] is not None:
        delta=b['quality']-a['quality']
        actual='IMPROVE' if delta>.05 else 'WORSEN' if delta<-.05 else 'CONTINUE'
    return {'market_target':actual,'candidates':truths,'sources':_source_bindings(source_archive,m),
            'formed_at':_now().isoformat(),'request_failures':failures,
            'scope':'PROVIDER_SCOPE_EXACT_CENT_PRICE_AND_SAME_QUALITY_FORMULA_NOT_EXCHANGE_CERTIFIED'}
