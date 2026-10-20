"""Isolated, no-writeback raw-source field proposals for every unknown shape.

Only original signal/code scope and same/previous official-session raw bars.
Returns candidate patches, never modifies a bound engine or frozen score.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
from collections import Counter,defaultdict
import joblib
import numpy as np
import pandas as pd
import short_term_thinking_dataset_builder as b
from public_outcome_sources_v293 import parse_daily
from audit_current_shape import CONTEXT, OUT, inspect_one

PRICE_CONTEXT=ROOT/'research/model_upgrade_validation/early_signal_ohlc_20261013T1303Z/complete9245_provisional_price_context_DELIVERY.json'


def main():
    context=b.read(PRICE_CONTEXT)
    bykey={(r['date'],r['code']):r for r in context['rows']}
    artifact=joblib.load(CONTEXT);engine=joblib.load(artifact['engine_path'])
    old=pd.read_csv(OUT/'all9315_exact_daily_qualification.csv',dtype={'code':str})
    unknown=old[old.stored_current_price_shape.isna()].copy()
    sessions=engine['sessions'];indices={d:i for i,d in enumerate(sessions)}
    needed={(r.date,r.code) for r in unknown.itertuples()}
    needed|={(sessions[indices[d]-1],c) for d,c in needed.copy() if indices[d]}
    cols=[x for x in ['open','high','low','close','pre_close','volume','turnover','is_suspended','trade_status'] if x in engine['bars']]
    selected=pd.DataFrame(sorted(needed),columns=['date','code']).merge(engine['bars'][['date','code']+cols],on=['date','code'],how='inner',validate='one_to_one')
    prices={(r[0],r[1]):r[2:] for r in selected.itertuples(index=False,name=None)}
    pa=engine['attempts'];pa=pa[pa.pool_assignment.eq('CLOSING_LIMIT_UP')]
    pools={(r.signal_date,r.code):dict(close=r.close) for r in pa[['signal_date','code','close']].itertuples(index=False)}
    # No current or future result data are opened. Request/source eligibility is
    # determined from the immutable original price-context cohort, not labels.
    report_cache={};payload_cache={};bindings={str(PRICE_CONTEXT.resolve()):b.sha(PRICE_CONTEXT),str(CONTEXT.resolve()):b.sha(CONTEXT),str(Path(__file__).resolve()):b.sha(__file__),str((OUT/'audit_current_shape.py').resolve()):b.sha(OUT/'audit_current_shape.py'),str((ROOT/'public_outcome_sources_v293.py').resolve()):b.sha(ROOT/'public_outcome_sources_v293.py')}
    evidence=[];fields=[];results=[];limitations=[]
    for r in unknown.itertuples():
        item=bykey.get((r.date,r.code),{})
        sources=item.get('sources',[])
        candidate_days=defaultdict(list)
        for source in sources:
            if not source.get('raw_path'):
                limitations.append(dict(date=r.date,code=r.code,reason='NORMALIZED_ONLY_NOT_RAW_QUALIFIED',normalized_file=source.get('normalized_file')))
                continue
            path=Path(source['raw_path']);report_path=Path(source['report_path']);params=source.get('request',{}).get('param','').split(',')
            if len(params)!=6 or params[0]!=(('sh' if r.code.startswith('6') else 'sz')+r.code) or params[1]!='day' or params[-1]!='':raise ValueError('original unadjusted source request mismatch')
            if not params[2]<=r.date<=params[3]:raise ValueError('signal is outside original request')
            for p,h in [(path,source['raw_sha256']),(report_path,source['report_sha256'])]:
                if str(p) not in bindings:
                    if b.sha(p)!=h:raise ValueError('original source hash mismatch: '+str(p))
                    bindings[str(p)]=h
            if str(report_path) not in report_cache:
                report=b.read(report_path)
                if 'full_long_interval' in str(report_path):scope_path=report_path.parent/'candidate_scope_NO_OUTCOMES.csv'
                elif 'long_interval_recovery' in str(report_path):scope_path=report_path.parent/'recovery_scope_NO_OUTCOMES.csv'
                elif 'full_august' in str(report_path):scope_path=report_path.parent/'august_cohort_LABEL_ONLY.csv'
                else:scope_path=ROOT/'results/v278/frozen_september_20261001T165403Z/frozen_september_cohort.csv'
                # The September legacy cohort is resolved from its module's
                # actual constant rather than guessing an equivalent file.
                if 'september_unadjusted' in str(report_path):
                    import historical_ohlc_archive_v278 as sep
                    scope_path=Path(sep.POOL)
                scope_hash=report.get('artifact_hashes',{}).get(scope_path.name,report.get('source_pool_sha256'))
                if not scope_hash or b.sha(scope_path)!=scope_hash:raise ValueError('original request scope hash mismatch')
                scope=pd.read_csv(scope_path,dtype={'code':str})
                allowed=set(zip(scope[scope.boards.ge(2)].date,scope[scope.boards.ge(2)].code))
                bindings[str(scope_path.resolve())]=scope_hash
                report_cache[str(report_path)]=(report,allowed)
            report,allowed=report_cache[str(report_path)]
            if (r.date,r.code) not in allowed:raise ValueError('outside original dated candidate cohort')
            matches=[x for x in report['audit'] if x.get('symbol')==params[0] and x.get('kind','daily')=='daily']
            if len(matches)!=1:raise ValueError('ambiguous original daily receipt')
            receipt=matches[0]
            if receipt.get('http_status')!=200 or receipt.get('status') not in ['RAW_PARSED','UNADJUSTED_DAY_AVAILABLE']:raise ValueError('unsuccessful raw source')
            if receipt.get('raw_sha256',receipt.get('sha256'))!=source['raw_sha256'] or receipt.get('parameters')!=source['request']:raise ValueError('receipt raw binding mismatch')
            if str(path) not in payload_cache:
                raw=b.read(path);parsed=parse_daily(raw,r.code,params[2],params[3])
                symbol=params[0]
                pointers={x[0]:i for i,x in enumerate(raw['data'][symbol]['day'])}
                payload_cache[str(path)]=(parsed,pointers)
            parsed,pointers=payload_cache[str(path)]
            prev=sessions[indices[r.date]-1] if indices[r.date] else None
            # Only same day and its immediate official predecessor are emitted.
            for day in [r.date,prev]:
                if not day or day not in parsed['days']:continue
                value=parsed['days'][day]
                x=dict(signal_date=r.date,date=day,code=r.code,raw_path=str(path),raw_sha256=source['raw_sha256'],report_path=str(report_path),report_sha256=source['report_sha256'],
                    original_requested_start=params[2],original_requested_end=params[3],original_signal_code_scope=True,
                    raw_pointer=f"data.{params[0]}.day[{pointers[day]}]",raw_field_order=['date','open','close','high','low','volume_native'],
                    available_at_actual=source.get('available_at'),requested_at=receipt.get('requested_at',receipt.get('requested_at_utc')),
                    strict_original_signal_time_eligible=False,volume_unit='PROVIDER_NATIVE_TENCENT_DAY_NOT_SHARES_CERTIFIED',
                    adjusted=False,**{k:value[k]/100 for k in ['open','high','low','close']},volume=value['volume_native'])
                evidence.append(x);candidate_days[day].append(x)
        patches={};conflicts=[]
        for day,candidates in candidate_days.items():
            original=prices.get((day,r.code));original=dict(zip(cols,original)) if original is not None else {}
            # Each OHLC/volume column must agree across all bound raw responses.
            proposed=original.copy()
            for field in ['open','high','low','close','volume']:
                values={b.finite(x[field]) for x in candidates}
                if len(values)!=1:
                    conflicts.append(dict(date=day,field=field,reason='RAW_SOURCES_DISAGREE',values=sorted(x for x in values if x is not None)));continue
                value=next(iter(values));before=b.finite(original.get(field))
                if before is not None:
                    # Provider-native raw volume must never replace existing
                    # share-volume or get divided against it.
                    if field!='volume' and before!=value:
                        conflicts.append(dict(date=day,field=field,reason='EXISTING_NONMISSING_PRICE_DISAGREES',existing=before,raw=value))
                    continue
                if value is None or value<=0:continue
                proposed[field]=value
                fields.append(dict(signal_date=r.date,date=day,code=r.code,field=field,existing_value=None,proposed_value=value,
                    policy='FILL_MISSING_ONLY_RAW_AGREEMENT_NO_WRITEBACK',actual_source_count=len(candidates),source_paths=[x['raw_path'] for x in candidates],
                    raw_field_index={'open':1,'close':2,'high':3,'low':4,'volume':5}[field],
                    strict_original_time_eligible=False,late_obtained_research_only=True,
                    unit='PROVIDER_NATIVE_TENCENT_DAY_NOT_SHARES_CERTIFIED' if field=='volume' else 'CNY_UNADJUSTED_EXACT_CENTS'))
            if original!=proposed:patches[(day,r.code)]=tuple(proposed.get(k) for k in cols)
        scenario=prices|patches
        after=inspect_one(r.date,r.code,scenario,pools,sessions,indices,cols)
        patch_fields=[x for x in fields if x['signal_date']==r.date and x['code']==r.code]
        # Conflicts invalidate a source repair even if a missing-field scenario
        # would incidentally pass daily(). No altered input is exported as truth.
        candidate_resolved=after['replay_current_price_shape'] is not None and not conflicts
        results.append(dict(date=r.date,code=r.code,name=r.name,original_root_cause=r.primary_root_cause,
            raw_source_count=len(sources),raw_evidence_days=len(candidate_days),candidate_field_count=len(patch_fields),
            missing_fields_scenario_status=after['replay_daily_status'],missing_fields_scenario_shape=after['replay_current_price_shape'],
            candidate_resolved=candidate_resolved,source_conflicts=conflicts,
            # Volume-ratio features need a same-provider/unit interval. This
            # audit only qualifies daily trade presence, never a mixed ratio.
            volume_ratio_feature_update_authorized=False,writeback_performed=False))
    results=pd.DataFrame(results)
    results['source_conflicts']=results.source_conflicts.map(lambda x:json.dumps(x,ensure_ascii=False))
    results.to_csv(OUT/'all244_unknown_raw_repair_candidate_status.csv',index=False,encoding='utf-8-sig')
    b.write_new(OUT/'raw_same_and_adjacent_evidence.json',dict(formed_at=b.now(),rows=evidence,scope='Only original signal/code cohort, same and adjacent official previous day; later dates not read for qualification',source_bindings=bindings))
    b.write_new(OUT/'missing_field_candidate_patches_NO_WRITEBACK.json',dict(formed_at=b.now(),rows=fields,unknown_normalized_only=limitations,policy='Candidate evidence only, no model refit, no frozen scores or labels changed; no mixed-volume relative feature authorization'))
    sep=pd.read_csv(OUT/'september35_unknown_exact_root_causes.csv',dtype={'code':str})
    sep=sep.merge(results,on=['date','code'],validate='one_to_one',suffixes=('','_repair'))
    sep.to_csv(OUT/'september35_raw_repair_candidates_NO_WRITEBACK.csv',index=False,encoding='utf-8-sig')
    output_paths=[OUT/'all244_unknown_raw_repair_candidate_status.csv',OUT/'raw_same_and_adjacent_evidence.json',OUT/'missing_field_candidate_patches_NO_WRITEBACK.json',OUT/'september35_raw_repair_candidates_NO_WRITEBACK.csv']
    summary=dict(formed_at=b.now(),all_unknown=244,all_unknown_candidate_resolved=int(results.candidate_resolved.sum()),
        by_original_cause=results.groupby('original_root_cause').agg(rows=('code','size'),candidate_resolved=('candidate_resolved','sum')).reset_index().to_dict('records'),
        september35_candidate_resolved=int(sep.candidate_resolved.sum()),september35_scenario_shapes=sep.missing_fields_scenario_shape.value_counts(dropna=False).to_dict(),
        september35_remaining=int((~sep.candidate_resolved).sum()),candidate_fields=len(fields),same_adjacent_raw_evidence_rows=len(evidence),
        receipt_semantics='Late actual receipt or unknown retained; not original-time eligible, no role/exchange/fill truth',
        no_fits=True,no_network=True,no_original_writeback=True,source_bindings=bindings,output_bindings={str(p.resolve()):b.sha(p) for p in output_paths})
    # JSON keys may not contain NaN; explicit string presentation.
    summary['september35_scenario_shapes']={str(k):v for k,v in summary['september35_scenario_shapes'].items()}
    b.write_new(OUT/'raw_repair_candidate_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
