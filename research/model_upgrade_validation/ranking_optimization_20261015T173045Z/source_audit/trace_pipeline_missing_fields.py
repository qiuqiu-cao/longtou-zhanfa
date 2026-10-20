"""Trace missing field creation without changing any historical data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
import joblib
import pandas as pd
import short_term_thinking_dataset_builder as b
from audit_current_shape import OUT, CONTEXT


def main():
    audit=pd.read_csv(OUT/'september35_unknown_exact_root_causes.csv',dtype={'code':str})
    keys=set(zip(audit.date,audit.code))|set(zip(audit.previous_official_date,audit.code))
    path=ROOT/'data/v3/imported/daily-bars.csv'
    fields=['date','code','open','high','low','close','pre_close','volume']
    pieces=[]
    for block in pd.read_csv(path,usecols=fields,dtype={'code':str},chunksize=200000):
        subset=block[[key in keys for key in zip(block.date,block.code)]]
        if len(subset):pieces.append(subset)
    original=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=fields)
    if original.duplicated(['date','code']).any():raise ValueError('original imported keys duplicated')
    price_context=b.read(ROOT/'research/model_upgrade_validation/early_signal_ohlc_20261013T1303Z/complete9245_provisional_price_context_DELIVERY.json')
    prior={(r['date'],r['code']):r for r in price_context['rows']}
    artifact=joblib.load(CONTEXT);engine=joblib.load(artifact['engine_path'])
    current=engine['bars'];current=current[[key in keys for key in zip(current.date,current.code)]][fields]
    originals={tuple(x[:2]):dict(zip(fields[2:],x[2:])) for x in original.itertuples(index=False,name=None)}
    present={tuple(x[:2]):dict(zip(fields[2:],x[2:])) for x in current.itertuples(index=False,name=None)}
    rows=[]
    for day,code in sorted(keys):
        old=originals.get((day,code));now=present.get((day,code));entry=prior.get((day,code),{})
        signals=[x for x in audit.itertuples() if x.code==code and day in [x.date,x.previous_official_date]]
        reason='IMPORTED_KEY_ABSENT_AND_NO_CURRENT_KEY' if old is None and now is None else 'IMPORTED_KEY_ABSENT_OHLC_ONLY_SIGNAL_INJECTION' if old is None and now is not None and all(pd.isna(now.get(k)) for k in ['volume','pre_close']) else 'ORIGINAL_IMPORTED_KEY_REMOVED_BY_LATER_DAILY_INTERVAL_REPLACEMENT' if old is not None and now is None else 'PRESENT_ORIGINAL_OR_REPLACED_KEY'
        rows.append(dict(date=day,code=code,related_signals=[r.date for r in signals],original_imported_present=old is not None,current_engine_present=now is not None,
            early_context_signal_price_sources=len(entry.get('sources',[])),early_context_status=entry.get('status'),pipeline_root_cause=reason,
            **{'original_'+k:old.get(k) if old else None for k in fields[2:]},**{'engine_'+k:now.get(k) if now else None for k in fields[2:]}))
    frame=pd.DataFrame(rows)
    frame['related_signals']=frame.related_signals.map(lambda x:json.dumps(x,ensure_ascii=False))
    frame.to_csv(OUT/'september35_current_and_adjacent_imported_pipeline_trace.csv',index=False,encoding='utf-8-sig')
    bindings={str(path.resolve()):b.sha(path),str(Path(__file__).resolve()):b.sha(__file__),str((ROOT/'short_term_thinking_dataset_builder.py').resolve()):b.sha(ROOT/'short_term_thinking_dataset_builder.py'),str((ROOT/'short_term_thinking_current.py').resolve()):b.sha(ROOT/'short_term_thinking_current.py')}
    b.write_new(OUT/'pipeline_trace_summary.json',dict(formed_at=b.now(),signal_count=35,unique_current_and_adjacent_keys=len(frame),pipeline_key_counts=frame.pipeline_root_cause.value_counts().to_dict(),
        code_evidence=['short_term_thinking_dataset_builder.load_existing(): reindex union adds only signal OHLC; no previous bar or volume added','short_term_thinking_current.build(): removes all old bars for any current code from 2026-08-31, then uses bounded current raw response rows; response may start later than old covered date'],
        fact_inference='Key-level before/after field comparison proves exact absence/injection/removal; which builder executed each step is established by existing module pipeline, not an invented raw receipt',
        source_bindings=bindings,output_sha256=b.sha(OUT/'september35_current_and_adjacent_imported_pipeline_trace.csv')))
    print(frame.pipeline_root_cause.value_counts().to_json())


if __name__=='__main__':main()
