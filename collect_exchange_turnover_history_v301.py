"""Collect a fixed official-session amount cohort, retaining every failed request."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from collect_exchange_turnover_v301 import fetch
from market_interval_research_v300 import CALENDAR,ROOT,sha,write_json


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
    test=[d for d in sessions if '2026-01-01'<=d<='2026-09-30']
    prior=[d for d in sessions if d<test[0]][-121:]
    days=prior+test
    initial=ROOT/'data/v301/turnover_daily_probe_20261003_v1'
    parent=json.loads((initial/'manifest.json').read_text(encoding='utf-8'))
    existing=set()
    for r in parent['requests']:
        if r['status']=='RECEIVED' and sha(initial/r['raw_path'])==r['sha256']:
            existing.add((r['venue'],r['requested_date']))
    jobs=[(v,d) for d in days for v in ['SSE','SZSE'] if (v,d) not in existing]
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'all_requested_dates':days,
        'historical_context_dates':len(prior),'signal_dates':len(test),
        'fresh_venue_day_requests':len(jobs),'workers':4,'timeouts_seconds':[5,8],'automatic_retries':0,
        'parent_archive':str(initial.resolve()),'parent_manifest_sha256':sha(initial/'manifest.json'),
        'source_hashes':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'collect_exchange_turnover_v301.py',CALENDAR]},
        'same_scope_non_st_pool_reconstruction':False,'forward_evidence':False}
    write_json(out/'collection_protocol.json',protocol)
    records=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for item in executor.map(lambda pair:fetch(out,*pair),jobs):
            records.append(item)
            with (out/'receipt_journal.jsonl').open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(item,ensure_ascii=False)+'\n')
            if len(records)%40==0 or len(records)==len(jobs):
                print(json.dumps({'finished':len(records),'planned':len(jobs),
                    'received':sum(r['status']=='RECEIVED' for r in records)},ensure_ascii=False),flush=True)
    report={'version':'v301-fixed-official-turnover-history-receipts-1',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requested_dates':sorted(set(d for _,d in jobs)),
        'requests':records,'calendar_sha256':sha(CALENDAR),'script_sha256':sha(__file__),
        'collection_protocol_sha256':sha(out/'collection_protocol.json'),
        'genuine_historical_availability_certified':False,'forward_evidence':False}
    write_json(out/'manifest.json',report)
    return {'status':'BOUNDED_COLLECTION_COMPLETED','fresh_requests':len(records),
            'received':sum(r['status']=='RECEIVED' for r in records),'all_dates_including_parent':len(days)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output),ensure_ascii=False))
