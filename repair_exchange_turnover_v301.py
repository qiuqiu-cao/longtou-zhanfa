"""One explicit retry per recorded missing receipt; original attempts remain immutable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from collect_exchange_turnover_v301 import fetch
from market_interval_research_v300 import ROOT,CALENDAR,sha,write_json

PARENTS=[ROOT/'data/v301/turnover_daily_probe_20261003_v1',ROOT/'data/v301/turnover_history_receipts_20261003_v1']
NORMALIZED=ROOT/'data/v301/turnover_history_observations_20261003_v1'


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((NORMALIZED/'report.json').read_text(encoding='utf-8'))
    for path,digest in report['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('parent original source altered')
    records={};sources=dict(report['source_hashes'])
    sources[str((NORMALIZED/'report.json').resolve())]=sha(NORMALIZED/'report.json')
    for parent in PARENTS:
        manifest=json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
        for r in manifest['requests']:
            key=(r['venue'],r['requested_date'])
            if key in records:raise ValueError('duplicate parent key')
            records[key]=r
            if r['status']=='RECEIVED':
                path=parent/r['raw_path']
                if sha(path)!=r['sha256']:raise ValueError('parent raw altered')
                (out/r['raw_path']).write_bytes(path.read_bytes())
    jobs=sorted(key for key,r in records.items() if r['status']=='UNAVAILABLE')
    if not jobs or len(jobs)>20:raise ValueError('explicit small missing-request repair required')
    write_json(out/'repair_protocol.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'repair_keys':[list(k) for k in jobs],'automatic_retries':0,'one_attempt_each':True,
        'original_failed_attempts_preserved_in_parent':True,'parent_source_hashes':sources})
    with ThreadPoolExecutor(max_workers=4) as executor:
        attempts=list(executor.map(lambda pair:fetch(out,*pair),jobs))
    write_json(out/'repair_attempts.json',attempts)
    for r in attempts:
        key=(r['venue'],r['requested_date'])
        if key not in jobs:raise ValueError('repair outside frozen missing keys')
        records[key]=r
    manifest={'version':'v301-official-turnover-receipts-with-one-explicit-gap-repair',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requested_dates':sorted(set(k[1] for k in records)),
        'requests':[records[k] for k in sorted(records)],'calendar_sha256':sha(CALENDAR),
        'script_sha256':sha(__file__),'parent_source_hashes':sources,
        'repair_protocol_sha256':sha(out/'repair_protocol.json'),'repair_attempts_sha256':sha(out/'repair_attempts.json'),
        'new_requests':len(attempts),'new_received':sum(r['status']=='RECEIVED' for r in attempts),
        'original_604_calls_and_failures_retained':True,'forward_evidence':False}
    write_json(out/'manifest.json',manifest)
    return {'new_requests':len(attempts),'new_received':manifest['new_received'],'logical_venue_days':len(records)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args().output)))
