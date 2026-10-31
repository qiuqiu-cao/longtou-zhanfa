"""Outcome-blind full-code acquisition inventory and exact-date THS ingestion."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse,parse_qs
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PUBLIC=ROOT/'research/first_board_public_theme_sources_20261030'
sys.path.insert(0,str(PUBLIC))
from public_theme_sources import parse_ths,sha,validate_detail_link,as_provider


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def inventory():
    plan=load(HERE/'COLLECTION_PLAN.json')
    cached={r['code']:ROOT/r['path'] for r in plan['cached_success']}
    sources,events=[],[]
    for code in plan['all_codes']:
        p=cached.get(code,HERE/'raw'/f'ths_{code}.json')
        if not p.exists():raise ValueError('collection unfinished: '+code)
        rows,status=parse_ths(p)
        status['acquisition']='REUSED_ORIGINAL_READABLE_PAGE' if code in cached else 'FULL_CODE_LIST_NEW_ATTEMPT'
        sources.append(status);events+=rows
    save(HERE/'ALL_SOURCE_INVENTORY.json',sources)
    save(HERE/'ALL_EVENTS_PENDING_DATE_CHECK.json',events)
    # Reuse original checks for exact source references. Do not treat a new page
    # with a shifting ten-event window as the same ordered version.
    queue=[{k:r[k] for k in ['code','signal_date','source_ref','detail_link_id']} for r in events if r['code'] not in cached]
    save(HERE/'LINK_CHECK_QUEUE.json',queue)
    print(json.dumps({'stocks':len(sources),'status':dict(Counter(r['status'] for r in sources)),
                      'events':len(events),'new_links':len(queue)}))


def link_checks():
    checks={}
    for directory in [PUBLIC/'link_checks',HERE/'link_checks']:
        for p in sorted(directory.glob('*.json')):
            obj=load(p);s=obj['result'].get('value','')
            if not isinstance(s,str):continue
            for part in s.split('-'*80):
                m=re.search(r'Source: click\((\{[^}]+\})\)',part)
                if not m:continue
                origin=json.loads(m[1]);key=(origin['ref_id'],origin['id'])
                # No averaging or picking a favorable conflicting destination.
                if key in checks and checks[key]['result']['value']!=part:
                    old=checks[key]
                    urls=lambda t:set(re.findall(r'&w=([^\s)\";]+)',t))
                    if urls(old['result']['value'])!=urls(part):old['conflicting_destinations']=True
                    continue
                checks[key]={'source_ref':key[0],'detail_link_id':key[1],
                    'result':{'status':'fulfilled','value':part},'check_path':str(p.relative_to(ROOT)),
                    'check_sha256':sha(p),'conflicting_destinations':False}
    return checks


def validate_observed_target(row,check):
    """The tool exposes the clicked href even if its destination has no cache.

This verifies the original page link's code/date, not the destination's content.
The theme text was already obtained from the original timeline page.
"""
    if (check.get('source_ref'),check.get('detail_link_id'))!=(row['source_ref'],row['detail_link_id']):return False
    result=check.get('result',{})
    if result.get('status')!='fulfilled' or not isinstance(result.get('value'),str):return False
    urls=re.findall(r'https://www\.iwencai\.com/[^\s)\";]+',result['value'])
    expected=row['code']+' '+row['signal_date'].replace('-','')+' 涨停原因'
    queries=[]
    for url in urls:
        # Tool error prose appends ':' after a URL; it is not part of the href.
        q=parse_qs(urlparse(url.rstrip(':')).query).get('w',[])
        queries+=q
    return bool(queries) and all(q==expected for q in queries)


def finalize():
    events=load(HERE/'ALL_EVENTS_PENDING_DATE_CHECK.json');checks=link_checks()
    for r in events:
        v=checks.get((r['source_ref'],r['detail_link_id']))
        good=bool(v and not v['conflicting_destinations'] and validate_observed_target(r,v))
        r['event_date_binding']='DETAIL_LINK_DATE_VERIFIED' if good else 'UNVERIFIED_OR_DATE_MISMATCH'
        if v:r.update(check_path=v['check_path'],check_sha256=v['check_sha256'],
                      target_content_fetched='Failed to fetch' not in v['result']['value'],
                      validation_scope='ORIGINAL_PAGE_LINK_TARGET_CODE_AND_DATE_ONLY_NOT_TARGET_BODY')
    dest=HERE/'sources_v2';dest.mkdir(exist_ok=False)
    save(dest/'THS_DATED_EVENTS.json',events)
    provider=as_provider(events)
    provider['themes']=provider.themes.map(lambda t:json.dumps(t,ensure_ascii=False))
    provider.to_csv(dest/'THS_PROVIDER_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    pool=pd.read_csv(ROOT/'research/first_board_top3_40_20261030/results/FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv',
                     dtype={'code':str},float_precision='round_trip')
    status={r['code']:r['status'] for r in load(HERE/'ALL_SOURCE_INVENTORY.json')}
    parsed={(e['signal_date'],e['code']):e for e in events}
    rows=[]
    for r in pool.itertuples(index=False):
        e=parsed.get((r.signal_date,r.code));s=status[r.code]
        category='VERIFIED_DATE_THS_TEXT' if e and e['event_date_binding']=='DETAIL_LINK_DATE_VERIFIED' else (
            'EVENT_DATE_LINK_UNVERIFIED' if e else 'NOT_IN_VISIBLE_HISTORY_OR_UNKNOWN' if s=='PARSED_ORDERED_TIMELINE' else s)
        rows.append({'signal_date':r.signal_date,'code':r.code,'source_status':category,
            'theme_raw':e['theme_raw'] if e else None,'actual_historical_eligible':False,
            'source_path':e['source_path'] if e else None,'source_sha256':e['source_sha256'] if e else None})
    coverage=pd.DataFrame(rows)
    coverage.to_csv(dest/'FULL7254_SOURCE_COVERAGE_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    coverage.assign(observed=coverage.source_status.eq('VERIFIED_DATE_THS_TEXT')).groupby('signal_date').agg(
        candidates=('code','size'),same_date_text=('observed','sum')).to_csv(dest/'DAILY_SOURCE_COVERAGE.csv',encoding='utf-8-sig')
    summary={'source_codes':len(status),'rows':len(pool),'dates':pool.signal_date.nunique(),
        'status':dict(Counter(coverage.source_status)),'events':len(events),'event_date_binding':dict(Counter(e['event_date_binding'] for e in events)),
        'all_source_queries_complete':True,'all_candidate_date_sources_complete':bool(coverage.source_status.eq('VERIFIED_DATE_THS_TEXT').all()),
        'source_limitations':['ten-event visible history','non-random unavailable pages','retrospectively retrieved versions',
            'full-period candidate code universe is not full market theme universe','forum case evidence not backfilled across history']}
    save(dest/'SOURCE_COVERAGE_REPORT.json',summary);print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','finalize'])
    inventory() if p.parse_args().action=='inventory' else finalize()
