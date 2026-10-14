"""Bounded official daily turnover receipts. Parsing and scope approval are separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests
from market_interval_research_v300 import sha,write_json,CALENDAR


def request_spec(venue,day):
    if venue=='SSE':
        return 'https://query.sse.com.cn/commonQuery.do', {
            'sqlId':'COMMON_SSE_SJ_GPSJ_CJGK_MRGK_C','PRODUCT_CODE':'01,02,03,11,17',
            'type':'inParams','SEARCH_DATE':day},'https://www.sse.com.cn/'
    if venue=='SZSE':
        return 'https://www.szse.cn/api/report/ShowReport/data',{
            'SHOWTYPE':'JSON','CATALOGID':'1803_sczm','TABKEY':'tab1',
            'txtQueryDate':day,'PAGENO':'1'},'https://www.szse.cn/market/overview/index.html'
    raise ValueError('venue invalid')


def fetch(out,venue,day):
    url,params,referer=request_spec(venue,day)
    item={'venue':venue,'requested_date':day,'url':url,'parameters':params,
          'requested_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        r=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0','Referer':referer},timeout=(5,8))
        raw=out/(venue+'_'+day+'.response');raw.write_bytes(r.content)
        item.update(received_at_utc=datetime.now(timezone.utc).isoformat(),http_status=r.status_code,
            sha256=sha(raw),raw_path=raw.name,bytes=len(r.content),status='RECEIVED' if r.ok else 'HTTP_REJECTED')
    except requests.RequestException as e:
        item.update(status='UNAVAILABLE',error=str(e)[:220],finished_at_utc=datetime.now(timezone.utc).isoformat())
    return item


def run(out,days):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    calendar=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))
    if not days or len(days)>400 or len(days)!=len(set(days)) or not set(days).issubset(calendar['sessions']):
        raise ValueError('days must be unique official sessions, maximum 400')
    jobs=[(venue,day) for day in sorted(days) for venue in ['SSE','SZSE']]
    with ThreadPoolExecutor(max_workers=4) as executor:
        records=list(executor.map(lambda pair:fetch(out,*pair),jobs))
    report={'version':'v301-official-daily-turnover-receipts-1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'requested_dates':sorted(days),'requests':records,'calendar_sha256':sha(CALENDAR),
        'script_sha256':sha(__file__),'parsed_observations_authorized':False,
        'genuine_historical_availability_certified':False,'forward_evidence':False}
    write_json(out/'manifest.json',report)
    return [{'venue':r['venue'],'day':r['requested_date'],'status':r['status'],'bytes':r.get('bytes')} for r in records]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--days',nargs='+',required=True)
    a=p.parse_args();print(json.dumps(run(a.output,a.days),ensure_ascii=False))
