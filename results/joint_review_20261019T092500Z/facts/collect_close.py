"""Bounded exact-date public closing receipts for the joint review, no fits."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import requests
from bs4 import BeautifulSoup
import public_outcome_sources_v293 as public

OUT=Path(__file__).resolve().parent
DATES=['2026-10-16','2026-10-19']


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def fetch(job):
    date,kind,url,params=job
    path=OUT/'raw'/(date+'_'+kind+('.html' if kind=='alternative' else '.json'))
    record=dict(date=date,kind=kind,url=url,parameters=params,requested_at=now(),max_attempts=1)
    try:
        response=requests.get(url,params=params,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'})
        path.write_bytes(response.content)
        record.update(available_at=now(),http_status=response.status_code,raw_path=str(path),raw_sha256=sha(path))
        response.raise_for_status()
        record['status']='RAW_RECEIVED'
    except Exception as exc:record.update(status='UNAVAILABLE',error=str(exc)[:450],completed_at=now())
    return record


def parse_page(record):
    if record['status']!='RAW_RECEIVED':raise ValueError('page unavailable')
    path=Path(record['raw_path']);date=record['date']
    body=BeautifulSoup(path.read_bytes(),'html.parser')
    selected=body.select_one('#limitUpDateInput')
    if not selected or selected.get('value')!=date:raise ValueError('wrong selected date')
    text=body.get_text(' ',strip=True)
    stamp=re.search(re.escape(date)+r'\s+(\d\d:\d\d)',text)
    if not stamp or stamp.group(1)<'15:00':raise ValueError('page is an intraday snapshot, not closing')
    tables=body.find_all('table')
    if len(tables)!=3:raise ValueError('expected complete three page tables')
    ladder={}
    for group in body.select('[data-ladder-height]'):
        for anchor in group.select('article a'):
            code=re.search(r'/([0-9]{6})/',anchor['href']).group(1)
            if code in ladder:raise ValueError('duplicate ladder')
            ladder[code]=int(group['data-ladder-height'])
    rows=[];counts={}
    for kind,table in zip(['limit_up','failed_board','limit_down'],tables):
        seen=set();counts[kind]=len(table.find_all('tr'))-1
        for index,tr in enumerate(table.find_all('tr')[1:]):
            cells=tr.find_all('td');anchor=cells[0].find('a')
            code=re.search(r'/([0-9]{6})/',anchor['href']).group(1);name=anchor.get_text(strip=True)
            if code in seen:raise ValueError('duplicate pool code')
            seen.add(code)
            scope=code.startswith(public.MAINBOARD) and 'ST' not in name.upper() and '退' not in name
            summary=cells[2].find('summary');summary=summary.get_text(' ',strip=True) if summary else cells[2].get_text(' ',strip=True)
            rows.append(dict(date=date,code=code,name=name,kind=kind,boards=ladder.get(code,1) if kind=='limit_up' else None,
                streak_scope='EXPLICIT_PAGE_LADDER_NOT_ROLLING_X_DAYS_Y_BOARDS',row=index,scope=scope,
                reported_pattern=cells[1].get_text(' ',strip=True),reason_summary_UNVERIFIED=summary,
                turnover_pct=float(tr.get('data-sort-turnover','nan')),reported_times=cells[3].get_text(' ',strip=True),
                reported_open_count=cells[4].get_text(' ',strip=True),reported_seal=cells[5].get_text(' ',strip=True),
                raw_row_sha256=hashlib.sha256(str(tr).encode()).hexdigest()))
    return dict(date=date,snapshot_time=stamp.group(1),ladder=ladder,rows=rows,raw_pool_counts=counts,
        source_path=str(path),source_sha256=sha(path),available_at=record['available_at'],
        scope='COMPLETE_RETURNED_ALTERNATIVE_PAGE_NOT_EXCHANGE_WIDE_OR_SECURITY_CERTIFICATION',
        business_summary_is_unverified_AI_text=True)


def daily_one(code):
    symbol=('sh' if code.startswith('6') else 'sz')+code
    record=fetch((code,'daily',public.DAY_URL,{'param':f'{symbol},day,2026-09-18,2026-10-19,100,'}))
    record['code']=code;record['kind']='daily';record.pop('date',None)
    data=None
    if record['status']=='RAW_RECEIVED':
        try:
            data=public.parse_daily(json.loads(Path(record['raw_path']).read_bytes()),code,'2026-09-18','2026-10-19')
            data.update(available_at=record['available_at'],raw_path=record['raw_path'])
        except Exception as exc:record.update(parse_status='UNAVAILABLE',parse_error=str(exc)[:400])
    return record,data


def main():
    (OUT/'raw').mkdir(exist_ok=False)
    jobs=[]
    for date in DATES:
        jobs.append((date,'alternative','https://toupage.com/stocks/limit-up',{'date':date}))
        for kind,endpoint in public.ENDPOINTS.items():
            jobs.append((date,kind,public.EM_BASE+endpoint,dict(ut='7eea3edcaed734bea9cbfc24409ed989',dpt='wz.ztzt',
                Pageindex='0',pagesize='10000',sort='fbt:asc' if kind in {'limit_up','failed_board'} else 'fund:asc' if kind=='limit_down' else 'zs:desc',date=date.replace('-',''))))
    with ThreadPoolExecutor(max_workers=5) as executor:records=list(executor.map(fetch,jobs))
    pages={};primary={};failures=[]
    for record in records:
        try:
            if record['kind']=='alternative':
                result=parse_page(record);pages[record['date']]=result
                write(OUT/(record['date']+'_complete_page_pools.json'),result)
            elif record['status']=='RAW_RECEIVED':
                result=public.parse_pool(json.loads(Path(record['raw_path']).read_bytes()),record['date'],record['kind'])
                primary[(record['date'],record['kind'])]=result
                write(OUT/(record['date']+'_'+record['kind']+'_primary.json'),result)
            else:raise ValueError(record.get('error','unavailable'))
        except Exception as exc:failures.append(dict(date=record['date'],kind=record['kind'],error=str(exc)[:450]))
    codes={r['code'] for p in pages.values() for r in p['rows'] if r['scope']}
    codes|={'000530','001318','002343','002696','601811','605189','600825','002242'}
    for pool in primary.values():codes|={r['code'] for r in pool['rows']}
    write(OUT/'pool_capture_manifest.json',dict(formed_at=now(),requests=records,failures=failures,
        pages_qualified=sorted(pages),primary_qualified=[list(k) for k in primary],requested_daily_codes=sorted(codes),max_requests_per_endpoint=1))
    normalized={};daily_records=[]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for record,data in executor.map(daily_one,sorted(codes)):
            daily_records.append(record)
            if data:normalized[record['code']]=data
    write(OUT/'normalized_daily.json',normalized)
    write(OUT/'daily_manifest.json',dict(formed_at=now(),requests=daily_records,requested_codes=sorted(codes),
        daily_start='2026-09-18',daily_end='2026-10-19',adjusted=False,
        max_attempts_per_code=1,successful_raw_reconstructions=len(normalized),
        exact19_count=sum('2026-10-19' in x['days'] for x in normalized.values()),
        source_bindings={str(Path(r['raw_path'])):r['raw_sha256'] for r in records+daily_records if r.get('raw_path')}))
    print(json.dumps(dict(pages={d:dict(counts=x['raw_pool_counts'],ladder=x['ladder'],time=x['snapshot_time']) for d,x in pages.items()},
                         requested_codes=len(codes),daily_received=len(normalized),failures=failures),ensure_ascii=False))


if __name__=='__main__':main()
