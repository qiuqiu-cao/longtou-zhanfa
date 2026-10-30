"""Fresh bounded exact-date collection; old models and artifacts read-only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'results/joint_review_20261019T092500Z'
spec=importlib.util.spec_from_file_location('prior_close_collector',OLD/'facts/collect_close.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.OUT=OUT/'facts'
DAY='2026-10-27'

def daily_one(code):
    symbol=('sh' if code.startswith('6') else 'sz')+code
    r=c.fetch((code,'daily',c.public.DAY_URL,{'param':f'{symbol},day,2026-09-18,{DAY},100,'}))
    r['code']=code;r['kind']='daily';r.pop('date',None);data=None
    if r['status']=='RAW_RECEIVED':
        try:
            data=c.public.parse_daily(json.loads(Path(r['raw_path']).read_bytes()),code,'2026-09-18',DAY)
            data.update(available_at=r['available_at'],raw_path=r['raw_path'])
        except Exception as e:r.update(parse_status='UNAVAILABLE',parse_error=str(e)[:300])
    return r,data

def main():
    (c.OUT/'raw').mkdir(parents=True,exist_ok=False)
    jobs=[(DAY,'alternative','https://toupage.com/stocks/limit-up',{'date':DAY})]
    for kind,endpoint in c.public.ENDPOINTS.items():
        jobs.append((DAY,kind,c.public.EM_BASE+endpoint,dict(ut='7eea3edcaed734bea9cbfc24409ed989',dpt='wz.ztzt',Pageindex='0',pagesize='10000',sort='fbt:asc' if kind in {'limit_up','failed_board'} else 'fund:asc' if kind=='limit_down' else 'zs:desc',date=DAY.replace('-',''))))
    with ThreadPoolExecutor(max_workers=5) as e:records=list(e.map(c.fetch,jobs))
    pages={};primary={};failures=[]
    for r in records:
        try:
            if r['kind']=='alternative':
                v=c.parse_page(r);pages[DAY]=v;c.write(c.OUT/(DAY+'_complete_page_pools.json'),v)
            elif r['status']=='RAW_RECEIVED':
                v=c.public.parse_pool(json.loads(Path(r['raw_path']).read_bytes()),DAY,r['kind'])
                primary[r['kind']]=v;c.write(c.OUT/(DAY+'_'+r['kind']+'_primary.json'),v)
            else:raise ValueError(r.get('error','unavailable'))
        except Exception as x:failures.append(dict(kind=r['kind'],error=str(x)[:350]))
    oldpage=json.loads((ROOT/'results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json').read_text(encoding='utf-8'))
    codes={r['code'] for p in pages.values() for r in p['rows'] if r['scope']}
    codes|={r['code'] for r in oldpage['rows']}
    codes|={'002185','600825','600584','002156','000529','002696','002242','001318','000530','601811','605189'}
    for v in primary.values():codes|={r['code'] for r in v['rows']}
    c.write(c.OUT/'pool_capture_manifest.json',dict(formed_at=c.now(),requests=records,failures=failures,pages_qualified=list(pages),primary_qualified=list(primary),prior_page=str(ROOT/'results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json'),requested_daily_codes=sorted(codes),max_attempts=1))
    print(json.dumps(dict(stage='POOLS_READY',page_counts={d:v['raw_pool_counts'] for d,v in pages.items()},ladder={d:v['ladder'] for d,v in pages.items()},failures=failures,requested_codes=len(codes)),ensure_ascii=False),flush=True)
    normalized={};requests=[]
    with ThreadPoolExecutor(max_workers=6) as e:
        for r,d in e.map(daily_one,sorted(codes)):
            requests.append(r)
            if d:normalized[r['code']]=d
    c.write(c.OUT/'normalized_daily.json',normalized)
    c.write(c.OUT/'daily_manifest.json',dict(formed_at=c.now(),requests=requests,requested_codes=sorted(codes),daily_start='2026-09-18',daily_end=DAY,adjusted=False,max_attempts_per_code=1,exact_day=sum(DAY in d['days'] for d in normalized.values())))
    print(json.dumps(dict(page_counts={d:v['raw_pool_counts'] for d,v in pages.items()},ladder={d:v['ladder'] for d,v in pages.items()},failures=failures,requested=len(codes),daily_received=len(normalized),exact_day=sum(DAY in d['days'] for d in normalized.values())),ensure_ascii=False))

if __name__=='__main__':main()

