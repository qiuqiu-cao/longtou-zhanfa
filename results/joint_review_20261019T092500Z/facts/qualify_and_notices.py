"""Reconcile exact price rows, preserve source conflicts, and read issuer notices."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import requests
import fitz
import pandas as pd
import public_outcome_sources_v293 as public

OUT=Path(__file__).resolve().parent
DAY='2026-10-19'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def upper(c):return int((Decimal(c)*Decimal('1.1')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))


def notices(code):
    url='https://np-anotice-stock.eastmoney.com/api/security/ann'
    params=dict(sr='-1',page_size='20',page_index='1',ann_type='A',client_source='web',stock_list=code,begin_time='2026-10-14',end_time=DAY)
    result=dict(code=code,url=url,parameters=params,requested_at=now(),max_attempts=1)
    try:
        response=requests.get(url,params=params,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'})
        path=OUT/'notices'/(code+'_index.raw.json');path.write_bytes(response.content)
        result.update(available_at=now(),http_status=response.status_code,raw_path=str(path),raw_sha256=sha(path))
        response.raise_for_status();raw=response.json();items=(raw.get('data')or{}).get('list')
        if not isinstance(items,list):raise ValueError('index list unavailable')
        result.update(status='INDEX_RECEIVED_NO_CLEARANCE',items=[{k:item.get(k) for k in ['art_code','title','notice_date','codes','columns']} for item in items])
    except Exception as exc:result.update(status='UNAVAILABLE',error=str(exc)[:350])
    return result


def pdf_one(job):
    code,item=job;art=item['art_code'];url=f'https://pdf.dfcfw.com/pdf/H2_{art}_1.pdf'
    result=dict(code=code,title=item['title'],notice_date=item['notice_date'],url=url,requested_at=now(),max_attempts=1,
                source_kind='ISSUER_DOCUMENT_MIRROR_NOT_EXCHANGE_HOST_CERTIFICATION',first_publication_time=None)
    try:
        response=requests.get(url,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'});response.raise_for_status()
        if not response.content.startswith(b'%PDF'):raise ValueError('not PDF')
        path=OUT/'notices'/(code+'_'+art+'.pdf');path.write_bytes(response.content)
        with fitz.open(stream=response.content,filetype='pdf') as doc:
            text='\n'.join(p.get_text(sort=True) for p in doc)
        if len(text.strip())<80:raise ValueError('no readable text')
        txt=path.with_suffix('.txt');txt.write_text(text,encoding='utf-8')
        result.update(status='PDF_TEXT_RECEIVED',available_at=now(),pdf_path=str(path),text_path=str(txt),pdf_sha256=sha(path),text_sha256=sha(txt))
    except Exception as exc:result.update(status='UNAVAILABLE',error=str(exc)[:350],completed_at=now())
    return result


def main():
    pages={d:read(OUT/(d+'_complete_page_pools.json')) for d in ['2026-10-16',DAY]}
    daily=read(OUT/'normalized_daily.json');manifest=read(OUT/'daily_manifest.json')
    requests_by_code={r['code']:r for r in manifest['requests']}
    for code,record in daily.items():
        request=requests_by_code[code]
        if sha(record['raw_path'])!=request['raw_sha256']:raise ValueError('raw daily hash changed')
        reconstructed=public.parse_daily(read(record['raw_path']),code,'2026-09-18',DAY)
        if any(reconstructed[k]!=record[k] for k in ['code','days','quote','adjusted']):raise ValueError('raw daily reconstruction changed')
    spec=importlib.util.spec_from_file_location('quote_adapter',ROOT/'results/score_update_20261014T1153Z/supplemental_score_update.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);adapter.DAY=DAY
    rows=[];conflicts=[];source_proof={str(OUT/'normalized_daily.json'):sha(OUT/'normalized_daily.json'),str(OUT/'daily_manifest.json'):sha(OUT/'daily_manifest.json')}
    for r in pages[DAY]['rows']:
        if not r['scope']:continue
        code=r['code'];item=daily[code];bar=item['days'][DAY];before=item['days']['2026-10-16']
        amount,turnover,_=adapter.quote_fields(code,item)
        q=item['quote']['values'];cap=q['upper_candidate'];low=q['lower_candidate']
        status='CONSISTENT'
        if (bar['close']==cap)!=(r['kind']=='limit_up'):conflicts.append(dict(code=code,kind=r['kind'],reason='pool-close conflict'));status='CONFLICT'
        if r['kind']=='failed_board' and bar['high']!=cap:conflicts.append(dict(code=code,reason='no daily cap touch'));status='CONFLICT'
        if r['kind']=='limit_down' and bar['close']!=low:conflicts.append(dict(code=code,reason='not lower close'));status='CONFLICT'
        if abs(turnover-r['turnover_pct'])>.011:conflicts.append(dict(code=code,reason='turnover mismatch'));status='CONFLICT'
        streak=None
        if r['kind']=='limit_up' and r['boards']>=2:
            dates=['2026-10-15','2026-10-16',DAY] if r['boards']==3 else ['2026-10-16',DAY]
            calendar=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions']
            pairs=[(calendar[calendar.index(d)-1],d) for d in dates]
            streak=all(p in item['days'] and d in item['days'] and item['days'][d]['close']==upper(item['days'][p]['close']) for p,d in pairs)
        row=dict(r,close=bar['close']/100,open=bar['open']/100,high=bar['high']/100,low=bar['low']/100,
            previous_close=before['close']/100,change_pct=(bar['close']/before['close']-1)*100,
            amount_cny=amount,quote_turnover_pct=turnover,upper=cap/100,lower=low/100,
            one_price=bar['open']==bar['high']==bar['low']==bar['close'],
            volume_ratio_previous_official=bar['volume_native']/before['volume_native'] if before['volume_native'] else None,
            amplitude_pct=(bar['high']-bar['low'])/before['close']*100,
            consecutive_official_sessions_checked=streak,quote_observed_at=item['quote']['observed_at'],
            quote_amount_turnover_semantics='POSITIONAL_INFERRED_CROSSCHECKED_NOT_EXCHANGE_CERTIFIED',qualification=status)
        rows.append(row);source_proof[item['raw_path']]=sha(item['raw_path'])
    primary=read(OUT/'2026-10-19_failed_board_primary.json')
    page_codes={r['code'] for r in rows if r['kind']=='failed_board'}
    primary_codes={r['code'] for r in primary['rows']}
    extra=[]
    for r in primary['rows']:
        if r['code'] not in page_codes:
            extra.append(dict(r,source_status='PRIMARY_ONLY_CURRENT_DAILY_NOT_IN_INITIAL_REQUEST_CODES_RETAIN_UNKNOWN'))
    qualified=dict(formed_at=now(),signal_date=DAY,target_date='2026-10-20',rows=rows,full_mainboard_page_counts={k:sum(r['kind']==k for r in rows) for k in ['limit_up','failed_board','limit_down']},
        relay=[r for r in rows if r['kind']=='limit_up' and r['boards']>=2],conflicts=conflicts,
        failed_source_discrepancy=dict(primary_only=extra,page_only=sorted(page_codes-primary_codes),not_silently_merged=True),
        exact_daily_received=158,source_bindings=source_proof|{pages[DAY]['source_path']:pages[DAY]['source_sha256']},
        raw_AI_business_text_is_not_factual_evidence=True,source_wide_completeness=False,all_original_candidates_retained=True,
        model_outputs_masked=True,new_fits=0)
    write(OUT/'qualified_current_facts_SCORES_MASKED.json',qualified)
    pd.DataFrame(rows).to_csv(OUT/'完整收盘首板炸板跌停事实_遮分.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(counts=qualified['full_mainboard_page_counts'],relay=len(qualified['relay']),conflicts=conflicts,extra_failed=extra),ensure_ascii=False))
    (OUT/'notices').mkdir(exist_ok=False)
    codes=sorted({r['code'] for r in rows if r['kind']=='limit_up'}|{'600825'})
    with ThreadPoolExecutor(max_workers=6) as executor:indexes=list(executor.map(notices,codes))
    write(OUT/'notices/index_manifest.json',dict(formed_at=now(),rows=indexes,full_closing_candidates=len(codes)))
    # Focus document retrieval on every relay, meaningful new first-board branch,
    # the old highest anchor and potential formal risk announcements.
    focus={r['code'] for r in qualified['relay']}|{'600825','002343','003021','603119','603286','605488','600487','605255','002886'}
    jobs=[]
    for entry in indexes:
        if entry.get('status')!='INDEX_RECEIVED_NO_CLEARANCE':continue
        items=entry['items']
        selected=[x for x in items if any(word in x['title'] for word in ['异常波动','风险提示','业绩预告','业绩情况','立案','停牌','复牌'])]
        if entry['code'] in focus and not selected:selected=items[:1]
        for item in selected[:2]:jobs.append((entry['code'],item))
    with ThreadPoolExecutor(max_workers=6) as executor:pdfs=list(executor.map(pdf_one,jobs))
    write(OUT/'notices/fulltext_manifest.json',dict(formed_at=now(),rows=pdfs,max_attempts_per_document=1))
    print(json.dumps(dict(indexes_received=sum(x['status']=='INDEX_RECEIVED_NO_CLEARANCE' for x in indexes),pdfs_received=sum(x['status']=='PDF_TEXT_RECEIVED' for x in pdfs),documents=[(x['code'],x['title'],x['status']) for x in pdfs]),ensure_ascii=False))


if __name__=='__main__':main()
