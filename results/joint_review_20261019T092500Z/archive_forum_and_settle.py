"""Append source receipts and evaluate elapsed windows; never rewrite old calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from bs4 import BeautifulSoup
import relay_multiday_scenarios as multiday
OUT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def cap(c):return int((Decimal(c)*Decimal('1.10')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
def fetch(job):
 key,url=job;r=dict(key=key,url=url,requested_at=now(),max_attempts=1)
 try:
  resp=requests.get(url,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'})
  p=OUT/'forum'/(key+'.raw.html');p.write_bytes(resp.content)
  r.update(received_at=now(),http_status=resp.status_code,raw_path=str(p.resolve()),sha256=sha(p));resp.raise_for_status()
  soup=BeautifulSoup(resp.content,'html.parser')
  for tag in soup(['script','style','nav']):tag.decompose()
  text=soup.get_text('\n',strip=True);q=p.with_suffix('.txt');q.write_text(text,encoding='utf-8')
  r.update(status='TEXT_CAPTURED_REQUIRES_AUTHOR_AND_TIME_REVIEW',text_path=str(q.resolve()),text_sha256=sha(q))
 except Exception as e:r.update(status='UNAVAILABLE',error=str(e)[:400],completed_at=now())
 return r
def metrics(rows,field):
 judged=[r for r in rows if r.get(field) is not None and r.get('actual_close_limit_up') is not None]
 tp=sum(r[field] and r['actual_close_limit_up'] for r in judged);fp=sum(r[field] and not r['actual_close_limit_up'] for r in judged)
 fn=sum(not r[field] and r['actual_close_limit_up'] for r in judged);tn=len(judged)-tp-fp-fn
 positives=sum(r['actual_close_limit_up'] is True for r in rows)
 return dict(full_count=len(rows),judged_count=len(judged),abstained=len(rows)-len(judged),TP=tp,FP=fp,FN=fn,TN=tn,accuracy=(tp+tn)/len(judged) if judged else None,
  full_pool_recall=tp/positives if positives else None,precision=tp/(tp+fp) if tp+fp else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
def main():
 (OUT/'forum').mkdir(exist_ok=True)
 jobs=[('linyao_1016','https://m.tgb.cn/a/2vWrtLGJL9K'),('ale_1019','https://caifuhao.eastmoney.com/news/20261019053500445417840?from=guba'),('counterpoint_1016','https://www.tgb.cn/a/2vWNJbv2XAs')]
 if (OUT/'forum/limited_capture_receipt.json').exists():received=read(OUT/'forum/limited_capture_receipt.json')['requests']
 else:
  received=list(ThreadPoolExecutor(max_workers=3).map(fetch,jobs));write(OUT/'forum/limited_capture_receipt.json',dict(formed_at=now(),requests=received,new_training_truth=False))
 daily=read(OUT/'facts/normalized_daily.json');facts=read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json')
 item=daily['603337'];a=item['days']['2026-10-19'];q=item['quote']['values']
 correction=dict(formed_at=now(),code='603337',name='杰克科技',original_annotation='PRIMARY_ONLY_CURRENT_DAILY_NOT_IN_INITIAL_REQUEST_CODES_RETAIN_UNKNOWN',
  correction='EXACT_CURRENT_DAILY_IS_PRESENT; PRIMARY_FAILED_ASSIGNMENT_CONFLICTS_WITH_CLOSE_AT_UPPER; ALTERNATIVE_PAGE_OMITS_STOCK',daily=a,quote=q,
  actual_price_close_limit_up=a['close']==q['upper_candidate'],pool_assignment='SOURCE_CONFLICT_RETAIN_OUTSIDE_39_PAGE_COHORT',
  original_file_preserved=True,source_binding={str((OUT/'facts/qualified_current_facts_SCORES_MASKED.json').resolve()):sha(OUT/'facts/qualified_current_facts_SCORES_MASKED.json'),str(Path(item['raw_path']).resolve()):sha(item['raw_path'])})
 if not (OUT/'facts/603337_source_conflict_APPEND_CORRECTION.json').exists():write(OUT/'facts/603337_source_conflict_APPEND_CORRECTION.json',correction)
 old=read(ROOT/'results/daily_review_20261015T093600Z/independent_forward_views_BEFORE_MODEL.json')
 scores=read(ROOT/'results/daily_review_20261015T093600Z/BASE_scores_AFTER_INDEPENDENT_REVIEW.json');sm={r['code']:r for r in scores['rows']}
 rows=[]
 for r in old['rows']:
  code=r['code'];before=daily[code]['days']['2026-10-15'];after=daily[code]['days']['2026-10-16'];upper=cap(before['close']);call=r.get('direction',r.get('independent_direction',r.get('promotion_direction')))
  # Preserve exact author vocabulary; no UNKNOWN-to-negative conversion.
  if call is None:
   for k in ['promotion_judgement','next_close_limit_up_call','promotion_tendency','promotion_call','next_day_direction']:
    if k in r:call=r[k];break
  if call is None:raise ValueError('inspect original call vocabulary before settlement: '+str(r.keys()))
  pred=True if call in ['PROMOTE','YES','偏晋级'] else False if call in ['FAIL','NO','偏断板'] else None
  rows.append(dict(code=code,name=r['name'],original_direction=call,own_predicted=pred,BASE_predicted=sm[code]['predicted_positive'],signal_close=before['close']/100,target_close=after['close']/100,target_high=after['high']/100,
    upper_price=upper/100,actual_close_limit_up=after['close']==upper,actual_intraday_touch=after['high']==upper,return_pct=(after['close']/before['close']-1)*100,
    source='EXACT_CENT_UNADJUSTED_RAW_DAILY_10PCT_CAP_CROSSCHECK_NOT_EXCHANGE_ATTESTATION'))
 write(OUT/'settlement/15_to16_full6_APPEND_ONLY.json',dict(formed_at=now(),rows=rows,own=metrics(rows,'own_predicted'),BASE=metrics(rows,'BASE_predicted'),new_formal_labels=0,old_records_changed=False,source_bindings={str((ROOT/'results/daily_review_20261015T093600Z/independent_forward_views_BEFORE_MODEL.json').resolve()):sha(ROOT/'results/daily_review_20261015T093600Z/independent_forward_views_BEFORE_MODEL.json'),str((OUT/'facts/normalized_daily.json').resolve()):sha(OUT/'facts/normalized_daily.json')}))
 record=ROOT/'research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z/current_forecast/15日收盘_16日优先与2_3_5日情景_已看模型.json'
 bars=pd.DataFrame([dict(code=c,date=d,**{k:r[k]/100 for k in ['open','high','low','close']},volume=r['volume_native']) for c,it in daily.items() for d,r in it['days'].items()])
 sessions=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions']
 result=multiday.evaluate(read(record),bars,sessions,now());result['source_bindings']={str(record.resolve()):sha(record),str((OUT/'facts/normalized_daily.json').resolve()):sha(OUT/'facts/normalized_daily.json')};write(OUT/'settlement/15_multiday_mature_asof19_APPEND_ONLY.json',result)
 pool16=read(OUT/'facts/2026-10-16_complete_page_pools.json');nexts={r['code']:r for r in facts['relay']};rr=[]
 for r in pool16['rows']:
  if r['kind']!='limit_up' or not r['scope'] or r['boards']<2:continue
  prev=daily[r['code']]['days']['2026-10-16'];cur=daily[r['code']]['days']['2026-10-19'];upper=cap(prev['close'])
  rr.append(dict(code=r['code'],name=r['name'],reported_signal_boards=r['boards'],suspension_gap=r['code']=='600825',reported_target_boards=nexts.get(r['code'],{}).get('boards'),close19=cur['close']/100,high19=cur['high']/100,upper19=upper/100,actual_close_limit_up=cur['close']==upper,actual_touch=cur['high']==upper))
 write(OUT/'settlement/16_to19_full7_APPEND_ONLY.json',dict(formed_at=now(),rows=rr,full_count=len(rr),promoted=sum(r['actual_close_limit_up'] for r in rr),new_formal_maturity=0,original_frozen_records_changed=False))
 print(json.dumps(dict(own=metrics(rows,'own_predicted'),BASE=metrics(rows,'BASE_predicted'),multiday_known=result['new_mature_labels'],prior16_pool=len(rr),forum=[r['status'] for r in received]),ensure_ascii=False))
if __name__=='__main__':main()
