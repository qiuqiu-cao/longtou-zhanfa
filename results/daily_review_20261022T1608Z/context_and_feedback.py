from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,ROUND_HALF_UP
import importlib.util,json,hashlib,sys
import requests
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json');d=read(OUT/'facts/normalized_daily.json')
raw=read(OUT/'facts/raw/2026-10-22_limit_up.json')['data']['pool'];lut={x['c']:x for x in raw}
for row in f['rows']:
 z=lut.get(row['code'])
 if z:
  past=[v for k,v in sorted(d.get(row['code'],{}).get('days',{}).items()) if k<'2026-10-22'];cur=d.get(row['code'],{}).get('days',{}).get('2026-10-22')
  mean=sum(v['volume_native'] for v in past[-5:])/5 if len(past)>=5 else None
  ratio=cur['volume_native']/mean if cur and mean else None
  thin=z['ltsz']<=3e9 and z['hs']<5 and (row['one_price'] is True or ratio is not None and ratio<1)
  row.update(reported_float_cap_cny=z['ltsz'],reported_total_cap_cny=z['tshare'],primary_amount_cny=z['amount'],volume_ratio_previous5=ratio,
    original_cap_screen=z['ltsz']<=9326859896.52,thin_proxy_excluded=thin,screen_status='EXCLUDED_CAP' if z['ltsz']>9326859896.52 else 'EXCLUDED_THIN_PROXY' if thin else 'WITHIN_ORIGINAL_RESEARCH_SCREEN',risk_proxy_not_fund_identity=True)
write(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json',f)
old=read(ROOT/'results/joint_review_20261021T084800Z/facts/qualified_full_facts_SCORES_MASKED.json');current={x['code']:x for x in f['rows']}
feedback=[]
for x in old['rows']:
 if x['kind']!='limit_up':continue
 ds=d.get(x['code'],{}).get('days',{});a=ds.get('2026-10-21');b=ds.get('2026-10-22')
 feedback.append(dict(code=x['code'],name=x['name'],prior_boards=x['boards'],promoted=int(current.get(x['code'],{}).get('kind')=='limit_up') if a and b else None,
  return_pct=(b['close']/a['close']-1)*100 if a and b else None,open_gap_pct=(b['open']/a['close']-1)*100 if a and b else None))
layers={}
for h in [1,2,3,4]:
 q=[x for x in feedback if x['prior_boards']==h];k=[x for x in q if x['promoted'] is not None]
 layers[str(h)]=dict(total=len(q),known=len(k),unknown=len(q)-len(k),promoted=sum(x['promoted'] for x in k),mean_return_pct=sum(x['return_pct'] for x in k)/len(k) if k else None,loss7_count=sum(x['return_pct']<=-7 for x in k))
write(OUT/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json',dict(formed_at=now(),prior_date='2026-10-21',signal_date='2026-10-22',layers=layers,rows=feedback,missing_prices_not_zero=True))
# Mature original top choices transparently; never modify the previous forecast.
previous=read(ROOT/'results/joint_review_20261021T084800Z/JOINT_FINAL_PREDICTION_BEFORE_NEXT_SESSION.json')
settled=[]
for code in previous['top3']:
 x=next(v for v in feedback if v['code']==code)
 settled.append(dict(x,old_forecast_unchanged=True))
write(OUT/'facts/PRIOR_TOP3_SETTLEMENT_APPEND_ONLY.json',dict(formed_at=now(),source=str(ROOT/'results/joint_review_20261021T084800Z/JOINT_FINAL_PREDICTION_BEFORE_NEXT_SESSION.json'),top3=settled,correct_known=sum(x['promoted'] for x in settled if x['promoted'] is not None),complete_slots=3,multi_day_windows_not_all_mature=True))
jobs=[('forum_sheng','https://caifuhao.eastmoney.com/news/20261022001743609853790'),('forum_kunlun','https://caifuhao.eastmoney.com/news/20261021200207344442080'),('forum_panqian','https://www.jiuyangongshe.com/a/4eqptks4rm8')]
def article(job):
 key,url=job;r=dict(key=key,url=url,requested_at=now(),attempts=1)
 try:
  response=requests.get(url,timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'});response.raise_for_status()
  path=OUT/'context'/(key+'.html');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content)
  soup=BeautifulSoup(response.content,'html.parser');text=soup.get_text('\n',strip=True);txt=path.with_suffix('.txt');txt.write_text(text,encoding='utf-8')
  r.update(status='RAW_TEXT_RECEIVED_BODY_REQUIRES_REVIEW',available_at=now(),raw_path=str(path),raw_sha256=sha(path),text_path=str(txt))
 except Exception as e:r.update(status='UNAVAILABLE',error=str(e)[:300])
 return r
with ThreadPoolExecutor(max_workers=3) as e:articles=list(e.map(article,jobs))
write(OUT/'context/limited_forum_capture.json',dict(formed_at=now(),records=articles,body_comments_time_separate=True,no_popularity_or_author_identity_certifies_accuracy=True))
spec=importlib.util.spec_from_file_location('notice_helpers',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py');n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=OUT/'context';n.DAY='2026-10-22';(n.OUT/'notices').mkdir(parents=True,exist_ok=True)
with ThreadPoolExecutor(max_workers=6) as e:indexes=list(e.map(n.notices,['603278','603028','603280','600261','002724','600293']))
write(OUT/'context/notices/index_manifest.json',dict(formed_at=now(),records=indexes,limited_to_focus_and_execution_risk=True))
selected=[]
for entry in indexes:
 for item in [z for z in entry.get('items',[]) if any(s in z['title'] for s in ['停牌','复牌','立案','风险提示','异常波动'])][:1]:selected.append((entry['code'],item))
with ThreadPoolExecutor(max_workers=4) as e:docs=list(e.map(n.pdf_one,selected))
write(OUT/'context/notices/documents_manifest.json',dict(formed_at=now(),records=docs,empty_index_does_not_certify_clearance=True))
print(json.dumps(dict(layers=layers,prior_top3=settled,forum=[{'key':x['key'],'status':x['status']} for x in articles],notices=[{'code':x['code'],'status':x['status'],'items':[z['title'] for z in x.get('items',[])]} for x in indexes],documents=[{'code':x['code'],'title':x['title'],'status':x['status']} for x in docs]),ensure_ascii=True))
