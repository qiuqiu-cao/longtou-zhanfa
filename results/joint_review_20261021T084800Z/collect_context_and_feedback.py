from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.util,json,hashlib,sys
import pandas as pd

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('context_helpers',ROOT/'results/daily_review_20261020T101630Z/capture_context.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.OUT=OUT
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

jobs=[('forum_xiaoqi20','https://m.tgb.cn/a/2w31Z4KV77h'),('forum_cike15','https://m.tgb.cn/a/2vUNIn9W0UH'),('forum_ayi20','https://www.jiuyangongshe.com/a/3y8n00z1uzb'),('market_close','https://finance.eastmoney.com/a/202610213902615867.html'),('limit_close','https://finance.eastmoney.com/a/202610213902596617.html')]
with ThreadPoolExecutor(max_workers=5) as e:records=list(e.map(c.article,jobs))
write(OUT/'context/capture_manifest.json',{'formed_at':c.now(),'records':records,'forum_scope':'Three selected originals only. Bounded date/author searches did not yield a qualified Oct21 closing original. Oct20/Oct15 bodies remain dated background; no later comments inherit main-post time. Search snippets and raw bodies may differ; not a complete forum survey.'})
facts=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json');daily=read(OUT/'facts/normalized_daily.json')
old=read(ROOT/'results/daily_review_20261020T101630Z/facts/qualified_full_facts_SCORES_MASKED.json')
raw=read(OUT/'facts/raw/2026-10-21_limit_up.json')['data']['pool'];lookup={r['c']:r for r in raw}
enriched=[]
for r in facts['rows']:
 v=dict(r);z=lookup.get(r['code'])
 if z:v.update(reported_float_cap_cny=z['ltsz'],reported_total_cap_cny=z['tshare'],screen_cap_9326859896_52=z['ltsz']<=9326859896.52,reported_rolling_pattern=z['zttj'],pool_seal_field_status='PROVIDER_REPORTED_NO_LEVEL2_OR_QUEUE_VALIDATION')
 enriched.append(v)
write(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json',dict(facts,rows=enriched,derived_enrichment_at=c.now()))
pd.DataFrame([{k:v for k,v in r.items() if k!='source_rows'} for r in enriched]).to_csv(OUT/'facts/完整67只_事实与市值_遮分.csv',index=False,encoding='utf-8-sig')
feedback=[]
cur={r['code']:r for r in facts['rows']}
for r in old['rows']:
 if r['kind']!='limit_up':continue
 bars=daily.get(r['code'],{}).get('days',{});a=bars.get('2026-10-20');b=bars.get('2026-10-21')
 if not a or not b:ret=None;win=None;gap=None
 else:
  ret=(b['close']/a['close']-1)*100;gap=(b['open']/a['close']-1)*100
  win=int(cur.get(r['code'],{}).get('kind')=='limit_up')
 feedback.append(dict(code=r['code'],name=r['name'],prior_boards=r['boards'],return_pct=ret,open_gap_pct=gap,promoted=win))
layers={}
for layer in [1,2,3,4]:
 rows=[r for r in feedback if r['prior_boards']==layer];known=[r for r in rows if r['promoted'] is not None]
 layers[str(layer)]={'total':len(rows),'known':len(known),'unknown':len(rows)-len(known),'promoted':sum(r['promoted'] for r in known),'mean_close_return_pct':sum(r['return_pct'] for r in known)/len(known) if known else None,'loss7_count':sum(r['return_pct']<=-7 for r in known)}
write(OUT/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json',dict(formed_at=c.now(),signal_date='2026-10-21',prior_date='2026-10-20',layers=layers,rows=feedback,current_mainboard_seal_rate=44/(44+18),all_returned_seal_rate=50/(50+19),current_ladder={str(h):sum(r.get('boards')==h for r in facts['rows'] if r['kind']=='limit_up') for h in [1,2,3,4]},complete_prior_mainboard_cohort=True))
# Limited issuer index check for focus stock; no broad corporate-action research.
nsp=importlib.util.spec_from_file_location('issuer_helpers',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
n=importlib.util.module_from_spec(nsp);nsp.loader.exec_module(n);n.OUT=OUT/'context';n.DAY='2026-10-21';(n.OUT/'notices').mkdir(exist_ok=True)
idx=n.notices('002086');write(OUT/'context/东方海洋_公告索引.json',idx)
selected=[x for x in idx.get('items',[]) if any(w in x['title'] for w in ['停牌','复牌','立案','异常','风险','业绩','处罚'])][:2]
with ThreadPoolExecutor(max_workers=2) as e:docs=list(e.map(n.pdf_one,[('002086',x) for x in selected]))
write(OUT/'context/东方海洋_公告正文收据.json',{'formed_at':c.now(),'docs':docs,'empty_index_is_not_full_clearance':True})
print(json.dumps({'layers':layers,'context_records':[{k:r.get(k) for k in ['key','status','characters','text_path']} for r in records],'notice_index':idx.get('items'),'notice_docs':docs},ensure_ascii=False))
