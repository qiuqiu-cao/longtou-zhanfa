from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util,json
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('notice_helper',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=OUT/'context';n.DAY='2026-10-24';(n.OUT/'notices').mkdir(parents=True,exist_ok=True)
p=json.loads((OUT/'facts/2026-10-23_complete_page_pools.json').read_text(encoding='utf-8-sig'))
codes=sorted({x['code'] for x in p['rows'] if x['kind']=='limit_up' and x['scope']})
with ThreadPoolExecutor(max_workers=6) as e:records=list(e.map(n.notices,codes))
n.write(n.OUT/'notices/index_manifest.json',dict(formed_at=n.now(),signal_date='2026-10-23',end_date_requested=n.DAY,post_close_information_separate=True,rows=records))
focus={'603278','002724','600838','603028','603280','603933','603089','603528','002535','000678','600215','603081','603677','002026','605488','002354','002343'}
for r in records:
 items=[x for x in r.get('items',[]) if str(x.get('notice_date',''))>='2026-10-22' or any(w in x.get('title','') for w in ['立案','停牌','监管','风险提示','异动'])]
 if r['code'] in focus or any(any(w in x.get('title','') for w in ['立案','停牌','退市','监管']) for x in items):
  print(json.dumps({'code':r['code'],'status':r['status'],'items':[{k:x.get(k) for k in ['art_code','title','notice_date']} for x in items]},ensure_ascii=False))
