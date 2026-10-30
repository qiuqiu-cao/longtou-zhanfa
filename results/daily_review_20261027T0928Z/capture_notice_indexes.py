from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util,json
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('notice_helper',ROOT/'results/joint_review_20261019T092500Z/facts/qualify_and_notices.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
n.OUT=OUT/'context';n.DAY='2026-10-28';(n.OUT/'notices').mkdir(parents=True,exist_ok=True)
p=json.loads((OUT/'facts/2026-10-27_complete_page_pools.json').read_text(encoding='utf-8-sig'))
codes=sorted({x['code'] for x in p['rows'] if x['kind']=='limit_up' and x['scope']}|{'603278','002724','603677','002009','603081','603280','603933'})
with ThreadPoolExecutor(max_workers=6) as e:records=list(e.map(n.notices,codes))
n.write(n.OUT/'notices/index_manifest.json',dict(formed_at=n.now(),signal_date='2026-10-27',end_date_requested=n.DAY,post_close_information_separate=True,rows=records))
print('NOTICE_INDEXES',len(records),'received',sum(r['status']=='INDEX_RECEIVED_NO_CLEARANCE' for r in records))

