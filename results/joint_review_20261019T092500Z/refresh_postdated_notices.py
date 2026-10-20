"""Read currently available notices dated tomorrow; real receipt is today.

An evening issuer release can bear the next official day's cover date. It is
not valid to declare no new news from an index filtered to today's cover date.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
OUT=Path(__file__).resolve().parent
def main():
 codes=sorted({r['code'] for r in b.read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json')['relay']}|{'600487'})
 dest=OUT/'facts/evening_postdated20_currently_public';(dest/'notices').mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('evening_postdated_notices',OUT/'facts/qualify_and_notices.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=dest;m.DAY='2026-10-20'
 old=b.read(OUT/'facts/notices/index_manifest.json')['rows']+b.read(OUT/'facts/evening_refresh_20261019T142830Z/notices/index_manifest.json')['rows'];known={}
 for r in old:known.setdefault(r['code'],set()).update(a['art_code'] for a in r.get('items',[]))
 with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(m.notices,codes))
 delta=[];jobs=[]
 for r in rows:
  items=[a for a in r.get('items',[]) if a['art_code'] not in known.get(r['code'],set())]
  delta.append(dict(code=r['code'],status=r['status'],new_currently_received=items))
  for a in items[:2]:jobs.append((r['code'],a))
 b.write_new(dest/'notices/index_manifest.json',dict(formed_at=b.now(),real_signal_date='2026-10-19',search_end_cover_date='2026-10-20',reason='19日17:59已报道保隆异常公告，19封面日期窗口未返回，检查当晚已公开但封面日期20日的新增原文',current_real_receipt_not_future_retrieval=True,rows=rows,max_attempts_per_code=1))
 with ThreadPoolExecutor(max_workers=6) as ex:pdfs=list(ex.map(m.pdf_one,jobs))
 b.write_new(dest/'notices/fulltext_manifest.json',dict(formed_at=b.now(),rows=pdfs))
 result=dict(formed_at=b.now(),real_date='2026-10-19',delta=delta,pdfs=pdfs,original_predictions_changed=False,new_fits=0)
 b.write_new(dest/'delta_receipt.json',result);print(json.dumps(dict(formed_at=result['formed_at'],delta=delta,documents=[{k:r.get(k) for k in ['code','title','status','text_path']} for r in pdfs]),ensure_ascii=False))
if __name__=='__main__':main()
