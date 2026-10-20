"""Bounded cover-date20 check for the remaining complete first-board pool."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
OUT=Path(__file__).resolve().parent
def main():
 facts=b.read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json');already=b.read(OUT/'facts/evening_postdated20_currently_public/notices/index_manifest.json')
 covered={r['code'] for r in already['rows']};codes=sorted(r['code'] for r in facts['rows'] if r['kind']=='limit_up' and r['code'] not in covered)
 assert len(codes)==27
 dest=OUT/'facts/evening_remaining27_firstboards';(dest/'notices').mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('night_complete27',OUT/'facts/qualify_and_notices.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=dest;m.DAY='2026-10-20'
 old=b.read(OUT/'facts/notices/index_manifest.json')['rows']+b.read(OUT/'facts/evening_refresh_20261019T142830Z/notices/index_manifest.json')['rows'];known={}
 for r in old:known.setdefault(r['code'],set()).update(a['art_code'] for a in r.get('items',[]))
 with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(m.notices,codes))
 delta=[];jobs=[]
 for r in rows:
  items=[a for a in r.get('items',[]) if a['art_code'] not in known.get(r['code'],set())]
  delta.append(dict(code=r['code'],status=r['status'],newly_acquired_not_necessarily_newly_published=items))
  priority=[a for a in items if any(t in a['title'] for t in ['异常','风险','业绩','立案','停牌','复牌','合同','订单','减持','重组','股权变更'])]
  for a in priority[:2]:jobs.append((r['code'],a))
 b.write_new(dest/'notices/index_manifest.json',dict(formed_at=b.now(),remaining_codes=codes,real_signal_date='2026-10-19',search_end_cover_date='2026-10-20',rows=rows,max_attempts_per_code=1,priority='risk/earnings/event before generic governance documents'))
 with ThreadPoolExecutor(max_workers=6) as ex:pdfs=list(ex.map(m.pdf_one,jobs))
 b.write_new(dest/'notices/fulltext_manifest.json',dict(formed_at=b.now(),rows=pdfs))
 receipt=dict(formed_at=b.now(),rows=delta,documents=pdfs,complete39_indexes_attempted=True,original_predictions_unchanged=True,new_fits=0)
 b.write_new(dest/'receipt.json',receipt)
 print(json.dumps(dict(formed_at=receipt['formed_at'],remaining27_status=[(r['code'],r['status']) for r in rows],delta=delta,documents=[{k:r.get(k) for k in ['code','title','status','text_path']} for r in pdfs]),ensure_ascii=False))
if __name__=='__main__':main()
